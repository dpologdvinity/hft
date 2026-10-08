"""Evaluate the pre-registered overnight drift and weekly reversal rules.

Implements docs/specs/2026-10-08-multiday-preregistration.md exactly, on
candlebench's cached 1-minute bars. The holdout split runs only after the
development report is committed, so neither rule can be adjusted after seeing it.

    .venv/bin/python scripts/multiday_rules.py --bars ~/git/stock-analyzer/.cache/bars \
        --split development --output artifacts/multiday/development.json
"""

import argparse
import json
import subprocess
import sys
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intraday_momentum import local_minutes

from hft.data import atomic_json
from hft.training import reserved_final_sessions

HOLDOUT_START = date(2025, 10, 3)
REFERENCE_ETFS = frozenset({"SPY", "QQQ", "IWM", "DIA"})
RESULTS_DOC = "docs/multiday-results.md"
OPEN, ENTRY, LAST = 570, 958, 959  # minutes after midnight ET: 09:30, 15:58, 15:59
HOLD_SESSIONS, LOSERS = 5, 5
SLIPPAGE_BPS, FEE_PER_SHARE = 1.0, 0.005
RESAMPLES, SEED = 10_000, 0
EPOCH = date(1970, 1, 1).toordinal()


def symbol_sessions(path):
    """{date: (09:30 open, 15:58 close)} for usable sessions, and every trading date."""
    table = pq.read_table(path, columns=["timestamp", "open", "close"])
    stamps = table.column("timestamp").cast("int64").to_numpy()
    opens, closes = table.column("open").to_numpy(), table.column("close").to_numpy()
    days, minutes = local_minutes(stamps)

    def at(minute, values):
        mask = minutes == minute
        return dict(zip(days[mask].tolist(), values[mask].tolist(), strict=True))

    first, entry, last = at(OPEN, opens), at(ENTRY, closes), at(LAST, closes)
    usable = {
        date.fromordinal(EPOCH + int(d)): (first[d], entry[d])
        for d in last
        if d in first and d in entry
    }
    trading = {date.fromordinal(EPOCH + int(d)) for d in np.unique(days)}
    return usable, trading


def side_cost_bps(half_spread_bps, price, multiple):
    return multiple * (half_spread_bps + SLIPPAGE_BPS + FEE_PER_SHARE / price * 1e4)


def interval(values, rng, low, high):
    means = rng.choice(values, size=(RESAMPLES, len(values))).mean(axis=1)
    return [float(v) for v in np.percentile(means, [low, high])]


def summary(periods, rng):
    """`periods`: [(start date, value in bp)]; equal weight across periods."""
    if not periods:
        return {"periods": 0}
    values = np.array([v for _, v in sorted(periods)])
    return {
        "periods": len(values),
        "mean_bps": float(values.mean()),
        "ci95_bps": interval(values, rng, 2.5, 97.5),
        "ci97_5_bonferroni_bps": interval(values, rng, 1.25, 98.75),
        "hit_rate": float((values > 0).mean()),
    }


def load(bars_dir):
    spreads = json.loads((bars_dir / "quoted_spreads.json").read_text())
    reserved = {(s, d) for s, d in reserved_final_sessions()}
    data, calendar, excluded = {}, set(), 0
    for path in sorted((bars_dir / "alpaca" / "1m").glob("*.parquet")):
        symbol = path.stem
        if f"{symbol}|open" not in spreads or f"{symbol}|close" not in spreads:
            continue
        usable, trading = symbol_sessions(path)
        for day in [d for d in usable if (symbol, d.isoformat()) in reserved]:
            del usable[day]
            excluded += 1
        data[symbol] = usable
        calendar |= trading
    return data, sorted(calendar), spreads, excluded


def in_split(day, split):
    return (day >= HOLDOUT_START) == (split == "holdout")


def overnight(data, calendar, spreads, symbols, split, multiple):
    rule, control = {}, {}
    for symbol in symbols:
        usable, hs_open, hs_close = (
            data[symbol],
            spreads[f"{symbol}|open"],
            spreads[f"{symbol}|close"],
        )
        for today, tomorrow in pairwise(calendar):
            if not in_split(today, split) or today not in usable or tomorrow not in usable:
                continue
            (open_, close), (next_open, _) = usable[today], usable[tomorrow]
            night = (next_open / close - 1) * 1e4 - side_cost_bps(hs_close, close, multiple)
            night -= side_cost_bps(hs_open, next_open, multiple)
            day = (close / open_ - 1) * 1e4 - side_cost_bps(hs_open, open_, multiple)
            day -= side_cost_bps(hs_close, close, multiple)
            rule.setdefault(today, []).append(night)
            control.setdefault(today, []).append(day)
    return (
        [(d, float(np.mean(v))) for d, v in rule.items()],
        [(d, float(np.mean(v))) for d, v in control.items()],
    )


def weekly_reversal(data, calendar, spreads, symbols, split, multiple):
    rule, control, excess = [], [], []
    for i in range(HOLD_SESSIONS, len(calendar) - HOLD_SESSIONS, HOLD_SESSIONS):
        before, start, end = calendar[i - HOLD_SESSIONS], calendar[i], calendar[i + HOLD_SESSIONS]
        if not in_split(start, split):
            continue
        ranked, returns = [], {}
        for symbol in symbols:
            usable = data[symbol]
            if not all(d in usable for d in (before, start, end)):
                continue
            entry, exit_ = usable[start][1], usable[end][1]
            cost = side_cost_bps(spreads[f"{symbol}|close"], entry, multiple)
            cost += side_cost_bps(spreads[f"{symbol}|close"], exit_, multiple)
            returns[symbol] = (exit_ / entry - 1) * 1e4 - cost
            ranked.append((entry / usable[before][1] - 1, symbol))
        if len(ranked) < LOSERS:
            continue
        losers = [s for _, s in sorted(ranked)[:LOSERS]]
        held = float(np.mean([returns[s] for s in losers]))
        everyone = float(np.mean(list(returns.values())))
        rule.append((start, held))
        control.append((start, everyone))
        excess.append((start, held - everyone))
    return rule, control, excess


def report(data, calendar, spreads, excluded, split):
    rng = np.random.default_rng(SEED)
    stocks = sorted(s for s in data if s not in REFERENCE_ETFS)
    etfs = sorted(s for s in data if s in REFERENCE_ETFS)
    result = {
        "split": split,
        "reserved_sessions_excluded": excluded,
        "stocks": stocks,
        "etfs": etfs,
        "rules": {},
    }
    for multiple in (0.0, 1.0, 3.0):
        tag = f"@{multiple:g}x_cost"
        night, day = overnight(data, calendar, spreads, stocks, split, multiple)
        result["rules"]["overnight" + tag] = summary(night, rng)
        result["rules"]["control_intraday" + tag] = summary(day, rng)
        etf_night, _ = overnight(data, calendar, spreads, etfs, split, multiple)
        result["rules"]["etf_overnight" + tag] = summary(etf_night, rng)
        held, universe, excess = weekly_reversal(data, calendar, spreads, stocks, split, multiple)
        result["rules"]["weekly_reversal" + tag] = summary(held, rng)
        result["rules"]["control_universe_week" + tag] = summary(universe, rng)
        result["rules"]["weekly_reversal_excess" + tag] = summary(excess, rng)
    starts = [d for d, _ in overnight(data, calendar, spreads, stocks, split, 1.0)[0]]
    result["first_period"] = min(starts).isoformat() if starts else None
    result["last_period"] = max(starts).isoformat() if starts else None
    return result


def holdout_allowed():
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", RESULTS_DOC],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return tracked.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars", type=Path, required=True, help="candlebench .cache/bars")
    parser.add_argument("--split", choices=["development", "holdout"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "holdout" and not holdout_allowed():
        parser.error(f"commit the development report ({RESULTS_DOC}) before the holdout")
    data, calendar, spreads, excluded = load(args.bars.expanduser())
    result = report(data, calendar, spreads, excluded, args.split)
    atomic_json(args.output, result)
    print(
        f"{args.split}: {len(result['stocks'])} stocks, periods {result['first_period']} "
        f"to {result['last_period']}, reserved excluded {excluded}"
    )
    for name, value in result["rules"].items():
        if not value["periods"]:
            print(f"  {name:<36} no periods")
            continue
        low, high = value["ci95_bps"]
        b_low, b_high = value["ci97_5_bonferroni_bps"]
        print(
            f"  {name:<36} n {value['periods']:>4} mean {value['mean_bps']:+8.2f} bp "
            f"95% [{low:+7.2f}, {high:+7.2f}] 97.5% [{b_low:+7.2f}, {b_high:+7.2f}] "
            f"hit {value['hit_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
