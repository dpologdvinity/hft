"""Evaluate the pre-registered intraday momentum rule on cached 1-minute bars.

Implements docs/specs/2026-10-08-intraday-momentum-preregistration.md exactly:
buy at 15:30 ET when the 09:30-10:00 return is positive, sell at the 15:58 bar
close, net of the measured NBBO half-spread on both sides, 1 bp slippage per side
and $0.005 per share per side. The holdout split runs only after the development
report is committed, so the rule cannot be adjusted after seeing it.

    .venv/bin/python scripts/intraday_momentum.py --bars ~/git/stock-analyzer/.cache/bars \
        --split development --output artifacts/intraday-momentum/development.json
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.data import atomic_json
from hft.training import reserved_final_sessions

NEW_YORK = ZoneInfo("America/New_York")
HOLDOUT_START = date(2025, 10, 3)
REFERENCE_ETFS = frozenset({"SPY", "QQQ", "IWM", "DIA"})
RESULTS_DOC = "docs/intraday-momentum-results.md"
OPEN, SIGNAL_END, ENTRY, EXIT, LAST = 570, 599, 930, 958, 959  # minutes after midnight ET
SLIPPAGE_BPS, FEE_PER_SHARE = 1.0, 0.005
RESAMPLES, SEED = 10_000, 0
MINUTE_US = 60_000_000
DAY_US = 86_400_000_000


def local_minutes(stamps_us):
    """Local New York date ordinal and minute of day for UTC microsecond stamps."""
    utc_days = stamps_us // DAY_US
    offsets = {}
    for day in np.unique(utc_days):
        noon = datetime(1970, 1, 1, 12, tzinfo=UTC) + timedelta(days=int(day))
        offsets[day] = int(noon.astimezone(NEW_YORK).utcoffset().total_seconds()) * 1_000_000
    local = stamps_us + np.array([offsets[d] for d in utc_days], dtype=np.int64)
    return local // DAY_US, (local % DAY_US) // MINUTE_US


def symbol_days(path):
    """Per-day signal, last-half-hour return and entry price for one symbol."""
    table = pq.read_table(path, columns=["timestamp", "open", "close"])
    stamps = table.column("timestamp").cast("int64").to_numpy()
    opens = table.column("open").to_numpy()
    closes = table.column("close").to_numpy()
    days, minutes = local_minutes(stamps)

    def at(minute, values):
        mask = minutes == minute
        return dict(zip(days[mask].tolist(), values[mask].tolist(), strict=True))

    first, signal_end = at(OPEN, opens), at(SIGNAL_END, closes)
    entry, exit_, last = at(ENTRY, opens), at(EXIT, closes), at(LAST, closes)
    rows, previous_close = [], None
    for day in sorted(last):  # full sessions only: half-days have no 15:59 bar
        if all(day in m for m in (first, signal_end, entry, exit_)):
            rows.append(
                {
                    "date": date.fromordinal(date(1970, 1, 1).toordinal() + int(day)),
                    "r_first": signal_end[day] / first[day] - 1,
                    "r_overnight_first": (
                        None if previous_close is None else signal_end[day] / previous_close - 1
                    ),
                    "r_last": exit_[day] / entry[day] - 1,
                    "entry": entry[day],
                }
            )
        previous_close = last[day]
    return rows


def round_trip_cost_bps(half_spread_bps, price, multiple=1.0):
    return multiple * (2 * half_spread_bps + 2 * SLIPPAGE_BPS + 2 * FEE_PER_SHARE / price * 1e4)


def by_date(trades):
    """Equal-weight mean of each date's trades, in basis points, ordered by date."""
    grouped = {}
    for t in trades:
        grouped.setdefault(t["date"], []).append(t["value"])
    return np.array([np.mean(v) for _, v in sorted(grouped.items())])


def summary(trades, rng):
    values = by_date(trades)
    if not len(values):
        return {"dates": 0, "trades": 0}
    means = rng.choice(values, size=(RESAMPLES, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    raw = np.array([t["value"] for t in trades])
    return {
        "dates": len(values),
        "trades": len(raw),
        "mean_bps": float(values.mean()),
        "ci95_bps": [float(low), float(high)],
        "hit_rate": float((raw > 0).mean()),
    }


def evaluate(bars_dir, split):
    spreads = json.loads((bars_dir / "quoted_spreads.json").read_text())
    reserved = {(s, d) for s, d in reserved_final_sessions()}
    groups = {"stocks": {}, "etfs": {}}
    excluded = 0
    for path in sorted((bars_dir / "alpaca" / "1m").glob("*.parquet")):
        symbol = path.stem
        half_spread = spreads.get(f"{symbol}|close")
        if half_spread is None:
            continue
        for row in symbol_days(path):
            if (symbol, row["date"].isoformat()) in reserved:
                excluded += 1
                continue
            if (row["date"] >= HOLDOUT_START) != (split == "holdout"):
                continue
            row["symbol"], row["half_spread_bps"] = symbol, half_spread
            group = "etfs" if symbol in REFERENCE_ETFS else "stocks"
            groups[group].setdefault(symbol, []).append(row)
    return groups, excluded


def trades_for(rows, *, condition, cost_multiple):
    out = []
    for r in rows:
        if condition(r):
            cost = round_trip_cost_bps(r["half_spread_bps"], r["entry"], cost_multiple)
            out.append(
                {"date": r["date"], "symbol": r["symbol"], "value": r["r_last"] * 1e4 - cost}
            )
    return out


def report(groups, excluded, split):
    rng = np.random.default_rng(SEED)
    result = {"split": split, "reserved_sessions_excluded": excluded, "groups": {}}
    rules = {
        "momentum": lambda r: r["r_first"] > 0,
        "control_every_day": lambda r: True,
        "negative_first_half_hour": lambda r: r["r_first"] <= 0,
        "momentum_with_overnight": lambda r: (r["r_overnight_first"] or 0) > 0,
    }
    for group, symbols in groups.items():
        rows = [r for rs in symbols.values() for r in rs]
        dates = sorted({r["date"] for r in rows})
        out = {
            "symbols": sorted(symbols),
            "first_date": dates[0].isoformat() if dates else None,
            "last_date": dates[-1].isoformat() if dates else None,
        }
        for multiple in (1.0, 3.0):
            for name, condition in rules.items():
                trades = trades_for(rows, condition=condition, cost_multiple=multiple)
                out[f"{name}@{multiple:g}x_cost"] = summary(trades, rng)
        gross = trades_for(rows, condition=rules["momentum"], cost_multiple=0.0)
        out["momentum_gross"] = summary(gross, rng)
        out["momentum_by_symbol_bps"] = {
            s: round(
                float(
                    np.mean(
                        [
                            t["value"]
                            for t in trades_for(rs, condition=rules["momentum"], cost_multiple=1.0)
                        ]
                        or [np.nan]
                    )
                ),
                2,
            )
            for s, rs in sorted(symbols.items())
        }
        result["groups"][group] = out
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
    groups, excluded = evaluate(args.bars.expanduser(), args.split)
    result = report(groups, excluded, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    for group, out in result["groups"].items():
        print(f"{group}: {len(out['symbols'])} symbols, {out['first_date']} to {out['last_date']}")
        for key, value in out.items():
            if "@" in key or key == "momentum_gross":
                ci = value.get("ci95_bps", [float("nan")] * 2)
                print(
                    f"  {key:<38} dates {value['dates']:>4} trades {value['trades']:>6} "
                    f"mean {value.get('mean_bps', float('nan')):+7.2f} bp "
                    f"[{ci[0]:+6.2f}, {ci[1]:+6.2f}] hit {value.get('hit_rate', float('nan')):.2f}"
                )


if __name__ == "__main__":
    main()
