"""Evaluate the pre-registered analyst-upgrade day trade.

Implements docs/specs/2026-10-09-analyst-upgrade-preregistration.md: buy at 09:31 the
symbols with net analyst-up events in their pre-open news, sell five minutes before
the close, with the ML day trader's fill simulator and costs. Minute data is loaded
one symbol at a time to keep memory small. The test split runs only with --final,
once, after the validation report is committed.

    .venv/bin/python scripts/analyst_upgrade.py --split validation
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.ml.datasets import split_mask
from hft.ml.download import load_calendar
from hft.ml.features import daily_frame, minute_store
from hft.ml.fills import SLIPPAGE_BPS, half_spread_bps, simulate_day
from hft.ml.news import read_news
from hft.ml.news_features import NEWS_FEATURES, news_features
from hft.ml.train import _costs, _holding
from hft.ml.universe import UNIVERSE

RESULTS_DOC = "docs/analyst-upgrade-results.md"
REGISTRY = ROOT / ".state" / "analyst-upgrade-final.json"
SLOTS = 5
RESAMPLES, SEED = 10_000, 0
ANALYST = NEWS_FEATURES.index("news_analyst_net")


def clustered_interval(dates, values, rng):
    """95% interval of the mean per trade, resampling whole dates."""
    dates, values = np.asarray(dates), np.asarray(values, float)
    if not len(values):
        return [0.0, 0.0]
    unique, inverse = np.unique(dates, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    picks = rng.integers(0, len(unique), size=(RESAMPLES, len(unique)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def summary(dates, values, rng):
    values = np.asarray(values, float)
    if not len(values):
        return {"trades": 0}
    low, high = clustered_interval(dates, values, rng)
    return {
        "trades": len(values),
        "mean_bp": float(values.mean() * 1e4),
        "ci95_bp": [low * 1e4, high * 1e4],
        "hit_rate": float((values > 0).mean()),
        "passes": bool(low > 0),
    }


def evaluate(root, split):
    symbols = [s.ticker for s in UNIVERSE if not s.trade_only]
    calendar = load_calendar(root)
    frame = daily_frame(root, symbols)
    news = news_features(read_news(root), symbols, frame.dates, calendar)[..., ANALYST]
    days = split_mask(frame.dates, split)
    up = (news > 0) & frame.valid & days[:, None]
    down = (news < 0) & frame.valid & days[:, None]
    costs = _costs(symbols)
    trades = {name: [] for name in ("long", "long_3x", "short_mirror")}
    picks_by_day = {}
    for s, symbol in enumerate(symbols):
        wanted = np.flatnonzero(up[:, s] | down[:, s])
        if not len(wanted):
            continue
        data = minute_store(root, [symbol], calendar)[symbol]
        rows = {d: r for r, d in enumerate(data.dates)}
        for d in wanted:
            row = rows.get(frame.dates[d])
            if row is None:
                continue
            prices = data.open[row][data.valid[row]]
            spread = half_spread_bps(costs[s], float(prices[0])) if len(prices) else costs[s]
            hold = np.full(390, np.inf)
            last = int(data.last_minute[row])
            if up[d, s]:
                for name, multiple in (("long", 1.0), ("long_3x", 3.0)):
                    trade = simulate_day(
                        data.open[row],
                        data.valid[row],
                        hold,
                        enter=0.0,
                        half_spread_bps=spread,
                        last_minute=last,
                        cost_multiple=multiple,
                    )
                    if trade is not None:
                        trades[name].append((frame.dates[d], trade.net_return))
                        if name == "long":
                            picks_by_day.setdefault(d, []).append(
                                (-news[d, s], s, trade.net_return)
                            )
            else:
                gross = simulate_day(
                    data.open[row],
                    data.valid[row],
                    hold,
                    enter=0.0,
                    half_spread_bps=0.0,
                    last_minute=last,
                    cost_multiple=0.0,
                )
                if gross is not None:
                    cost = 2 * (spread + SLIPPAGE_BPS) / 1e4
                    trades["short_mirror"].append((frame.dates[d], -gross.net_return - cost))
    rng = np.random.default_rng(SEED)
    result = {"split": split, "symbols": len(symbols)}
    for name, rows in trades.items():
        dates = [d for d, _ in rows]
        result[name] = summary(dates, [v for _, v in rows], rng)
    score_rows = np.flatnonzero(days)
    portfolio = []
    for d in score_rows:
        chosen = sorted(picks_by_day.get(d, []))[:SLOTS]
        portfolio.append(sum(v for _, _, v in chosen) / SLOTS)
    holding = _holding(root, symbols, frame, days, np.ones(len(symbols), bool))
    wealth = float(np.prod(1 + np.asarray(portfolio)) - 1)
    result["portfolio_total"] = wealth
    result["holding_total"] = float(np.prod(1 + holding) - 1)
    years = len(score_rows) / 252
    result["trades_per_year"] = result["long"].get("trades", 0) / max(years, 1e-9)
    result["first_day"], result["last_day"] = (
        str(frame.dates[score_rows[0]]),
        str(frame.dates[score_rows[-1]]),
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ml")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.split == "test":
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", RESULTS_DOC],
                cwd=ROOT,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
        if not tracked:
            parser.error(f"commit the validation report ({RESULTS_DOC}) before the test")
        if REGISTRY.exists():
            parser.error(f"the test split was already used: {REGISTRY.read_text()}")
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY.write_text(json.dumps({"started": datetime.now(UTC).isoformat()}) + "\n")
    result = evaluate(args.data, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
