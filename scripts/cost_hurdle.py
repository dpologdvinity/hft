"""Compare typical price moves at several horizons with the round-trip trading cost.

Development sessions only. For each session this samples the IEX mid-price on a
regular grid and reports the standard deviation of forward mid returns per
horizon, next to the simulator's round-trip cost: the median quoted spread, two
slippage charges and two per-share commissions (hft.account.Costs defaults).

A directional signal with information coefficient IC earns roughly
IC * sigma_h * sqrt(2/pi) gross per trade, so `sigma_h / cost` tells you how
strong a predictor must be at that horizon just to break even:
breakeven IC ~= cost / (sigma_h * 0.8).

    .venv/bin/python scripts/cost_hurdle.py --report artifacts/sweep/report.json \
        --cache data/sweep --output artifacts/sweep/hurdle.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.account import Costs
from hft.data import NS, atomic_json, load_session

HORIZONS_SECONDS = (5, 60, 300, 1800)
GRID_SECONDS = 5


def hurdle(session, costs):
    stamps, mid = session.quote_ns, (session.bid + session.ask) / 2
    grid = np.arange(session.open_ns, session.close_ns, GRID_SECONDS * NS)
    index = np.searchsorted(stamps, grid, side="right") - 1
    valid = index >= 0
    grid, prices = grid[valid], mid[index[valid]]
    spread = (session.ask - session.bid) / mid * 1e4
    price = float(np.median(mid))
    commission_bps = 2 * float(costs.commission) / price * 1e4
    cost = float(np.median(spread)) + 2 * float(costs.slippage_bps) + commission_bps
    row = {"median_mid": price, "round_trip_cost_bps": cost, "horizons": {}}
    for horizon in HORIZONS_SECONDS:
        step = horizon // GRID_SECONDS
        if len(prices) <= step:
            continue
        moves = np.log(prices[step:] / prices[:-step]) * 1e4
        sigma = float(np.std(moves))
        row["horizons"][str(horizon)] = {
            "sigma_bps": sigma,
            "mean_abs_bps": float(np.mean(np.abs(moves))),
            "breakeven_ic": cost / (sigma * np.sqrt(2 / np.pi)) if sigma else None,
        }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, nargs="+", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    costs = Costs()
    rows = []
    for report in args.report:
        for entry in json.loads(report.read_text())["rows"]:
            if entry.get("date") is None:
                continue
            symbol, date = entry["symbol"], entry["date"]
            manifest = args.cache / symbol.lower() / symbol / date / "manifest.json"
            row = hurdle(load_session(manifest, metadata="execution"), costs)
            row.update(symbol=symbol, date=date)
            rows.append(row)
            h = row["horizons"]
            print(
                f"{symbol:5} {date} cost {row['round_trip_cost_bps']:7.2f}bps  "
                + "  ".join(
                    f"{k}s sigma {v['sigma_bps']:6.1f} IC* {v['breakeven_ic']:.2f}"
                    for k, v in h.items()
                ),
                flush=True,
            )
    atomic_json(args.output, {"horizons_seconds": HORIZONS_SECONDS, "rows": rows})


if __name__ == "__main__":
    main()
