"""Evaluate overnight drift with auction execution on 2016-2024 daily bars.

Implements docs/specs/2026-10-08-overnight-auction-preregistration.md exactly.
Downloads Alpaca daily SIP bars (split- and dividend-adjusted) once into
`data/daily-bars/`, then reports the close-to-next-open return net of a 1 bp
per-side auction allowance and $0.005 per share per side.

    .venv/bin/python scripts/overnight_auction.py --output artifacts/overnight-auction.json
"""

import argparse
import json
import sys
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.data import atomic_json
from hft.training import reserved_final_sessions

STOCKS = [
    "AAPL",
    "ABBV",
    "ADBE",
    "AMD",
    "AMZN",
    "AVGO",
    "BA",
    "BAC",
    "C",
    "CAT",
    "COST",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "F",
    "GE",
    "GOOGL",
    "GS",
    "HD",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "LLY",
    "MA",
    "META",
    "MRK",
    "MS",
    "MSFT",
    "MU",
    "NFLX",
    "NVDA",
    "ORCL",
    "OXY",
    "PEP",
    "PFE",
    "QCOM",
    "SLB",
    "TSLA",
    "TXN",
    "UNH",
    "V",
    "WFC",
    "WMT",
    "XOM",
]
ETFS = ["SPY", "QQQ", "IWM", "DIA"]
FIRST_NIGHT, LAST_NIGHT = date(2016, 1, 4), date(2024, 10, 1)
SIDE_BPS, FEE_PER_SHARE = 1.0, 0.005
RESAMPLES, SEED = 10_000, 0


def download(symbol, cache_dir):
    path = cache_dir / f"{symbol}.json"
    if path.exists():
        return json.loads(path.read_text())
    from hft.history import DATA_BASE, ReadOnlyClient

    client, bars, token = ReadOnlyClient(), [], None
    while True:
        params = {
            "timeframe": "1Day",
            "start": FIRST_NIGHT.isoformat(),
            "end": "2024-10-03",
            "adjustment": "all",
            "feed": "sip",
            "limit": 10000,
        }
        if token:
            params["page_token"] = token
        payload = client.get(f"{DATA_BASE}/v2/stocks/{symbol}/bars", params)
        bars += [
            {"date": b["t"][:10], "open": b["o"], "close": b["c"]}
            for b in payload.get("bars") or []
        ]
        token = payload.get("next_page_token")
        if not token:
            break
    atomic_json(path, bars)
    return bars


def nights(bars, symbol, reserved):
    """(night start date, close-to-open, open-to-close same day, close, next open) rows."""
    rows = []
    for today, tomorrow in pairwise(bars):
        start = date.fromisoformat(today["date"])
        if not FIRST_NIGHT <= start <= LAST_NIGHT:
            continue
        if {(symbol, today["date"]), (symbol, tomorrow["date"])} & reserved:
            continue
        rows.append((start, today, tomorrow))
    return rows


def cost_bps(price, multiple):
    return multiple * (SIDE_BPS + FEE_PER_SHARE / price * 1e4)


def series(data, symbols, multiple, leg):
    by_night = {}
    for symbol in symbols:
        for start, today, tomorrow in data[symbol]:
            if leg == "overnight":
                buy, sell = today["close"], tomorrow["open"]
            else:  # intraday control: the same session's open to close
                buy, sell = today["open"], today["close"]
            value = (sell / buy - 1) * 1e4 - cost_bps(buy, multiple) - cost_bps(sell, multiple)
            by_night.setdefault(start, []).append(value)
    return {d: float(np.mean(v)) for d, v in by_night.items()}


def summary(values, rng):
    values = np.array([v for _, v in sorted(values.items())])
    if not len(values):
        return {"nights": 0}
    means = rng.choice(values, size=(RESAMPLES, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "nights": len(values),
        "mean_bps": float(values.mean()),
        "ci95_bps": [float(low), float(high)],
        "hit_rate": float((values > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "daily-bars")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reserved = {(s, d) for s, d in reserved_final_sessions()}
    data = {}
    for symbol in STOCKS + ETFS:
        data[symbol] = nights(download(symbol, args.cache), symbol, reserved)
    rng = np.random.default_rng(SEED)
    result = {"first_night": FIRST_NIGHT.isoformat(), "last_night": LAST_NIGHT.isoformat()}
    for group, symbols in (("stocks", STOCKS), ("etfs", ETFS)):
        out = {}
        for multiple in (0.0, 1.0, 3.0):
            for leg in ("overnight", "intraday"):
                out[f"{leg}@{multiple:g}x_cost"] = summary(
                    series(data, symbols, multiple, leg), rng
                )
        yearly = series(data, symbols, 1.0, "overnight")
        out["overnight_net_by_year"] = {
            str(year): summary({d: v for d, v in yearly.items() if d.year == year}, rng)
            for year in sorted({d.year for d in yearly})
        }
        result[group] = out
    atomic_json(args.output, result)
    for group in ("stocks", "etfs"):
        print(group)
        for key, value in result[group].items():
            if key == "overnight_net_by_year":
                for year, s in value.items():
                    low, high = s["ci95_bps"]
                    print(
                        f"  net overnight {year}: n {s['nights']:>3} mean {s['mean_bps']:+6.2f} "
                        f"[{low:+6.2f}, {high:+6.2f}]"
                    )
                continue
            low, high = value["ci95_bps"]
            print(
                f"  {key:<22} n {value['nights']:>4} mean {value['mean_bps']:+7.2f} bp "
                f"95% [{low:+7.2f}, {high:+7.2f}] hit {value['hit_rate']:.2f}"
            )


if __name__ == "__main__":
    main()
