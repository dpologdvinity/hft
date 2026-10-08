"""Summarize a paper run's completed trades from its hash-chained journals.

Reads `logs/trade/<run>/<SYMBOL>-<start_ns>.jsonl` (every restart of the run writes
a new file), verifies each hash chain, and reports per-stock and per-day results
from the broker fills recorded there. Paper fills are simulated by the broker and
rule-strategy runs never count as graduation evidence.

    .venv/bin/python scripts/paper_report.py --run overnight
"""

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.logs import iter_log

NEW_YORK = ZoneInfo("America/New_York")


def load_trades(log_dir):
    trades, seen = [], set()
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        symbol = path.name.split("-")[0]
        for row in iter_log(path):  # raises on a broken hash chain
            if row.get("event") != "trade":
                continue
            key = (symbol, row["entry_ns"], row["exit_ns"])
            if key in seen:
                continue
            seen.add(key)
            trades.append(
                {
                    "symbol": symbol,
                    "entry_ns": row["entry_ns"],
                    "exit_ns": row["exit_ns"],
                    "pnl": Decimal(str(row["pnl"])),
                    "notional": Decimal(str(row["opening_notional"])),
                }
            )
    return trades


def exit_day(ns):
    return datetime.fromtimestamp(ns / 1e9, UTC).astimezone(NEW_YORK).date().isoformat()


def report(trades):
    if not trades:
        return "no completed trades yet"
    lines = ["stock   trades   net $    mean bp/trade   winners"]
    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t["symbol"]].append(t)
    for symbol, rows in sorted(by_symbol.items()):
        pnl = sum(r["pnl"] for r in rows)
        bp = sum(r["pnl"] / r["notional"] for r in rows) / len(rows) * 10_000
        wins = sum(r["pnl"] > 0 for r in rows)
        lines.append(f"{symbol:<7} {len(rows):>6} {pnl:>+8.2f} {bp:>+15.2f} {wins:>9}")
    lines.append("")
    lines.append("exit day     trades   net $   mean bp/trade")
    by_day = defaultdict(list)
    for t in trades:
        by_day[exit_day(t["exit_ns"])].append(t)
    for day, rows in sorted(by_day.items()):
        pnl = sum(r["pnl"] for r in rows)
        bp = sum(r["pnl"] / r["notional"] for r in rows) / len(rows) * 10_000
        lines.append(f"{day}  {len(rows):>6} {pnl:>+7.2f} {bp:>+15.2f}")
    total = sum(t["pnl"] for t in trades)
    days = len(by_day)
    lines.append("")
    lines.append(f"total {total:+.2f} over {len(trades)} trades on {days} exit days")
    lines.append("Broker-simulated paper fills; a few days cannot establish an edge.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run name under logs/trade/")
    parser.add_argument("--logs", type=Path, default=ROOT / "logs" / "trade")
    args = parser.parse_args()
    print(report(load_trades(args.logs / args.run)))


if __name__ == "__main__":
    main()
