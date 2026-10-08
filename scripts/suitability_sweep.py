"""Screen many stocks for free-IEX data suitability on declared development dates.

For every (symbol, session) this downloads with the existing read-only acquirer,
then reports the unchanged five-second diagnostic (feed gaps and warmup-eligible
decisions, the same traversal as `hft data-quality`) plus event-gap counts at
coarser thresholds and the median quoted spread. Nothing is fitted or traded.

Reserved final-test dates are refused before any request: every frozen
experiment's final_test list and the global reservation registry are checked.

    .venv/bin/python scripts/suitability_sweep.py --symbols NVDA TSLA AMD \
        --dates 2026-06-04 2026-06-08 --cache data/sweep --report artifacts/sweep/report.json
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.cli import _read_credentials
from hft.data import NS, atomic_json, load_session
from hft.execution import Simulation
from hft.history import download_sessions
from hft.research import is_real_executable
from hft.training import reserved_final_sessions

GAP_THRESHOLDS_SECONDS = (5, 15, 60)


def measure(manifest_path):
    session = load_session(manifest_path, metadata="execution")
    stamps = np.union1d(session.quote_ns, session.trade_ns)
    edges = np.concatenate(([session.open_ns], stamps, [session.close_ns]))
    gaps = np.diff(edges)
    mid = (session.bid + session.ask) / 2
    spread_bps = np.median((session.ask - session.bid) / mid * 1e4) if len(mid) else None
    row = {
        "quotes": len(session.quote_ns),
        "trades": len(session.trade_ns),
        "real_executable": is_real_executable(session),
        "max_event_gap_seconds": float(gaps.max() / NS),
        "event_gaps_over": {str(s): int((gaps > s * NS).sum()) for s in GAP_THRESHOLDS_SECONDS},
        "median_spread_bps": None if spread_bps is None else float(spread_bps),
        "median_mid": float(np.median(mid)) if len(mid) else None,
        "eligible_decisions": 0,
        "ineligible_decisions": 0,
        "feed_gaps": None,
    }
    try:
        simulation = Simulation(session)
    except ValueError as exc:
        if str(exc) != "insufficient session warmup":
            raise
        row["reasons"] = ["insufficient-decision-warmup"]
        return row
    for bar in simulation.bars[61:]:
        key = "eligible_decisions" if simulation.decision_eligible else "ineligible_decisions"
        row[key] += 1
        simulation.advance_to(bar.end_ns)
    row["feed_gaps"] = simulation.gap_count
    total = row["eligible_decisions"] + row["ineligible_decisions"]
    row["eligible_fraction"] = row["eligible_decisions"] / total if total else None
    row["passes_five_second_contract"] = (
        row["real_executable"] and row["feed_gaps"] == 0 and row["eligible_decisions"] > 0
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--dates", nargs="+", required=True, help="exact development dates")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    symbols = [s.upper() for s in args.symbols]
    dates = sorted(args.dates)
    conflicts = {(s, d) for s in symbols for d in dates} & reserved_final_sessions()
    if conflicts:
        raise SystemExit(f"refusing reserved final-test sessions: {sorted(conflicts)}")
    report = json.loads(args.report.read_text()) if args.report.exists() else {"rows": []}
    report.update(symbols=symbols, dates=dates, feed="iex", fitted=False, paper_eligible=False)
    done = {(r["symbol"], r["date"]) for r in report["rows"]}
    _read_credentials()
    for symbol in symbols:
        if all((symbol, d) in done for d in dates):
            continue
        started = time.monotonic()
        try:
            index = download_sessions(symbol, dates[0], dates[-1], args.cache / symbol.lower())
        except Exception as exc:  # noqa: BLE001 - record and continue the sweep
            report["rows"].append({"symbol": symbol, "date": None, "error": repr(exc)})
            atomic_json(args.report, report)
            print(f"{symbol}: acquisition failed: {exc!r}", flush=True)
            continue
        acquired = time.monotonic() - started
        for entry in json.loads(index.read_text())["sessions"]:
            if entry["session_id"] not in dates or (symbol, entry["session_id"]) in done:
                continue
            begin = time.monotonic()
            row = measure(index.parent / entry["manifest"])
            row.update(
                symbol=symbol,
                date=entry["session_id"],
                acquisition_seconds=acquired,
                measure_seconds=time.monotonic() - begin,
            )
            report["rows"].append(row)
            atomic_json(args.report, report)
            print(
                f"{symbol} {row['date']}: {row['quotes']:>9} quotes, gaps>5s "
                f"{row['event_gaps_over']['5']:>5}, >60s {row['event_gaps_over']['60']:>4}, "
                f"spread {row['median_spread_bps']:.2f}bps, "
                f"pass5s={row.get('passes_five_second_contract')}",
                flush=True,
            )
            gc.collect()


if __name__ == "__main__":
    main()
