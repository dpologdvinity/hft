import importlib.util
from pathlib import Path

from hft.logs import EventLog

NS = 1_000_000_000
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paper_report.py"


def _module():
    spec = importlib.util.spec_from_file_location("paper_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_counts_each_trade_once_across_restarts(tmp_path):
    entry, exit_ = 1_759_867_200 * NS, 1_759_930_200 * NS  # 2025-10-07 20:00, 13:30 next day UTC
    for start in (1, 2):  # a restart journals the same trade again only if it re-reads it
        log = EventLog(tmp_path / f"KO-{start}.jsonl")
        log.write("start", event_ns=start)
        log.write(
            "trade",
            event_ns=exit_,
            entry_ns=entry,
            exit_ns=exit_,
            pnl="1.50",
            opening_notional="500",
        )
        log.close()
    log = EventLog(tmp_path / "BAC-1.jsonl")
    log.write("start", event_ns=1)
    log.write(
        "trade", event_ns=exit_, entry_ns=entry, exit_ns=exit_, pnl="-0.50", opening_notional="500"
    )
    log.close()
    module = _module()
    trades = module.load_trades(tmp_path)
    assert len(trades) == 2
    text = module.report(trades)
    assert "KO" in text and "+30.00" in text  # 1.50 on 500 = 30 bp
    assert "2025-10-08" in text
    assert "total +1.00 over 2 trades on 1 exit days" in text
