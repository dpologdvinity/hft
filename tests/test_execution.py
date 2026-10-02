from decimal import Decimal

from hft.account import Costs
from hft.calendar import SessionWindow
from hft.data import NS, synthetic_sessions
from hft.logs import read_log
from hft.paper import PaperEngine, replay_sessions


def test_replay_real_quote_fills_and_synthetic_provenance(tmp_path):
    sessions = synthetic_sessions(days=2, bars_per_day=100)
    engine = PaperEngine(
        lambda obs: 1,
        log_path=tmp_path / "run.jsonl",
        initial_cash=500,
        costs=Costs(0, 0),
        symbol=sessions[0].symbol,
        synthetic=True,
        source="replay",
    )
    replay_sessions(engine, sessions)
    rows = read_log(tmp_path / "run.jsonl")
    assert any(r["event"] == "fill" for r in rows)
    assert any(r["event"] == "decision" for r in rows)
    assert engine.account.position == 0
    assert rows[0]["synthetic"] is True
    assert rows[-1]["complete"] is False  # replay never becomes wall-clock evidence


def test_finish_never_invents_exit_and_records_inventory(tmp_path):
    engine = PaperEngine(lambda obs: 1, log_path=tmp_path / "run.jsonl", symbol="AAPL")
    from hft.account import Execution

    engine.account.apply(Execution("id", "order", 1, Decimal(".1"), Decimal(100), Decimal(0)))
    engine.finish()
    assert engine.account.position == Decimal(".1")
    assert read_log(tmp_path / "run.jsonl")[-1]["position"] == "0.1"


def test_timer_completes_bar_without_next_trade(tmp_path):
    engine = PaperEngine(lambda obs: 0, log_path=tmp_path / "run.jsonl", symbol="AAPL")
    start = 1_735_830_000 * NS
    engine.start_session(SessionWindow("2025-01-02", start, start + 400 * NS))
    engine.on_event(
        dict(T="q", S="AAPL", event_ns=start + NS, bp=100, ap=100, bs=1, **{"as": 1}), start + NS
    )
    engine.on_event(
        {"T": "t", "S": "AAPL", "event_ns": start + 2 * NS, "p": 100, "s": 2}, start + 2 * NS
    )
    engine.tick(start + 5 * NS)
    assert len(engine.history) == 1
    engine.finish()
