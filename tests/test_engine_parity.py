"""The C++ engine must reproduce the Python reference exactly."""

import importlib
import os

import numpy as np
import pytest
from engine_vectors import OPEN, SESSION, engine_vectors, quote, trade

from hft.data import synthetic_sessions
from hft.feed import BarAggregator, historical_timeline

FIELDS = [
    "start_ns",
    "end_ns",
    "quote_ns",
    "session_id",
    "tradable",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "vwap",
]


@pytest.fixture(scope="module")
def hftcore():
    try:
        return importlib.import_module("hftcore")
    except ImportError:
        if os.environ.get("HFT_REQUIRE_HFTCORE") == "1":
            raise
        pytest.skip("hftcore is not installed")


def _rows(bars):
    return [tuple(getattr(b, f) for f in FIELDS) for b in bars]


def _run_both(hftcore, session, symbol, timeline):
    python = BarAggregator(symbol, 5, session)
    native = hftcore.BarAggregator(symbol, 5, session.session_id, session.open_ns, session.close_ns)
    for step, (event, now) in enumerate(timeline):
        if event is None:
            expected, actual = _rows(python.advance_to(now)), native.advance_to(now)
        else:
            expected, actual = _rows(python.add(event, now)), native.add(event, now)
        assert [tuple(r) for r in actual] == expected, f"step {step}"
    assert native.quality() == dict(python.quality)


@pytest.mark.parametrize("vector", engine_vectors(), ids=lambda v: v[0])
def test_aggregator_matches_reference_on_vectors(hftcore, vector):
    _, session, symbol, timeline = vector
    _run_both(hftcore, session, symbol, timeline)


@pytest.mark.parametrize("seed", [1, 2])
def test_aggregator_matches_reference_on_sessions(hftcore, seed):
    session = synthetic_sessions(1, 200, seed=seed)[0]
    from hft.calendar import SessionWindow

    window = SessionWindow(session.session_id, session.open_ns, session.close_ns)
    _run_both(hftcore, window, session.symbol, list(historical_timeline(session)))


@pytest.mark.parametrize(
    "event",
    [
        {"T": "t", "S": "X", "event_ns": OPEN, "p": -1.0, "s": 1.0},
        {"T": "t", "S": "X", "event_ns": True, "p": 1.0, "s": 1.0},
        {"T": "q", "S": "X", "t": "2025-01-06 14:30:00Z", "bp": 1, "ap": 2, "bs": 1, "as": 1},
        {"T": "q", "S": "X", "t": 5, "bp": 1, "ap": 2, "bs": 1, "as": 1},
    ],
)
def test_invalid_events_raise_value_error_in_both(hftcore, event):
    python = BarAggregator("X", 5, SESSION)
    native = hftcore.BarAggregator("X", 5, SESSION.session_id, SESSION.open_ns, SESSION.close_ns)
    with pytest.raises(ValueError):
        python.add(event, OPEN)
    with pytest.raises(ValueError):
        native.add(event, OPEN)


def test_negative_arrival_and_float_clock_raise(hftcore):
    native = hftcore.BarAggregator("X", 5, SESSION.session_id, SESSION.open_ns, SESSION.close_ns)
    with pytest.raises(ValueError, match="arrival"):
        native.add(quote(OPEN)[0], -1)
    with pytest.raises(ValueError, match="integer"):
        native.advance_to(1.5)
    assert native.add(trade(OPEN)[0], OPEN) == []


def _engine_run(engine, session, timeline, now_ns=None):
    engine.start_session(session, now_ns)
    out = []
    for event, now in timeline:
        updates = engine.advance_to(now) if event is None else engine.on_event(event, now)
        for u in updates:
            if hasattr(u, "market"):
                market = None if u.market is None else u.market.view(np.uint32).tolist()
                out.append((u.bar, u.now_ns, u.accepted, u.reset, u.ready, market, u.action))
            else:
                out.append(u)
        out.append(("history", tuple(engine.history)))
    return out, dict(engine.quality), engine.warmup_after_ns


def _engines(hftcore, symbol, strategy):
    from hft.market_engine import PyMarketEngine
    from hft.market_engine_cpp import CppMarketEngine

    return PyMarketEngine(symbol, strategy), CppMarketEngine(symbol, strategy)


@pytest.mark.parametrize("strategy", ["model", "ema-crossover", "hold-day"])
@pytest.mark.parametrize("vector", engine_vectors(), ids=lambda v: v[0])
def test_market_engine_matches_reference_on_vectors(hftcore, vector, strategy):
    _, session, symbol, timeline = vector
    python, native = _engines(hftcore, symbol, strategy)
    assert _engine_run(native, session, timeline) == _engine_run(python, session, timeline)


@pytest.mark.parametrize("strategy", ["model", "ema-crossover", "hold-day"])
def test_market_engine_matches_reference_on_sessions(hftcore, strategy):
    from hft.calendar import SessionWindow

    data = synthetic_sessions(1, 200, seed=4)[0]
    window = SessionWindow(data.session_id, data.open_ns, data.close_ns)
    timeline = list(historical_timeline(data))
    python, native = _engines(hftcore, data.symbol, strategy)
    expected = _engine_run(python, window, timeline)
    assert sum(1 for row in expected[0] if len(row) == 7 and row[4]) > 50  # many decisions
    assert _engine_run(native, window, timeline) == expected
    late = data.open_ns + 300 * 1_000_000_000 + 7
    assert _engine_run(native, window, timeline, late) == _engine_run(
        python, window, timeline, late
    )


@pytest.mark.parametrize("strategy", ["model", "ema-crossover"])
def test_mark_gap_matches_reference(hftcore, strategy):
    from hft.calendar import SessionWindow

    data = synthetic_sessions(1, 200, seed=6)[0]
    window = SessionWindow(data.session_id, data.open_ns, data.close_ns)
    timeline = list(historical_timeline(data))
    gap_at = timeline[len(timeline) // 2][1]
    marked = [
        *timeline[: len(timeline) // 2],
        ("mark_gap", gap_at),
        *timeline[len(timeline) // 2 :],
    ]

    def run(engine):
        engine.start_session(window)
        rows = []
        for event, now in marked:
            if event == "mark_gap":
                engine.mark_gap(now)
                updates = []
            elif event is None:
                updates = engine.advance_to(now)
            else:
                updates = engine.on_event(event, now)
            rows += [(getattr(u, "bar", u), getattr(u, "action", None)) for u in updates]
            rows.append((tuple(engine.history), engine.warmup_after_ns))
        return rows

    python, native = _engines(hftcore, data.symbol, strategy)
    assert run(native) == run(python)
