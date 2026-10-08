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


def _frames():
    """Alpaca-format frames: quotes in lots, int trade ids, conditions, noise."""
    from datetime import UTC, datetime

    def stamp(ns, offset=False):
        text = datetime.fromtimestamp(ns // 10**9, UTC).strftime("%Y-%m-%dT%H:%M:%S")
        text += f".{ns % 10**9:09d}"
        if offset:
            hours = datetime.fromtimestamp(ns // 10**9 - 5 * 3600, UTC)
            return hours.strftime("%Y-%m-%dT%H:%M:%S") + f".{ns % 10**9:09d}-05:00"
        return text + "Z"

    # (frame, arrival_ns): arrivals trail event times by 50 us.
    frames = [
        ([{"T": "success", "msg": "authenticated"}], OPEN - 10**9),
        ([{"T": "subscription", "quotes": []}], OPEN - 10**9),
    ]
    for k in range(2000):
        t = OPEN + k * 400_000_000
        price = round(100 + 0.003 * (k % 211) - 0.002 * (k % 37), 2)
        frame = [
            {
                "T": "q",
                "S": "AAA",
                "bp": price - 0.01,
                "ap": price + 0.01,
                "bs": [3, 0.07, 1.5][k % 3],
                "as": 2,
                "t": stamp(t, k % 5 == 0),
                "z": "C",
            },
            {"T": "q", "S": "ZZZ", "bp": 1, "ap": 2, "bs": 1, "as": 1, "t": stamp(t)},
        ]
        if k % 4 == 0:
            frame.append(
                {
                    "T": "t",
                    "S": "AAA",
                    "i": k,
                    "p": price,
                    "s": 10 + k % 7,
                    "t": stamp(t + 1),
                    "c": ["@", "W"][: 1 + (k % 9 == 0)],
                    "z": "C",
                }
            )
        if k % 50 == 25:
            frame.append({"T": "c", "S": "AAA", "t": stamp(t + 2)})
        if k % 97 == 0:
            frame.append(7)  # malformed entry
        frames.append((frame, t + 50_000))
    frames.insert(10, ("not a list", OPEN + 3 * 400_000_000))
    return frames


def test_frame_router_matches_the_python_stream_path(hftcore):
    import json

    from hft.calendar import SessionWindow
    from hft.market_engine_cpp import CppMarketEngine

    window = SessionWindow("2025-01-06", OPEN, OPEN + 3600 * 10**9)
    python, native = _engines(hftcore, "AAA", "ema-crossover")
    python.start_session(window)
    native.start_session(window)
    router = hftcore.FrameRouter()
    router.add(native._native)
    assert isinstance(native, CppMarketEngine)
    expected, actual, malformed = [], [], 0
    for frame, arrival in _frames():
        raw = json.dumps(frame).encode()
        try:
            messages = json.loads(raw)
            assert isinstance(messages, list)
        except AssertionError:
            malformed += 1
            messages = []
        for message in messages:
            if not isinstance(message, dict):
                malformed += 1
                continue
            if message.get("T") in ("q", "t", "c", "x") and message.get("S") == "AAA":
                for u in python.on_event({**message, "arrival_ns": arrival}, arrival):
                    expected.append(_comparable(u))
        updates, quotes, bad = router.on_frame(raw, arrival)
        malformed -= bad
        for symbol, row in updates:
            assert symbol == "AAA"
            actual += [_comparable(u) for u in native._convert([row])]
        for _, event_ns, bid, ask, bid_size, ask_size in quotes:
            from hft.feed import parse_timestamp, quote_from_event

            source = next(
                m
                for m in messages
                if isinstance(m, dict) and m.get("T") == "q" and m.get("S") == "AAA"
            )
            reference = quote_from_event(source, arrival)
            assert event_ns == parse_timestamp(source["t"])
            assert (bid, ask, bid_size, ask_size) == (
                float(reference.bid),
                float(reference.ask),
                float(reference.bid_size),
                float(reference.ask_size),
            )
    assert malformed == 0
    assert actual == expected
    assert sum(1 for row in expected if row[0] == "bar" and row[5]) > 50  # decisions


def _comparable(update):
    if hasattr(update, "bar"):
        market = None if update.market is None else update.market.view(np.uint32).tolist()
        return (
            "bar",
            update.bar,
            update.now_ns,
            update.accepted,
            update.reset,
            update.ready,
            market,
            update.action,
        )
    return ("gap", update.now_ns, update.reason)


def test_frame_router_raises_stream_errors(hftcore):
    router = hftcore.FrameRouter()
    with pytest.raises(RuntimeError, match="stream error: 406"):
        router.on_frame(b'[{"T":"error","code":406,"msg":"connection limit"}]', OPEN)
