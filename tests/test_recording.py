"""Recorded provider events retain identity and arrival-time causality."""

import json

import numpy as np

from hft.data import load_dataset
from hft.feed import parse_timestamp
from hft.recording import prepare_recordings


def test_prepare_quote_without_provider_id_and_out_of_order_arrivals(tmp_path):
    calendar = [{"date": "2025-01-02", "open": "09:30", "close": "16:00"}]
    (tmp_path / "calendar.json").write_text(json.dumps({"calendar": calendar}))
    start = parse_timestamp("2025-01-02T14:30:00Z")
    quotes = [
        {
            "T": "q",
            "S": "AAPL",
            "t": "2025-01-02T14:30:00.000000001Z",
            "bp": 99.99,
            "ap": 100.01,
            "bs": 1,
            "as": 2,
            "c": ["R"],
            "arrival_ns": start + 10_000_000,
        },
        {
            "T": "q",
            "S": "AAPL",
            "t": "2025-01-02T14:30:00.200000001Z",
            "bp": 100.09,
            "ap": 100.11,
            "bs": 2,
            "as": 3,
            "c": ["R"],
            "arrival_ns": start + 250_000_000,
        },
        {
            "T": "q",
            "S": "AAPL",
            "t": "2025-01-02T14:30:00.100000001Z",
            "bp": 100.04,
            "ap": 100.06,
            "bs": 3,
            "as": 4,
            "c": ["R"],
            "arrival_ns": start + 300_000_000,
        },
    ]
    trades = [
        {
            "T": "t",
            "S": "AAPL",
            "t": "2025-01-02T14:30:00.020000001Z",
            "p": 100.0,
            "s": 1,
            "i": 1,
            "c": ["@"],
            "z": "C",
            "arrival_ns": start + 30_000_000,
        },
        {
            "T": "t",
            "S": "AAPL",
            "t": "2025-01-02T14:30:00.220000001Z",
            "p": 100.1,
            "s": 2,
            "i": 2,
            "c": ["@"],
            "z": "C",
            "arrival_ns": start + 270_000_000,
        },
    ]
    rows = sorted(quotes + trades, key=lambda row: row["arrival_ns"])
    (tmp_path / "raw-fixture.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = prepare_recordings(tmp_path)
    sessions = load_dataset(result["dataset"])
    assert len(sessions) == 1
    session = sessions[0]
    assert session.symbol == "AAPL"
    assert np.array_equal(session.quote_ns, start + np.array([1, 100_000_001, 200_000_001]))
    assert np.array_equal(
        session.quote_arrival_ns, start + np.array([10_000_000, 300_000_000, 250_000_000])
    )
    assert np.array_equal(session.trade_arrival_ns, start + np.array([30_000_000, 270_000_000]))
    assert session.manifest["quote_metadata"]["i"].null_count == 0
    events = list(session.iter_events())
    assert [event["arrival_ns"] for event in events] == [row["arrival_ns"] for row in rows]
    assert [event["event_ns"] for event in events if event["T"] == "q"] == [
        start + 1,
        start + 200_000_001,
        start + 100_000_001,
    ]
    assert session.manifest["arrival_times_available"] is True
    assert result["complete_sessions"] == 0
