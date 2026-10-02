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


def fixture_rows(tmp_path, *, gap=False, correction=False):
    from hft.data import NS

    start = parse_timestamp("2025-01-02T14:30:00Z")
    close = start + 23400 * NS
    calendar = [{"date": "2025-01-02", "open": "09:30", "close": "16:00"}]
    (tmp_path / "calendar.json").write_text(json.dumps({"calendar": calendar}))
    ticks = [start, close - NS] if gap else list(range(start, close, 5 * NS)) + [close - NS]
    rows = [
        {
            "T": "q",
            "S": "AAPL",
            "event_ns": stamp,
            "bp": 99.99,
            "ap": 100.01,
            "bs": 1,
            "as": 2,
            "arrival_ns": stamp + 1_000_000,
        }
        for stamp in ticks
    ]
    rows.append(
        {
            "T": "t",
            "S": "AAPL",
            "event_ns": start + NS,
            "p": 100,
            "s": 1,
            "i": 1,
            "arrival_ns": start + NS + 1_000_000,
        }
    )
    if correction:
        rows.append(
            {
                "T": "c",
                "S": "AAPL",
                "event_ns": start + 2 * NS,
                "oi": 1,
                "cp": 999,
                "arrival_ns": start + 2 * NS,
            }
        )
    rows.sort(key=lambda r: r["arrival_ns"])
    (tmp_path / "raw-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_endpoints_with_large_gap_cannot_be_complete(tmp_path):
    fixture_rows(tmp_path, gap=True)
    result = prepare_recordings(tmp_path)
    assert result["complete_sessions"] == 0
    session = load_dataset(result["dataset"])[0]
    assert session.manifest["quality"]["quote_event_gaps"] > 0
    assert session.manifest["quality"]["quote_arrival_gaps"] > 0


def test_full_coverage_and_correction_uncertainty(tmp_path):
    fixture_rows(tmp_path)
    result = prepare_recordings(tmp_path)
    assert result["complete_sessions"] == 1
    fixture_rows(tmp_path, correction=True)
    result = prepare_recordings(tmp_path)
    assert result["complete_sessions"] == 0
    assert load_dataset(result["dataset"])[0].manifest["quality"]["corrections"] == 1


def test_sort_memory_guard_precedes_full_table_read(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    from hft import recording

    fixture_rows(tmp_path, gap=True)
    monkeypatch.setattr(recording, "MAX_SORT_WORKING_BYTES", 1)
    calls = []
    monkeypatch.setattr(pq, "read_table", lambda *a, **kw: calls.append(True))
    with __import__("pytest").raises(ValueError, match="sorting memory budget"):
        prepare_recordings(tmp_path)
    assert not calls


def test_failed_reprepare_preserves_completed_dataset(tmp_path):
    fixture_rows(tmp_path)
    result = prepare_recordings(tmp_path)
    manifest = __import__("pathlib").Path(result["dataset"])
    before = manifest.read_bytes()
    (tmp_path / "raw-b.jsonl").write_text('{"T":"q"')
    with __import__("pytest").raises(ValueError, match="truncated"):
        prepare_recordings(tmp_path)
    assert manifest.read_bytes() == before
    assert load_dataset(manifest)[0].manifest["complete_session"]


def test_repeated_prepare_and_split_files_keep_identity(tmp_path):
    fixture_rows(tmp_path)
    lines = (tmp_path / "raw-a.jsonl").read_text().splitlines(keepends=True)
    (tmp_path / "raw-a.jsonl").write_text("".join(lines[:2000]))
    (tmp_path / "raw-b.jsonl").write_text("".join(lines[2000:]))
    first = prepare_recordings(tmp_path)
    manifest = __import__("pathlib").Path(first["dataset"])
    content = manifest.read_bytes()
    second = prepare_recordings(tmp_path)
    assert first == second and content == manifest.read_bytes()
    assert second["complete_sessions"] == 1


def test_arrival_gap_or_late_record_invalidates_coverage(tmp_path):
    fixture_rows(tmp_path)
    path = tmp_path / "raw-a.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[2000]["arrival_ns"] += 1_000_000_000
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = prepare_recordings(tmp_path)
    assert result["complete_sessions"] == 0
    quality = load_dataset(result["dataset"])[0].manifest["quality"]
    assert quality["late_or_future_events"] > 0 and quality["quote_arrival_gaps"] > 0
