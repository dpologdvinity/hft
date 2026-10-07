import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hft.data import load_dataset, save_dataset, synthetic_sessions


def test_raw_array_contract_and_roundtrip(tmp_path):
    sessions = synthetic_sessions(days=2, bars_per_day=80)
    s = sessions[0]
    assert s.synthetic and s.quote_ns.dtype == np.int64
    assert s.quote_ns[1] - s.quote_ns[0] == 100_000_000
    assert not s.bid.flags.writeable
    loaded = load_dataset(save_dataset(sessions, tmp_path))
    np.testing.assert_array_equal(loaded[0].bid, s.bid)
    assert list(loaded[0].iter_events()) == list(s.iter_events())


@pytest.mark.parametrize(
    "field,value",
    [("quote_ns", [2, 1]), ("bid", [float("nan"), 100]), ("ask", [0, 100]), ("bid_size", [-1, 1])],
)
def test_invalid_arrays(field, value):
    s = synthetic_sessions(days=1, bars_per_day=1)[0]
    with pytest.raises(ValueError):
        replace(s, **{field: np.asarray(value)})


def test_nonfinite_crossed_duplicates_and_null_timestamps():
    s = synthetic_sessions(days=1, bars_per_day=1)[0]
    for name, value in [("bid", float("inf")), ("bid", 0), ("trade_price", -1), ("ask_size", -1)]:
        a = getattr(s, name).copy()
        a[0] = value
        with pytest.raises(ValueError):
            replace(s, **{name: a})
    with pytest.raises(ValueError):
        replace(s, bid=s.ask + 1)
    bad = s.quote_ns.copy()
    bad[1] = bad[0]
    with pytest.raises(ValueError):
        replace(s, quote_ns=bad, bid=np.full(len(bad), 100.0), ask=np.full(len(bad), 101.0))
    bad = s.quote_ns.astype(object)
    bad[0] = None
    with pytest.raises(ValueError):
        replace(s, quote_ns=bad)


def test_arrival_metadata_preserves_causal_availability(tmp_path):
    from hft.data import NS, build_bars

    s = synthetic_sessions(days=1, bars_per_day=2)[0]
    arrival = s.quote_ns.copy()
    arrival[49] = s.open_ns + 6 * NS
    s = replace(s, manifest={**s.manifest, "quote_metadata": {"arrival_ns": arrival}})
    loaded = load_dataset(save_dataset([s], tmp_path))[0]
    np.testing.assert_array_equal(loaded.quote_arrival_ns, arrival)
    events = list(loaded.iter_events())
    assert all(
        b.get("arrival_ns", b["event_ns"]) >= a.get("arrival_ns", a["event_ns"])
        for a, b in itertools.pairwise(events)
    )
    assert build_bars(loaded)[0].quote_ns == s.open_ns + 4_800_000_000


def test_twenty_session_loading_aggregation_memory_budget(tmp_path):
    import resource

    from hft.data import build_bars

    sessions = load_dataset(save_dataset(synthetic_sessions(days=20, bars_per_day=80), tmp_path))
    assert len(sessions) == 20
    assert all(len(build_bars(s)) == 80 for s in sessions)
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 8 * 1024**3


def test_parquet_float_nanoseconds_rejected_before_cast(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from hft.data import atomic_json, checksum, load_session

    save_dataset(synthetic_sessions(1, 1), tmp_path)
    directory = tmp_path / "SYNTH/2025-01-06"
    q = directory / "quotes.parquet"
    table = pq.read_table(q)
    table = table.set_column(0, "quote_ns", pa.array(table["quote_ns"].to_numpy().astype(float)))
    pq.write_table(table, q)
    m = json.loads((directory / "manifest.json").read_text())
    m["partitions"]["quotes"]["sha256"] = checksum(q)
    atomic_json(directory / "manifest.json", m)
    with pytest.raises(ValueError, match="int64"):
        load_session(directory / "manifest.json")


def test_arrow_metadata_budget_preflight_precedes_session_load(tmp_path, monkeypatch):
    import pyarrow as pa

    from hft import data

    sessions = synthetic_sessions(2, 1)
    sessions = [
        replace(
            s,
            manifest={
                **s.manifest,
                "quote_metadata": {"raw_json": pa.array(["x" * 4000] * len(s.quote_ns))},
            },
        )
        for s in sessions
    ]
    path = save_dataset(sessions, tmp_path)
    monkeypatch.setattr(data, "MAX_DATASET_BYTES", 300_000)
    calls = []
    original = data._read_session

    def tracked(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(data, "_read_session", tracked)
    with pytest.raises(ValueError, match="memory budget"):
        data.load_dataset(path)
    assert len(calls) <= 1


def test_load_dataset_preflights_each_session_once(tmp_path, monkeypatch):
    from hft import data

    path = save_dataset(synthetic_sessions(3, 1), tmp_path)
    calls = []
    original = data._session_preflight

    def tracked(target, *args, **kwargs):
        calls.append(Path(target).parent.name)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(data, "_session_preflight", tracked)
    sessions = data.load_dataset(path)
    assert calls == [s.session_id for s in sessions]


def test_immutable_constructor_metadata_budget_before_copy(monkeypatch):
    import pyarrow as pa

    from hft import data

    s = synthetic_sessions(1, 1)[0]
    monkeypatch.setattr(data, "MAX_SESSION_BYTES", 1000)
    copies = []
    monkeypatch.setattr(data.np, "ascontiguousarray", lambda *args, **kw: copies.append(True))
    with pytest.raises(ValueError, match="memory budget"):
        replace(
            s,
            manifest={
                **s.manifest,
                "quote_metadata": {"raw_json": pa.array(["x" * 4000] * len(s.quote_ns))},
            },
        )
    assert not copies


def test_preflight_reserves_defensive_numeric_copy(tmp_path, monkeypatch):
    from hft import data

    p = save_dataset(synthetic_sessions(1, 1), tmp_path)
    monkeypatch.setattr(data, "MAX_WORKING_BYTES", 3000)
    with pytest.raises(ValueError, match="working memory budget"):
        data.load_dataset(p)


@pytest.mark.parametrize("empty", [False, True])
def test_metadata_column_bytes_preserves_total_accounting(empty):
    import pyarrow as pa

    from hft.data import NUMERIC_NAMES, metadata_bytes, metadata_column_bytes, session_bytes

    manifest = (
        {}
        if empty
        else {
            "quote_metadata": {
                "raw_json": pa.array(["x" * 10000, "y" * 10000]),
                "chunked": pa.chunked_array([[1], [2]], type=pa.int64()),
                "numpy": np.array([1, 2], dtype=np.int64),
                "fixture": [1, 2],
            },
            "trade_metadata": {"flags": pa.array([True, False])},
        }
    )
    expected = {"quote_metadata": {}, "trade_metadata": {}}
    if not empty:
        expected = {
            "quote_metadata": {"raw_json": 20008, "chunked": 16, "numpy": 16, "fixture": 16},
            "trade_metadata": {"flags": 1},
        }
    assert metadata_column_bytes(manifest) == expected
    assert metadata_bytes(manifest) == (20057 if not empty else 0)
    session = synthetic_sessions(1, 1)[0]
    session = replace(
        session,
        **{name: getattr(session, name)[:2] for name in NUMERIC_NAMES},
        manifest={**session.manifest, **manifest},
    )
    numeric = sum(getattr(session, name).nbytes for name in NUMERIC_NAMES)
    assert session_bytes(session) == numeric + (20057 if not empty else 0)
