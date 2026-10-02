import itertools
from dataclasses import replace

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
