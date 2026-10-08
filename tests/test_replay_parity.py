"""Native replay must reproduce the Python bar/ready/gap construction exactly."""

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_projection import _with_archive_metadata

from hft.data import load_dataset, save_dataset, synthetic_sessions
from hft.feed import historical_bars


@pytest.fixture(scope="module", autouse=True)
def hftcore():
    try:
        return importlib.import_module("hftcore")
    except ImportError:
        if os.environ.get("HFT_REQUIRE_HFTCORE") == "1":
            raise
        pytest.skip("hftcore is not installed")


def _assert_same(session):
    expected = historical_bars(session, implementation="python")
    actual = historical_bars(session, implementation="cpp")
    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])
    assert np.array_equal(actual[2], expected[2])
    return expected


@pytest.mark.parametrize("arrival", [False, True])
def test_replay_matches_on_archive_schema_sessions(tmp_path, arrival):
    session = _with_archive_metadata(synthetic_sessions(1, 200, seed=8)[0], arrival=arrival)
    for metadata in ("all", "execution"):
        (loaded,) = load_dataset(save_dataset([session], tmp_path / metadata), metadata=metadata)
        bars, _, _ = _assert_same(loaded)
        assert len(bars) > 100


def test_replay_matches_with_gaps_and_numeric_identities():
    session = synthetic_sessions(1, 200, seed=9)[0]  # no metadata: numeric identities
    keep = (session.quote_ns < session.open_ns + 100 * 10**9) | (
        session.quote_ns >= session.open_ns + 130 * 10**9
    )
    object.__setattr__(session, "quote_ns", session.quote_ns[keep])
    for name in ("bid", "ask", "bid_size", "ask_size"):
        object.__setattr__(session, name, getattr(session, name)[keep])
    keep = (session.trade_ns < session.open_ns + 100 * 10**9) | (
        session.trade_ns >= session.open_ns + 130 * 10**9
    )
    for name in ("trade_ns", "trade_price", "trade_size"):
        object.__setattr__(session, name, getattr(session, name)[keep])
    _, _, gaps = _assert_same(session)
    assert len(gaps) >= 1


def _development_sessions():
    """Real development days on disk; reserved final-test sessions are refused."""
    from hft.training import reserved_final_sessions

    root = Path(__file__).resolve().parents[1]
    candidates = [
        (root / "data/mcd/manifest.json", "2026-06-04"),
        (root / "data/free-data-probe-2026-10-04/nvda/manifest.json", "2026-06-04"),
    ]
    reserved = reserved_final_sessions()
    for manifest, day in candidates:
        if not manifest.exists():
            continue
        entries = json.loads(manifest.read_text())["sessions"]
        symbol = json.loads((manifest.parent / entries[0]["manifest"]).read_text())["symbol"]
        assert (symbol, day) not in reserved
        yield manifest, day


@pytest.mark.parametrize("source", list(_development_sessions()), ids=lambda s: s[1])
def test_replay_matches_on_real_development_sessions(source):
    manifest, day = source
    (session,) = load_dataset(manifest, session_ids=[day], metadata="execution")
    bars, _, _ = _assert_same(session)
    assert len(bars) > 1000
