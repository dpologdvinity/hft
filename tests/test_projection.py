"""The execution metadata view must change memory, never behavior."""

import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from hft import data
from hft.data import load_dataset, metadata_bytes, save_dataset, synthetic_sessions
from hft.execution import Simulation
from hft.research import _rollout


def _with_archive_metadata(session, *, arrival=False, quote_ids=None):
    """Attach the column layout written by hft.history to a synthetic session."""
    nq, nt = len(session.quote_ns), len(session.trade_ns)
    quotes = {
        "raw_bs": pa.array(session.bid_size / 100),
        "raw_as": pa.array(session.ask_size / 100),
        "raw_json": pa.array([json.dumps({"q": k, "pad": "x" * 200}) for k in range(nq)]),
        "i": pa.array(quote_ids if quote_ids is not None else [f"q{k}" for k in range(nq)]),
        "c": pa.array([["R"]] * nq, type=pa.list_(pa.string())),
        "x": pa.array(["V"] * nq),
        "z": pa.array(["A"] * nq),
        "bx": pa.array(["V" if k % 2 else "Q" for k in range(nq)]),
        "ax": pa.array(["V"] * nq),
    }
    # Every seventh trade carries a condition the bar filter excludes.
    trades = {
        "raw_json": pa.array([json.dumps({"t": k, "pad": "y" * 200}) for k in range(nt)]),
        "i": pa.array([f"t{k}" for k in range(nt)]),
        "c": pa.array([["W"] if k % 7 == 0 else ["@"] for k in range(nt)]),
        "x": pa.array(["V"] * nt),
        "z": pa.array(["C"] * nt),
        "bx": pa.array([None] * nt, type=pa.string()),
        "ax": pa.array([None] * nt, type=pa.string()),
    }
    if arrival:
        quotes["arrival_ns"] = pa.array(session.quote_ns + 1_000_000)
        trades["arrival_ns"] = pa.array(session.trade_ns + 2_000_000)
    return replace(
        session, manifest={**session.manifest, "quote_metadata": quotes, "trade_metadata": trades}
    )


def _toggle_policy(period=40):
    calls = []

    def policy(obs, env):
        calls.append(None)
        return (len(calls) // period) % 2

    return policy


def _load_both(tmp_path, session):
    path = save_dataset([session], tmp_path)
    (full,) = load_dataset(path)
    (projected,) = load_dataset(path, metadata="execution")
    return full, projected


@pytest.mark.parametrize("arrival", [False, True])
def test_execution_view_preserves_events_simulation_and_rollout(tmp_path, arrival):
    session = _with_archive_metadata(synthetic_sessions(1, 200, seed=7)[0], arrival=arrival)
    full, projected = _load_both(tmp_path, session)

    kept = {"i"} | ({"arrival_ns"} if arrival else set())
    assert set(projected.manifest["quote_metadata"]) == kept
    assert set(projected.manifest["trade_metadata"]) == kept | {"c", "z"}
    assert metadata_bytes(projected.manifest) < metadata_bytes(full.manifest) / 3
    assert projected.metadata_projection == "execution"
    assert full.metadata_projection == "all"

    dropped = {"raw_json", "raw_bs", "raw_as", "x", "bx", "ax"} | {"c", "z"}
    for a, b in zip(full.iter_events(), projected.iter_events(), strict=True):
        drop = dropped - ({"c", "z"} if a["T"] == "t" else set())
        assert b == {k: v for k, v in a.items() if k not in drop}

    sim_full, sim_projected = Simulation(full), Simulation(projected)
    assert sim_full.bars == sim_projected.bars
    assert sim_full.gap_count == sim_projected.gap_count
    for bar in sim_full.bars[61:]:
        assert sim_full.decision_eligible == sim_projected.decision_eligible
        sim_full.advance_to(bar.end_ns)
        sim_projected.advance_to(bar.end_ns)
    assert sim_full.gap_count == sim_projected.gap_count

    result_full, obs_full = _rollout([full], _toggle_policy())
    result_projected, obs_projected = _rollout([projected], _toggle_policy())
    assert result_full["trades"] > 0
    assert result_full == result_projected
    assert np.array_equal(obs_full, obs_projected)


def test_incomplete_quote_identities_keep_hashed_fallback_fields(tmp_path):
    # Null identities are rejected at load; an empty one falls back to a field hash.
    session = synthetic_sessions(1, 200, seed=7)[0]
    ids = [f"q{k}" for k in range(len(session.quote_ns))]
    ids[5] = ""
    full, projected = _load_both(tmp_path, _with_archive_metadata(session, quote_ids=ids))
    assert {"bx", "ax", "c", "z"} <= set(projected.manifest["quote_metadata"])
    assert "raw_json" not in projected.manifest["quote_metadata"]
    quotes = [e for e in full.iter_events() if e["T"] == "q"]
    projected_quotes = [e for e in projected.iter_events() if e["T"] == "q"]
    drop = {"raw_json", "raw_bs", "raw_as", "x"}
    assert projected_quotes == [{k: v for k, v in e.items() if k not in drop} for e in quotes]
    assert Simulation(full).bars == Simulation(projected).bars


def test_projected_sessions_cannot_be_resaved(tmp_path):
    session = _with_archive_metadata(synthetic_sessions(1, 200, seed=7)[0])
    _, projected = _load_both(tmp_path / "source", session)
    with pytest.raises(ValueError, match="projected"):
        save_dataset([projected], tmp_path / "copy")


def test_projection_reduces_preflight_budget_and_keeps_checksums(tmp_path, monkeypatch):
    session = _with_archive_metadata(synthetic_sessions(1, 200, seed=7)[0])
    path = save_dataset([session], tmp_path)
    full_bytes = metadata_bytes(load_dataset(path)[0].manifest)
    # A budget that only the projected view fits must still load it.
    monkeypatch.setattr(data, "MAX_SESSION_BYTES", full_bytes)
    with pytest.raises(ValueError, match="memory budget"):
        load_dataset(path)
    assert load_dataset(path, metadata="execution")
    quotes = tmp_path / session.symbol / session.session_id / "quotes.parquet"
    quotes.write_bytes(quotes.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(path, metadata="execution")


def test_unknown_metadata_projection_is_rejected(tmp_path):
    path = save_dataset(synthetic_sessions(1, 200), tmp_path)
    with pytest.raises(ValueError, match="projection"):
        load_dataset(path, metadata="numeric")
