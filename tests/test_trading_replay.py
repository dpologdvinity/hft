from dataclasses import replace
from decimal import Decimal

import pytest

from hft.data import save_dataset, synthetic_sessions
from hft.strategies import parse_strategy
from hft.trading.replay import ensure_session, find_cached_session, format_replay, replay_stock


def _real_like(tmp_path):
    session = synthetic_sessions(1, 300, seed=11)[0]
    manifest = {
        **session.manifest,
        "synthetic": False,
        "feed": "iex",
        "provenance": "alpaca-historical-iex",
    }
    session = replace(session, manifest=manifest)
    save_dataset([session], tmp_path / "data" / "x")
    return session


def test_replay_runs_the_bot_decision_path_with_simulated_fills(tmp_path):
    session = _real_like(tmp_path)
    manifest = find_cached_session(session.symbol, session.session_id, tmp_path / "data")
    assert manifest is not None
    row = replay_stock(manifest, 200, parse_strategy("hold-day"), tmp_path / "logs")
    assert row["decisions"] > 100 and row["long_decisions"] == row["decisions"]
    assert row["round_trips"] >= 1  # bought after warmup and sold in the pre-close window
    assert Decimal(row["position"]) == 0
    assert "total net" in format_replay([row])


def test_replay_refuses_synthetic_and_reserved_data(tmp_path, monkeypatch):
    session = synthetic_sessions(1, 300, seed=12)[0]
    save_dataset([session], tmp_path / "data" / "s")
    manifest = find_cached_session(session.symbol, session.session_id, tmp_path / "data")
    with pytest.raises(ValueError, match="real, verified"):
        replay_stock(manifest, 200, parse_strategy("hold-day"), tmp_path / "logs")
    monkeypatch.setattr(
        "hft.trading.replay.reserved_final_sessions", lambda: {("MCD", "2026-08-19")}
    )
    with pytest.raises(ValueError, match="reserved"):
        ensure_session("MCD", "2026-08-19", tmp_path / "data", download=lambda *a: None)


def test_replay_downloads_only_when_not_cached(tmp_path):
    session = _real_like(tmp_path)
    calls = []

    def download(*args):
        calls.append(args)
        raise AssertionError("should use the cache")

    found = ensure_session(session.symbol, session.session_id, tmp_path / "data", download)
    assert found.name == "manifest.json" and calls == []
