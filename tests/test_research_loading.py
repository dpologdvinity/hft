"""Research identities and phase boundaries must not inspect reserved prices."""

import json
from pathlib import Path

import pytest

from hft import data, research, training


def archive(root, days=80):
    return data.save_dataset(data.synthetic_sessions(days, 1), root / "data")


def test_freeze_never_opens_market_partitions(tmp_path, monkeypatch):
    source = archive(tmp_path)
    original = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.suffix != ".parquet", "freezing opened market prices"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    result = research.freeze_experiment(source, {}, tmp_path / "experiment.json")
    assert result["status"] == "frozen"
    assert len(result["development"]) == 50 and len(result["final_test"]) == 30
    assert result["final_test_consumed"] is False


def test_dataset_selected_sessions_never_read_unselected_prices(tmp_path):
    source = archive(tmp_path, 3)
    index = json.loads(source.read_text())
    final = source.parent / index["sessions"][-1]["manifest"]
    (final.parent / "quotes.parquet").write_bytes(b"unopened held-out prices")
    chosen = [entry["session_id"] for entry in index["sessions"][:2]]
    sessions = data.load_dataset(source, session_ids=chosen)
    assert [s.session_id for s in sessions] == chosen
    with pytest.raises(ValueError, match="checksum"):
        data.load_dataset(source, session_ids=[index["sessions"][-1]["session_id"]])


def test_search_setup_loads_only_development_prices(tmp_path, monkeypatch):
    from test_quality import frozen

    path, _ = frozen(tmp_path, bars=80)
    original = data.load_session
    loaded = []
    allowed = json.loads(path.read_text())["development"]

    def development_only(target, **kwargs):
        assert target.parent.name in allowed, "setup opened final-test prices"
        loaded.append(target.parent.name)
        return original(target, **kwargs)

    monkeypatch.setattr(data, "load_session", development_only)

    def search(sessions, manifest, *args, **kwargs):
        assert [s.session_id for s in sessions] == allowed
        return {"status": "development-loaded"}

    monkeypatch.setattr(training, "_search", search)
    assert training.run_search(path)["status"] == "development-loaded"
    assert set(loaded) == set(allowed)
    assert json.loads(path.read_text())["final_test_consumed"] is False


def test_final_prices_open_only_after_durable_reservation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from test_quality import frozen, refreeze

    path, _ = frozen(tmp_path, bars=80)
    development = json.loads(path.read_text())["development"]
    refreeze(
        path,
        symbol="SYNTH",
        feed="iex",
        config={"seeds": [42], "n_envs": 1},
        folds=[{"train": development[:1], "validation": development[1:]}],
    )
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(training, "TEST_REGISTRY", registry)
    final = json.loads(path.read_text())["final_test"][0]
    opened_final = []
    original = Path.open

    def reserved_only(target, *args, **kwargs):
        if target.suffix == ".parquet" and target.parent.name == final:
            assert registry.exists(), "final prices opened before reservation"
            assert json.loads(path.read_text())["final_test_consumed"] is True
            assert [r["session_id"] for r in json.loads(registry.read_text()).values()] == [final]
            opened_final.append(target)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reserved_only)

    def save(target):
        Path(target).write_bytes(b"selected fixture model")

    model = SimpleNamespace(
        num_timesteps=256,
        peak_rss_bytes=1,
        fit_complete=True,
        save=save,
        predict=lambda obs, **kwargs: (np.array(0), None),
    )
    monkeypatch.setattr(training, "fit_candidate", lambda *args, **kwargs: model)
    evaluated = []

    def evaluate(sessions, policy, **kwargs):
        evaluated.append([s.session_id for s in sessions])
        metric = {
            "return": 0.0,
            "turnover": 0.0,
            "max_drawdown": 0.0,
            "net_profit": 0.0,
            "sessions": len(sessions),
            "trades": 0,
            "expectancy": 0.0,
            "profit_factor": None,
        }
        return {
            "policy": metric,
            "intraday_long": metric,
            "ema_5_20": metric,
            "random": metric,
            "stress": metric,
            "edge": {"intraday_long": {"positive": False}},
        }, np.zeros((1, 17), dtype=np.float32)

    monkeypatch.setattr(training, "evaluate", evaluate)
    monkeypatch.setattr("hft.policy.export_bundle", lambda *args, **kwargs: None)
    result = training.run_search(path)
    assert result["status"] == "failed-edge"
    assert opened_final and evaluated[-1] == [final]
    assert all(final not in ids for ids in evaluated[:-1])
    assert result["paper_eligible"] is False
    previous = len(opened_final)
    assert training.run_search(path, resume=True) == result
    assert len(opened_final) == previous


def test_phase_loader_rejects_unreserved_final_without_price_reads(tmp_path, monkeypatch):
    source = archive(tmp_path)
    experiment = research.freeze_experiment(source, {}, tmp_path / "experiment.json")
    monkeypatch.setattr(data, "load_session", lambda *args, **kwargs: pytest.fail("prices opened"))
    with pytest.raises(ValueError, match="reserved"):
        research.load_phase_sessions(experiment, "final_test")


@pytest.mark.parametrize("ids", [["missing"], ["2025-01-06", "2025-01-06"]])
def test_selection_rejects_unknown_or_duplicate_dates_before_price_reads(
    tmp_path, monkeypatch, ids
):
    source = archive(tmp_path, 2)
    monkeypatch.setattr(data, "load_session", lambda *args, **kwargs: pytest.fail("prices opened"))
    with pytest.raises(ValueError, match="selection"):
        data.load_dataset(source, session_ids=ids)


def test_phase_loading_counts_already_resident_development_memory(tmp_path, monkeypatch):
    source = archive(tmp_path, 2)
    sessions = data.load_dataset(source)
    retained = data.session_bytes(sessions[0])
    monkeypatch.setattr(data, "MAX_DATASET_BYTES", retained + 1)
    with pytest.raises(ValueError, match="memory budget"):
        data.load_dataset(source, session_ids=[sessions[1].session_id], resident_bytes=retained)


@pytest.mark.parametrize("failure", ["checksum", "identity", "order", "escape", "partial"])
def test_freeze_rejects_invalid_archive_metadata_without_reading_prices(
    tmp_path, monkeypatch, failure
):
    source = archive(tmp_path, 3)
    index = json.loads(source.read_text())
    first = source.parent / index["sessions"][0]["manifest"]
    if failure == "checksum":
        first.write_text(first.read_text() + "\n")
    elif failure == "identity":
        index["sessions"][0]["session_id"] = "wrong-date"
    elif failure == "order":
        index["sessions"].reverse()
    elif failure == "escape":
        escaped = tmp_path / "escaped.json"
        escaped.write_bytes(first.read_bytes())
        index["sessions"][0].update(manifest="../escaped.json", sha256=data.checksum(escaped))
    else:
        index["status"] = "partial"
    data.atomic_json(source, index)
    monkeypatch.setattr(data, "load_session", lambda *args, **kwargs: pytest.fail("prices opened"))
    output = tmp_path / "experiment.json"
    with pytest.raises(ValueError):
        research.freeze_experiment(source, {}, output)
    assert not output.exists()


def test_freeze_preserves_existing_session_identity_hashes(tmp_path):
    source = archive(tmp_path, 3)
    previous = data.load_dataset(source)
    hashes = {
        s.session_id: research.canonical_hash(
            {k: v for k, v in s.manifest.items() if k not in ("quote_metadata", "trade_metadata")}
        )
        for s in previous
    }
    frozen = research.freeze_experiment(source, {}, tmp_path / "experiment.json")
    assert frozen["session_hashes"] == hashes


def test_phase_rejects_changed_dataset_identity_without_reading_prices(tmp_path, monkeypatch):
    source = archive(tmp_path, 3)
    frozen = research.freeze_experiment(source, {}, tmp_path / "experiment.json")
    source.write_text(source.read_text() + "\n")
    monkeypatch.setattr(data, "load_session", lambda *args, **kwargs: pytest.fail("prices opened"))
    with pytest.raises(ValueError, match="dataset identity"):
        research.load_phase_sessions(frozen, "development")


def test_phase_rejects_overlap_before_opening_final_prices(tmp_path, monkeypatch):
    source = archive(tmp_path)
    frozen = research.freeze_experiment(source, {}, tmp_path / "experiment.json")
    frozen["development"] = frozen["final_test"][:1]
    monkeypatch.setattr(
        data, "load_session", lambda *args, **kwargs: pytest.fail("final prices opened")
    )
    with pytest.raises(ValueError, match="development partition"):
        research.load_phase_sessions(frozen, "development")


def test_freeze_excludes_execution_metadata_views_from_archive_identity(tmp_path):
    source = archive(tmp_path, 1)
    index = json.loads(source.read_text())
    target = source.parent / index["sessions"][0]["manifest"]
    manifest = json.loads(target.read_text())
    original_hash = research.canonical_hash(manifest)
    manifest.update(quote_metadata={}, trade_metadata={})
    data.atomic_json(target, manifest)
    index["sessions"][0]["sha256"] = data.checksum(target)
    data.atomic_json(source, index)
    frozen = research.freeze_experiment(source, {}, tmp_path / "experiment.json")
    assert frozen["session_hashes"][manifest["session_id"]] == original_hash


def test_loading_includes_existing_process_memory_before_allocating(tmp_path, monkeypatch):
    import resource
    from types import SimpleNamespace

    source = archive(tmp_path, 1)
    monkeypatch.setattr(resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=8 * 1024 * 1024))
    monkeypatch.setattr(
        data, "load_session", lambda *args, **kwargs: pytest.fail("allocation began")
    )
    with pytest.raises(ValueError, match="working memory budget"):
        data.load_dataset(source)
