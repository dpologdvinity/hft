import json

import pytest

from hft.data import AGGREGATION_VERSION, checksum, save_dataset, synthetic_sessions
from hft.features import FEATURE_VERSION
from hft.research import canonical_hash


def frozen(root, *, gap=False):
    sessions = synthetic_sessions(3, 100, 42)
    if gap:
        session = sessions[0]
        for names in (
            ("quote_ns", "bid", "ask", "bid_size", "ask_size"),
            ("trade_ns", "trade_price", "trade_size"),
        ):
            stamps = getattr(session, names[0])
            keep = (stamps < session.open_ns + 100 * 10**9) | (
                stamps >= session.open_ns + 110 * 10**9
            )
            for name in names:
                object.__setattr__(session, name, getattr(session, name)[keep])
    for session in sessions:
        session.manifest.update(synthetic=False, feed="iex", provenance="alpaca-historical-iex")
    source = save_dataset(sessions, root / "data")
    dataset = json.loads(source.read_text())
    result = {
        "dataset_manifest": str(source),
        "dataset_hash": checksum(source),
        "feature_version": FEATURE_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "status": "frozen",
        "config": {"wall_seconds": 7200},
        "synthetic": False,
        "final_test_consumed": False,
        "development": [s.session_id for s in sessions[:2]],
        "final_test": [sessions[2].session_id],
        "session_hashes": {
            entry["session_id"]: canonical_hash(
                json.loads((source.parent / entry["manifest"]).read_text())
            )
            for entry in dataset["sessions"]
        },
    }
    result["experiment_hash"] = canonical_hash(result)
    path = root / "artifacts/experiment.json"
    path.parent.mkdir()
    path.write_text(json.dumps(result))
    return path, source


def test_quality_reads_development_only_and_records_exact_coverage(tmp_path, monkeypatch):
    from hft import data, research

    path, source = frozen(tmp_path)
    held_out = json.loads(path.read_text())["final_test"][0]
    entries = json.loads(source.read_text())["sessions"]
    (source.parent / entries[-1]["manifest"]).write_text("held-out must never be opened")
    original = data.load_session
    calls = []

    def load(target, **kwargs):
        assert held_out not in str(target)
        calls.append(target)
        return original(target, **kwargs)

    monkeypatch.setattr(data, "load_session", load)
    result = research.preflight_experiment(path)
    assert result["status"] == "data-ready"
    assert result["coverage_sessions"] == 2
    assert result["coverage_complete"] is True
    assert result["eligible_decisions"] > 0
    assert result["feed_gaps"] == 0 and result["live_comparable"] is True
    assert len(calls) == 2 and result["paper_eligible"] is False
    assert json.loads(path.read_text())["final_test_consumed"] is False
    assert json.loads((path.parent / "diagnostics.json").read_text()) == result


def test_expired_quality_budget_does_not_open_market_partitions(tmp_path, monkeypatch):
    from hft import data, research, training

    path, _ = frozen(tmp_path)
    monkeypatch.setattr(data, "load_session", lambda *args: pytest.fail("market partition opened"))
    with pytest.raises(TimeoutError, match="preflight"):
        research.preflight_experiment(path, deadline=0)
    output = path.parent / "search"
    output.mkdir()
    manifest = json.loads(path.read_text())
    (output / "budget.json").write_text(
        json.dumps(
            {
                "experiment_hash": manifest["experiment_hash"],
                "elapsed_seconds": 7200,
                "wall_seconds": 7200,
            }
        )
    )
    result = training.run_search(path, resume=True)
    assert result["status"] == "incomplete-search"
    assert json.loads(path.read_text())["final_test_consumed"] is False


def test_search_charges_setup_against_persisted_budget(tmp_path, monkeypatch):
    from hft import training

    experiment = {"experiment_hash": "frozen", "config": {"wall_seconds": 100}}
    (tmp_path / "budget.json").write_text(
        json.dumps(
            {
                "experiment_hash": "frozen",
                "elapsed_seconds": 25,
                "wall_seconds": 100,
            }
        )
    )
    monkeypatch.setattr(training.time, "monotonic", lambda: 100)

    def search(sessions, manifest, *args):
        assert manifest["_deadline"] == 155
        return {"status": "proved"}

    monkeypatch.setattr(training, "_search_impl", search)
    assert (
        training._search([], experiment, tmp_path, resume=True, setup_seconds=20)["status"]
        == "proved"
    )
    assert json.loads((tmp_path / "budget.json").read_text())["elapsed_seconds"] == 45


@pytest.mark.parametrize("status", ["paper-eligible", "failed-edge"])
def test_completed_report_survives_resume_with_exhausted_budget(tmp_path, monkeypatch, status):
    from hft import data, training

    path, _ = frozen(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["final_test_consumed"] = True
    path.write_text(json.dumps(manifest))
    output = path.parent / "search"
    output.mkdir()
    report = {
        "experiment_hash": manifest["experiment_hash"],
        "status": status,
        "test": {"policy": {}},
        "paper_eligible": status == "paper-eligible",
    }
    report_path = output / "research.json"
    report_path.write_text(json.dumps(report))
    before = report_path.read_bytes()
    (output / "budget.json").write_text(
        json.dumps(
            {
                "experiment_hash": manifest["experiment_hash"],
                "elapsed_seconds": 7200,
                "wall_seconds": 7200,
            }
        )
    )
    monkeypatch.setattr(data, "load_dataset", lambda *args: pytest.fail("market partitions opened"))
    assert training.run_search(path, resume=True) == report
    assert report_path.read_bytes() == before


def test_gap_stops_preflight_and_training_before_fit_or_held_out_reads(tmp_path, monkeypatch):
    from hft import data, training

    path, _ = frozen(tmp_path, gap=True)
    monkeypatch.setattr(data, "load_dataset", lambda *args: pytest.fail("full data loaded"))
    monkeypatch.setattr(training, "fit_candidate", lambda *args: pytest.fail("training started"))
    result = training.run_search(path)
    assert result["status"] == "insufficient-data"
    assert result["eligibility"]["reasons"] == ["runtime-feed-gap"]
    diagnostics = json.loads((path.parent / "diagnostics.json").read_text())
    assert diagnostics["feed_gaps"] > 0
    assert diagnostics["coverage_sessions"] == 1
    assert diagnostics["coverage_complete"] is False
    assert diagnostics["live_comparable"] is False
    assert json.loads(path.read_text())["final_test_consumed"] is False


def test_quality_rejects_changed_identity_and_partitions(tmp_path):
    from hft.research import preflight_experiment

    path, source = frozen(tmp_path)
    entry = json.loads(source.read_text())["sessions"][0]
    manifest = source.parent / entry["manifest"]
    parts = json.loads(manifest.read_text())["partitions"]
    (manifest.parent / parts["quotes"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        preflight_experiment(path)
    changed = json.loads(path.read_text())
    changed["development"] = changed["final_test"]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity"):
        preflight_experiment(path)
