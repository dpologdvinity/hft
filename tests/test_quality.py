import json

import pytest

from hft.data import AGGREGATION_VERSION, checksum, save_dataset, synthetic_sessions
from hft.features import FEATURE_VERSION
from hft.research import canonical_hash


def frozen(root, *, gap=False, bars=100, provenance="alpaca-historical-iex", metadata=False):
    sessions = synthetic_sessions(3, bars, 42)
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
    if metadata:
        import pyarrow as pa

        for session in sessions:
            session.manifest["quote_metadata"] = {
                "raw_json": pa.array(['{"source":"fixture"}'] * len(session.quote_ns))
            }
            session.manifest["trade_metadata"] = {"flags": [1] * len(session.trade_ns)}
    for session in sessions:
        session.manifest.update(synthetic=False, feed="iex", provenance=provenance)
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


def refreeze(path, **changes):
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    manifest.pop("experiment_hash")
    manifest["experiment_hash"] = canonical_hash(manifest)
    path.write_text(json.dumps(manifest))


def test_all_development_continues_after_gap_without_reading_final_test(tmp_path, monkeypatch):
    from hft import data, research

    path, source = frozen(tmp_path, gap=True)
    entries = json.loads(source.read_text())["sessions"]
    (source.parent / entries[-1]["manifest"]).write_text("never open reserved data")
    calls = []
    original = data.load_session

    def load(target, **kwargs):
        assert target != (source.parent / entries[-1]["manifest"]).resolve()
        calls.append(target)
        return original(target, **kwargs)

    monkeypatch.setattr(data, "load_session", load)
    default = research.preflight_experiment(path)
    assert default["inspection_mode"] == "stop-first-failure"
    assert default["coverage_sessions"] == 1 and default["coverage_complete"] is False
    calls.clear()
    result = research.preflight_experiment(path, inspect_all=True)
    assert result["inspection_mode"] == "all-development"
    assert result["coverage_sessions"] == 2 and result["coverage_complete"] is True
    assert len(calls) == 2
    assert result["status"] == "insufficient-data"
    assert result["live_comparable"] is False and result["paper_eligible"] is False
    assert result["sessions"][0]["reasons"] == ["runtime-feed-gap"]
    assert result["sessions"][1]["reasons"] == []
    for key in ("eligible_decisions", "ineligible_decisions", "feed_gaps"):
        assert result[key] == sum(row[key] for row in result["sessions"])
    assert json.loads(path.read_text())["final_test_consumed"] is False


def test_full_inspection_preserves_published_report_on_later_corruption(tmp_path):
    from hft import research

    path, source = frozen(tmp_path, gap=True)
    research.preflight_experiment(path)
    report = path.parent / "diagnostics.json"
    before = report.read_bytes()
    entry = json.loads(source.read_text())["sessions"][1]
    target = source.parent / entry["manifest"]
    part = json.loads(target.read_text())["partitions"]["quotes"]["path"]
    (target.parent / part).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        research.preflight_experiment(path, inspect_all=True)
    assert report.read_bytes() == before


@pytest.mark.parametrize("change", ["duplicate", "overlap", "identity", "escaped", "index"])
def test_full_inspection_still_validates_partition_identity(tmp_path, monkeypatch, change):
    from hft import data, research

    path, source = frozen(tmp_path)
    manifest = json.loads(path.read_text())
    if change == "duplicate":
        refreeze(path, development=manifest["development"] * 2)
    elif change == "overlap":
        refreeze(path, development=manifest["final_test"])
    elif change == "identity":
        manifest["development"] = manifest["final_test"]
        path.write_text(json.dumps(manifest))
    else:
        dataset = json.loads(source.read_text())
        if change == "escaped":
            dataset["sessions"][0]["manifest"] = "../outside.json"
        else:
            dataset["sessions"].append(dataset["sessions"][0])
        source.write_text(json.dumps(dataset))
        refreeze(path, dataset_hash=checksum(source))
    monkeypatch.setattr(data, "load_session", lambda *a, **k: pytest.fail("tick load"))
    with pytest.raises(ValueError, match="partition|identity|escapes|index"):
        research.preflight_experiment(path, inspect_all=True)


@pytest.mark.parametrize("expire_after", [2, 40])
def test_full_inspection_deadline_blocks_later_load_and_publication(
    tmp_path, monkeypatch, expire_after
):
    from hft import data, research

    path, _ = frozen(tmp_path)
    research.preflight_experiment(path)
    report = path.parent / "diagnostics.json"
    before = report.read_bytes()
    calls = []
    original = data.load_session

    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    ticks = 0

    def clock():
        nonlocal ticks
        ticks += 1
        return 0 if ticks <= expire_after else 2

    monkeypatch.setattr(data, "load_session", load)
    monkeypatch.setattr(research.time, "monotonic", clock)
    with pytest.raises(TimeoutError, match="preflight"):
        research.preflight_experiment(path, inspect_all=True, deadline=1)
    assert len(calls) == 1
    assert report.read_bytes() == before


@pytest.mark.parametrize("case", ["good", "empty", "short", "provenance"])
def test_quality_ratios_and_empty_development(tmp_path, case):
    from hft import research

    path, _ = frozen(
        tmp_path,
        bars=10 if case == "short" else 100,
        provenance="unknown" if case == "provenance" else "alpaca-historical-iex",
    )
    if case == "empty":
        refreeze(path, development=[])
    result = research.preflight_experiment(path, inspect_all=True)
    assert result["coverage_complete"] is True and result["paper_eligible"] is False
    assert result["status"] == ("data-ready" if case == "good" else "insufficient-data")
    expected_reasons = {
        "good": [],
        "empty": ["insufficient-history"],
        "short": ["insufficient-decision-warmup"],
        "provenance": ["unverified-data-provenance"],
    }
    assert result["reasons"] == expected_reasons[case]
    for row in [result, *result["sessions"]]:
        total = row["eligible_decisions"] + row["ineligible_decisions"]
        assert row["eligible_fraction"] == (row["eligible_decisions"] / total if total else None)
    if case == "short":
        assert all(row["eligible_fraction"] is None for row in result["sessions"])


@pytest.mark.parametrize("gap", [False, True])
def test_quality_reports_memory_without_changing_events(tmp_path, monkeypatch, gap):
    import resource
    import weakref
    from types import SimpleNamespace

    from hft import data, research

    path, _ = frozen(tmp_path, gap=gap, metadata=True)
    original_load = data.load_session
    original_size = data.session_bytes
    prior = None
    numeric = []

    def load(*args, **kwargs):
        nonlocal prior
        if prior is not None:
            assert prior() is None, "previous session retained across loads"
        session = original_load(*args, **kwargs)
        prior = weakref.ref(session.bid)
        numeric.append(sum(getattr(session, name).nbytes for name in data.NUMERIC_NAMES))
        return session

    def size(session):
        before = list(session.iter_events())
        result = original_size(session)
        assert list(session.iter_events()) == before
        return result

    monkeypatch.setattr(data, "load_session", load)
    monkeypatch.setattr(data, "session_bytes", size)
    monkeypatch.setattr(resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=12345))
    result = research.preflight_experiment(path, inspect_all=True)
    for row, expected_numeric in zip(result["sessions"], numeric, strict=True):
        memory = row["memory"]
        for key in ("numeric_bytes", "metadata_bytes", "retained_bytes", "process_peak_rss_bytes"):
            assert isinstance(memory[key], int) and memory[key] >= 0
        assert memory["numeric_bytes"] == expected_numeric
        assert memory["metadata_bytes"] > 0
        assert memory["metadata_bytes"] == sum(
            sum(columns.values()) for columns in memory["metadata_columns_bytes"].values()
        )
        assert memory["retained_bytes"] == memory["numeric_bytes"] + memory["metadata_bytes"]
        assert memory["process_peak_rss_bytes"] == 12345 * 1024


@pytest.mark.parametrize("case", ["last-decision", "last-warmup", "empty", "default-last-decision"])
def test_preflight_deadline_preserves_report_after_final_work(tmp_path, monkeypatch, case):
    from hft import execution, research

    path, _ = frozen(
        tmp_path, gap=case == "last-decision", bars=10 if case == "last-warmup" else 100
    )
    research.preflight_experiment(path)
    report = path.parent / "diagnostics.json"
    before = report.read_bytes()
    last = json.loads(path.read_text())["development"][-1]
    expired = False
    if case == "empty":
        refreeze(path, development=[])
        expired = True
    elif case == "last-warmup":
        original = execution.Simulation

        def simulation(session):
            nonlocal expired
            try:
                return original(session)
            finally:
                if session.session_id == last:
                    expired = True

        monkeypatch.setattr(execution, "Simulation", simulation)
    else:
        original = execution.Simulation.advance_to

        def advance_to(simulation, now_ns):
            nonlocal expired
            result = original(simulation, now_ns)
            if simulation.data.session_id == last and now_ns == simulation.bars[-1].end_ns:
                expired = True
            return result

        monkeypatch.setattr(execution.Simulation, "advance_to", advance_to)
    monkeypatch.setattr(research.time, "monotonic", lambda: 2 if expired else 0)
    with pytest.raises(TimeoutError, match="preflight"):
        research.preflight_experiment(path, inspect_all=case != "default-last-decision", deadline=1)
    assert report.read_bytes() == before
