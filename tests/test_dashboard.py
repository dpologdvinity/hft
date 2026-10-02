import hashlib
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hft.dashboard import Dashboard, create_server


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def workspace(root):
    source = root / "data/mcd/manifest.json"
    sessions = []
    for date, price in [("2026-06-04", 234.5), ("2026-09-30", 999.0)]:
        part = source.parent / date / "trades.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"trade_price": [price]}), part)
        manifest = part.parent / "manifest.json"
        write_json(
            manifest,
            {
                "partitions": {
                    "trades": {
                        "path": "trades.parquet",
                        "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
                    }
                }
            },
        )
        sessions.append(
            {
                "session_id": date,
                "manifest": f"{date}/manifest.json",
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    write_json(source, {"status": "complete", "sessions": sessions})
    experiment = root / "artifacts/mcd/experiment.json"
    write_json(
        experiment,
        {
            "status": "frozen",
            "symbol": "MCD",
            "feed": "iex",
            "synthetic": False,
            "dataset_manifest": str(source),
            "dataset_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
            "session_ids": ["2026-06-04", "2026-09-30"],
            "development": ["2026-06-04"],
            "final_test": ["2026-09-30"],
            "final_test_consumed": False,
            "folds": [{}],
            "config": {
                "initial_cash": 500,
                "timesteps": 100000,
                "seeds": [42, 43],
                "wall_seconds": 7200,
                "secret": "never-publish-this",
            },
        },
    )
    return experiment


def test_dashboard_uses_real_development_prices_and_hides_test_and_secrets(tmp_path):
    workspace(tmp_path)
    result = Dashboard(tmp_path).snapshot()
    assert result["market_history"] == [{"date": "2026-06-04", "close": 234.5}]
    assert result["dataset"]["final_test_sessions"] == 1
    assert result["training"]["status"] == "idle"
    assert result["training"]["data_validated"] is False
    assert result["training"]["paper_eligible"] is False
    assert result["training"]["steps_completed"] is None
    assert "999.0" not in json.dumps(result)
    assert "never-publish-this" not in json.dumps(result)


def test_market_history_rejects_overlapping_test_dates_before_reading(tmp_path):
    path = workspace(tmp_path)
    experiment = json.loads(path.read_text())
    experiment["development"] += experiment["final_test"]
    write_json(path, experiment)
    result = Dashboard(tmp_path).snapshot()
    assert result["market_history"] == []
    assert result["warnings"]
    assert "999.0" not in json.dumps(result)


@pytest.mark.parametrize("damage", ["trades", "manifest", "missing"])
def test_cached_market_history_detects_changed_development_files(tmp_path, damage):
    workspace(tmp_path)
    dashboard = Dashboard(tmp_path)
    assert dashboard.snapshot()["market_history"] == [{"date": "2026-06-04", "close": 234.5}]
    root = tmp_path / "data/mcd/2026-06-04"
    if damage == "trades":
        pq.write_table(pa.table({"trade_price": [1234.0]}), root / "trades.parquet")
    elif damage == "manifest":
        (root / "manifest.json").write_text("{}")
    else:
        (root / "trades.parquet").unlink()
    result = dashboard.snapshot()
    assert result["market_history"] == []
    assert result["warnings"] and result["training"]["data_validated"] is False


def test_empty_workspace_and_escaped_source_fail_closed(tmp_path):
    assert Dashboard(tmp_path).snapshot()["dataset"] is None
    experiment = workspace(tmp_path)
    data = json.loads(experiment.read_text())
    data["dataset_manifest"] = str(tmp_path.parent / ".env")
    write_json(experiment, data)
    result = Dashboard(tmp_path).snapshot()
    assert result["market_history"] == []
    assert result["warnings"]


def test_failed_coverage_is_visible_without_final_test_results(tmp_path):
    path = workspace(tmp_path)
    experiment = json.loads(path.read_text())
    experiment["experiment_hash"] = "frozen"
    write_json(path, experiment)
    write_json(
        path.parent / "diagnostics.json",
        {
            "experiment_hash": "frozen",
            "status": "insufficient-data",
            "coverage_sessions": 1,
            "development_sessions": 52,
            "coverage_complete": False,
            "feed_gaps": 935,
            "live_comparable": False,
        },
    )
    write_json(
        path.parent / "search/research.json",
        {
            "status": "insufficient-data",
            "paper_eligible": False,
            "eligibility": {"reasons": ["runtime-feed-gap"]},
            "test": {"policy": {"net_profit": 999}},
        },
    )
    result = Dashboard(tmp_path).snapshot()
    assert result["training"]["data_validated"] is False
    assert result["training"]["held_out_evaluated"] is False
    assert "five-second" in result["training"]["message"]
    assert result["runs"][0]["details"]["decision_coverage"]["feed_gaps"] == 935
    assert "result" not in result["runs"][0]["details"]


def test_paper_metrics_require_valid_real_journal_and_never_enable_trading(tmp_path):
    from hft.logs import EventLog

    workspace(tmp_path)
    path = tmp_path / "logs/dry-run/session.jsonl"
    log = EventLog(path)
    log.write(
        "start",
        source="dry-run",
        synthetic=False,
        initial_cash=100,
        account_state={"cash": "100", "position": "0"},
    )
    log.write("equity", equity="100")
    log.write("equity", equity="99")
    log.write("trade", pnl="-1")
    log.write("session", date="2026-10-02", complete=True, equity="99", position="0")
    log.close()
    dashboard = Dashboard(tmp_path)
    paper = dashboard.snapshot()["paper"]
    assert paper["metrics"]["net_profit"] == -1
    assert paper["metrics"]["max_drawdown"] == 0.01
    assert paper["metrics"]["trades"] == 1
    assert paper["sessions"][0]["cash"] == 99.0
    assert paper["sessions"][0]["position"] == 0.0
    assert paper["eligible"] is False
    path.write_text(path.read_text().replace('"equity":"99"', '"equity":"999"'))
    result = dashboard.snapshot()
    assert result["paper"]["metrics"] is None
    assert result["warnings"]


def test_restarted_paper_run_drawdown_starts_at_observed_equity(tmp_path):
    from hft.logs import EventLog

    workspace(tmp_path)
    log = EventLog(tmp_path / "logs/dry-run/restarted.jsonl")
    log.write(
        "start",
        source="dry-run",
        synthetic=False,
        initial_cash=500,
        account_state={"cash": "90", "position": "1"},
    )
    log.write("equity", equity="120")
    log.write("equity", equity="118")
    log.write("session", date="2026-10-02", complete=False, equity="90", position="1")
    log.close()
    result = Dashboard(tmp_path).snapshot()["paper"]
    assert result["metrics"]["net_profit"] is None
    assert result["metrics"]["max_drawdown"] == pytest.approx(2 / 120)
    assert result["sessions"][0]["cash"] == 90
    assert result["sessions"][0]["position"] == 1


def test_http_serves_only_built_assets_and_read_only_api(tmp_path):
    workspace(tmp_path)
    built = tmp_path / "frontend/dist"
    built.mkdir(parents=True)
    (built / "index.html").write_text("<html>HFT</html>")
    (tmp_path / ".env").write_text("SECRET=value")
    (built / "leak.txt").symlink_to(tmp_path / ".env")
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/dashboard") as response:
            assert json.load(response)["read_only"] is True
            assert response.headers["Cache-Control"] == "no-store"
        assert b"HFT" in urlopen(base).read()
        for path in ("/.env", "/../.env", "/leak.txt"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
        for headers in ({"Host": "attacker.invalid"}, {"Origin": "https://attacker.invalid"}):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/api/dashboard", headers=headers))
            assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/dashboard", data=b"{}", method="POST"))
        assert error.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
