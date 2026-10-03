import os
import subprocess
import sys


def command(*args):
    return subprocess.run(
        [sys.executable, "-m", "hft", *args],
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if not k.startswith("ALPACA_")},
    )


def test_help_lists_the_operator_commands():
    result = command("--help")
    assert result.returncode == 0
    assert "smoke" in result.stdout and "live" in result.stdout


def test_unknown_command_fails():
    assert command("not-a-command").returncode == 2


def test_live_without_activation_fails_before_credentials_or_network():
    result = command("live")
    assert result.returncode != 0
    assert "enable-live" in result.stderr


def test_data_probe_without_credentials_is_actionable():
    result = command("data-probe", "--symbol", "AAPL", "--session", "2026-09-28")
    assert result.returncode != 0
    assert "ALPACA" in result.stderr


def test_report_preserves_insufficient_history_result(tmp_path):
    import json

    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"status": "insufficient-data", "reasons": ["need more sessions"]}))
    result = command("report", "--experiment", str(path))
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "insufficient-data"


def test_data_quality_help_explains_all_development():
    result = command("data-quality", "--help")
    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "--all-development" in help_text
    assert "inspect every development session" in help_text
    assert "report failures" in help_text
    assert "without fitting or reading final-test data" in help_text


def quality_dispatch(monkeypatch, *, all_development):
    from pathlib import Path

    from hft import cli, research

    for key in os.environ:
        if key.startswith("ALPACA_"):
            monkeypatch.delenv(key)

    def preflight(path, *, inspect_all):
        assert path == Path("artifacts/example/experiment.json")
        assert inspect_all is all_development
        return {"inspection_mode": "all-development" if inspect_all else "stop-first-failure"}

    monkeypatch.setattr(research, "preflight_experiment", preflight)
    args = ["data-quality", "--experiment", "artifacts/example/experiment.json"]
    if all_development:
        args.append("--all-development")
    return cli._handle(cli._parser().parse_args(args))


def test_data_quality_defaults_to_early_stop(monkeypatch):
    assert quality_dispatch(monkeypatch, all_development=False) == {
        "inspection_mode": "stop-first-failure"
    }


def test_data_quality_explicitly_requests_all_development(monkeypatch):
    assert quality_dispatch(monkeypatch, all_development=True) == {
        "inspection_mode": "all-development"
    }
