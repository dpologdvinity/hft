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
