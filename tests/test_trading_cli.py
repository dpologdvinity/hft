import json
from decimal import Decimal

import pytest

from hft.cli import main
from hft.strategies import parse_strategy
from hft.trading.command import LIVE_REFUSAL, parse_symbols, run_identity, run_paths
from hft.trading.status import format_status


@pytest.mark.parametrize(
    "argv",
    [
        ["trade", "--symbols", "NVDA=200"],
        ["trade", "--paper", "--live", "--symbols", "NVDA=200"],
    ],
)
def test_real_money_is_refused(argv, capsys):
    assert main(argv) == 1
    assert LIVE_REFUSAL in capsys.readouterr().err


def test_symbol_budget_parsing():
    assert parse_symbols(["nvda=200", "AAPL=100.50"]) == {
        "NVDA": Decimal(200),
        "AAPL": Decimal("100.50"),
    }
    for bad in (
        ["NVDA"],
        ["NVDA=0"],
        ["NVDA=-5"],
        ["NVDA=abc"],
        ["NVDA=1", "NVDA=2"],
        ["1BAD=5"],
        [],
        [f"S{i}=1".replace("S", "A") for i in range(31)],
    ):
        with pytest.raises(ValueError):
            parse_symbols(bad)


def test_run_names_are_validated(tmp_path):
    state, logs = run_paths("my-run_1", tmp_path)
    assert state == tmp_path / ".state" / "trade" / "my-run_1"
    assert logs == tmp_path / "logs" / "trade" / "my-run_1"
    for bad in ("../escape", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            run_paths(bad, tmp_path)


def test_identity_changes_with_every_input():
    base = {
        "daily_loss": 0.02,
        "max_drawdown": 0.05,
        "engine": "cpp",
        "engine_version": "0.1.0",
        "feed": "iex",
    }
    budgets, strategy = {"NVDA": Decimal(200)}, parse_strategy("hold-day")
    reference = run_identity(budgets, strategy, **base)
    assert reference == run_identity(dict(budgets), strategy, **base)
    variants = [
        run_identity({"NVDA": Decimal(201)}, strategy, **base),
        run_identity(budgets, parse_strategy("ema-crossover"), **base),
        run_identity(budgets, strategy, **{**base, "daily_loss": 0.03}),
        run_identity(budgets, strategy, **{**base, "engine": "python"}),
    ]
    assert len({reference, *variants}) == 5


def test_status_reads_state_and_journal_tail(tmp_path):
    assert "no paper trading runs" in format_status(tmp_path)
    state_dir = tmp_path / ".state" / "trade" / "demo"
    log_dir = tmp_path / "logs" / "trade" / "demo"
    state_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    book = {
        "account": {"position": "1.500000", "cash": "50.00", "initial_cash": "200"},
        "risk": {"halted": None},
        "pending": None,
    }
    (state_dir / "x.json").write_text(json.dumps({"state": {"books": {"NVDA": book}}}))
    journal = [{"event": "equity", "equity": "201.25", "wall_ns": 1}, {"event": "timer"}]
    (log_dir / "NVDA-1.jsonl").write_text("\n".join(json.dumps(r) for r in journal) + "\n")
    text = format_status(tmp_path, "demo")
    assert "NVDA" in text and "201.25" in text and "1.25" in text and "trading" in text
