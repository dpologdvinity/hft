from decimal import Decimal

import pytest

from hft.account import Account, Costs
from hft.calendar import SessionWindow
from hft.data import NS, Quote
from hft.risk import RiskGateway, Snapshot
from hft.sizing import SizingConfig, make_budget_intent, make_intent
from hft.trading.budget import AccountGuard, budget_gateway, budget_intent_factory

NOW = 1_735_830_000 * NS
SESSION = SessionWindow("2025-01-02", NOW - NS, NOW + 1000 * NS)
QUOTE = Quote("q", NOW, NOW, Decimal("99.99"), Decimal("100.01"), Decimal(500), Decimal(500))
LIMITS = SizingConfig(slippage_bps=0, fee_per_share=0)


def _snapshot(account):
    equity = account.mark(QUOTE.mid)
    return Snapshot(
        QUOTE, SESSION, account.position, equity, account.initial_cash, account.cash, ()
    )


def test_budget_entry_uses_the_whole_stock_budget():
    account = Account(200, Costs(0, 0))
    intent = make_budget_intent(1, account, QUOTE, LIMITS, NOW, budget=200, symbol="NVDA")
    assert intent.side == "buy" and intent.symbol == "NVDA"
    assert intent.quantity * intent.limit_price <= Decimal(200)
    assert intent.quantity * intent.limit_price > Decimal("199.9")
    fraction = make_intent(1, account, QUOTE, LIMITS, NOW, symbol="NVDA")
    assert fraction.quantity * fraction.limit_price <= Decimal(20)  # default 10% path unchanged


def test_budget_entry_never_exceeds_cash_or_minimum_notional():
    account = Account(50, Costs(0, 0))
    intent = make_budget_intent(1, account, QUOTE, LIMITS, NOW, budget=200)
    assert intent.quantity * intent.limit_price <= Decimal(50)
    assert make_budget_intent(1, account, QUOTE, LIMITS, NOW, budget="0.5") is None
    assert budget_intent_factory(200)(1, account, QUOTE, LIMITS, NOW).side == "buy"


def test_entry_cap_replaces_the_fraction_limit_only_when_set():
    account = Account(200, Costs(0, 0))
    intent = make_budget_intent(1, account, QUOTE, LIMITS, NOW, budget=200)
    assert budget_gateway(200).evaluate(intent, _snapshot(account), NOW).allowed
    assert RiskGateway().evaluate(intent, _snapshot(account), NOW).reason == "exposure_limit"
    assert budget_gateway(100).evaluate(intent, _snapshot(account), NOW).reason == "exposure_limit"
    with pytest.raises(ValueError):
        RiskGateway(entry_cap=0)


def test_wide_spread_blocks_entries_but_never_exits():
    wide = Quote("w", NOW, NOW, Decimal("99.90"), Decimal("100.10"), Decimal(500), Decimal(500))
    account = Account(200, Costs(0, 0))
    snapshot = Snapshot(wide, SESSION, 0, Decimal(200), Decimal(200), Decimal(200), ())
    buy = make_budget_intent(1, account, wide, LIMITS, NOW, budget=100)
    assert budget_gateway(200).evaluate(buy, snapshot, NOW).reason == "wide_spread"  # 20 bp
    assert budget_gateway(200, max_spread_bps=25).evaluate(buy, snapshot, NOW).allowed
    assert RiskGateway(entry_cap=200).evaluate(buy, snapshot, NOW).allowed  # off unless set
    assert budget_gateway(200).evaluate(buy, _snapshot(account), NOW).allowed  # 2 bp
    account.execute(1, 100, 0, NOW)
    held = Snapshot(wide, SESSION, account.position, Decimal(200), Decimal(200), account.cash, ())
    sell = make_budget_intent(0, account, wide, LIMITS, NOW, budget=100)
    assert budget_gateway(200).evaluate(sell, held, NOW).allowed
    with pytest.raises(ValueError):
        RiskGateway(max_entry_spread_bps=0)


def test_account_guard_daily_loss_resets_and_drawdown_latches():
    guard = AccountGuard(1000, daily_loss=0.02, max_drawdown=0.05)
    guard.start_session(Decimal(1000))
    assert guard.observe(Decimal(990)) is None
    assert guard.observe(Decimal(980)) == "account_daily_loss"
    guard.start_session(Decimal(980))
    assert guard.observe(Decimal(975)) is None  # new session clears the daily halt
    assert guard.observe(Decimal(950)) == "account_drawdown"
    guard.start_session(Decimal(1100))
    assert guard.halted == "account_drawdown"  # latched for the rest of the run
    with pytest.raises(ValueError):
        AccountGuard(0)
