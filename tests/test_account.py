from decimal import Decimal as D

import pytest

from hft.account import Account, Execution


def test_fractional_roundtrip_and_dedup():
    a = Account(500)
    buy = Execution("a", "buy", 1, D(".5"), D("100.02"), D(".01"))
    a.apply(buy)
    a.apply(buy)
    assert a.cash == D("449.980")
    assert a.mark(D(100)) == D("499.980")
    a.apply(Execution("b", "sell", 2, D("-.2"), D(101), D(".004")))
    assert a.cash == D("470.176")
    assert a.realized_pnl == D(".188")
    assert not a.completed_trades
    a.apply(Execution("c", "sell", 3, D("-.3"), D(101), D(".006")))
    assert a.cash == D("500.470")
    assert len(a.completed_trades) == 1
    assert a.completed_trades[0].pnl == D(".470")
    with pytest.raises(ValueError):
        a.apply(Execution("a", "buy", 1, D(".6"), D("100.02"), D(".01")))


def test_invalid_execution_is_atomic():
    a = Account(500)
    for q, p, f in [
        ("-.1", "100", "0"),
        ("6", "100", "0"),
        ("NaN", "100", "0"),
        ("1", "0", "0"),
        ("1", "100", "-1"),
    ]:
        with pytest.raises(ValueError):
            a.apply(Execution(q, "x", 1, D(q), D(p), D(f)))
    assert a.cash == 500 and a.position == 0


def test_cash_flows_are_not_trading_pnl():
    a = Account(500)
    a.record_cash_flow(100, 1)
    assert a.cash == a.initial_cash == 600 and a.realized_pnl == 0
    a.record_cash_flow(-50, 2)
    assert a.cash == a.initial_cash == 550 and not a.completed_trades
