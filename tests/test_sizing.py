from decimal import Decimal as D

from hft.account import Account
from hft.data import Quote
from hft.sizing import SizingConfig, make_intent


def test_capital_fraction_fee_headroom_and_no_rebalance():
    q = Quote("q", 1, None, D(100), D("100.02"), D(10), D(10))
    for capital in (500, 5000):
        a = Account(capital)
        intent = make_intent(1, a, q, SizingConfig(), 1)
        assert intent.quantity * (intent.limit_price + D(".005")) <= D(str(capital)) * D(".10")
        a.execute(intent.quantity, intent.limit_price, intent.quantity * D(".005"), 1)
        assert make_intent(1, a, q, SizingConfig(), 2) is None
        assert make_intent(0, a, q, SizingConfig(), 2).quantity == a.position
