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


def test_off_grid_limit_encloses_configured_slippage_without_changing_fill_cost():
    from dataclasses import replace

    from hft.calendar import SessionWindow
    from hft.execution import LocalExecution
    from hft.risk import RiskGateway, Snapshot

    q = Quote("q", 10**9, None, D("99.999999"), D("100.012345"), D(10), D(10))
    a = Account(500)
    limits = SizingConfig(price_precision="0.0001")
    buy = make_intent(1, a, q, limits, 10**9)
    assert buy.limit_price >= q.ask * D("1.0001")
    assert buy.quantity * (buy.limit_price + D(".005")) <= D(50)
    g = RiskGateway()
    session = SessionWindow("s", 10**9, 1000 * 10**9, None)
    e = LocalExecution(a, g, latency_ms=75)
    snap = Snapshot(q, session, a.position, a.mark(q.mid), a.initial_cash, a.cash, ())
    assert e.submit(buy, snap, 10**9).allowed
    next_q = replace(q, quote_id="next", event_ns=10**9 + 100_000_000)
    fills = e.on_quote(next_q, next_q.event_ns, session=session)
    assert fills[0].price == q.ask * D("1.0001")
    sell = make_intent(0, a, next_q, limits, next_q.event_ns)
    assert sell.limit_price <= q.bid * D(".9999")
