from dataclasses import replace
from decimal import Decimal as D

from hft.calendar import SessionWindow
from hft.data import Quote
from hft.risk import RiskGateway, Snapshot
from hft.sizing import OrderIntent

NS = 10**9


def snap(equity=500, position=0, now=100 * NS, session_id="day", open_ns=1):
    return Snapshot(
        Quote(str(now), now, None, D(100), D(100), D(10), D(10)),
        SessionWindow(session_id, open_ns, open_ns + 1000 * NS, None),
        D(position),
        D(equity),
        D(500),
        D(500),
        (),
    )


def intent(side="buy", quantity=".4", now=100 * NS):
    return OrderIntent(str(now), "SPY", side, D(quantity), D(100), now)


def test_daily_inclusive_latch_and_restart_peak():
    g = RiskGateway()
    g.observe(snap(), 100 * NS)
    assert g.observe(snap(495, 1), 100 * NS).reason == "daily_loss"
    assert not g.observe(snap(500), 100 * NS).allowed
    g.observe(snap(500, session_id="next", open_ns=1000 * NS), 1100 * NS)
    assert g.halted is None
    g.observe(snap(475, session_id="next", open_ns=1000 * NS), 1100 * NS)
    restored = RiskGateway.from_state(g.to_state())
    assert restored.permanent_halt
    assert (
        restored.observe(snap(500, session_id="third", open_ns=2000 * NS), 2100 * NS).reason
        == "max_drawdown"
    )


def test_freshness_rate_inventory_and_exits():
    g = RiskGateway()
    s = snap()
    assert g.evaluate(intent(), s, 100 * NS).allowed
    assert g.evaluate(intent(), s, 102 * NS).allowed
    assert g.evaluate(intent(), s, 102 * NS + 1).reason == "stale_quote"
    assert g.evaluate(intent(), s, 100 * NS - 250_000_001).reason == "stale_quote"
    for i in range(5):
        g.record_order(100 * NS + i)
    assert g.evaluate(intent(), s, 100 * NS + 5).reason == "order_rate_limit"
    held = replace(s, position=D(".5"))
    assert g.evaluate(intent("sell"), held, 100 * NS).allowed
    assert g.evaluate(intent("sell", ".6"), held, 100 * NS).reason == "inventory_limit"
    assert g.evaluate(intent(quantity=".51"), s, 100 * NS).reason == "exposure_limit"


def test_blackout_preclose_and_older_session():
    g = RiskGateway(blackouts=((100 * NS, 101 * NS),))
    s = snap()
    assert g.evaluate(intent(), s, 100 * NS).reason == "news_blackout"
    assert g.evaluate(
        intent(), replace(s, quote=replace(s.quote, event_ns=101 * NS)), 101 * NS
    ).allowed
    late = snap(now=940 * NS)
    assert g.evaluate(intent(), late, 940 * NS + 1).reason == "pre_close"
    g.observe(snap(session_id="next", open_ns=1000 * NS), 1100 * NS)
    assert g.observe(s, 100 * NS).reason == "older_session"


def test_atr_spike_excludes_itself_and_pauses_for_minute():
    from hft.data import build_bars, synthetic_sessions

    rows = list(build_bars(synthetic_sessions(days=1, bars_per_day=80)[0]))
    rows = [replace(b, open=100, high=100.01, low=99.99, close=100) for b in rows]
    rows[60] = replace(rows[60], high=105, low=95)
    s = replace(snap(), history=tuple(rows[:61]))
    g = RiskGateway()
    assert g.evaluate(intent(), s, 100 * NS).reason == "volatility_breaker"
    assert g.evaluate(intent("sell", ".2"), replace(s, position=D(".2")), 100 * NS).allowed
    s = replace(s, quote=replace(s.quote, event_ns=160 * NS))
    assert g.evaluate(intent(), s, 160 * NS).allowed


def test_maintenance_cushion_and_no_order_observation():
    g = RiskGateway()
    s = snap(position=".6")
    assert g.observe(s, 100 * NS).allowed
    s = replace(s, position=D(".751"))
    d = g.observe(s, 100 * NS)
    assert d.requires_flatten and d.reason == "maintenance_exposure"
