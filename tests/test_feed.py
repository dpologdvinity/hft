import pytest

from hft.calendar import SessionWindow
from hft.data import NS, build_bars, synthetic_sessions
from hft.feed import BarAggregator, parse_timestamp

BASE = parse_timestamp("2025-01-02T14:30:00Z")


def q(seconds, price=100):
    return {
        "T": "q",
        "S": "X",
        "event_ns": BASE + int(seconds * NS),
        "bp": price - 0.01,
        "ap": price + 0.01,
        "bs": 1,
        "as": 2,
    }


def t(seconds, price, size=1, **kw):
    return {"T": "t", "S": "X", "event_ns": BASE + int(seconds * NS), "p": price, "s": size, **kw}


def agg():
    return BarAggregator("X", session=SessionWindow("2025-01-02", BASE, BASE + 20 * NS))


def test_exact_ns_and_aware_timestamp():
    assert parse_timestamp("2025-01-02T14:30:00.123456789Z") == BASE + 123456789
    with pytest.raises(ValueError):
        parse_timestamp("2025-01-02T14:30:00")


def test_half_open_trades_and_quote_lots():
    a = agg()
    for e in [q(0), t(1, 100, 2), t(4, 101, 3), q(4.9, 101)]:
        assert a.add(e) == []
    b = a.add(t(5, 999))[0]
    assert (b.open, b.close, b.volume) == (100, 101, 5)
    assert b.quote_ns == BASE + 4_900_000_000 and b.bid_size == 100
    assert b.end_ns == BASE + 5 * NS and b.tradable


def test_no_trade_stale_and_correction_immutable():
    a = agg()
    a.add(q(0))
    a.add(t(1, 100, i=1))
    b = a.advance_to(BASE + 5 * NS)[0]
    assert not b.tradable
    a.add(t(1, 999, i=1), arrival_ns=BASE + 6 * NS)
    a.add({"T": "c", "S": "X", "event_ns": BASE + 6 * NS, "oi": 1, "cp": 999})
    c = a.advance_to(BASE + 10 * NS)[0]
    assert b.close == c.close == 100 and c.volume == 0
    assert a.quality["late_closed"] == 1 and a.quality["corrections"] == 1


def test_reorder_open_interval_and_historical_stream_parity():
    a = agg()
    a.add(q(0))
    a.add(t(2, 101))
    a.add(t(1.9, 100), arrival_ns=BASE + 2 * NS)
    assert a.advance_to(BASE + 5 * NS)[0].close == 101
    s = synthetic_sessions(days=1, bars_per_day=5)[0]
    live = BarAggregator(s.symbol, session=SessionWindow(s.session_id, s.open_ns, s.close_ns))
    bars = []
    for e in s.iter_events():
        bars.extend(live.add(e))
    bars.extend(live.advance_to(s.close_ns))
    assert tuple(bars) == build_bars(s)


def test_excluded_close_condition_and_too_late():
    a = agg()
    a.add(q(0))
    a.add(t(1, 100))
    a.add(t(2, 999, c=["I"]))
    a.add(t(0.5, 888), arrival_ns=BASE + 3 * NS)
    b = a.advance_to(BASE + 5 * NS)[0]
    assert b.close == 100 and b.volume == 1
    assert a.quality["filtered_trades"] == 1 and a.quality["late_tolerance"] == 1


def test_quote_identity_distinguishes_same_timestamp_price_and_depth():
    from hft.feed import quote_from_event

    one = q(1)
    two = {**one, "ap": 101.0}
    three = {**one, "as": 3}
    assert len({quote_from_event(e).quote_id for e in (one, two, three)}) == 3
    assert quote_from_event(one).quote_id == quote_from_event(dict(one)).quote_id
    shares = {**one, "bs": 100.0, "as": 200.0, "sizes_in_shares": True}
    assert quote_from_event(one).quote_id == quote_from_event(shares).quote_id
