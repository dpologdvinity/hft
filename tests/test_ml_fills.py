import numpy as np
import pytest

from hft.ml.fills import simulate_day

COST = (1.5 + 1) / 1e4


def _day(n=390):
    opens = np.linspace(100, 101, n)
    return opens, np.ones(n, bool), np.full(n, -1.0)


def test_entry_fills_at_the_next_minutes_open_never_the_decision_minute():
    opens, valid, signal = _day()
    signal[10] = 1.0
    trade = simulate_day(opens, valid, signal, enter=0.5, half_spread_bps=1.5, last_minute=385)
    assert trade.entry_minute == 11
    assert trade.exit_minute == 12  # signal is negative again at minute 11: sell next open
    expected = opens[12] * (1 - COST) / (opens[11] * (1 + COST)) - 1
    assert trade.net_return == pytest.approx(expected)


def test_missing_minutes_are_skipped_for_fills():
    opens, valid, signal = _day()
    signal[10:20] = 1.0
    valid[11] = False
    trade = simulate_day(opens, valid, signal, enter=0.5, half_spread_bps=1.5, last_minute=385)
    assert trade.entry_minute == 12


def test_positions_are_flat_by_the_last_minute_and_trade_once():
    opens, valid, signal = _day()
    signal[:] = 1.0
    trade = simulate_day(opens, valid, signal, enter=0.5, half_spread_bps=1.5, last_minute=200)
    assert trade.entry_minute == 1 and trade.exit_minute == 200
    signal[:] = -1.0
    signal[[5, 50]] = 1.0  # a second chance after the exit is ignored
    trade = simulate_day(opens, valid, signal, enter=0.5, half_spread_bps=1.5, last_minute=200)
    assert (trade.entry_minute, trade.exit_minute) == (6, 7)


def test_costs_are_paid_on_both_sides_and_no_signal_means_no_trade():
    opens, valid, signal = np.full(390, 100.0), np.ones(390, bool), np.full(390, -1.0)
    signal[0] = 1.0
    trade = simulate_day(opens, valid, signal, enter=0.5, half_spread_bps=1.5, last_minute=385)
    assert trade.net_return == pytest.approx((1 - COST) / (1 + COST) - 1)
    assert (
        simulate_day(opens, valid, -signal * 0 - 1, enter=0.5, half_spread_bps=1.5, last_minute=385)
        is None
    )


def test_no_entry_too_late_to_fill_before_the_last_minute():
    opens, valid, signal = _day()
    signal[199] = 1.0
    assert (
        simulate_day(opens, valid, signal, enter=0.5, half_spread_bps=1.5, last_minute=200) is None
    )


def test_a_cost_multiple_scales_both_sides():
    opens, valid, signal = np.full(390, 100.0), np.ones(390, bool), np.full(390, -1.0)
    signal[0] = 1.0
    trade = simulate_day(
        opens, valid, signal, enter=0.5, half_spread_bps=1.5, last_minute=385, cost_multiple=3
    )
    assert trade.net_return == pytest.approx((1 - 3 * COST) / (1 + 3 * COST) - 1)


def test_cheap_stocks_pay_at_least_half_a_cent():
    from hft.ml.fills import half_spread_bps

    assert half_spread_bps(1.5, 100.0) == 1.5
    assert half_spread_bps(1.5, 5.0) == pytest.approx(10.0)  # half of one cent on $5
