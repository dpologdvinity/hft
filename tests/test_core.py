from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hft.account import Account, Costs
from hft.data import Bar, MarketData, synthetic_data
from hft.env import TradingEnv
from hft.features import observation


def bars(n=100, price=100.0):
    start = 1_735_826_400_000_000_000  # 2025-01-02 14:00 UTC
    return [Bar(start + i * 1_000_000_000, price, price, price, price,
                100, price - .05, price + .05, 100, 100) for i in range(n)]


def test_spread_fees_reversals_and_equity_reconcile():
    a = Account(1000, Costs(commission=.01, slippage_bps=0))
    q = bars()[0]
    a.target(1, q)
    assert a.cash == pytest.approx(899.94)
    assert a.equity(100) == pytest.approx(999.94)
    assert a.target(1, q) is None
    a.target(-1, q)
    assert a.closed_pnl == pytest.approx([-.12])
    a.target(0, q)
    assert a.cash == pytest.approx(999.76)
    assert sum(a.closed_pnl) == pytest.approx(a.cash - 1000)
    assert a.position == 0


def test_invalid_data_and_configuration_fail_closed():
    for change in [{'bid': 101}, {'close': float('nan')}, {'volume': -1}, {'high': 99}]:
        with pytest.raises(ValueError):
            MarketData([bars()[0], replace(bars()[1], **change)])
    with pytest.raises(ValueError):
        MarketData([bars()[0], bars()[0]])
    with pytest.raises(ValueError):
        Costs(slippage_bps=-1)
    with pytest.raises(ValueError):
        Account(float('nan'))


def test_causal_features_and_stream_batch_parity():
    rows = bars()
    rows[59] = replace(rows[59], close=101, high=101, ask=101.05, bid=100.95)
    a = Account()
    before = observation(rows[:61], a)
    assert before.shape == (15,)
    assert before.dtype == np.float32
    rows[80] = replace(rows[80], close=500, high=500, ask=500.05, bid=499.95)
    np.testing.assert_array_equal(before, observation(rows[:61], a))
    np.testing.assert_array_equal(before, observation(rows[0:61], a))
    assert before[0] == pytest.approx(np.log(100 / 101), abs=1e-7)


def test_action_fills_at_next_quote_and_terminal_liquidation_costs():
    rows = bars(63)
    rows[61] = replace(rows[61], open=110, high=110, low=110, close=110,
                       bid=109.95, ask=110.05)
    env = TradingEnv(MarketData(rows), costs=Costs(.01, 0), episode_steps=10)
    env.reset(options={'start': 60})
    _, reward, terminated, truncated, info = env.step(1)
    assert env.account.entry_price == pytest.approx(110.05)
    assert env.account.equity(110) == pytest.approx(9999.94)
    assert reward == pytest.approx(-.06)  # costs once, reward in dollars
    assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(1)
    assert truncated
    assert env.account.position == 0
    assert env.account.cash == pytest.approx(9989.88)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_parquet_roundtrip_and_seeded_augmentation(tmp_path):
    data = synthetic_data(days=2, bars_per_day=100, seed=7)
    path = tmp_path / 'prices.parquet'
    data.write(path)
    loaded = MarketData.read(path)
    assert loaded.synthetic and loaded.symbol == 'SYNTH'
    assert loaded.rows == data.rows
    env = TradingEnv(loaded, augment=True, episode_steps=10)
    first, _ = env.reset(seed=9)
    second, _ = env.reset(seed=9)
    np.testing.assert_array_equal(first, second)
    check_env(TradingEnv(loaded, episode_steps=10), skip_render_check=True)
