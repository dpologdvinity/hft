from dataclasses import replace
from decimal import Decimal as D

import numpy as np
import pytest

from hft.account import Account, Costs
from hft.calendar import SessionWindow
from hft.data import build_bars, synthetic_sessions
from hft.env import TradingEnv
from hft.features import FEATURE_NAMES, observation
from hft.risk import RiskGateway


def test_invalid_configuration_fails_closed():
    with pytest.raises(ValueError):
        Costs(slippage_bps=-1)
    with pytest.raises(ValueError):
        Account(float("nan"))
    with pytest.raises(ValueError):
        TradingEnv([])


def test_manual_causal_features_future_invariance():
    session = synthetic_sessions(days=1, bars_per_day=100)[0]
    rows = list(build_bars(session))
    rows[59] = replace(rows[59], close=101, high=max(rows[59].high, 101))
    rows[60] = replace(
        rows[60], close=100, low=min(rows[60].low, 100), high=max(rows[60].high, 100)
    )
    account = Account(500)
    window = SessionWindow(session.session_id, session.open_ns, session.close_ns, None)
    risk = RiskGateway()
    before = observation(rows[:61], account, risk, window)
    assert before.shape == (17,) and len(FEATURE_NAMES) == 17
    assert before.dtype == np.float32
    assert before[0] == pytest.approx(np.log(100 / 101), abs=1e-7)
    assert before[10] == 0 and before[12] == 1 and before[14] == 1 and before[15] == 1
    rows[80] = replace(rows[80], close=500, high=500)
    np.testing.assert_array_equal(before, observation(rows[:61], account, risk, window))


def test_flat_and_constant_market_costs_once():
    session = synthetic_sessions(days=1, bars_per_day=80)[0]
    session = replace(
        session,
        bid=np.full(len(session.bid), 99.99),
        ask=np.full(len(session.ask), 100.01),
        trade_price=np.full(len(session.trade_price), 100.0),
    )
    env = TradingEnv(session, costs=Costs(0.005, 1), episode_steps=3)
    env.reset()
    rewards = []
    for action in (1, 0, 0):
        rewards.append(env.step(action)[1])
    assert env.done and env.account.position == 0
    loss = float(env.account.cash - D(500))
    assert sum(rewards) == pytest.approx(10000 * loss * 1.1 / 500, abs=1e-10)
    assert len(env.account.completed_trades) == 1
    assert env.account.completed_trades[0].pnl == env.account.cash - D(500)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_seeded_augmentation():
    session = synthetic_sessions(days=1, bars_per_day=100)[0]
    env = TradingEnv(session, augment=True, episode_steps=10)
    first, _ = env.reset(seed=9)
    second, _ = env.reset(seed=9)
    np.testing.assert_array_equal(first, second)


def test_feature_streaming_deque_matches_historical_tuple():
    from collections import deque

    from hft.feed import BarAggregator

    session = synthetic_sessions(days=1, bars_per_day=80)[0]
    bars = build_bars(session)
    window = SessionWindow(session.session_id, session.open_ns, session.close_ns, None)
    aggregator = BarAggregator(session.symbol, session=window)
    streamed = []
    for event in session.iter_events():
        streamed.extend(aggregator.add(event))
    streamed.extend(aggregator.advance_to(session.close_ns))
    account = Account(500)
    risk = RiskGateway()
    window = SessionWindow(session.session_id, session.open_ns, session.close_ns, None)
    np.testing.assert_array_equal(
        observation(bars[:61], account, risk, window),
        observation(deque(streamed[:61]), account, risk, window),
    )
