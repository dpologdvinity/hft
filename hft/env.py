"""Next-row execution with explicit spread, commissions and terminal flattening."""

import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .account import Account, Costs
from .data import MarketData, augment as augment_data, session_day
from .features import FEATURE_NAMES, LOOKBACK, TARGETS, observation


class TradingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, data: MarketData, initial_cash=10_000.0, costs=None,
                 episode_steps=512, random_start=False, augment=False,
                 drawdown_penalty=0.0, downside_penalty=0.0):
        if len(data) < LOOKBACK + 2 or episode_steps < 1:
            raise ValueError('insufficient data or invalid episode length')
        if not all(math.isfinite(p) and p >= 0 for p in (drawdown_penalty, downside_penalty)):
            raise ValueError('reward penalties must be nonnegative and finite')
        self.source = data
        self.initial_cash, self.costs = initial_cash, costs or Costs()
        self.episode_steps, self.random_start, self.augment = episode_steps, random_start, augment
        self.drawdown_penalty, self.downside_penalty = drawdown_penalty, downside_penalty
        self.action_space = spaces.Discrete(3)
        bound = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-bound, bound, shape=(len(FEATURE_NAMES),),
                                            dtype=np.float32)
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        max_start = max(LOOKBACK, len(self.source) - self.episode_steps - 1)
        start = int(options.get('start', self.np_random.integers(LOOKBACK, max_start + 1)
                                if self.random_start else LOOKBACK))
        if not LOOKBACK <= start < len(self.source) - 1:
            raise ValueError('episode start has insufficient context or future rows')
        stop = min(start + self.episode_steps, len(self.source) - 1)
        if self.augment:
            self.data = augment_data(self.source.subset(start - LOOKBACK, stop + 1), self.np_random)
            self.index, self.stop = LOOKBACK, len(self.data) - 1
        else:
            self.data, self.index, self.stop = self.source, start, stop
        self.account = Account(self.initial_cash, self.costs)
        self.peak, self.last_drawdown, self.done = self.initial_cash, 0.0, False
        return self._observation(), self._info()

    def _observation(self):
        return observation(self.data.rows[max(0, self.index - LOOKBACK):self.index + 1], self.account)

    def _info(self):
        row = self.data.rows[self.index]
        return {'equity': self.account.equity(row.close), 'position': self.account.position,
                'timestamp': row.timestamp, 'fills': len(self.account.fills)}

    def step(self, action):
        if self.done:
            raise RuntimeError('reset before stepping a completed episode')
        if not self.action_space.contains(action):
            raise ValueError('invalid action')
        current, following = self.data.rows[self.index:self.index + 2]
        before = self.account.equity(current.close)
        if session_day(current.timestamp) != session_day(following.timestamp):
            self.account.target(0, current)
        else:
            self.account.target(TARGETS[int(action)], following)
        self.index += 1
        terminated = self.account.equity(following.close) <= 0
        truncated = self.index >= self.stop
        # Flatten before crossing a session boundary, including the last available row.
        end_session = (self.index == len(self.data) - 1 or
                       session_day(following.timestamp) !=
                       session_day(self.data.rows[self.index + 1].timestamp))
        if terminated or truncated or end_session:
            self.account.target(0, following)
        equity = self.account.equity(following.close)
        pnl = equity - before
        self.peak = max(self.peak, equity)
        drawdown = self.peak - equity
        reward = (pnl - self.drawdown_penalty * max(0, drawdown - self.last_drawdown)
                  - self.downside_penalty * min(pnl, 0) ** 2)
        self.last_drawdown, self.done = drawdown, terminated or truncated
        return self._observation(), float(reward), bool(terminated), bool(truncated), self._info()
