"""Gym wrapper around the shared causal quote simulation."""

from dataclasses import replace
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .account import Costs
from .execution import Simulation
from .features import FEATURE_NAMES, observation


class TradingEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        data,
        initial_cash=500,
        costs=None,
        episode_steps=512,
        random_start=False,
        augment=False,
        latency_ms=75,
    ):
        if episode_steps < 1:
            raise ValueError("invalid episode length")
        self.sessions = (data,) if hasattr(data, "quote_ns") else tuple(data)
        if not self.sessions:
            raise ValueError("no sessions")
        self.initial_cash = initial_cash
        self.costs = costs or Costs()
        self.episode_steps = episode_steps
        self.random_start = random_start
        self.augment = augment
        self.latency_ms = latency_ms
        self.action_space = spaces.Discrete(2)
        bound = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            -bound, bound, shape=(len(FEATURE_NAMES),), dtype=np.float32
        )
        self.done = True
        self._session_index = -1

    @property
    def account(self):
        return self.simulation.account

    @property
    def history(self):
        return self.simulation.history

    @property
    def session(self):
        return self.simulation.session

    @property
    def index(self):
        return self.simulation.index

    @property
    def remaining_steps(self):
        return self.stop - self.index

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self._session_index = -1
        self._session_index = (
            int(self.np_random.integers(len(self.sessions)))
            if self.random_start
            else (self._session_index + 1) % len(self.sessions)
        )
        data = self.sessions[self._session_index]
        if self.augment:
            # Causal independent volume jitter only, with raw prices and executable depth unchanged.
            data = replace(
                data,
                trade_size=data.trade_size * self.np_random.uniform(0.8, 1.2, len(data.trade_size)),
            )
        self.simulation = Simulation(
            data, self.initial_cash, self.costs, latency_ms=self.latency_ms
        )
        max_start = max(60, len(self.simulation.bars) - self.episode_steps - 1)
        start = int(
            options.get(
                "start", self.np_random.integers(60, max_start + 1) if self.random_start else 60
            )
        )
        self.simulation.reset(start)
        self.stop = min(start + self.episode_steps, len(self.simulation.bars) - 1)
        if self.stop <= start:
            raise ValueError("insufficient future bars")
        self.done = False
        self.peak = float(self.account.initial_cash)
        self.last_drawdown = 0.0
        return self._observation(), self._info()

    def _observation(self):
        s = self.simulation
        return observation(s.history, s.account, s.risk, s.session, now_ns=s.now_ns, quote=s.quote)

    def _info(self):
        s = self.simulation
        e = s.execution
        return {
            "equity": float(s.snapshot().equity),
            "position": float(s.account.position),
            "timestamp": s.now_ns,
            "session_id": s.session.session_id,
            "fills": len(s.account.fills),
            "rejects": e.rejects,
            "unfilled": e.unfilled,
            "latencies": list(e.latencies),
            "incomplete": bool(self.done and (s.account.position or e.pending)),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        s = self.simulation
        before = float(s.snapshot().equity)
        # On the final interval an exit has actual future quotes available, not a fictional terminal fill.
        final = s.index + 1 >= self.stop
        s.submit_action(0 if final else int(action))
        next_ns = s.bars[s.index + 1].end_ns
        s.advance_to(next_ns)
        equity = float(s.snapshot().equity)
        self.peak = max(self.peak, equity)
        drawdown = self.peak - equity
        reward = (
            10000
            * ((equity - before) - 0.1 * max(0, drawdown - self.last_drawdown))
            / float(s.account.initial_cash)
        )
        self.last_drawdown = drawdown
        terminated = equity <= 0
        truncated = s.index >= self.stop
        self.done = terminated or truncated
        if self.done:
            s.execution.cancel()
        return self._observation(), float(reward), bool(terminated), bool(truncated), self._info()
