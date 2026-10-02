"""Synchronous pre-order guards; emergency position reductions remain possible."""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .data import NS, session_day


@dataclass(frozen=True)
class RiskConfig:
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.05
    max_exposure: float = 1000.0
    max_leverage: float = 1.0
    max_orders_per_minute: int = 5
    max_quote_age_seconds: float = 2.0

    def __post_init__(self):
        values = (
            self.max_daily_loss,
            self.max_drawdown,
            self.max_exposure,
            self.max_leverage,
            self.max_quote_age_seconds,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("risk limits must be positive and finite")
        if self.max_daily_loss >= 1 or self.max_drawdown >= 1 or self.max_orders_per_minute < 1:
            raise ValueError("invalid loss or order-rate limit")


class RiskGateway:
    def __init__(self, config=None, blackouts=()):
        self.config = config or RiskConfig()
        self.blackouts = tuple(blackouts)
        if any(start >= stop for start, stop in self.blackouts):
            raise ValueError("blackout intervals must be increasing")
        self.order_times = deque()
        self.day, self.day_start, self.peak = None, None, None
        self.halted = None
        self.permanent_halt = False

    def observe(self, quote, account):
        equity = account.equity(quote.mid)
        day = session_day(quote.timestamp)
        if self.day != day:
            self.day, self.day_start = day, equity
            if not self.permanent_halt:
                self.halted = None
        self.peak = max(equity, self.peak if self.peak is not None else account.initial_cash)
        if equity <= self.day_start * (1 - self.config.max_daily_loss):
            self.halted = "daily_loss"
        if equity <= self.peak * (1 - self.config.max_drawdown):
            self.halted, self.permanent_halt = "max_drawdown", True

    def check(self, target, quote, account, now, *, history=(), count_rate=True):
        if isinstance(target, bool) or target not in (-1, 0, 1):
            return "invalid_target"
        self.observe(quote, account)
        age = (now - quote.timestamp) / NS
        if age < -0.25 or age > self.config.max_quote_age_seconds:
            return "stale_quote"
        reduction = target == 0 or (
            target * account.position > 0 and abs(target) < abs(account.position)
        )
        if reduction:
            return None
        if self.halted:
            return self.halted
        if any(start <= now < stop for start, stop in self.blackouts):
            return "news_blackout"
        exposure = abs(target) * quote.ask
        if exposure > self.config.max_exposure:
            return "exposure_limit"
        equity = account.equity(quote.mid)
        if equity <= 0 or exposure > equity * self.config.max_leverage:
            return "leverage_limit"
        if len(history) >= 61:
            rows = list(history)
            ranges = np.array(
                [
                    max(b.high - b.low, abs(b.high - a.close), abs(b.low - a.close))
                    for a, b in zip(rows, rows[1:])
                ]
            )
            atr = np.convolve(ranges, np.ones(14) / 14, mode="valid")
            baseline = atr[:-1]
            if len(baseline) and atr[-1] > baseline.mean() + 3 * baseline.std() + 1e-12:
                return "volatility_breaker"
        while self.order_times and now - self.order_times[0] >= 60 * NS:
            self.order_times.popleft()
        if count_rate and len(self.order_times) >= self.config.max_orders_per_minute:
            return "order_rate_limit"
        return None

    def record_order(self, now):
        self.order_times.append(now)
