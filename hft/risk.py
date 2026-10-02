"""Immutable risk snapshots and persistent inclusive loss latches."""

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from .account import decimal

NS = 1_000_000_000


@dataclass(frozen=True)
class RiskConfig:
    max_daily_loss: float = 0.01
    max_drawdown: float = 0.05
    entry_allocation: float = 0.10
    maintenance_exposure: float = 0.15
    max_orders_per_minute: int = 5
    max_quote_age_seconds: float = 2.0
    max_future_seconds: float = 0.25
    pre_close_seconds: float = 60.0
    order_expiry_seconds: float = 5.0

    def __post_init__(self):
        for v in vars(self).values():
            if not decimal(v).is_finite() or v <= 0:
                raise ValueError("invalid risk configuration")
        if (
            self.max_daily_loss >= 1
            or self.max_drawdown >= 1
            or self.entry_allocation > 0.10
            or self.maintenance_exposure > 1
        ):
            raise ValueError("invalid loss or exposure limit")


@dataclass(frozen=True)
class Snapshot:
    quote: object
    session: object
    position: Decimal
    equity: Decimal
    initial_equity: Decimal
    available_cash: Decimal
    history: tuple

    def __post_init__(self):
        object.__setattr__(self, "history", tuple(self.history))
        for name in ("position", "equity", "initial_equity", "available_cash"):
            value = decimal(getattr(self, name))
            if not value.is_finite():
                raise ValueError("nonfinite snapshot")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str | None = None
    requires_flatten: bool = False


class RiskGateway:
    def __init__(self, config=None, blackouts=()):
        self.config = config or RiskConfig()
        self.blackouts = tuple(blackouts)
        if any(a >= b for a, b in blackouts):
            raise ValueError("invalid blackout")
        self.order_times = deque()
        self.day = None
        self.day_start = None
        self.peak = None
        self.halted = None
        self.permanent_halt = False
        self.last_session_open = -1
        self.breaker_until = 0
        self._atr_bar = None

    def freshness(self, quote, now_ns):
        if quote is None:
            return False
        age = now_ns - quote.event_ns
        return (
            -int(self.config.max_future_seconds * NS)
            <= age
            <= int(self.config.max_quote_age_seconds * NS)
        )

    def observe(self, snapshot, now_ns):
        s = snapshot
        if s.session.open_ns < self.last_session_open:
            return RiskDecision(False, "older_session", s.position > 0)
        if s.session.session_id != self.day:
            self.day = s.session.session_id
            self.last_session_open = s.session.open_ns
            self.day_start = s.equity
            if not self.permanent_halt:
                self.halted = None
        self.peak = max(s.equity, self.peak if self.peak is not None else s.initial_equity)
        if s.equity <= self.day_start * (1 - decimal(self.config.max_daily_loss)):
            self.halted = "daily_loss"
        if s.equity <= self.peak * (1 - decimal(self.config.max_drawdown)):
            self.halted = "max_drawdown"
            self.permanent_halt = True
        while self.order_times and now_ns - self.order_times[0] >= 60 * NS:
            self.order_times.popleft()
        if len(s.history) >= 61 and s.history[-1].timestamp != self._atr_bar:
            self._atr_bar = s.history[-1].timestamp
            ranges = np.array(
                [
                    max(b.high - b.low, abs(b.high - a.close), abs(b.low - a.close))
                    for a, b in zip(s.history, s.history[1:])
                ]
            )
            atr = np.convolve(ranges, np.ones(14) / 14, mode="valid")
            prior = atr[:-1]
            if len(prior) and atr[-1] > prior.mean() + 3 * prior.std() + 1e-12:
                self.breaker_until = now_ns + 60 * NS
        if self.halted:
            return RiskDecision(False, self.halted, s.position > 0)
        if s.quote is not None and s.position * s.quote.mid > s.equity * decimal(
            self.config.maintenance_exposure
        ):
            return RiskDecision(False, "maintenance_exposure", s.position > 0)
        if now_ns >= s.session.close_ns - int(self.config.pre_close_seconds * NS):
            return RiskDecision(False, "pre_close", s.position > 0)
        if not self.freshness(s.quote, now_ns):
            return RiskDecision(False, "stale_quote")
        return RiskDecision(True)

    def evaluate(self, intent, snapshot, now_ns, *, count_rate=True):
        observed = self.observe(snapshot, now_ns)
        q, p = decimal(intent.quantity), decimal(intent.limit_price)
        if (
            not q.is_finite()
            or not p.is_finite()
            or q <= 0
            or p <= 0
            or intent.side not in ("buy", "sell")
        ):
            return RiskDecision(False, "invalid_intent")
        if intent.side == "sell":
            if q > snapshot.position:
                return RiskDecision(False, "inventory_limit")
            if not self.freshness(snapshot.quote, now_ns):
                return RiskDecision(False, "stale_quote")
            if not snapshot.session.open_ns <= now_ns < snapshot.session.close_ns:
                return RiskDecision(False, "market_closed")
            return RiskDecision(True)
        if not observed.allowed:
            return observed
        if not snapshot.session.open_ns <= now_ns < snapshot.session.close_ns:
            return RiskDecision(False, "market_closed")
        if snapshot.position > 0:
            return RiskDecision(False, "already_long")
        if any(a <= now_ns < b for a, b in self.blackouts):
            return RiskDecision(False, "news_blackout")
        if now_ns < self.breaker_until:
            return RiskDecision(False, "volatility_breaker")
        if q * p > self.day_start * decimal(self.config.entry_allocation):
            return RiskDecision(False, "exposure_limit")
        if q * p > snapshot.available_cash or q * p > snapshot.equity:
            return RiskDecision(False, "cash_limit")
        if count_rate and len(self.order_times) >= self.config.max_orders_per_minute:
            return RiskDecision(False, "order_rate_limit")
        return RiskDecision(True)

    def record_order(self, now_ns):
        self.order_times.append(int(now_ns))

    def state_dict(self):
        return {
            "day": self.day,
            "day_start": str(self.day_start) if self.day_start is not None else None,
            "peak": str(self.peak) if self.peak is not None else None,
            "halted": self.halted,
            "permanent_halt": self.permanent_halt,
            "last_session_open": self.last_session_open,
            "breaker_until": self.breaker_until,
            "order_times": list(self.order_times),
        }

    def load_state(self, state):
        for key in ("day", "halted", "permanent_halt", "last_session_open", "breaker_until"):
            setattr(self, key, state[key])
        self.day_start = decimal(state["day_start"]) if state["day_start"] is not None else None
        self.peak = decimal(state["peak"]) if state["peak"] is not None else None
        self.order_times = deque(state["order_times"])

    to_state = state_dict
    restore = load_state

    @classmethod
    def from_state(cls, state, config=None, blackouts=()):
        gateway = cls(config, blackouts)
        gateway.load_state(state)
        return gateway
