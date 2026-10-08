"""Version 3 causal market, account and risk observation contract.

The 17 feature names are preserved. Decision histories contain at most 61
contiguous, actually published bars after the latest observed feed gap. The
environment masks ineligible observations with zeros and prohibits decisions.
"""

import math

import numpy as np

from .account import decimal

LOOKBACK = 60
FEATURE_VERSION = 3
TARGETS = (0, 1)
FEATURE_NAMES = (
    "log_return_1",
    "log_return_5",
    "log_return_15",
    "log_return_60",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "relative_spread",
    "quote_imbalance",
    "trailing_vwap_deviation",
    "inventory",
    "unrealized_fraction",
    "cash_fraction",
    "session_remaining",
    "seconds_since_fill",
    "remaining_order_slots",
    "quote_freshness",
)


def _validated(history):
    history = tuple(history)
    if len(history) < 61:
        raise ValueError("need 61 completed bars")
    b = history[-1]
    if any(r.session_id != b.session_id for r in history[-61:]):
        raise ValueError("history crosses sessions")
    return history, b


def market_features(history):
    """Features 0-9: returns, candle shape and microstructure from completed bars."""
    history, b = _validated(history)
    returns = [math.log(b.close / history[-1 - w].close) for w in (1, 5, 15, 60)]
    span = max(b.high - b.low, 1e-12)
    candles = [
        (b.close - b.open) / span,
        (b.high - max(b.open, b.close)) / span,
        (min(b.close, b.open) - b.low) / span,
    ]
    tail = history[-60:]
    volume = sum(r.volume for r in tail)
    vwap = sum(r.vwap * r.volume for r in tail) / volume if volume else b.close
    depth = b.bid_size + b.ask_size
    micro = [
        (b.ask - b.bid) / b.mid,
        (b.bid_size - b.ask_size) / depth if depth else 0,
        b.close / vwap - 1,
    ]
    return np.asarray(returns + candles + micro, dtype=np.float32)


def state_features(history, account, risk_state, session_window, *, now_ns=None, quote=None):
    """Features 10-16: account, session and risk state at the decision time."""
    history, b = _validated(history)
    now = b.end_ns if now_ns is None else now_ns
    mid = decimal(quote.mid if quote is not None else b.mid)
    equity = account.mark(mid)
    if equity <= 0:
        raise ValueError("nonpositive strategy equity")
    remaining = (session_window.close_ns - now) / (session_window.close_ns - session_window.open_ns)
    last = account.last_fill_ns
    seconds = 1 if last is None else np.clip((now - last) / 60e9, 0, 1)
    config = risk_state.config
    order_count = sum(now - t < 60e9 for t in risk_state.order_times)
    fresh = (
        risk_state.freshness(quote, now)
        if quote is not None
        else b.tradable
        and -250_000_000 <= now - b.quote_ns <= int(config.max_quote_age_seconds * 1e9)
    )
    state = [
        float(account.position * mid / equity),
        float(account.unrealized(mid) / equity),
        float(account.cash / equity),
        np.clip(remaining, 0, 1),
        seconds,
        max(0, config.max_orders_per_minute - order_count) / config.max_orders_per_minute,
        float(fresh),
    ]
    return np.asarray(state, dtype=np.float32)


def assemble(market, state):
    result = np.concatenate([market, state])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite observation")
    return result


def observation(history, account, risk_state, session_window, *, now_ns=None, quote=None):
    history = tuple(history)
    market = market_features(history)
    state = state_features(history, account, risk_state, session_window, now_ns=now_ns, quote=quote)
    return assemble(market, state)
