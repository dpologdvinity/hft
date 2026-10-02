"""A fixed causal state contract used unchanged by training and live inference."""

import math
import numpy as np

from .data import local_time

LOOKBACK = 60
FEATURE_NAMES = (
    'log_return_1', 'log_return_5', 'log_return_15', 'log_return_60',
    'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'prior_trend',
    'relative_spread', 'quote_imbalance', 'trailing_vwap_deviation',
    'inventory', 'unrealized_fraction', 'cash_fraction', 'session_remaining',
)
FEATURE_VERSION = 1
TARGETS = (0, 1, -1)


def observation(history, account) -> np.ndarray:
    if len(history) < LOOKBACK + 1:
        raise ValueError('need 61 completed bars for causal features')
    b = history[-1]
    returns = [math.log(b.close / history[-1 - w].close) for w in (1, 5, 15, 60)]
    span = max(b.high - b.low, 1e-12)
    candles = [(b.close - b.open) / span, (b.high - max(b.close, b.open)) / span,
               (min(b.close, b.open) - b.low) / span, float(np.sign(returns[2]))]
    tail = list(history)[-60:]
    volume = sum(r.volume for r in tail)
    vwap = sum(r.close * r.volume for r in tail) / volume if volume else b.close
    depth = b.bid_size + b.ask_size
    micro = [(b.ask - b.bid) / b.mid, (b.bid_size - b.ask_size) / depth if depth else 0,
             b.close / vwap - 1]
    clock = local_time(b.timestamp)
    elapsed = clock.hour * 3600 + clock.minute * 60 + clock.second - 9.5 * 3600
    state = [account.position, account.unrealized(b.close) / account.initial_cash,
             account.cash / account.initial_cash, float(np.clip(1 - elapsed / 23_400, 0, 1))]
    result = np.asarray(returns + candles + micro + state, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite observation')
    return result
