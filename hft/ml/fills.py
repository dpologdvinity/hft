"""One causal long round trip per symbol per day, filled on later minute opens.

`signal[t]` is a prediction known at the end of minute t, so any order it causes
fills at the next minute that has a real bar. Each side pays the symbol's estimated
half-spread plus 1 bp of slippage. The position is sold by `last_minute` (the
session's close minus five minutes) at the latest.
"""

from dataclasses import dataclass

import numpy as np

SLIPPAGE_BPS = 1.0
TICK = 0.01  # quotes move in whole cents, so a spread is never under one cent


def half_spread_bps(estimate_bps, price):
    """The symbol's estimated half-spread, but at least half a cent at this price."""
    return max(estimate_bps, TICK / 2 / price * 1e4)


@dataclass(frozen=True)
class Trade:
    entry_minute: int
    exit_minute: int
    net_return: float


def _next_valid(valid, after, until):
    """First minute in (after, until] with a real bar, or None."""
    candidates = np.flatnonzero(valid[after + 1 : until + 1])
    return int(after + 1 + candidates[0]) if len(candidates) else None


def simulate_day(
    minute_open, minute_valid, signal, *, enter, half_spread_bps, last_minute, cost_multiple=1.0
):
    cost = cost_multiple * (half_spread_bps + SLIPPAGE_BPS) / 1e4
    triggers = np.flatnonzero(signal[: last_minute - 1] > enter)
    if not len(triggers):
        return None
    entry = _next_valid(minute_valid, int(triggers[0]), last_minute - 1)
    if entry is None:
        return None
    exits = np.flatnonzero(signal[entry:last_minute] < 0)
    exit_ = _next_valid(minute_valid, entry + int(exits[0]), last_minute) if len(exits) else None
    if exit_ is None:
        exit_ = _next_valid(minute_valid, last_minute - 1, len(minute_valid) - 1)
    if exit_ is None:  # no bar at or after the last minute: sell at the last real one
        later = np.flatnonzero(minute_valid[entry + 1 :])
        # With no later bar at all, the position is closed at its own entry price: it
        # still pays both costs, so a bought position can never vanish from the results.
        exit_ = int(entry + 1 + later[-1]) if len(later) else entry
    bought = minute_open[entry] * (1 + cost)
    sold = minute_open[exit_] * (1 - cost)
    return Trade(entry, exit_, float(sold / bought - 1))
