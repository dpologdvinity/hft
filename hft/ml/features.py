"""Causal features for the ML day trader.

Daily features for session d are what is known at d's open: closes, highs, lows and
volumes up to d-1, plus d's opening price (the gap). The daily label is d's
open-to-close log return, the move a day trade can capture. A test changes d's
close and every later bar and checks that features for d do not move.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .download import read_bars
from .universe import UNIVERSE

NEW_YORK = ZoneInfo("America/New_York")
SHORT_HISTORY_SESSIONS = 250
RETURN_SPANS = (1, 5, 20, 60)
DAILY_FEATURES = (
    *(f"ret_{k}" for k in RETURN_SPANS),
    "vol_20",
    "gap",
    "rel_volume",
    "body",
    "upper_wick",
    "lower_wick",
    *(f"{m}_ret_{k}" for m in ("spy", "qqq") for k in (1, 5, 20)),
    "spy_gap",
    *(f"weekday_{k}" for k in range(5)),
    "short_history",
)


@dataclass
class DailyFrame:
    dates: np.ndarray  # (D,) datetime64[D]
    symbols: np.ndarray  # (S,) str
    x: np.ndarray  # (D, S, F) float32, NaN where unknown
    label: np.ndarray  # (D, S) float32, open-to-close log return
    valid: np.ndarray  # (D, S) bool: tradable and scoreable that day


def _day(ns):
    return np.datetime64(datetime.fromtimestamp(ns / 1e9, UTC).astimezone(NEW_YORK).date(), "D")


def _aligned(root, symbol, index):
    """o, h, l, c, v on the common calendar, NaN where the symbol has no bar."""
    table = read_bars(root, "1Day", symbol)
    out = np.full((5, len(index)), np.nan)
    if len(table):
        days = np.array([_day(t) for t in table.column("t").to_numpy()])
        rows = np.searchsorted(index, days)
        for k, name in enumerate("ohlcv"):
            out[k, rows] = table.column(name).to_numpy()
    return out


def _lag(values, k):
    """values[d - k], NaN for the first k days."""
    out = np.full_like(values, np.nan)
    out[k:] = values[:-k]
    return out


def _returns(close):
    """ret_k for day d: log(close[d-1] / close[d-1-k])."""
    log_close = np.log(close)
    return {k: _lag(log_close, 1) - _lag(log_close, 1 + k) for k in RETURN_SPANS}


def _rolling(values, window, func):
    out = np.full_like(values, np.nan)
    if len(values) >= window:
        out[window - 1 :] = func(sliding_window_view(values, window), axis=-1)
    return out


def daily_frame(root, symbols, first_dates=None) -> DailyFrame:
    universe = {s.ticker: s for s in UNIVERSE}
    if first_dates is None:
        first_dates = {t: s.first_date for t, s in universe.items()}
    trade_only = {t for t, s in universe.items() if s.trade_only}
    names = list(dict.fromkeys([*symbols, "SPY", "QQQ"]))
    days = set()
    for name in names:
        days |= {_day(t) for t in read_bars(root, "1Day", name).column("t").to_numpy()}
    index = np.array(sorted(days), dtype="datetime64[D]")
    bars = {name: _aligned(root, name, index) for name in names}

    market = []
    for name in ("SPY", "QQQ"):
        returns = _returns(bars[name][3])
        market += [returns[1], returns[5], returns[20]]
    spy_open, spy_close = bars["SPY"][0], bars["SPY"][3]
    market.append(np.log(spy_open) - _lag(np.log(spy_close), 1))
    weekday = (index.astype("datetime64[D]").view("int64") - 4) % 7  # 1970-01-05 is a Monday
    weekdays = [(weekday == k).astype(float) for k in range(5)]

    x = np.full((len(index), len(symbols), len(DAILY_FEATURES)), np.nan, dtype=np.float32)
    label = np.full((len(index), len(symbols)), np.nan, dtype=np.float32)
    valid = np.zeros((len(index), len(symbols)), dtype=bool)
    for s, name in enumerate(symbols):
        o, h, low, c, v = bars[name]
        returns = _returns(c)
        daily_return = np.log(c) - _lag(np.log(c), 1)
        prev = {k: _lag(arr, 1) for k, arr in (("o", o), ("h", h), ("l", low), ("c", c), ("v", v))}
        top, bottom = np.fmax(prev["o"], prev["c"]), np.fmin(prev["o"], prev["c"])
        first = np.datetime64(first_dates.get(name, index[0]), "D")
        usable = index >= first
        seen = np.cumsum(usable & np.isfinite(c))  # sessions with data since the first date
        columns = [
            *(returns[k] for k in RETURN_SPANS),
            _lag(_rolling(daily_return, 20, np.std), 1),
            np.log(o / prev["c"]),
            prev["v"] / _lag(_rolling(v, 20, np.mean), 1),
            np.log(prev["c"] / prev["o"]),
            np.log(prev["h"] / top),
            np.log(bottom / prev["l"]),
            *market,
            *weekdays,
            (seen < SHORT_HISTORY_SESSIONS).astype(float),
        ]
        x[:, s, :] = np.stack(columns, axis=-1)
        label[:, s] = np.log(c / o)
        valid[:, s] = (
            usable
            & np.isfinite(label[:, s])
            & np.isfinite(prev["c"])
            & (np.concatenate([[False], usable[:-1]]))
            & (name not in trade_only)
        )
    x[~valid] = np.nan
    return DailyFrame(index, np.array(symbols), x, label, valid)
