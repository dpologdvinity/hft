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


def daily_frame(root, symbols, first_dates=None, live=None) -> DailyFrame:
    """Daily features, labels and validity on the common calendar.

    `live=(day, {symbol: open})` appends a row for a session still in progress, built
    from bars up to the previous session plus today's opening prices, exactly as the
    backtest builds that day; its label is unknown (NaN) but the row is valid wherever
    the backtest row would be tradable.
    """
    universe = {s.ticker: s for s in UNIVERSE}
    if first_dates is None:
        first_dates = {t: s.first_date for t, s in universe.items()}
    trade_only = {t for t, s in universe.items() if s.trade_only}
    names = list(dict.fromkeys([*symbols, "SPY", "QQQ"]))
    days = set()
    for name in names:
        days |= {_day(t) for t in read_bars(root, "1Day", name).column("t").to_numpy()}
    if live is not None:
        days.add(np.datetime64(live[0], "D"))
    index = np.array(sorted(days), dtype="datetime64[D]")
    bars = {name: _aligned(root, name, index) for name in names}
    if live is not None:
        if index[-1] != np.datetime64(live[0], "D"):
            raise ValueError("the live day must follow every stored bar")
        for name in names:
            bars[name][:, -1] = np.nan
            bars[name][0, -1] = live[1].get(name, np.nan)

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
        has_bar = np.isfinite(c)
        if live is not None:
            has_bar[-1] = np.isfinite(o[-1])  # the live session counts once it has opened
        seen = np.cumsum(usable & has_bar)  # sessions with data since the first date
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
        if live is not None:  # tradable as the backtest would be, before the close is known
            valid[-1, s] = (
                usable[-1]
                and np.isfinite(o[-1])
                and np.isfinite(prev["c"][-1])
                and len(usable) > 1
                and usable[-2]
                and name not in trade_only
            )
    x[~valid] = np.nan
    return DailyFrame(index, np.array(symbols), x, label, valid)


# Minute data -----------------------------------------------------------------

MINUTES = 390
LAST_MINUTE_BEFORE_CLOSE = 5  # positions are flat five minutes before the close
CHANNELS = ("return", "relative_volume", "vwap_distance", "time_of_day", "opening_gap")


@dataclass
class MinuteData:
    dates: np.ndarray  # (n,) datetime64[D], sessions with at least one bar
    open: np.ndarray  # (n, 390) float32, NaN where there is no bar
    valid: np.ndarray  # (n, 390) bool
    last_minute: np.ndarray  # (n,) int: the minute by which positions are flat
    channels: np.ndarray  # (n, 390, len(CHANNELS)) float32, causal per minute


def _minute_channels(close, volume, vwap, valid, length, gap):
    """Per session row: return, relative volume, distance from VWAP, share of the session
    elapsed, and the opening gap (known once the session's first bar printed)."""
    n = len(close)
    filled = np.where(valid, close, np.nan)
    # Carry the last real close forward so a return spans any missing minutes.
    index = np.where(valid, np.arange(MINUTES), 0)
    np.maximum.accumulate(index, axis=1, out=index)
    carried = np.take_along_axis(np.nan_to_num(filled), index, axis=1)
    previous = np.concatenate([np.full((n, 1), np.nan), carried[:, :-1]], axis=1)
    first = valid & (np.cumsum(valid, axis=1) == 1)
    previous = np.where(first | (previous == 0), np.nan, previous)
    returns = np.where(valid & np.isfinite(previous), np.log(carried / previous), 0.0)
    volume = np.where(valid, volume, 0.0)
    seen = np.maximum(np.cumsum(valid, axis=1), 1)
    relative = np.where(
        valid, np.log1p(volume / np.maximum(np.cumsum(volume, axis=1) / seen, 1e-9)), 0.0
    )
    session_vwap = np.cumsum(np.where(valid, vwap, 0) * volume, axis=1) / np.maximum(
        np.cumsum(volume, axis=1), 1e-9
    )
    distance = np.where(
        valid & (session_vwap > 0),
        np.log(carried / np.where(session_vwap > 0, session_vwap, 1)),
        0.0,
    )
    time_of_day = np.arange(MINUTES)[None, :] / np.maximum(length, 1)[:, None]
    opened = np.cumsum(valid, axis=1) > 0
    gap_channel = np.where(opened, np.nan_to_num(gap)[:, None], 0.0)
    return np.stack([returns, relative, distance, time_of_day, gap_channel], axis=-1).astype(
        np.float32
    )


def minute_store(root, symbols, sessions) -> dict[str, MinuteData]:
    """Minute arrays per symbol on the exchange calendar `sessions` (SessionWindow list)."""
    opens = np.array([w.open_ns for w in sessions], dtype=np.int64)
    closes = np.array([w.close_ns for w in sessions], dtype=np.int64)
    store = {}
    for symbol in symbols:
        table = read_bars(root, "1Min", symbol)
        t = table.column("t").to_numpy()
        session = np.searchsorted(opens, t, side="right") - 1
        keep = (session >= 0) & (t < closes[np.clip(session, 0, None)])
        session, t = session[keep], t[keep]
        minute = ((t - opens[session]) // (60 * 1_000_000_000)).astype(np.int64)
        rows, row_of = np.unique(session, return_inverse=True)
        shape = (len(rows), MINUTES)
        values = {}
        for name in ("o", "c", "v", "vw"):
            array = np.full(shape, np.nan)
            array[row_of, minute] = table.column(name).to_numpy()[keep]
            values[name] = array
        valid = np.isfinite(values["c"])
        length = ((closes[rows] - opens[rows]) // (60 * 1_000_000_000)).astype(np.int64)
        first_open = np.array(
            [o[v][0] if v.any() else np.nan for o, v in zip(values["o"], valid, strict=True)]
        )
        last_close = np.array(
            [c[v][-1] if v.any() else np.nan for c, v in zip(values["c"], valid, strict=True)]
        )
        follows = np.concatenate([[False], np.diff(rows) == 1])  # previous calendar session
        previous = np.where(follows, np.concatenate([[np.nan], last_close[:-1]]), np.nan)
        gap = np.log(first_open / previous)
        store[symbol] = MinuteData(
            dates=np.array([sessions[r].session_id for r in rows], dtype="datetime64[D]"),
            open=values["o"].astype(np.float32),
            valid=valid,
            last_minute=length - LAST_MINUTE_BEFORE_CLOSE,
            channels=_minute_channels(values["c"], values["v"], values["vw"], valid, length, gap),
        )
    return store


def windows(store, symbols, keys, *, lookback=60, horizon=15):
    """x (N, lookback, C) up to and including each key's minute; y = log return from the
    next minute's open to the open `horizon` minutes later (NaN when either is missing).

    `keys` rows are (symbol index into `symbols`, session row, minute).
    """
    keys = np.asarray(keys, dtype=np.int64)
    x = np.zeros((len(keys), lookback, len(CHANNELS)), dtype=np.float32)
    y = np.full(len(keys), np.nan, dtype=np.float32)
    offsets = np.arange(-lookback + 1, 1)
    for s, symbol in enumerate(symbols):
        rows = np.flatnonzero(keys[:, 0] == s)
        if not len(rows):
            continue
        data = store[symbol]
        session, minute = keys[rows, 1], keys[rows, 2]
        idx = minute[:, None] + offsets
        inside = idx >= 0
        gathered = data.channels[session[:, None], np.clip(idx, 0, None)]
        x[rows] = np.where(inside[..., None], gathered, 0)
        start, end = minute + 1, minute + 1 + horizon
        ok = end < MINUTES
        a = data.open[session[ok], start[ok]]
        b = data.open[session[ok], end[ok]]
        label = np.full(len(rows), np.nan, dtype=np.float32)
        label[ok] = np.log(b / a)
        y[rows] = label
    return x, y
