from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hft.ml.download import SCHEMA
from hft.ml.features import DAILY_FEATURES, daily_frame

NS = 1_000_000_000


def _sessions(n, start=date(2019, 1, 2)):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _write(root, symbol, days, seed, tweak=None):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    rows = []
    for i, day in enumerate(days):
        o = close[i - 1] * np.exp(rng.normal(0, 0.003)) if i else close[0]
        c = close[i]
        h, low = max(o, c) * 1.004, min(o, c) * 0.996
        v = float(rng.integers(1000, 2000))
        row = {"o": o, "h": h, "l": low, "c": c, "v": v}
        if tweak:
            row = tweak(i, day, row)
        t = int(datetime(day.year, day.month, day.day, 5, tzinfo=UTC).timestamp()) * NS
        rows.append({"t": t, **row, "vw": row["c"], "n": 1})
    folder = root / "1Day" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, folder / "2019.parquet")


def _build(root, tweak_symbol=None, tweak=None):
    days = _sessions(120)
    for k, symbol in enumerate(["SPY", "QQQ", "NVDA", "PLTR"]):
        start = days if symbol != "PLTR" else days[80:]
        _write(root, symbol, start, k, tweak if symbol == tweak_symbol else None)
    return daily_frame(root, ["SPY", "QQQ", "NVDA", "PLTR"], first_dates={"PLTR": days[80]})


def test_features_for_a_day_never_use_that_days_close_or_later_bars(tmp_path):
    base = _build(tmp_path / "a")
    d = 90

    def tweak(i, day, row):
        if i == d:  # keep the open (known at decision time), change everything else
            return {
                **row,
                "c": row["c"] * 1.5,
                "h": row["h"] * 1.6,
                "l": row["l"] * 0.5,
                "v": row["v"] * 9,
            }
        if i > d:
            return {k: v * 2 for k, v in row.items()}
        return row

    changed = _build(tmp_path / "b", "NVDA", tweak)
    nvda = list(base.symbols).index("NVDA")
    np.testing.assert_array_equal(base.x[: d + 1, nvda], changed.x[: d + 1, nvda])
    assert base.label[d, nvda] != changed.label[d, nvda]  # the label is that day's move


def test_label_is_the_open_to_close_log_return(tmp_path):
    frame = _build(tmp_path)
    assert frame.x.shape[2] == len(DAILY_FEATURES)
    nvda = list(frame.symbols).index("NVDA")
    assert np.isfinite(frame.label[frame.valid[:, nvda], nvda]).all()


def test_symbols_are_invalid_before_their_first_usable_date(tmp_path):
    frame = _build(tmp_path)
    pltr = list(frame.symbols).index("PLTR")
    first = frame.dates.tolist().index(np.datetime64(_sessions(120)[80]))
    assert not frame.valid[:first, pltr].any()
    assert frame.valid[first + 1 :, pltr].any()
