from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hft.calendar import SessionWindow
from hft.ml.download import SCHEMA
from hft.ml.features import CHANNELS, minute_store, windows

NS = 1_000_000_000


def _ns(text):
    return int(datetime.fromisoformat(text).timestamp()) * NS


FULL = SessionWindow(
    "2019-07-02", _ns("2019-07-02T09:30:00-04:00"), _ns("2019-07-02T16:00:00-04:00")
)
HALF = SessionWindow(
    "2019-07-03", _ns("2019-07-03T09:30:00-04:00"), _ns("2019-07-03T13:00:00-04:00")
)


def _write(root, symbol, skip=(), tweak=None):
    rows, price = [], 100.0
    for w in (FULL, HALF):
        for m in range((w.close_ns - w.open_ns) // (60 * NS)):
            if (w.session_id, m) in skip:
                continue
            price *= 1.0001
            row = {
                "t": w.open_ns + m * 60 * NS,
                "o": price,
                "h": price,
                "l": price,
                "c": price,
                "v": 100.0 + m,
                "vw": price,
                "n": 1,
            }
            if tweak:
                row = tweak(w.session_id, m, row)
            rows.append(row)
    folder = root / "1Min" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), folder / "2019.parquet")


def test_half_days_end_five_minutes_before_their_real_close(tmp_path):
    _write(tmp_path, "NVDA")
    store = minute_store(tmp_path, ["NVDA"], [FULL, HALF])
    assert store["NVDA"].last_minute.tolist() == [385, 205]
    assert store["NVDA"].valid[1, 210:].sum() == 0


def test_missing_minutes_have_zero_return_and_volume(tmp_path):
    _write(tmp_path, "NVDA", skip={("2019-07-02", 30)})
    data = minute_store(tmp_path, ["NVDA"], [FULL, HALF])["NVDA"]
    assert not data.valid[0, 30]
    ret, volume = CHANNELS.index("return"), CHANNELS.index("relative_volume")
    assert data.channels[0, 30, ret] == 0 and data.channels[0, 30, volume] == 0


def test_a_window_never_reads_a_later_minute(tmp_path):
    _write(tmp_path / "a", "NVDA")

    def tweak(session, m, row):
        return (
            {**row, "c": row["c"] * 3, "v": row["v"] * 7, "vw": row["vw"] * 5} if m > 100 else row
        )

    _write(tmp_path / "b", "NVDA", tweak=tweak)
    keys = np.array([[0, 0, 100], [0, 0, 20]])
    xa, _ = windows(minute_store(tmp_path / "a", ["NVDA"], [FULL, HALF]), ["NVDA"], keys, horizon=5)
    xb, _ = windows(minute_store(tmp_path / "b", ["NVDA"], [FULL, HALF]), ["NVDA"], keys, horizon=5)
    assert xa.shape == (2, 60, len(CHANNELS))
    np.testing.assert_array_equal(xa, xb)
    assert (xa[1, :39] == 0).all()  # minutes before the open are padding


def test_labels_are_the_return_between_later_opens(tmp_path):
    _write(tmp_path, "NVDA")
    store = minute_store(tmp_path, ["NVDA"], [FULL, HALF])
    _, y = windows(store, ["NVDA"], np.array([[0, 0, 100]]), horizon=5)
    data = store["NVDA"]
    assert y[0] == np.float32(np.log(data.open[0, 106] / data.open[0, 101]))


def test_minutes_carry_the_opening_gap_and_the_share_of_the_session_elapsed(tmp_path):
    _write(tmp_path, "NVDA")
    data = minute_store(tmp_path, ["NVDA"], [FULL, HALF])["NVDA"]
    gap, time = CHANNELS.index("opening_gap"), CHANNELS.index("time_of_day")
    assert (data.channels[0, :, gap] == 0).all()  # no earlier session: unknown gap
    assert data.channels[1, 5, gap] == pytest.approx(np.log(1.0001), rel=1e-3)  # one tick up
    assert data.channels[1, 105, time] == pytest.approx(105 / 210)  # half day: 210 minutes
