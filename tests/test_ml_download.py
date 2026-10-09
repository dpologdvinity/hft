from datetime import UTC, date, datetime

import pytest

from hft.calendar import SessionWindow
from hft.ml.download import download_bars, read_bars

NS = 1_000_000_000


def _ns(text):
    return int(datetime.fromisoformat(text).timestamp()) * NS


def _stamp(text):
    return datetime.fromisoformat(text).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bar(text, price=100.0):
    return {
        "t": _stamp(text),
        "o": price,
        "h": price,
        "l": price,
        "c": price,
        "v": 10,
        "vw": price,
        "n": 3,
    }


WINDOWS = [
    SessionWindow("2016-11-25", _ns("2016-11-25T09:30:00-05:00"), _ns("2016-11-25T13:00:00-05:00")),
    SessionWindow("2016-11-28", _ns("2016-11-28T09:30:00-05:00"), _ns("2016-11-28T16:00:00-05:00")),
    SessionWindow("2017-01-03", _ns("2017-01-03T09:30:00-05:00"), _ns("2017-01-03T16:00:00-05:00")),
]


class FakeClient:
    def __init__(self, pages, fail_years=()):
        self.pages, self.fail_years, self.calls = pages, set(fail_years), []

    def get(self, url, params):
        year = int(params["start"][:4])
        self.calls.append((url, year, params.get("page_token")))
        if year in self.fail_years:
            self.fail_years.discard(year)
            raise RuntimeError("network down")
        pages = self.pages.get(year, [[]])
        index = int(params.get("page_token") or 0)
        token = str(index + 1) if index + 1 < len(pages) else None
        return {"bars": pages[index], "next_page_token": token}


MINUTES = {
    2016: [
        [_bar("2016-11-25T08:00:00-05:00"), _bar("2016-11-25T09:30:00-05:00")],  # pre-market
        [
            _bar("2016-11-25T12:59:00-05:00"),
            _bar("2016-11-25T13:30:00-05:00"),  # after 13:00
            _bar("2016-11-28T15:59:00-05:00"),
            _bar("2016-11-28T16:30:00-05:00"),
        ],
    ],
    2017: [[_bar("2017-01-03T09:30:00-05:00"), _bar("2017-01-03T09:31:00-05:00")]],
}


def _download(client, root):
    return download_bars(
        "NVDA", "1Min", date(2016, 1, 4), date(2017, 1, 3), root, client=client, windows=WINDOWS
    )


def test_minute_bars_keep_regular_session_minutes_across_pages(tmp_path):
    _download(FakeClient(MINUTES), tmp_path)
    table = read_bars(tmp_path, "1Min", "NVDA")
    times = table.column("t").to_pylist()
    assert times == sorted(times) and len(set(times)) == len(times)
    expected = [
        "2016-11-25T09:30:00-05:00",
        "2016-11-25T12:59:00-05:00",
        "2016-11-28T15:59:00-05:00",
        "2017-01-03T09:30:00-05:00",
        "2017-01-03T09:31:00-05:00",
    ]
    assert times == [_ns(t) for t in expected]  # extended hours and the half-day's 13:30 dropped


def test_an_interrupted_download_resumes_without_refetching_finished_years(tmp_path):
    client = FakeClient(MINUTES, fail_years={2017})
    with pytest.raises(RuntimeError):
        _download(client, tmp_path)
    assert (tmp_path / "1Min" / "NVDA" / "2016.parquet").exists()
    assert not (tmp_path / "1Min" / "NVDA" / "2017.parquet").exists()
    client.calls.clear()
    _download(client, tmp_path)
    assert {year for _, year, _ in client.calls} == {2017}  # 2016 was complete
    assert len(read_bars(tmp_path, "1Min", "NVDA")) == 5


def test_daily_bars_are_kept_whole(tmp_path):
    pages = {
        2016: [[_bar("2016-11-25T00:00:00-05:00")]],
        2017: [[_bar("2017-01-03T00:00:00-05:00")]],
    }
    download_bars(
        "SPY",
        "1Day",
        date(2016, 1, 4),
        date(2017, 1, 3),
        tmp_path,
        client=FakeClient(pages),
        windows=WINDOWS,
    )
    assert len(read_bars(tmp_path, "1Day", "SPY")) == 2
