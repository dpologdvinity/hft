from datetime import date

import pytest

from hft.ml.news import download_news, read_news


def _article(i, stamp, symbols=("NVDA",), headline="Nvidia upgraded to Buy"):
    return {
        "id": i,
        "created_at": stamp,
        "updated_at": stamp,
        "headline": headline,
        "summary": "",
        "source": "benzinga",
        "symbols": list(symbols),
    }


class FakeClient:
    def __init__(self, articles, fail_months=()):
        self.articles, self.fail_months, self.calls = articles, set(fail_months), []

    def get(self, url, params):
        assert url.endswith("/v1beta1/news")
        month = params["start"][:7]
        self.calls.append((month, params.get("page_token")))
        if month in self.fail_months:
            self.fail_months.discard(month)
            raise RuntimeError("network down")
        rows = [a for a in self.articles if params["start"] <= a["created_at"] < params["end"]]
        start = int(params.get("page_token") or 0)
        page = rows[start : start + 2]  # two per page to exercise pagination
        token = str(start + 2) if start + 2 < len(rows) else None
        return {"news": page, "next_page_token": token}


ARTICLES = [
    _article(1, "2023-01-03T12:00:00Z"),
    _article(2, "2023-01-03T13:00:00Z", ("AAPL", "NVDA")),
    _article(3, "2023-01-20T08:00:00Z"),
    _article(4, "2023-02-01T09:00:00Z", ("JPM",)),
    _article(5, "2023-02-02T09:00:00Z"),
]


def test_news_is_stored_per_month_across_pages_without_duplicates(tmp_path):
    download_news(
        ["NVDA", "AAPL", "JPM"],
        date(2023, 1, 1),
        date(2023, 2, 28),
        tmp_path,
        client=FakeClient(ARTICLES + [ARTICLES[0]]),
    )
    table = read_news(tmp_path)
    assert table.column("id").to_pylist() == [1, 2, 3, 4, 5]
    assert table.column("symbols").to_pylist()[1] == ["AAPL", "NVDA"]
    times = table.column("created_ns").to_pylist()
    assert times == sorted(times)


def test_an_interrupted_download_resumes_at_the_missing_month(tmp_path):
    client = FakeClient(ARTICLES, fail_months={"2023-02"})
    with pytest.raises(RuntimeError):
        download_news(["NVDA"], date(2023, 1, 1), date(2023, 3, 31), tmp_path, client=client)
    client.calls.clear()
    download_news(["NVDA"], date(2023, 1, 1), date(2023, 3, 31), tmp_path, client=client)
    assert {month for month, _ in client.calls} == {"2023-02", "2023-03"}
    assert len(read_news(tmp_path)) == 5
