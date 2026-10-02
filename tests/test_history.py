import json

import pytest

from hft.data import load_dataset
from hft.history import HistoryError, ReadOnlyClient, download_sessions, probe_access

CAL = [{"date": "2025-01-02", "open": "09:30", "close": "16:00"}]
QUOTE = {
    "t": "2025-01-02T14:30:00.123456789Z",
    "bp": 99.99,
    "ap": 100.01,
    "bs": 1,
    "as": 2,
    "c": ["R"],
}
TRADE = {"t": "2025-01-02T14:30:00.123456789Z", "p": 100.0, "s": 1, "i": 1, "c": ["@"], "z": "C"}


class Fake:
    def __init__(self):
        self.calls = []
        self.fail = False

    def get(self, url, params):
        self.calls.append((url, dict(params)))
        if url.endswith("/calendar"):
            return CAL
        if self.fail and url.endswith("/trades"):
            raise HistoryError("interrupted")
        kind = "quotes" if url.endswith("/quotes") else "trades"
        row = QUOTE if kind == "quotes" else TRADE
        if params.get("page_token") == "next":
            return {kind: [], "next_page_token": None}
        return {kind: [row], "next_page_token": "next"}


def test_download_pagination_resume_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr("hft.history._check_disk", lambda *args: None)
    f = Fake()
    p = download_sessions("X", "2025-01-02", "2025-01-02", tmp_path, client=f)
    s = load_dataset(p)[0]
    assert s.bid_size[0] == 100 and s.quote_ns[0] % 1_000_000_000 == 123456789
    assert s.manifest["quote_metadata"]["c"].to_pylist() == [["R"]]
    calls = [params for url, params in f.calls if "/stocks/" in url]
    assert len(calls) == 4 and all(c["feed"] == "iex" and c["sort"] == "asc" for c in calls)
    assert calls[0]["start"] == "2025-01-02T14:30:00Z"
    f.calls = []
    download_sessions("X", "2025-01-02", "2025-01-02", tmp_path, client=f)
    assert not any("/stocks/" in url for url, _ in f.calls)
    (tmp_path / "X/2025-01-02/quotes.parquet").write_bytes(b"corrupt")
    f.calls = []
    download_sessions("X", "2025-01-02", "2025-01-02", tmp_path, client=f)
    assert any(url.endswith("/quotes") for url, _ in f.calls)


def test_interrupted_download_partial_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("hft.history._check_disk", lambda *args: None)
    f = Fake()
    f.fail = True
    with pytest.raises(HistoryError):
        download_sessions("X", "2025-01-02", "2025-01-02", tmp_path, client=f)
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "partial"
    f.fail = False
    f.calls = []
    download_sessions("X", "2025-01-02", "2025-01-02", tmp_path, client=f)
    assert not any(url.endswith("/quotes") for url, _ in f.calls)


def test_loop_and_holiday(tmp_path, monkeypatch):
    monkeypatch.setattr("hft.history._check_disk", lambda *args: None)

    class Loop(Fake):
        def get(self, url, params):
            result = super().get(url, params)
            if isinstance(result, dict):
                result["next_page_token"] = "next"
            return result

    with pytest.raises(HistoryError, match="pagination"):
        download_sessions("X", "2025-01-02", "2025-01-02", tmp_path, client=Loop())

    class Holiday:
        def get(self, url, params):
            assert url.endswith("calendar")
            return []

    assert probe_access("X", "2025-01-01", client=Holiday())["status"] == "no_session"


def test_probe_counts_and_no_order_endpoints():
    f = Fake()
    p = probe_access("X", "2025-01-02", client=f)
    assert p["quotes"] == p["trades"] == 1 and p["feed"] == "iex"
    assert all(url.endswith(("/calendar", "/quotes", "/trades")) for url, _ in f.calls)


def test_readonly_transport_retry_budget_and_nonretryable():
    from urllib.error import HTTPError

    now = [0.0]
    sleeps = []
    calls = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    def transport(url, params, headers):
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError(url, 429, "rate", {"Retry-After": "2"}, None)
        return {"ok": True}

    c = ReadOnlyClient("fake", "fake", transport=transport, clock=lambda: now[0], sleep=sleep)
    assert c.get("https://data.alpaca.markets/v2/stocks/X/quotes", {}) == {"ok": True}
    assert 2 in sleeps
    for i in range(151):
        c.get("https://paper-api.alpaca.markets/v2/calendar", {})
    assert now[0] >= 60
    with pytest.raises(HistoryError, match="allowlist"):
        c.get("https://paper-api.alpaca.markets/v2/orders", {})

    def denied(url, params, headers):
        raise HTTPError(url, 403, "denied", {}, None)

    c = ReadOnlyClient("fake", "fake", transport=denied, sleep=sleep)
    with pytest.raises(HistoryError, match="403"):
        c.get("https://data.alpaca.markets/v2/stocks/X/trades", {})


def test_disk_guard(tmp_path, monkeypatch):
    from collections import namedtuple

    from hft.history import _check_disk

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("hft.history.shutil.disk_usage", lambda _: usage(10, 9, 1))
    with pytest.raises(HistoryError, match="disk"):
        _check_disk(tmp_path, 100)


@pytest.mark.parametrize("code", [401, 403])
def test_credentials_and_entitlement_failure_no_retry(code):
    from urllib.error import HTTPError

    calls = []

    def denied(url, params, headers):
        calls.append(url)
        raise HTTPError(url, code, "denied", {}, None)

    c = ReadOnlyClient("fake", "fake", transport=denied)
    result = probe_access("X", "2025-01-02", client=c)
    assert result["status"] == "unavailable" and str(code) in result["reason"]
    assert len(calls) == 1


def test_transient_get_retry_and_empty_session():
    from urllib.error import HTTPError

    count = [0]

    def transport(url, params, headers):
        count[0] += 1
        if count[0] == 1:
            raise HTTPError(url, 503, "temporary", {}, None)
        return (
            CAL
            if url.endswith("calendar")
            else {"quotes": [], "trades": [], "next_page_token": None}
        )

    c = ReadOnlyClient("fake", "fake", transport=transport, sleep=lambda _: None)
    assert probe_access("X", "2025-01-02", client=c)["status"] == "empty_session"
    assert count[0] == 4
