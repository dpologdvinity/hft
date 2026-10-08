import asyncio
import json

import pytest

from hft.trading.stream import StreamAuthError, StreamStats, stream_events


class FakeSocket:
    def __init__(self, frames, sent):
        self.frames, self.sent = list(frames), sent

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, text):
        self.sent.append(json.loads(text))

    async def recv(self):
        if not self.frames:
            raise ConnectionError("socket closed")
        frame = self.frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame if isinstance(frame, str) else json.dumps(frame)


AUTH = [[{"T": "success", "msg": "connected"}], [{"T": "success", "msg": "authenticated"}]]


def _quote(symbol):
    return {"T": "q", "S": symbol, "t": "2025-01-06T14:30:00Z", "bp": 1, "ap": 2, "bs": 1, "as": 1}


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")


def _collect(sessions, limit, **kwargs):
    sent, delays, stats = [], [], StreamStats()
    sockets = iter(sessions)

    def connect(url):
        assert url.endswith("/v2/iex")
        return FakeSocket(next(sockets), sent)

    async def sleep(seconds):
        delays.append(seconds)

    async def run():
        out = []
        agen = stream_events(
            ["NVDA", "AAPL"], connect=connect, sleep=sleep, clock=lambda: 7, stats=stats, **kwargs
        )
        async for item in agen:
            out.append(item)
            if len(out) >= limit:
                await agen.aclose()
                break
        return out

    return asyncio.run(run()), sent, delays, stats


def test_one_subscription_for_all_symbols_and_filtering():
    frames = [*AUTH, [_quote("NVDA"), _quote("MSFT"), {"T": "subscription"}], "not json", "{}"]
    frames += [[5, _quote("AAPL")]]
    out, sent, _, stats = _collect([frames], 3)
    assert sent[1] == {
        "action": "subscribe",
        "quotes": ["NVDA", "AAPL"],
        "trades": ["NVDA", "AAPL"],
    }
    assert out[0] == ("connected", None, 7)
    assert [o[1]["S"] for o in out[1:]] == ["NVDA", "AAPL"]
    assert out[1][1]["arrival_ns"] == 7
    assert stats.malformed == 3  # bad JSON, a non-list frame, a non-object message


def test_reconnects_with_backoff_after_a_dropped_socket():
    first = [*AUTH, [_quote("NVDA")]]  # then the socket closes
    out, _, delays, stats = _collect([first, [ConnectionError("refused")], [*AUTH]], 4)
    kinds = [o[0] for o in out]
    assert kinds == ["connected", "event", "disconnected", "connected"]
    assert delays == [1, 2]
    assert stats.reconnects == 2


def test_authentication_failure_raises_without_retry():
    frames = [[{"T": "error", "code": 402, "msg": "auth failed"}]]
    with pytest.raises(StreamAuthError, match="402"):
        _collect([frames], 1)


def test_symbol_limit_and_credentials_required(monkeypatch):
    async def first(agen):
        return await agen.__anext__()

    with pytest.raises(ValueError, match="1-30"):
        asyncio.run(first(stream_events([f"S{i}" for i in range(31)])))
    monkeypatch.delenv("ALPACA_API_KEY")
    with pytest.raises(ValueError, match="required"):
        asyncio.run(first(stream_events(["NVDA"])))
