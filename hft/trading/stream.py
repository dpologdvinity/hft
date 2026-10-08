"""One Alpaca market-data stream for several stocks, reconnecting on failure."""

import asyncio
import json
import os
import time
from dataclasses import dataclass

MAX_SYMBOLS = 30  # Alpaca Basic (free) real-time subscription limit
BACKOFF = (1, 2, 4, 8, 16, 32, 60)
EVENT_TYPES = frozenset({"q", "t", "c", "x"})


class StreamAuthError(RuntimeError):
    """Credentials were refused; retrying cannot help."""


@dataclass
class StreamStats:
    malformed: int = 0
    reconnects: int = 0


def _default_connect(url):
    from websockets.asyncio.client import connect

    return connect(url, open_timeout=10, close_timeout=2, max_queue=64)


async def stream_events(
    symbols,
    *,
    feed="iex",
    connect=_default_connect,
    clock=time.time_ns,
    sleep=asyncio.sleep,
    backoff=BACKOFF,
    stats=None,
    recv_timeout=30,
):
    """Yield ("connected"|"disconnected", None, ns) and ("event", message, arrival_ns).

    Control messages and unsubscribed symbols are skipped; malformed frames are
    counted in `stats.malformed` and skipped. Authentication errors raise
    `StreamAuthError`; any other failure reconnects after the next backoff delay.
    """
    symbols = list(dict.fromkeys(symbols))
    if not symbols or len(symbols) > MAX_SYMBOLS:
        raise ValueError(f"stream needs 1-{MAX_SYMBOLS} symbols")
    if feed not in ("iex", "sip", "test"):
        raise ValueError("feed must be iex, sip or test")
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
    stats = stats if stats is not None else StreamStats()
    wanted = set(symbols)
    attempt = 0
    while True:
        connected = False
        try:
            async with connect(f"wss://stream.data.alpaca.markets/v2/{feed}") as ws:
                await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
                await _authenticate(ws)
                await ws.send(
                    json.dumps({"action": "subscribe", "quotes": symbols, "trades": symbols})
                )
                connected, attempt = True, 0
                yield "connected", None, clock()
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                    arrival = clock()
                    try:
                        messages = json.loads(raw)
                        if not isinstance(messages, list):
                            raise TypeError
                    except (TypeError, ValueError):
                        stats.malformed += 1
                        continue
                    for message in messages:
                        if not isinstance(message, dict):
                            stats.malformed += 1
                            continue
                        kind = message.get("T")
                        if kind == "error":
                            raise RuntimeError(f"market data stream error: {message.get('code')}")
                        if kind in EVENT_TYPES and message.get("S") in wanted:
                            yield "event", {**message, "arrival_ns": arrival}, arrival
        except StreamAuthError:
            raise
        except (OSError, RuntimeError, TimeoutError, ValueError):
            # websockets' ConnectionClosed derives from Exception; caught below.
            pass
        except Exception as error:
            if type(error).__module__.split(".")[0] != "websockets":
                raise
        if connected:
            yield "disconnected", None, clock()
        stats.reconnects += 1
        await sleep(backoff[min(attempt, len(backoff) - 1)])
        attempt += 1


async def _authenticate(ws):
    while True:
        for message in json.loads(await asyncio.wait_for(ws.recv(), timeout=10)):
            if message.get("T") == "error":
                raise StreamAuthError(f"market data authentication failed: {message.get('code')}")
            if message.get("msg") == "authenticated":
                return
