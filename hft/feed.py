"""Alpaca quote/trade events -> completed, causal OHLCV bars."""

import asyncio
import json
import math
import os
import re
from datetime import datetime

from .data import NS, Bar


def parse_timestamp(value: str) -> int:
    match = re.fullmatch(r"(.*T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})", value)
    if not match:
        raise ValueError("timestamp must be timezone-aware ISO8601")
    base, fractional, zone = match.groups()
    stamp = datetime.fromisoformat(base + ("+00:00" if zone == "Z" else zone))
    return int(stamp.timestamp()) * NS + int((fractional or "").ljust(9, "0"))


def quote_from_event(event):
    mid = (float(event["bp"]) + float(event["ap"])) / 2
    return Bar(
        parse_timestamp(event["t"]),
        mid,
        mid,
        mid,
        mid,
        0,
        float(event["bp"]),
        float(event["ap"]),
        float(event["bs"]),
        float(event["as"]),
    )


class BarAggregator:
    def __init__(self, symbol, bar_seconds=1):
        if bar_seconds not in (1, 5):
            raise ValueError("bar_seconds must be 1 or 5")
        self.symbol, self.width = symbol, bar_seconds * NS
        self.bucket, self.quote = None, None
        self.prices, self.volume, self.last_trade = [], 0.0, -1

    def _bar(self):
        if self.quote is None:
            return None
        prices = self.prices or [self.quote.mid]
        return Bar(
            self.bucket + self.width,
            prices[0],
            max(prices),
            min(prices),
            prices[-1],
            self.volume,
            self.quote.bid,
            self.quote.ask,
            self.quote.bid_size,
            self.quote.ask_size,
        )

    def add(self, event):
        if event.get("S") != self.symbol or event.get("T") not in ("q", "t"):
            return None
        stamp = parse_timestamp(event["t"])
        bucket = stamp // self.width * self.width
        if self.bucket is not None and bucket < self.bucket:
            raise ValueError("out-of-order market data")
        completed = None
        if self.bucket != bucket:
            completed = self._bar() if self.bucket is not None else None
            self.bucket, self.prices, self.volume, self.last_trade = bucket, [], 0.0, -1
        if event["T"] == "q":
            q = quote_from_event(event)
            if self.quote is None or q.timestamp >= self.quote.timestamp:
                self.quote = q
        else:
            price, size = float(event["p"]), float(event["s"])
            if not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size < 0:
                raise ValueError("invalid trade")
            if stamp < self.last_trade:
                raise ValueError("out-of-order trade")
            self.prices.append(price)
            self.volume += size
            self.last_trade = stamp
        return completed


async def alpaca_events(symbol, feed="iex", timeout=5):
    from websockets.asyncio.client import connect

    if feed not in ("iex", "sip", "test"):
        raise ValueError("feed must be iex, sip or test")
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
    async with connect(
        f"wss://stream.data.alpaca.markets/v2/{feed}",
        open_timeout=10,
        close_timeout=2,
        max_queue=16,
    ) as ws:
        await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
        authenticated = False
        while not authenticated:
            for message in json.loads(await asyncio.wait_for(ws.recv(), timeout=10)):
                if message.get("T") == "error":
                    raise RuntimeError(f"market data authentication failed: {message.get('code')}")
                authenticated |= message.get("msg") == "authenticated"
        await ws.send(json.dumps({"action": "subscribe", "quotes": [symbol], "trades": [symbol]}))
        while True:
            for message in json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout)):
                if message.get("T") == "error":
                    raise RuntimeError(f"market data stream error: {message.get('code')}")
                if message.get("T") in ("q", "t") and message.get("S") == symbol:
                    yield message
