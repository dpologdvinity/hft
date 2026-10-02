"""Causal, immutable completed bars for both historical and live event streams.

Open-interval events may reorder by <=250ms, if received before publication.
No grace period delays decisions: at end_ns publish from information available
then. Late closed-bar events and corrections are counted, never rewrite history.
All OHLCV uses the provider's close-eligible trade subset, rather than claiming
exact replication of minute bars with field-specific eligibility rules.
"""

import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal

from .calendar import SessionWindow
from .data import NS, Bar, Quote

# Alpaca's published close exclusions (CTS A/B vs UTP C).
# https://alpaca.markets/learn/stock-minute-bars
CLOSE_EXCLUDED = frozenset("479CGHIMNPQRTUVZ")
CONDITION_SOURCE = "https://alpaca.markets/learn/stock-minute-bars"


def parse_timestamp(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("timestamp must be RFC3339 text")  # noqa: TRY004 - parser contract
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})", value
    )
    if not match:
        raise ValueError("timestamp must be timezone-aware RFC3339")
    base, fraction, zone = match.groups()
    dt = datetime.fromisoformat(base + ("+00:00" if zone == "Z" else zone))
    delta = dt.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * NS + int((fraction or "").ljust(9, "0"))


def event_timestamp(event):
    stamp = event.get("event_ns")
    if stamp is None:
        return parse_timestamp(event.get("t"))
    if not isinstance(stamp, int) or isinstance(stamp, bool) or stamp <= 0:
        raise ValueError("event_ns must be positive integer nanoseconds")
    return stamp


def quote_from_event(event, arrival_ns=None) -> Quote:
    stamp = event_timestamp(event)
    factor = Decimal(1) if event.get("sizes_in_shares") else Decimal(100)
    bid, ask = Decimal(str(event["bp"])), Decimal(str(event["ap"]))
    bid_size, ask_size = Decimal(str(event["bs"])) * factor, Decimal(str(event["as"])) * factor
    identity = event.get("quote_id") or event.get("i")
    if identity is None:
        payload = [
            event.get("S", ""),
            stamp,
            *[format(v.normalize(), "f") for v in (bid, ask, bid_size, ask_size)],
            event.get("bx") or "",
            event.get("ax") or "",
        ]
        identity = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    return Quote(str(identity), stamp, arrival_ns, bid, ask, bid_size, ask_size)


class BarAggregator:
    def __init__(
        self,
        symbol,
        bar_seconds=5,
        session: SessionWindow | None = None,
        late_tolerance_ns=250_000_000,
        max_events=100_000,
    ):
        if bar_seconds not in (1, 5):
            raise ValueError("bar_seconds must be 1 or 5")
        self.symbol = symbol
        self.width = bar_seconds * NS
        self.session = session
        self.late_tolerance_ns = late_tolerance_ns
        self.max_events = max_events
        self.start = session.open_ns if session else None
        self.last_quote = None
        self.quality = Counter()
        self.alerts = []
        self._quotes = []
        self._trades = []
        self._seen = set()
        self._last_price = None
        self._max_event = 0
        self._clock = 0

    @property
    def quote(self):
        return self.last_quote

    def _count(self, reason):
        self.quality[reason] += 1
        if len(self.alerts) < 100:
            self.alerts.append(reason)

    def add(self, event, arrival_ns=None) -> list[Bar]:
        if event.get("S") != self.symbol:
            return []
        kind = event.get("T")
        if kind not in ("q", "t", "c", "x"):
            return []
        stamp = event_timestamp(event)
        arrival = arrival_ns if arrival_ns is not None else event.get("arrival_ns", stamp)
        if not isinstance(arrival, int) or arrival < 0:
            raise ValueError("invalid arrival_ns")
        if self.start is None:
            self.start = stamp // self.width * self.width
        if kind in ("c", "x"):
            self._count("corrections" if kind == "c" else "cancellations")
            return self.advance_to(arrival)
        # Validate even discarded ticks, to avoid disguising corrupted inputs.
        if kind == "q":
            payload = quote_from_event(event, arrival)
        else:
            price, size = float(event["p"]), float(event["s"])
            if not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size < 0:
                raise ValueError("invalid trade")
            payload = (price, size)
        completed = self.advance_to(arrival)
        if self.session and not self.session.contains(stamp):
            self._count("outside_session")
            return completed
        if stamp < self.start:
            self._count("late_closed")
            return completed
        if stamp > arrival + 250_000_000:
            self._count("future_event")
            return completed
        if (
            self._max_event - stamp > self.late_tolerance_ns
            or arrival - stamp > self.late_tolerance_ns
        ):
            self._count("late_tolerance")
            return completed
        if stamp >= self.start + self.width:
            # Future timestamps inside allowed clock skew remain pending until interval end.
            pass
        identity = (
            kind,
            str(
                event.get(
                    "i",
                    event.get(
                        "quote_id",
                        (stamp, payload.bid, payload.ask, payload.bid_size, payload.ask_size)
                        if kind == "q"
                        else (stamp, *payload),
                    ),
                )
            ),
        )
        if identity in self._seen:
            self._count("duplicates")
            return completed
        if len(self._quotes) + len(self._trades) >= self.max_events:
            self._count("buffer_overflow")
            raise ValueError("bounded event buffer exhausted")
        self._seen.add(identity)
        self._max_event = max(self._max_event, stamp)
        if kind == "q":
            self._quotes.append(payload)
            if self.last_quote is None or stamp >= self.last_quote.event_ns:
                self.last_quote = payload
        else:
            conditions = event.get("c", [])
            excluded = CLOSE_EXCLUDED | ({"W"} if event.get("z", "C") == "C" else {"B"})
            if any(c in excluded for c in conditions):
                self._count("filtered_trades")
            else:
                self._trades.append((stamp, len(self._trades), *payload))
        return completed

    def advance_to(self, now_ns) -> list[Bar]:
        if not isinstance(now_ns, int):
            raise ValueError("clock must be integer nanoseconds")  # noqa: TRY004 - data validation contract
        if now_ns < self._clock:
            self._count("clock_reversal")
            return []
        self._clock = now_ns
        completed = []
        if self.start is None:
            return completed
        limit = min(now_ns, self.session.close_ns) if self.session else now_ns
        while self.start + self.width <= limit:
            end = self.start + self.width
            eligible = sorted(
                (t for t in self._trades if self.start <= t[0] < end), key=lambda t: (t[0], t[1])
            )
            quotes = [q for q in self._quotes if q.event_ns < end]
            quote = max(quotes, key=lambda q: q.event_ns) if quotes else None
            if quote is None and self.last_quote is not None and self.last_quote.event_ns < end:
                quote = self.last_quote
            prices = [t[2] for t in eligible]
            if prices:
                self._last_price = prices[-1]
            if quote is not None and self._last_price is not None:
                p = prices or [self._last_price]
                volume = sum(t[3] for t in eligible)
                vwap = sum(t[2] * t[3] for t in eligible) / volume if volume else self._last_price
                completed.append(
                    Bar(
                        start_ns=self.start,
                        end_ns=end,
                        quote_ns=quote.event_ns,
                        session_id=self.session.session_id if self.session else "",
                        tradable=end - quote.event_ns <= 2 * NS
                        and quote.bid_size > 0
                        and quote.ask_size > 0,
                        open=p[0],
                        high=max(p),
                        low=min(p),
                        close=p[-1],
                        volume=volume,
                        bid=float(quote.bid),
                        ask=float(quote.ask),
                        bid_size=float(quote.bid_size),
                        ask_size=float(quote.ask_size),
                        vwap=vwap,
                    )
                )
            else:
                self._count("missing_context")
            self._trades = [t for t in self._trades if t[0] >= end]
            future_quotes = [q for q in self._quotes if q.event_ns >= end]
            self._quotes = ([quote] if quote is not None else []) + future_quotes
            self._seen.clear()
            self.start = end
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
                if message.get("T") in ("q", "t", "c", "x") and message.get("S") == symbol:
                    yield {**message, "arrival_ns": time.time_ns()}
