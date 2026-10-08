"""`hftcore`-backed market engine with the exact `PyMarketEngine` interface."""

from collections import deque

import numpy as np

from .data import Bar
from .features import LOOKBACK
from .market_engine import BarUpdate, GapEvent

BAR_FIELDS = (
    "start_ns",
    "end_ns",
    "quote_ns",
    "session_id",
    "tradable",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "vwap",
)


def bar_from_row(row):
    return Bar(**dict(zip(BAR_FIELDS, row, strict=True)))


class CppMarketEngine:
    implementation = "cpp"

    def __init__(self, symbol: str, strategy: str = "model", bar_seconds: int = 5):
        import hftcore

        self.version = hftcore.version()
        self.symbol, self.strategy, self.bar_seconds = symbol, strategy, bar_seconds
        self._native = hftcore.MarketEngine(symbol, strategy, bar_seconds)
        # Python-side mirror of the native history so per-quote snapshots stay cheap.
        self.history = deque(maxlen=LOOKBACK + 1)
        self.session = None

    @property
    def aggregator(self):
        return self if self.session is not None else None

    @property
    def start(self):
        return self._native.aggregator_start()

    @property
    def quality(self):
        return self._native.quality()

    @property
    def last_event_ns(self):
        return self._native.last_event_ns

    @property
    def last_tick_ns(self):
        return self._native.last_tick_ns

    @property
    def warmup_after_ns(self):
        return self._native.warmup_after_ns

    def start_session(self, session, now_ns=None):
        self.session = session
        self._native.start_session(session.session_id, session.open_ns, session.close_ns, now_ns)
        self.history.clear()

    def end_session(self):
        self.session = None
        self._native.end_session()
        self.history.clear()

    def mark_gap(self, now_ns):
        self._native.mark_gap(now_ns)
        self.history.clear()

    def reset_history(self):
        self._native.reset_history()
        self.history.clear()

    def on_event(self, event, arrival_ns):
        return self._convert(self._native.on_event(event, arrival_ns))

    def advance_to(self, now_ns):
        return self._convert(self._native.advance_to(now_ns))

    @property
    def native(self):
        """The `hftcore.MarketEngine`, for routing raw frames with `hftcore.FrameRouter`."""
        return self._native

    def convert(self, rows):
        """Python updates for native update rows, keeping the history mirror current."""
        return self._convert(rows)

    def _convert(self, rows):
        updates = []
        for row in rows:
            if row[0] == "gap":
                self.history.clear()
                updates.append(GapEvent(self.symbol, row[1], row[2]))
                continue
            _, bar_row, now_ns, accepted, reset, ready, market, action = row
            bar = bar_from_row(bar_row)
            if accepted:
                if reset:
                    self.history.clear()
                self.history.append(bar)
            market = None if market is None else np.asarray(market, dtype=np.float32)
            updates.append(
                BarUpdate(self.symbol, bar, now_ns, accepted, reset, ready, market, action)
            )
        return updates
