"""Market side of the streaming engine: bars, decision history, gaps and signals.

`PyMarketEngine` is the reference implementation; the C++ engine (`hftcore`)
must reproduce its outputs exactly. Account, risk and order handling stay in
`hft.paper.PaperEngine`, which reacts to the updates returned here.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .data import NS, Bar
from .features import LOOKBACK, market_features
from .feed import BarAggregator
from .strategies import RULE_STRATEGIES, rule_action

GAP_NS = 5 * NS


@dataclass(frozen=True)
class GapEvent:
    symbol: str
    now_ns: int
    reason: str  # "feed_gap" (arrival silence) or "runtime_sleep" (clock jump)


@dataclass(frozen=True)
class BarUpdate:
    symbol: str
    bar: Bar
    now_ns: int
    accepted: bool  # False: the bar starts before the warmup boundary and is ignored
    reset: str | None  # "bar_gap" when non-contiguous history was cleared first
    ready: bool  # history holds LOOKBACK + 1 bars, so a decision is due
    market: np.ndarray | None  # market_features(history) when ready
    action: int | None  # rule decision when ready and the strategy is a rule


class PyMarketEngine:
    implementation = "python"
    version = "1"

    def __init__(self, symbol: str, strategy: str = "model", bar_seconds: int = 5):
        if strategy != "model" and strategy not in RULE_STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.symbol, self.strategy, self.bar_seconds = symbol, strategy, bar_seconds
        self.history = deque(maxlen=LOOKBACK + 1)
        self.session = self.aggregator = None
        self.last_event_ns = self.last_tick_ns = None
        self.warmup_after_ns = 0

    @property
    def quality(self):
        return self.aggregator.quality if self.aggregator else {}

    def start_session(self, session, now_ns=None):
        width = self.bar_seconds * NS
        self.session = session
        self.aggregator = BarAggregator(self.symbol, self.bar_seconds, session)
        if now_ns is not None and now_ns > session.open_ns:
            self.aggregator.start = max(session.open_ns, now_ns // width * width)
        self.history.clear()
        self.last_event_ns = self.last_tick_ns = None
        self.warmup_after_ns = session.open_ns

    def end_session(self):
        self.session = self.aggregator = None
        self.history.clear()

    def reset_history(self):
        self.history.clear()

    def on_event(self, event, arrival_ns):
        updates = []
        if self.last_event_ns is not None and arrival_ns - self.last_event_ns > GAP_NS:
            self.history.clear()
            self.warmup_after_ns = arrival_ns
            updates.append(GapEvent(self.symbol, arrival_ns, "feed_gap"))
        self.last_event_ns = arrival_ns
        updates += [self._bar(b, arrival_ns) for b in self.aggregator.add(event, arrival_ns)]
        return updates

    def advance_to(self, now_ns):
        updates = []
        if self.last_tick_ns is not None and now_ns - self.last_tick_ns > GAP_NS:
            self.history.clear()
            self.warmup_after_ns = now_ns
            updates.append(GapEvent(self.symbol, now_ns, "runtime_sleep"))
        self.last_tick_ns = now_ns
        updates += [self._bar(b, now_ns) for b in self.aggregator.advance_to(now_ns)]
        return updates

    def _bar(self, bar, now_ns):
        # Carried bars published when an outage ends cannot rebuild fresh warmup.
        if bar.start_ns < self.warmup_after_ns:
            return BarUpdate(self.symbol, bar, now_ns, False, None, False, None, None)
        reset = None
        if self.history and bar.end_ns - self.history[-1].end_ns != self.bar_seconds * NS:
            self.history.clear()
            reset = "bar_gap"
        self.history.append(bar)
        ready = len(self.history) >= LOOKBACK + 1
        market = market_features(self.history) if ready else None
        action = (
            rule_action(self.strategy, self.history) if ready and self.strategy != "model" else None
        )
        return BarUpdate(self.symbol, bar, now_ns, True, reset, ready, market, action)


def make_market_engine(symbol, strategy="model", *, implementation="auto", bar_seconds=5):
    if implementation not in ("auto", "python"):
        raise ValueError(f"unknown engine implementation {implementation!r}")
    return PyMarketEngine(symbol, strategy, bar_seconds)
