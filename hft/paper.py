"""Delayed local fills, causal policy decisions and durable execution event logs."""

import json
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from .account import Account, Costs
from .data import NS, session_day
from .features import LOOKBACK, TARGETS, observation
from .risk import RiskGateway


@dataclass(frozen=True)
class Pending:
    target: int
    due: int
    signal_mid: float


class PaperBroker:
    def __init__(self, account, risk, latency_ms=75):
        if not 0 < latency_ms <= 10_000:
            raise ValueError("paper latency must be positive and <= 10 seconds")
        self.account, self.risk = account, risk
        self.latency_ns = int(latency_ms * 1_000_000)
        self.pending, self.last_rejection = None, None
        self.last_signal_mid = None

    def submit(self, target, quote, now, *, history=()):
        if self.pending is not None or target == self.account.position:
            return None
        reason = self.risk.check(target, quote, self.account, now, history=history)
        self.last_rejection = reason
        if reason:
            return reason
        self.pending = Pending(target, now + self.latency_ns, quote.mid)
        self.risk.record_order(now)
        return None

    def on_quote(self, quote, now, *, history=()):
        self.risk.observe(quote, self.account)
        if self.risk.halted:
            self.pending = None
            self.last_rejection = self.risk.halted
            return self.flatten(quote) if self.account.position else None
        if not self.pending or now < self.pending.due:
            return None
        pending, self.pending = self.pending, None
        self.last_rejection = self.risk.check(
            pending.target, quote, self.account, now, history=history, count_rate=False
        )
        if self.last_rejection:
            return None
        self.last_signal_mid = pending.signal_mid
        return self.account.target(pending.target, quote)

    def flatten(self, quote):
        self.pending = None
        self.last_signal_mid = quote.mid
        return self.account.target(0, quote)


class EventLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", buffering=1)  # do not overwrite or combine runs
        self.run_id = uuid.uuid4().hex

    def write(self, event, **values):
        self.file.write(
            json.dumps({"event": event, "run_id": self.run_id, **values}, allow_nan=False) + "\n"
        )

    def close(self):
        self.file.close()


class PaperEngine:
    def __init__(
        self,
        policy,
        *,
        log_path,
        initial_cash=10_000.0,
        costs=None,
        risk=None,
        symbol="UNKNOWN",
        synthetic=False,
        source="replay",
        latency_ms=75,
        bar_seconds=1,
        metadata=None,
        broker=None,
    ):
        self.policy, self.symbol = policy, symbol
        self.account = broker.account if broker else Account(initial_cash, costs or Costs())
        self.risk = broker.risk if broker else risk or RiskGateway()
        self.broker = broker or PaperBroker(self.account, self.risk, latency_ms)
        self.history = deque(maxlen=LOOKBACK + 1)
        self.bar_seconds = bar_seconds
        self.log = EventLog(log_path)
        self.source, self.synthetic = source, synthetic
        self.last_bar, self.last_quote, self.day_start = None, None, None
        self.day_trades, self.day_prices = 0, []
        self.inference_us = []
        self.closed = False
        self.log.write(
            "start",
            timestamp=time.time_ns(),
            symbol=symbol,
            synthetic=synthetic,
            source=source,
            initial_cash=self.account.initial_cash,
            costs=asdict(self.account.costs),
            risk=asdict(self.risk.config),
            metadata=metadata or {},
            bar_seconds=bar_seconds,
        )

    def _fill(self, fill):
        if isinstance(fill, list):
            for execution in fill:
                self._fill(execution)
            return
        if fill:
            ref = self.broker.last_signal_mid or fill.price
            slip = (fill.price / ref - 1) * (1 if fill.quantity > 0 else -1) * 10_000
            self.log.write(
                "fill",
                **asdict(fill),
                position=self.account.position,
                cash=self.account.cash,
                slippage_bps=slip,
            )
            for trade in self.account.trades[self.day_trades :]:
                self.log.write("trade", **asdict(trade), timestamp=trade.closed)
            self.day_trades = len(self.account.trades)

    def on_quote(self, quote, now=None):
        now = quote.timestamp if now is None else now
        self.last_quote = quote
        before = self.risk.halted
        fill = self.broker.on_quote(quote, now, history=self.history)
        self._fill(fill)
        if self.risk.halted and self.risk.halted != before:
            self.log.write("halt", timestamp=now, reason=self.risk.halted)
        if self.broker.last_rejection:
            self.log.write("rejection", timestamp=now, reason=self.broker.last_rejection)
            self.broker.last_rejection = None

    def on_bar(self, bar, now=None, execution_quote=None):
        now = bar.timestamp if now is None else now
        if self.last_bar and bar.timestamp <= self.last_bar.timestamp:
            raise ValueError("bars must advance monotonically")
        if self.last_bar and session_day(bar.timestamp) != session_day(self.last_bar.timestamp):
            self._fill(self.broker.flatten(self.last_quote or self.last_bar))
            self._session(self.last_bar, complete=False)
            self.day_start, self.day_prices = None, []
        if self.last_bar and session_day(bar.timestamp) == session_day(self.last_bar.timestamp):
            if bar.timestamp - self.last_bar.timestamp > 2 * self.bar_seconds * NS:
                self.history.clear()
                self.broker.pending = None
                self.log.write("gap", timestamp=now)
        self.on_quote(execution_quote or bar, now)
        self.last_bar = bar
        self.day_start = self.day_start or bar.timestamp
        self.day_prices.append(bar.close)
        self.history.append(bar)
        if len(self.history) >= LOOKBACK + 1:
            start = time.perf_counter_ns()
            action = self.policy(observation(self.history, self.account))
            elapsed = (time.perf_counter_ns() - start) / 1000
            self.inference_us.append(elapsed)
            if isinstance(action, bool) or action not in (0, 1, 2):
                raise ValueError("invalid policy action")
            target = TARGETS[int(action)]
            reason = self.broker.submit(target, execution_quote or bar, now, history=self.history)
            self.log.write(
                "decision",
                timestamp=now,
                action=int(action),
                target=target,
                rejected=reason,
                inference_us=elapsed,
            )
        self.log.write(
            "equity",
            timestamp=bar.timestamp,
            equity=self.account.equity(bar.close),
            position=self.account.position,
            cash=self.account.cash,
        )

    def _session(self, bar, complete):
        import numpy as np

        changes = np.diff(np.log(self.day_prices)) if len(self.day_prices) > 1 else np.array([0.0])
        volatility = float(np.sqrt(np.sum(changes**2)))
        trend = abs(self.day_prices[-1] / self.day_prices[0] - 1) if self.day_prices else 0
        regime = (
            "high_volatility" if volatility > 0.01 else "trending" if trend > 0.005 else "choppy"
        )
        self.log.write(
            "session",
            timestamp=bar.timestamp,
            date=session_day(bar.timestamp),
            started=self.day_start,
            complete=complete,
            regime=regime,
            equity=self.account.equity(bar.close),
            position=self.account.position,
        )

    def close_session(self, quote, complete=True):
        self._fill(self.broker.flatten(quote))
        if self.last_bar:
            self._session(self.last_bar, complete)
        self.day_start, self.day_prices, self.last_bar = None, [], None
        self.history.clear()

    def finish(self, quote=None, *, complete=False):
        if self.closed:
            return
        try:
            if quote:
                self._fill(self.broker.flatten(quote))
                if self.last_bar:
                    self._session(self.last_bar, complete)
            self.log.write(
                "finish",
                timestamp=time.time_ns(),
                complete=complete,
                position=self.account.position,
                cash=self.account.cash,
                equity=self.account.equity(quote.mid) if quote else self.account.cash,
            )
        finally:
            self.closed = True
            self.log.close()
