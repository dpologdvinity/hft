"""Causal delayed displayed-liquidity fills shared by replay and local runtime."""

from collections import deque
from dataclasses import asdict, replace
from decimal import Decimal

import numpy as np

from .account import Account, Costs, Execution, decimal
from .calendar import SessionWindow
from .data import build_bars
from .feed import quote_from_event
from .risk import RiskConfig, RiskDecision, RiskGateway, Snapshot
from .sizing import OrderIntent, SizingConfig, make_intent

NS = 1_000_000_000


class LocalExecution:
    def __init__(self, account, risk, latency_ms=75):
        if latency_ms < 0:
            raise ValueError("negative latency")
        self.account, self.risk = account, risk
        self.latency_ns = int(latency_ms * 1e6)
        self.pending = None
        self._remaining = Decimal(0)
        self._due = 0
        self._seen = set()
        self._seen_order = deque()
        self._last_quote_ns = -1
        self.rejects = 0
        self.unfilled = 0
        self.latencies = []
        self.last_decision = RiskDecision(True)

    @property
    def remaining(self):
        return self._remaining

    def to_state(self):
        pending = None if self.pending is None else asdict(self.pending)
        if pending is not None:
            pending["quantity"] = str(pending["quantity"])
            pending["limit_price"] = str(pending["limit_price"])
        return {"pending": pending, "remaining": str(self._remaining), "due_ns": self._due}

    def restore(self, state):
        pending = state["pending"]
        self.pending = (
            None
            if pending is None
            else OrderIntent(
                **{
                    **pending,
                    "quantity": decimal(pending["quantity"]),
                    "limit_price": decimal(pending["limit_price"]),
                }
            )
        )
        self._remaining = decimal(state["remaining"])
        self._due = int(state["due_ns"])
        if (
            self._remaining < 0
            or not self._remaining.is_finite()
            or self.pending is not None
            and self._remaining > self.pending.quantity
        ):
            raise ValueError("invalid pending execution state")

    def submit(self, intent, snapshot, now_ns):
        if self.pending is not None:
            return RiskDecision(False, "order_outstanding")
        decision = self.risk.evaluate(intent, snapshot, now_ns)
        self.last_decision = decision
        if not decision.allowed:
            self.rejects += 1
            return decision
        self.pending = intent
        self._remaining = intent.quantity
        self._due = now_ns + self.latency_ns
        if intent.side == "buy":
            self.risk.record_order(now_ns)
        return decision

    def cancel(self):
        if self.pending is not None:
            self.unfilled += 1
        self.pending = None
        self._remaining = Decimal(0)

    def on_quote(self, quote, now_ns, history=(), session=None):
        if quote.quote_id in self._seen or quote.event_ns < self._last_quote_ns:
            return ()
        self._last_quote_ns = quote.event_ns
        self._seen.add(quote.quote_id)
        self._seen_order.append(quote.quote_id)
        if len(self._seen_order) > 4096:
            self._seen.remove(self._seen_order.popleft())
        if self.pending is None:
            return ()
        intent = self.pending
        if now_ns - intent.created_ns >= int(self.risk.config.order_expiry_seconds * NS):
            self.cancel()
            return ()
        if now_ns < self._due or quote.event_ns < self._due:
            return ()
        if session is None:
            raise ValueError("session required for fill risk revalidation")
        snap = Snapshot(
            quote,
            session,
            self.account.position,
            self.account.mark(quote.mid),
            self.account.initial_cash,
            self.account.cash,
            tuple(history),
        )
        # Revalidation does not count this outstanding entry against its own rate slot.
        observed = self.risk.observe(snap, now_ns)
        if intent.side == "buy" and not observed.allowed:
            self.rejects += 1
            self.last_decision = observed
            self.cancel()
            return ()
        if intent.side == "buy" and self.account.position > 0:
            # Remaining partials retain the original total entry budget.
            position = self.account.position
            exposure = position * self.account.entry_price
            check = replace(snap, position=Decimal(0), available_cash=snap.available_cash)
            validation = replace(intent, quantity=self._remaining)
            decision = self.risk.evaluate(validation, check, now_ns, count_rate=False)
            if exposure + self._remaining * intent.limit_price > self.risk.day_start * decimal(
                self.risk.config.entry_allocation
            ):
                decision = RiskDecision(False, "exposure_limit")
        else:
            validation = replace(intent, quantity=self._remaining)
            decision = self.risk.evaluate(validation, snap, now_ns, count_rate=False)
        if not decision.allowed:
            self.rejects += 1
            self.last_decision = decision
            self.cancel()
            return ()
        slip = decimal(self.account.costs.slippage_bps) / 10000
        price = quote.ask * (1 + slip) if intent.side == "buy" else quote.bid * (1 - slip)
        if (
            intent.side == "buy"
            and price > intent.limit_price
            or intent.side == "sell"
            and price < intent.limit_price
        ):
            return ()
        depth = quote.ask_size if intent.side == "buy" else quote.bid_size
        quantity = min(self._remaining, depth)
        if quantity <= 0:
            return ()
        fee = quantity * decimal(self.account.costs.commission)
        if intent.side == "buy":
            quantity = min(
                quantity, self.account.cash / (price + decimal(self.account.costs.commission))
            )
        if quantity <= 0:
            self.cancel()
            return ()
        fee = quantity * decimal(self.account.costs.commission)
        execution = Execution(
            f"{intent.client_order_id}:{quote.quote_id}",
            intent.client_order_id,
            now_ns,
            quantity if intent.side == "buy" else -quantity,
            price,
            fee,
        )
        self.account.apply(execution)
        self._remaining -= quantity
        self.latencies.append((now_ns - intent.created_ns) / 1e6)
        if self._remaining == 0:
            self.pending = None
        return (execution,)


class Simulation:
    def __init__(
        self,
        session,
        initial_cash=500,
        costs=None,
        risk_config=None,
        sizing_config=None,
        latency_ms=75,
    ):
        costs = costs or Costs()
        risk_config = risk_config or RiskConfig()
        sizing_config = sizing_config or SizingConfig()
        self.data = session
        self._arrival = getattr(session, "quote_arrival_ns", None)
        self._available = (
            session.quote_ns
            if self._arrival is None
            else np.maximum(session.quote_ns, self._arrival)
        )
        self._qorder = np.argsort(self._available, kind="stable")
        self._ordered_available = self._available[self._qorder]
        self.session = SessionWindow(session.session_id, session.open_ns, session.close_ns, None)
        self.initial_cash = initial_cash
        self.costs = costs
        self.risk_config = risk_config
        self.sizing_config = replace(
            sizing_config, fee_per_share=costs.commission, slippage_bps=costs.slippage_bps
        )
        self.latency_ms = latency_ms
        self.bars = build_bars(session)
        self.reset()

    def _quote(self, index):
        d = self.data
        arrival = int(self._arrival[index]) if self._arrival is not None else None
        metadata = d.manifest.get("quote_metadata", {})
        event = {}
        for name in ("i", "quote_id", "bx", "ax"):
            if name in metadata:
                value = metadata[name][index]
                event[name] = value.as_py() if hasattr(value, "as_py") else value
        event.update(
            event_ns=int(d.quote_ns[index]),
            bp=decimal(d.bid[index]),
            ap=decimal(d.ask[index]),
            bs=decimal(d.bid_size[index]),
            sizes_in_shares=True,
        )
        event["as"] = decimal(d.ask_size[index])
        return quote_from_event(event, arrival)

    def reset(self, start_bar=60):
        if not 60 <= start_bar < len(self.bars):
            raise ValueError("insufficient session warmup")
        self.account = Account(self.initial_cash, self.costs)
        self.risk = RiskGateway(self.risk_config)
        self.execution = LocalExecution(self.account, self.risk, self.latency_ms)
        self.index = start_bar
        self.now_ns = self.bars[start_bar].end_ns
        self.history = tuple(self.bars[: start_bar + 1])
        self._qi = int(np.searchsorted(self._ordered_available, self.now_ns, side="left"))
        eligible = self._qorder[: self._qi]
        latest = int(eligible[np.argmax(self.data.quote_ns[eligible])]) if len(eligible) else None
        self.quote = self._quote(latest) if latest is not None else None
        self.risk.observe(self.snapshot(), self.now_ns)
        self.equity_curve = [float(self.snapshot().equity)]
        return self.snapshot()

    def snapshot(self):
        mark = self.quote.mid if self.quote is not None else decimal(self.history[-1].close)
        return Snapshot(
            self.quote,
            self.session,
            self.account.position,
            self.account.mark(mark),
            self.account.initial_cash,
            self.account.cash,
            self.history,
        )

    def submit_action(self, action, now_ns=None):
        now = self.now_ns if now_ns is None else int(now_ns)
        decision = self.risk.observe(self.snapshot(), now)
        if decision.requires_flatten:
            action = 0
            self.execution.cancel()
        if self.execution.pending is not None:
            return RiskDecision(False, "order_outstanding")
        if self.quote is None:
            return RiskDecision(False, "stale_quote")
        intent = make_intent(
            action,
            self.account,
            self.quote,
            self.sizing_config,
            now,
            symbol=self.data.symbol,
            emergency=decision.requires_flatten,
        )
        if intent is None:
            return decision
        return self.execution.submit(intent, self.snapshot(), now)

    def advance_to(self, next_decision_ns):
        if next_decision_ns < self.now_ns:
            raise ValueError("simulation clock cannot go backward")
        while (
            self._qi < len(self.data.quote_ns)
            and int(self._ordered_available[self._qi]) < next_decision_ns
        ):
            quote = self._quote(int(self._qorder[self._qi]))
            self._qi += 1
            now = max(quote.event_ns, quote.arrival_ns or quote.event_ns)
            if self.quote is not None and quote.event_ns < self.quote.event_ns:
                continue
            self.quote = quote
            while self.index + 1 < len(self.bars) and self.bars[self.index + 1].end_ns <= now:
                self.index += 1
                self.history = tuple(self.bars[: self.index + 1])
            self.now_ns = now
            decision = self.risk.observe(self.snapshot(), now)
            if decision.requires_flatten:
                if self.execution.pending is not None and self.execution.pending.side == "buy":
                    self.execution.cancel()
                if self.execution.pending is None:
                    self.submit_action(0, now)
            self.execution.on_quote(quote, now, self.history, self.session)
            self.equity_curve.append(float(self.snapshot().equity))
        self.now_ns = int(next_decision_ns)
        while (
            self.index + 1 < len(self.bars) and self.bars[self.index + 1].end_ns <= next_decision_ns
        ):
            self.index += 1
            self.history = tuple(self.bars[: self.index + 1])
        if (
            self.execution.pending is not None
            and self.now_ns - self.execution.pending.created_ns
            >= int(self.risk.config.order_expiry_seconds * NS)
        ):
            self.execution.cancel()
        self.risk.observe(self.snapshot(), self.now_ns)
        return self.snapshot()
