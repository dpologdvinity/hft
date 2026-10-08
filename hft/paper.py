"""One causal streaming engine for replay, dry runs and broker decisions."""

import time
from dataclasses import asdict

from .account import Account, Costs
from .execution import LocalExecution
from .features import assemble, state_features
from .feed import historical_timeline, quote_from_event
from .logs import EventLog
from .market_engine import GapEvent, make_market_engine
from .risk import RiskGateway, Snapshot
from .sizing import SizingConfig, make_intent

NS = 1_000_000_000
PaperBroker = LocalExecution


class PaperEngine:
    def __init__(
        self,
        policy,
        *,
        log_path,
        initial_cash=500,
        costs=None,
        risk=None,
        sizing=None,
        symbol="AAPL",
        synthetic=False,
        source="replay",
        latency_ms=75,
        bar_seconds=5,
        metadata=None,
        account=None,
        submit_callback=None,
        outstanding=None,
        clock=time.time_ns,
        market_engine=None,
        intent_factory=make_intent,
        log_market=True,
    ):
        self.policy, self.symbol = policy, symbol
        self.account = account or Account(initial_cash, costs or Costs())
        self.risk = risk or RiskGateway()
        self.sizing = sizing or SizingConfig(
            fee_per_share=self.account.costs.commission,
            slippage_bps=self.account.costs.slippage_bps,
        )
        self.execution = LocalExecution(self.account, self.risk, latency_ms)
        self.submit_callback = submit_callback
        self.outstanding = outstanding or (lambda: self.execution.pending is not None)
        self.market = market_engine or make_market_engine(symbol, bar_seconds=bar_seconds)
        self.last_update = None
        self.intent_factory = intent_factory
        # Raw market rows make journals replayable evidence; trade runs skip them.
        self.log_market = log_market
        self.log = EventLog(log_path, clock=clock)
        self.source, self.synthetic = source, synthetic
        self.session = self.quote = None
        self.bar_seconds = bar_seconds
        self.closed = False
        self.session_complete = False
        self.completed_sessions = 0
        self.first_event_ns = None
        self._trade_count = len(self.account.trades)
        self.log.write(
            "start",
            source=source,
            synthetic=synthetic,
            symbol=symbol,
            initial_cash=self.account.initial_cash,
            costs=asdict(self.account.costs),
            risk=asdict(self.risk.config),
            sizing=asdict(self.sizing),
            metadata=metadata or {},
            bar_seconds=bar_seconds,
            latency_ms=latency_ms,
            account_state=self.account.to_state(),
        )

    def start_session(self, session, now_ns=None):
        if self.session is not None:
            self.end_session(complete=False)
        self.session = session
        self.market.start_session(session, now_ns)
        self.quote = None
        self.first_event_ns = None
        self.session_complete = now_ns is None or now_ns <= session.open_ns + NS
        self.account.session_start_equity = self.account.cash
        self.log.write(
            "session_start", event_ns=now_ns or session.open_ns, calendar=asdict(session)
        )

    def snapshot(self):
        if self.session is None:
            raise ValueError("no active exchange session")
        equity = self.account.mark(self.quote.mid) if self.quote else self.account.cash
        return Snapshot(
            self.quote,
            self.session,
            self.account.position,
            equity,
            self.account.initial_cash,
            self.account.cash,
            tuple(self.history),
        )

    def record_fills(self, executions):
        for execution in executions:
            self.log.write("fill", event_ns=execution.timestamp_ns, execution=asdict(execution))
        for trade in self.account.trades[self._trade_count :]:
            self.log.write("trade", event_ns=trade.exit_ns, **asdict(trade))
        self._trade_count = len(self.account.trades)
        if executions and self.session and self.quote:
            self.risk.observe(
                self.snapshot(),
                time.time_ns()
                if self.source == "broker-paper" or self.source == "live"
                else executions[-1].timestamp_ns,
            )
            self.log.write(
                "equity",
                event_ns=executions[-1].timestamp_ns,
                equity=self.snapshot().equity,
                cash=self.account.cash,
                position=self.account.position,
            )

    def _submit(self, action, now_ns, *, emergency=False):
        if self.quote is None or self.outstanding():
            return None
        intent = self.intent_factory(
            action,
            self.account,
            self.quote,
            self.sizing,
            now_ns,
            symbol=self.symbol,
            emergency=emergency,
        )
        if intent is None:
            return None
        decision = self.risk.evaluate(intent, self.snapshot(), now_ns)
        if decision.allowed:
            if self.submit_callback:
                self.submit_callback(intent)
            else:
                decision = self.execution.submit(intent, self.snapshot(), now_ns)
            self.log.write(
                "order", event_ns=now_ns, intent=asdict(intent), allowed=decision.allowed
            )
        else:
            self.log.write("rejection", event_ns=now_ns, reason=decision.reason)
        return decision.reason

    def on_quote(self, quote, now_ns):
        if self.quote is not None and quote.event_ns < self.quote.event_ns:
            self.log.write("quality", event_ns=now_ns, reason="older_quote")
            return
        self.quote = quote
        before = self.risk.halted
        decision = self.risk.observe(self.snapshot(), now_ns)
        if self.risk.halted and self.risk.halted != before:
            self.log.write("halt", event_ns=now_ns, reason=self.risk.halted)
        if decision.requires_flatten:
            if (
                not self.submit_callback
                and self.execution.pending
                and self.execution.pending.side == "buy"
            ):
                self.execution.cancel()
            self._submit(0, now_ns, emergency=True)
        if not self.submit_callback:
            self.record_fills(
                self.execution.on_quote(quote, now_ns, tuple(self.history), self.session)
            )
        if not self.log_market:
            return  # equity is still journaled once per bar
        self.log.write(
            "equity",
            event_ns=now_ns,
            equity=self.snapshot().equity,
            cash=self.account.cash,
            position=self.account.position,
        )

    # Market state lives in the market engine; these views keep existing callers working.
    @property
    def history(self):
        return self.market.history

    @property
    def aggregator(self):
        return self.market.aggregator

    @property
    def last_event_ns(self):
        return self.market.last_event_ns

    @property
    def last_tick_ns(self):
        return self.market.last_tick_ns

    def on_bar(self, update):
        if not update.accepted:
            return
        bar, now_ns = update.bar, update.now_ns
        if update.reset:
            self.session_complete = False
            self.log.write("gap", event_ns=now_ns, reason=update.reset)
        decision = self.risk.observe(self.snapshot(), now_ns)
        if decision.requires_flatten:
            self._submit(0, now_ns, emergency=True)
        if update.ready:
            self.last_update = update
            start = time.perf_counter_ns()
            action = self.policy(
                assemble(
                    update.market,
                    state_features(
                        tuple(self.history),
                        self.account,
                        self.risk,
                        self.session,
                        now_ns=now_ns,
                        quote=self.quote,
                    ),
                )
            )
            if isinstance(action, bool) or action not in (0, 1):
                raise ValueError("invalid policy action")
            reason = self._submit(
                0 if decision.requires_flatten else action,
                now_ns,
                emergency=decision.requires_flatten,
            )
            self.log.write(
                "decision",
                event_ns=now_ns,
                bar_end_ns=bar.end_ns,
                action=int(action),
                rejected=reason,
                inference_us=(time.perf_counter_ns() - start) / 1000,
            )
        self.log.write(
            "equity",
            event_ns=now_ns,
            equity=self.snapshot().equity,
            cash=self.account.cash,
            position=self.account.position,
        )

    def on_event(self, event, arrival_ns):
        if self.session is None:
            raise ValueError("start a session before market events")
        if self.log_market:
            self.log.write(
                "market", event_ns=arrival_ns, payload={**event, "arrival_ns": arrival_ns}
            )
        updates = self.market.on_event(event, arrival_ns)
        self.first_event_ns = self.first_event_ns or arrival_ns
        for update in updates:
            if isinstance(update, GapEvent):
                self.session_complete = False
                self.log.write("gap", event_ns=arrival_ns, reason=update.reason)
                if not self.submit_callback:
                    self.execution.cancel()
            else:
                self.on_bar(update)
        if event.get("T") == "q":
            quote = quote_from_event(event, arrival_ns)
            if self.session.contains(quote.event_ns) and quote.event_ns <= arrival_ns + 250_000_000:
                self.on_quote(quote, arrival_ns)

    def tick(self, now_ns):
        if self.session is None:
            return
        updates = self.market.advance_to(now_ns)
        completed = [u for u in updates if not isinstance(u, GapEvent)]
        for gap in updates:
            if isinstance(gap, GapEvent):
                self.session_complete = False
                self.log.write("gap", event_ns=now_ns, reason=gap.reason)
                self.risk.halted = "runtime_gap"
                if not self.submit_callback:
                    self.execution.cancel()
        if completed or now_ns >= self.session.close_ns - int(
            self.risk.config.pre_close_seconds * NS
        ):
            self.log.write("timer", event_ns=now_ns)
        for update in completed:
            self.on_bar(update)
        if self.quote:
            decision = self.risk.observe(self.snapshot(), now_ns)
            if decision.requires_flatten:
                self._submit(0, now_ns, emergency=True)
        if (
            not self.submit_callback
            and self.execution.pending
            and (
                now_ns - self.execution.pending.created_ns
                >= self.risk.config.order_expiry_seconds * NS
            )
        ):
            self.execution.cancel()

    def end_session(self, *, complete=False, reconciled=True):
        if self.session is None:
            return
        fresh_end = (
            self.last_event_ns is not None and self.last_event_ns >= self.session.close_ns - 2 * NS
        )
        full_start = (
            self.first_event_ns is not None and self.first_event_ns <= self.session.open_ns + 5 * NS
        )
        valid = (
            complete
            and self.session_complete
            and fresh_end
            and full_start
            and reconciled
            and self.account.position == 0
            and not self.outstanding()
        )
        if valid:
            self.completed_sessions += 1
        self.log.write(
            "session",
            event_ns=self.session.close_ns,
            date=self.session.session_id,
            calendar=asdict(self.session),
            complete=valid,
            reconciled=reconciled,
            equity=self.account.cash,
            position=self.account.position,
            first_event_ns=self.first_event_ns,
            last_event_ns=self.last_event_ns,
        )
        self.session = None
        self.market.end_session()

    def finish(self, *, complete=False, reconciled=True):
        if self.closed:
            return
        try:
            self.end_session(complete=complete, reconciled=reconciled)
            self.finished_complete = bool(
                complete
                and self.source == "broker-paper"
                and reconciled
                and self.account.position == 0
                and not self.outstanding()
            )
            self.log.write(
                "finish",
                complete=self.finished_complete,
                position=self.account.position,
                cash=self.account.cash,
                unresolved=self.outstanding(),
                reconciled=reconciled,
            )
        finally:
            self.closed = True
            self.log.close()


def replay_sessions(engine, sessions):
    from .calendar import SessionWindow

    for data in sessions:
        engine.start_session(SessionWindow(data.session_id, data.open_ns, data.close_ns))
        for event, now in historical_timeline(data, engine.bar_seconds):
            if event is None:
                engine.tick(now)
            else:
                engine.on_event(event, now)
        engine.end_session(complete=True)
    engine.finish(complete=False)
    return {
        "cash": str(engine.account.cash),
        "position": str(engine.account.position),
        "trades": len(engine.account.trades),
        "log": str(engine.log.path),
        "synthetic": engine.synthetic,
        "source": engine.source,
    }
