"""Paper-trade several stocks on one account for as long as the run lasts.

Per stock: a `PaperEngine` (market engine, account ledger, budget risk gateway)
whose orders go to that stock's `SymbolBook`. Account-wide: one market-data
stream, an `AccountGuard`, polling, reconciliation and session boundaries.
Feed silence or a disconnect blocks entries instead of stopping the run.
"""

import asyncio
import time
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path

from ..calendar import session_at
from ..market_engine import make_market_engine
from ..paper import PaperEngine
from ..runtime import _flatten
from ..sizing import SizingConfig
from ..strategies import StrategySpec
from .budget import AccountGuard, budget_intent_factory
from .stream import stream_events

NS = 1_000_000_000
SILENCE_NS = 5 * NS
BACKLOG_NS = NS


@dataclass(frozen=True)
class TradeConfig:
    name: str
    symbols: tuple[str, ...]
    budgets: dict
    strategy: StrategySpec
    log_dir: Path
    run_dir: Path
    daily_loss: float = 0.02
    max_drawdown: float = 0.05
    feed: str = "iex"
    engine: str = "auto"


class TradeRunner:
    def __init__(
        self,
        config,
        broker,
        windows,
        *,
        stream_factory=stream_events,
        clock=time.time_ns,
        sleep=asyncio.sleep,
        policy_factory=None,
    ):
        self.config, self.broker, self.windows = config, broker, windows
        self.stream_factory, self.clock, self.sleep = stream_factory, clock, sleep
        self.connected = False
        self.started_ns = clock()
        self.queued = {s: [] for s in config.symbols}
        self.silent = set()
        self.guard = AccountGuard(
            sum((Decimal(str(b)) for b in config.budgets.values()), Decimal(0)),
            config.daily_loss,
            config.max_drawdown,
        )
        self.engines = {}
        for symbol in config.symbols:
            self.engines[symbol] = self._engine(symbol, policy_factory)

    def _engine(self, symbol, policy_factory):
        book = self.broker.book(symbol)
        strategy = self.config.strategy
        market = make_market_engine(
            symbol,
            strategy.name if strategy.is_rule else "model",
            implementation=self.config.engine,
        )
        queued = self.queued[symbol]

        def submit(intent):
            if intent.side == "buy" and (not self.connected or symbol in self.silent):
                return  # entries need a live, fresh feed; exits always proceed
            queued.append(intent)

        engine = PaperEngine(
            None,
            log_path=self.config.log_dir / f"{symbol}-{self.started_ns}.jsonl",
            initial_cash=book.account.initial_cash,
            costs=book.account.costs,
            risk=book.risk,
            sizing=SizingConfig(
                fee_per_share=book.account.costs.commission,
                slippage_bps=book.account.costs.slippage_bps,
            ),
            account=book.account,
            symbol=symbol,
            synthetic=False,
            source="trade-paper",
            metadata={
                "run": self.config.name,
                "strategy": strategy.name,
                "strategy_version": strategy.version,
                "engine": market.implementation,
                "account_id": self.broker.account_id,
            },
            submit_callback=submit,
            outstanding=lambda: bool(queued) or book.pending is not None,
            clock=self.clock,
            market_engine=market,
            intent_factory=budget_intent_factory(book.account.initial_cash),
            log_market=False,
        )
        if policy_factory is not None:
            engine.policy = policy_factory(symbol)
        elif strategy.is_rule:
            engine.policy = lambda observation: engine.last_update.action
        else:
            from ..policy import OnnxPolicy

            engine.policy = OnnxPolicy(strategy.bundle, symbol=symbol, feed=self.config.feed)
        return engine

    # Session boundaries -------------------------------------------------
    def _start_session(self, session, now):
        self.broker.start_session()
        for engine in self.engines.values():
            engine.start_session(session, now)
        self.guard.start_session(self._equity())
        self.silent.clear()

    async def _end_session(self):
        for symbol, engine in self.engines.items():
            await _flatten(self.broker.book(symbol), engine)
            engine.tick(engine.session.close_ns)
            engine.end_session(complete=True, reconciled=True)

    def _equity(self):
        total = Decimal(0)
        for engine in self.engines.values():
            quote = engine.quote
            total += engine.account.mark(quote.mid) if quote else engine.account.cash
        return total

    # Per-iteration work --------------------------------------------------
    def _route(self, event, now):
        engine = self.engines.get(event.get("S"))
        if engine is None or engine.session is None:
            return
        arrival = event.get("arrival_ns") or now
        if now - arrival > BACKLOG_NS:
            engine.market.reset_history()
            engine.session_complete = False
            self.queued[event["S"]].clear()
            engine.risk.halted = "runtime_gap"
            engine.log.write("gap", event_ns=now, reason="processing_backlog")
            return
        self.silent.discard(event["S"])
        engine.on_event(event, arrival)

    def _check_silence(self, now):
        for symbol, engine in self.engines.items():
            last = engine.last_event_ns or engine.session.open_ns
            if now - last > SILENCE_NS:
                engine.market.mark_gap(now)
                if symbol not in self.silent:
                    self.silent.add(symbol)
                    engine.session_complete = False
                    engine.log.write("gap", event_ns=now, reason="feed_silence")

    def _apply_guard(self):
        reason = self.guard.observe(self._equity())
        if reason:
            for engine in self.engines.values():
                if engine.risk.halted != reason:
                    engine.risk.halted = reason
                    engine.risk.permanent_halt |= reason == "account_drawdown"
                    engine.log.write("halt", event_ns=self.clock(), reason=reason)

    async def _submit_queued(self):
        for symbol, queued in self.queued.items():
            if not queued:
                continue
            intent = queued.pop(0)
            engine, book = self.engines[symbol], self.broker.book(symbol)
            snapshot = replace(engine.snapshot())
            decision = await asyncio.to_thread(book.submit, intent, lambda s=snapshot: s)
            engine.log.write(
                "broker_submit",
                event_ns=self.clock(),
                allowed=decision.allowed,
                reason=decision.reason,
            )

    async def _poll(self):
        for symbol, execution in await asyncio.to_thread(self.broker.poll):
            self.engines[symbol].record_fills((execution,))

    # Main loop -----------------------------------------------------------
    async def run(self, *, until_ns=None):
        queue = asyncio.Queue(maxsize=4096)

        async def receive():
            async for kind, message, arrival in self.stream_factory(
                list(self.config.symbols), feed=self.config.feed
            ):
                if kind == "event":
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull:
                        raise RuntimeError("market queue overflow; stop and reconcile") from None
                else:
                    self.connected = kind == "connected"
                    for engine in self.engines.values():
                        if engine.session:
                            engine.log.write("stream", event_ns=arrival, status=kind)

        task = asyncio.create_task(receive())
        for symbol, execution in self.broker.recovered_fills:
            self.engines[symbol].record_fills((execution,))
        last_poll = last_refresh = 0
        active = None
        try:
            while until_ns is None or self.clock() < until_ns:
                if task.done():
                    task.result()
                now = self.clock()
                session = session_at(self.windows, now)
                if active and (not session or session.session_id != active.session_id):
                    await self._end_session()
                    active = None
                if session and not active:
                    self._start_session(session, now)
                    active = session
                while not queue.empty():
                    self._route(queue.get_nowait(), self.clock())
                now = self.clock()
                if active:
                    self._check_silence(now)
                    for engine in self.engines.values():
                        engine.tick(now)
                    self._apply_guard()
                await self._submit_queued()
                if now - last_poll >= NS:
                    await self._poll()
                    last_poll = now
                if now - last_refresh >= 5 * NS:
                    await asyncio.to_thread(self.broker.refresh)
                    await asyncio.to_thread(self.broker.reconcile)
                    self.broker.save()
                    last_refresh = now
                await self.sleep(0.1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self.shutdown(active)
        return self.summary()

    async def shutdown(self, active):
        errors = []
        for symbol, engine in self.engines.items():
            try:
                if active:
                    await _flatten(self.broker.book(symbol), engine)
            except (RuntimeError, ValueError, OSError) as exc:
                errors.append(f"{symbol}: {exc}")
                engine.log.write("reconcile_error", reason=str(exc))
            engine.finish(complete=False, reconciled=not errors)
        if errors:
            raise RuntimeError("shutdown unresolved: " + "; ".join(errors))

    def summary(self):
        return {
            symbol: {
                "position": str(engine.account.position),
                "cash": str(engine.account.cash),
                "trades": len(engine.account.trades),
                "halted": engine.risk.halted,
            }
            for symbol, engine in self.engines.items()
        }
