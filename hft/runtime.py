"""Bounded quote intake independent of serialized broker REST operations."""

import asyncio
import math
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .account import Account
from .calendar import calendar_from_records, session_at
from .feed import alpaca_events, quote_from_event
from .history import CALENDAR_URL, ReadOnlyClient
from .paper import PaperEngine
from .risk import RiskGateway
from .sizing import OrderIntent
from .state import AccountStateStore

NS = 1_000_000_000


def fetch_calendar(client=None):
    today = datetime.now(ZoneInfo("America/New_York")).date()
    params = {
        "start": (today - timedelta(days=120)).isoformat(),
        "end": (today + timedelta(days=14)).isoformat(),
    }
    if client and hasattr(client, "request"):
        from urllib.parse import urlencode

        return client.request("GET", "/v2/calendar?" + urlencode(params))
    return (client or ReadOnlyClient()).get(CALENDAR_URL, params)


async def receive_market(stream, queue, latest):
    async for event in stream:
        latest["event"] = event
        if event.get("T") == "q":
            quote = quote_from_event(event, event.get("arrival_ns") or time.time_ns())
            if "quote" not in latest or quote.event_ns >= latest["quote"].event_ns:
                latest["quote"] = quote
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            raise RuntimeError("market queue overflow; stop and reconcile") from None
    raise RuntimeError("market data stream closed")


async def _receive_updates(client, queue):
    from .broker import trade_updates

    async for update in trade_updates(client):
        try:
            queue.put_nowait(update)
        except asyncio.QueueFull:
            raise RuntimeError("broker update queue overflow") from None
    raise RuntimeError("trade update stream closed")


async def _flatten(broker, engine):
    deadline = time.monotonic() + 20
    await asyncio.to_thread(broker.cancel)
    while broker.pending and time.monotonic() < deadline:
        engine.record_fills(await asyncio.to_thread(broker.poll))
        if broker.pending:
            await asyncio.sleep(0.1)
    if broker.pending:
        raise RuntimeError("cancel outcome unresolved; broker order may remain open")
    await asyncio.to_thread(broker.refresh)
    if broker.account.position:
        if not broker.market_clock.get("is_open"):
            raise RuntimeError("market closed with inventory; inspect broker account")
        intent = OrderIntent(
            uuid.uuid4().hex,
            broker.symbol,
            "sell",
            broker.account.position,
            engine.quote.bid if engine.quote else 1,
            time.time_ns(),
            True,
        )
        decision = await asyncio.to_thread(
            broker.submit, intent, engine.snapshot if engine.session else None
        )
        if not decision.allowed:
            raise RuntimeError("emergency exit rejected: " + str(decision.reason))
        while broker.pending and time.monotonic() < deadline:
            engine.record_fills(await asyncio.to_thread(broker.poll))
            if broker.pending:
                await asyncio.sleep(0.1)
    if broker.pending or broker.account.position:
        raise RuntimeError("exit unresolved; inspect broker orders and inventory")
    await asyncio.to_thread(broker.reconcile)


async def run_stream(
    policy, config, *, mode="dry-run", log_path, duration=None, broker=None, stream=None
):
    if duration is not None and (duration <= 0 or not math.isfinite(duration)):
        raise ValueError("duration must be positive and finite")
    calendar = await asyncio.to_thread(fetch_calendar, broker.client if broker else None)
    windows = calendar_from_records(calendar)
    log_path = Path(log_path)
    from .data import atomic_json

    atomic_json(
        log_path.parent / "calendar.json", {"calendar": calendar, "provenance": "alpaca-calendar"}
    )
    queue = asyncio.Queue(maxsize=4096)
    updates = asyncio.Queue(maxsize=1024)
    latest = {}
    queued = []

    def outstanding():
        return bool(queued) or bool(broker.pending if broker else engine.execution.pending)

    metadata = {
        **policy.metadata,
        "contract_hash": policy.manifest["contract_hash"],
        "account_id": broker.account_id if broker else "local-" + policy.metadata["sha256"],
    }
    engine = PaperEngine(
        policy,
        log_path=log_path,
        initial_cash=config.capital,
        costs=config.costs,
        risk=broker.risk if broker else RiskGateway(config.risk),
        sizing=config.sizing,
        account=broker.account if broker else None,
        symbol=config.symbol,
        synthetic=False,
        source=mode,
        latency_ms=config.latency_ms,
        metadata=metadata,
        submit_callback=queued.append if broker else None,
        outstanding=outstanding,
    )
    local_store = None
    if not broker:
        local_store = AccountStateStore(metadata["account_id"], mode)
        local_store.__enter__()
        try:
            saved = local_store.load()
            if saved:
                if saved["contract_hash"] != policy.manifest["contract_hash"]:
                    raise ValueError("local state contract mismatch")
                engine.account = Account.from_state(saved["account"])
                engine.risk.restore(saved["risk"])
                engine.execution.account = engine.account
                engine.execution.cancel()  # no actual local order survives a stopped process
            engine._trade_count = len(engine.account.trades)
        except BaseException:
            local_store.__exit__()
            engine.finish()
            raise
    tasks = [
        asyncio.create_task(
            receive_market(
                stream or alpaca_events(config.symbol, config.feed, timeout=None), queue, latest
            )
        )
    ]
    if broker:
        tasks.append(asyncio.create_task(_receive_updates(broker.client, updates)))
    started = time.monotonic()
    last_poll = last_refresh = last_save = 0.0
    clean = False
    cleanup_error = None
    try:
        if broker:
            engine.record_fills(broker.recovered_fills)
        while duration is None or time.monotonic() - started < duration:
            for task in tasks:
                if task.done():
                    task.result()
            now = time.time_ns()
            session = session_at(windows, now)
            if engine.session and (not session or session.session_id != engine.session.session_id):
                if broker:
                    await _flatten(broker, engine)
                engine.tick(engine.session.close_ns)
                engine.end_session(complete=True, reconciled=True)
                if broker and mode == "live":
                    broker.live_sessions += 1
                    broker._save()
                    if broker.live_sessions >= 10:
                        raise RuntimeError(
                            "ten live canary sessions complete; operator review required"
                        )
            if session and not engine.session:
                engine.start_session(session, now)
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                event = None
            now = time.time_ns()
            if event and engine.session:
                arrival = event.get("arrival_ns") or now
                if now - arrival > NS:
                    engine.market.reset_history()
                    engine.session_complete = False
                    queued.clear()
                    engine.risk.halted = "runtime_gap"
                    if broker:
                        await asyncio.to_thread(broker.cancel)
                    engine.log.write("gap", event_ns=now, reason="processing_backlog")
                    if not broker:
                        engine.execution.cancel()
                else:
                    engine.on_event(event, arrival)
            if engine.session:
                if now - (engine.last_event_ns or engine.session.open_ns) > 5 * NS:
                    raise RuntimeError("market feed gap during an open session")
                engine.tick(now)
            if broker:
                while not updates.empty():
                    update = updates.get_nowait()
                    engine.record_fills(broker.apply_update(update["order"], allow_old=True))
                if queued:
                    intent = queued.pop(0)

                    def current_snapshot():
                        snap = engine.snapshot()
                        quote = latest.get("quote")
                        return replace(
                            snap,
                            quote=quote,
                            equity=engine.account.mark(quote.mid) if quote else engine.account.cash,
                        )

                    decision = await asyncio.to_thread(broker.submit, intent, current_snapshot)
                    engine.log.write(
                        "broker_submit",
                        event_ns=time.time_ns(),
                        allowed=decision.allowed,
                        reason=decision.reason,
                    )
                monotonic = time.monotonic()
                if monotonic - last_poll >= 1:
                    engine.record_fills(await asyncio.to_thread(broker.poll))
                    last_poll = monotonic
                    if broker.pending and (
                        now - broker.pending.intent.created_ns
                        >= broker.risk.config.order_expiry_seconds * NS
                        or broker.risk.halted
                    ):
                        await asyncio.to_thread(broker.cancel)
                if monotonic - last_refresh >= 5:
                    await asyncio.to_thread(broker.refresh)
                    await asyncio.to_thread(broker.reconcile)
                    broker._save()
                    last_refresh = monotonic
            if local_store and time.monotonic() - last_save >= 1:
                local_store.save(
                    {
                        "contract_hash": policy.manifest["contract_hash"],
                        "account": engine.account.to_state(),
                        "risk": engine.risk.to_state(),
                    }
                )
                last_save = time.monotonic()
        clean = True
    except asyncio.CancelledError:
        # Operator stop: preserve completed sessions, but never certify a partial one.
        clean = True
        if engine.session:
            engine.session_complete = False
        engine.log.write("stopped", event_ns=time.time_ns(), reason="operator cancellation")
    except BaseException as exc:
        engine.session_complete = False
        engine.log.write(
            "halt", event_ns=time.time_ns(), reason=type(exc).__name__ + ": " + str(exc)
        )
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        reconciled = True
        try:
            if broker:
                await _flatten(broker, engine)
            elif engine.account.position:
                # No fresh future quote now exists: retain and report rather than invent liquidation.
                clean = False
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            reconciled = False
            clean = False
            cleanup_error = exc
            engine.log.write("reconcile_error", reason=str(exc))
        if local_store:
            try:
                local_store.save(
                    {
                        "contract_hash": policy.manifest["contract_hash"],
                        "account": engine.account.to_state(),
                        "risk": engine.risk.to_state(),
                    }
                )
            finally:
                local_store.__exit__()
        engine.finish(
            complete=clean
            and bool(
                (engine.session and time.time_ns() >= engine.session.close_ns)
                or not engine.session
                and engine.completed_sessions
            ),
            reconciled=reconciled,
        )
        if broker:
            broker.close()
    if cleanup_error:
        raise RuntimeError("shutdown unresolved: " + str(cleanup_error))
    return {
        "log": str(log_path),
        "cash": str(engine.account.cash),
        "position": str(engine.account.position),
        "trades": len(engine.account.trades),
        "complete": engine.finished_complete,
        "run_completed": clean,
        "mode": mode,
    }
