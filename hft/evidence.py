"""Recompute graduation from immutable bundles and actual, complete paper journals."""

import itertools
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from .account import Account, Costs, Execution, decimal
from .calendar import calendar_from_records, load_calendar
from .feed import quote_from_event
from .logs import canonical_hash, iter_log
from .metrics import EquitySummary, metrics
from .risk import RiskGateway

NS = 1_000_000_000


def _passes_stats(result, *, count=True):
    pf = result.get("profit_factor")
    return (
        result.get("net_profit", 0) > 0
        and result.get("expectancy", 0) > 0
        and result.get("max_drawdown", 1) < 0.05
        and (
            not count
            or result.get("trades", 0) >= 100
            and (pf == "infinite" or isinstance(pf, (int, float)) and pf >= 1.3)
        )
    )


def _shadow(model_path, paths, dates, metadata, max_notional):
    from .account import Costs
    from .calendar import SessionWindow
    from .configuration import runtime_config
    from .paper import PaperEngine
    from .policy import OnnxPolicy

    selected = set(dates)
    config = runtime_config(metadata, max_notional=max_notional)
    config = replace(
        config,
        costs=Costs(config.costs.commission * 2, 3),
        latency_ms=250,
        sizing=replace(config.sizing, fee_per_share=config.costs.commission * 2, slippage_bps=3),
    )
    with tempfile.TemporaryDirectory(prefix="hft-shadow-") as directory:
        path = Path(directory) / "shadow.jsonl"
        engine = PaperEngine(
            OnnxPolicy(model_path),
            log_path=path,
            initial_cash=config.capital,
            costs=config.costs,
            sizing=config.sizing,
            symbol=config.symbol,
            source="conservative-shadow",
            latency_ms=config.latency_ms,
            risk=RiskGateway(config.risk),
            bar_seconds=config.bar_seconds,
        )
        for journal in paths:
            active = False
            for row in iter_log(journal):
                if row["event"] == "session_start":
                    session = SessionWindow(**row["calendar"])
                    active = session.session_id in selected
                    if active:
                        engine.start_session(session)
                elif row["event"] == "market" and active:
                    engine.on_event(row["payload"], row["event_ns"])
                elif row["event"] == "timer" and active:
                    engine.tick(row["event_ns"])
                elif row["event"] == "session" and active:
                    engine.tick(row["calendar"]["close_ns"])
                    engine.end_session(complete=True)
                    active = False
        incomplete = engine.account.position != 0 or engine.outstanding()
        engine.finish()
        daily = [config.capital]
        curve = EquitySummary(config.capital)
        pnl = []
        for row in iter_log(path):
            if row["event"] == "equity":
                curve.add(row["equity"])
            if row["event"] == "session":
                daily.append(float(row["equity"]))
                incomplete |= not row["complete"]
            if row["event"] == "trade":
                pnl.append(float(row["pnl"]))
        result = metrics(daily, pnl)
        result["max_drawdown"] = curve.drawdown
        result["sample_counts"]["intraday_marks"] = curve.marks
        result["incomplete"] = incomplete
        result["execution_hash"] = config.execution_hash
        return result


def graduate(
    model_path,
    logs_path,
    *,
    calendar_path=None,
    calendar_records=None,
    now_ns=None,
    max_notional=None,
):
    from .policy import validate_bundle
    from .research import eligibility

    now = time.time_ns() if now_ns is None else now_ns
    result = {"passed": False, "reasons": [], "sessions": 0, "trades": 0, "evaluated_ns": now}
    reasons = result["reasons"]
    try:
        bundle = validate_bundle(model_path)
    except (OSError, ValueError, KeyError, TypeError):
        reasons.append("invalid-model")
        return result
    metadata = bundle["metadata"]
    research = bundle.get("research", {})
    if metadata.get("synthetic_training") is not False:
        reasons.append("synthetic-model")
    try:
        gate = eligibility(
            research["test"],
            synthetic=research["synthetic"],
            primary_control=research["primary_control"],
            real_executable_data=research["eligibility"]["real_executable_data"],
        )
        if not gate["passed"] or research.get("paper_eligible") is not True:
            reasons.append("research-gate-failed")
    except (KeyError, TypeError, ValueError):
        reasons.append("missing-research-evidence")
    if calendar_records is not None:
        windows = calendar_from_records(calendar_records)
    else:
        calendar_path = Path(calendar_path) if calendar_path else Path(logs_path) / "calendar.json"
        try:
            windows = load_calendar(calendar_path)
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append("missing-exchange-calendar")
            return result
    if not windows:
        reasons.append("missing-exchange-calendar")
        return result
    if windows[-1].close_ns < now:
        reasons.append("calendar-does-not-cover-current-time")
    summaries = {}
    paths = []
    account_ids = set()
    seen_trades = set()
    for path in sorted(Path(logs_path).glob("*.jsonl")):
        try:
            header = None
            trades = []
            curve = None
            issues = []
            dates = []
            finish = None
            quote = previous_quote = None
            ledger = None
            trade_cursor = 0
            first_event = last_event = None
            session_start = None
            for row in iter_log(path):
                event = row["event"]
                if header is None:
                    header = row
                    contract = row.get("metadata", {})
                    if (
                        event != "start"
                        or row.get("source") != "broker-paper"
                        or row.get("synthetic") is not False
                        or contract.get("contract_hash") != bundle["contract_hash"]
                        or contract.get("sha256") != metadata.get("sha256")
                        or contract.get("execution_hash") != metadata.get("execution_hash")
                        or row.get("symbol") != metadata.get("symbol")
                    ):
                        raise ValueError("incompatible paper journal")
                    account_ids.add(contract.get("account_id"))
                    for key in ("costs", "risk", "sizing", "latency_ms", "bar_seconds"):
                        if row.get(key) != metadata.get(key):
                            raise ValueError("journal execution contract mismatch")
                    ledger = Account(metadata["initial_cash"], Costs(**metadata["costs"]))
                    saved = row["account_state"]
                    if saved.get("cash_flows"):
                        raise ValueError("unreconciled external cash flows")
                    for fill in saved["fills"]:
                        ledger.apply(
                            Execution(
                                **{
                                    **fill,
                                    "signed_quantity": decimal(fill["signed_quantity"]),
                                    "price": decimal(fill["price"]),
                                    "fee": decimal(fill["fee"]),
                                }
                            )
                        )
                    if ledger.cash != decimal(saved["cash"]) or ledger.position != decimal(
                        saved["position"]
                    ):
                        raise ValueError("opening ledger mismatch")
                    trade_cursor = len(ledger.trades)
                    curve = EquitySummary(ledger.cash)
                if event == "session_start":
                    first_event = last_event = None
                    session_start = float(ledger.cash)
                    curve = EquitySummary(session_start)
                if event == "fill":
                    fill = row["execution"]
                    ledger.apply(
                        Execution(
                            **{
                                **fill,
                                "signed_quantity": decimal(fill["signed_quantity"]),
                                "price": decimal(fill["price"]),
                                "fee": decimal(fill["fee"]),
                            }
                        )
                    )
                    if quote:
                        curve.add(ledger.mark(quote.mid))
                if event in ("gap", "halt", "recovery_error", "reconcile_error"):
                    issues.append(event)
                if event == "market":
                    if row["event_ns"] is None or abs(row["wall_ns"] - row["event_ns"]) > 2 * NS:
                        issues.append("non-realtime-event")
                    if last_event is not None and row["event_ns"] - last_event > 5 * NS:
                        issues.append("coverage-gap")
                    first_event = first_event or row["event_ns"]
                    last_event = row["event_ns"]
                    if row["payload"].get("T") == "q":
                        candidate = quote_from_event(row["payload"], row["event_ns"])
                        if candidate.event_ns <= row["event_ns"] + 250_000_000 and (
                            quote is None or candidate.event_ns >= quote.event_ns
                        ):
                            previous_quote = quote
                            quote = candidate
                            curve.add(ledger.mark(quote.mid))
                if event == "trade":
                    identity = tuple(row["execution_ids"])
                    if identity in seen_trades:
                        raise ValueError("duplicate completed trade")
                    if trade_cursor >= len(ledger.trades):
                        raise ValueError("invented completed trade")
                    expected = ledger.trades[trade_cursor]
                    trade_cursor += 1
                    if (
                        expected.execution_ids != identity
                        or expected.pnl != decimal(row["pnl"])
                        or expected.entry_ns != row["entry_ns"]
                        or expected.exit_ns != row["exit_ns"]
                    ):
                        raise ValueError("completed trade does not reconcile")
                    seen_trades.add(identity)
                    trades.append(float(expected.pnl))
                if event == "equity":
                    # Bar publication precedes applying its boundary quote. Both marks
                    # are causal; independent quote extrema determine actual drawdown.
                    candidates = [ledger.mark(q.mid) for q in (quote, previous_quote) if q]
                    candidates = candidates or [ledger.cash]
                    if min(abs(value - decimal(row["equity"])) for value in candidates) > decimal(
                        ".000001"
                    ):
                        raise ValueError("equity mark does not reconcile")
                    if (
                        decimal(row["cash"]) != ledger.cash
                        or decimal(row["position"]) != ledger.position
                    ):
                        raise ValueError("equity ledger does not reconcile")
                if event == "session":
                    sid = row["date"]
                    dates.append(sid)
                    if sid in summaries:
                        raise ValueError("duplicate paper session")
                    w = next((w for w in windows if w.session_id == sid), None)
                    calendar_ok = (
                        w
                        and row["calendar"]["open_ns"] == w.open_ns
                        and row["calendar"]["close_ns"] == w.close_ns
                    )
                    clock_ok = w and abs(row["wall_ns"] - w.close_ns) <= 30 * NS
                    coverage = (
                        w
                        and first_event is not None
                        and first_event <= w.open_ns + 5 * NS
                        and last_event is not None
                        and last_event >= w.close_ns - 2 * NS
                    )
                    if ledger.position != decimal(row["position"]) or ledger.cash != decimal(
                        row["equity"]
                    ):
                        raise ValueError("session ledger does not reconcile")
                    complete = (
                        coverage
                        and row.get("complete") is True
                        and row.get("reconciled") is True
                        and float(row.get("position", 1)) == 0
                        and not issues
                        and calendar_ok
                        and clock_ok
                    )
                    summaries[sid] = {
                        "complete": bool(complete),
                        "equity": float(row["equity"]),
                        "pnl": trades[:],
                        "curve": curve,
                        "path": path,
                        "starting_equity": session_start,
                    }
                    trades.clear()
                    curve = None
                    issues.clear()
                if event == "finish":
                    finish = row
            if (
                not finish
                or finish.get("complete") is not True
                or finish.get("unresolved")
                or not finish.get("reconciled")
            ):
                for sid in dates:
                    summaries[sid]["complete"] = False
            paths.append(path)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            reasons.append("invalid-paper-journal")
            return result
    if len(account_ids) != 1 or None in account_ids:
        reasons.append("inconsistent-paper-account")
    dates = sorted(summaries)[-30:]
    result["sessions"] = len(dates)
    if len(dates) < 30:
        reasons.append("insufficient-consecutive-paper-sessions")
    if dates:
        calendar_ids = [w.session_id for w in windows]
        try:
            first, last = calendar_ids.index(dates[0]), calendar_ids.index(dates[-1])
            if calendar_ids[first : last + 1] != dates:
                reasons.append("nonconsecutive-paper-sessions")
            elapsed = [w for w in windows if dates[-1] < w.session_id and w.close_ns <= now]
            if len(elapsed) > 5:
                reasons.append("paper-evidence-expired")
            if windows[last].close_ns > now:
                reasons.append("future-paper-evidence")
        except ValueError:
            reasons.append("calendar-coverage-missing")
        if any(not summaries[d]["complete"] for d in dates):
            reasons.append("incomplete-paper-session")
        daily = [summaries[dates[0]]["starting_equity"]] + [summaries[d]["equity"] for d in dates]
        if any(
            abs(summaries[b]["starting_equity"] - summaries[a]["equity"]) > 1e-6
            for a, b in itertools.pairwise(dates)
        ):
            reasons.append("discontinuous-paper-ledger")
        pnl = [p for d in dates for p in summaries[d]["pnl"]]
        curve = EquitySummary(daily[0])
        for d in dates:
            curve.combine(summaries[d]["curve"])
        try:
            stats = metrics(daily, pnl)
            stats["max_drawdown"] = curve.drawdown
            stats["sample_counts"]["intraday_marks"] = curve.marks
            result.update(paper=stats, trades=stats["trades"], dates=dates)
            if not _passes_stats(stats):
                reasons.append("paper-performance-gate-failed")
        except ValueError:
            reasons.append("invalid-paper-statistics")
    result.update(
        model_hash=metadata.get("sha256"),
        contract_hash=bundle["contract_hash"],
        paper_execution_hash=metadata.get("execution_hash"),
    )
    if not reasons:
        selected_paths = list(dict.fromkeys(summaries[d]["path"] for d in dates))
        try:
            shadow = _shadow(model_path, selected_paths, dates, metadata, None)
            result["shadow"] = shadow
            if shadow["incomplete"] or not _passes_stats(shadow, count=False):
                reasons.append("conservative-shadow-failed")
            if max_notional is not None:
                canary = _shadow(model_path, selected_paths, dates, metadata, max_notional)
                result["canary_shadow"] = canary
                result["canary_execution_hash"] = canary["execution_hash"]
                if canary["incomplete"] or not _passes_stats(canary, count=False):
                    reasons.append("canary-shadow-failed")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            reasons.append("shadow-evidence-unavailable")
    result["passed"] = not reasons
    result["evidence_hash"] = canonical_hash(result)
    return result
