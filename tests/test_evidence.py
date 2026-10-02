from hft.evidence import graduate
from hft.logs import EventLog


def test_missing_model_and_empty_logs_fail_closed(tmp_path):
    result = graduate(tmp_path / "no-model", tmp_path / "logs")
    assert result["passed"] is False
    assert "invalid-model" in result["reasons"]


def test_synthetic_replay_and_forged_pass_dict_cannot_graduate(tmp_path, monkeypatch):
    import hft.policy

    monkeypatch.setattr(
        hft.policy,
        "validate_bundle",
        lambda path: {
            "metadata": {"synthetic_training": True},
            "research": {"paper_eligible": True},
            "contract_hash": "x",
        },
    )
    log = EventLog(tmp_path / "logs" / "run.jsonl")
    log.write("start", source="replay", synthetic=True, metadata={"passed": True})
    log.write("finish", complete=True)
    log.close()
    result = graduate(tmp_path / "model", tmp_path / "logs")
    assert not result["passed"]
    assert "synthetic-model" in result["reasons"]
    assert "missing-exchange-calendar" in result["reasons"]


def paper_fixture(tmp_path, monkeypatch, days=30):
    from dataclasses import asdict
    from decimal import Decimal

    import hft.evidence
    import hft.policy
    from hft.account import Account, Costs, Execution
    from hft.data import atomic_json
    from hft.risk import RiskConfig
    from hft.sizing import SizingConfig

    start = 1_735_830_000 * 1_000_000_000
    calendar = [
        {
            "session_id": f"2025-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
            "open_ns": start + i * 86400 * 10**9,
            "close_ns": start + i * 86400 * 10**9 + 600 * 10**9,
        }
        for i in range(days + 12)
    ]
    metadata = {
        "synthetic_training": False,
        "initial_cash": 500,
        "symbol": "AAPL",
        "feed": "iex",
        "sha256": "model",
        "execution_hash": "execution",
        "costs": asdict(Costs(0, 0)),
        "risk": asdict(RiskConfig()),
        "sizing": asdict(SizingConfig(fee_per_share=0, slippage_bps=0)),
        "latency_ms": 75,
        "bar_seconds": 5,
    }
    stat = {
        "net_profit": 20,
        "expectancy": 0.1,
        "max_drawdown": 0.01,
        "trades": 120,
        "profit_factor": 2,
        "gross_profit": 40,
        "gross_loss": 20,
        "sessions": 30,
        "incomplete": False,
    }
    research = {
        "synthetic": False,
        "paper_eligible": True,
        "primary_control": "ema_5_20",
        "eligibility": {"real_executable_data": True},
        "test": {
            "policy": stat,
            "stress": stat,
            "edge": {"ema_5_20": {"positive": True, "lower": 0.001}},
            "random": {"matched_fill_frequency": True},
        },
    }
    monkeypatch.setattr(
        hft.policy,
        "validate_bundle",
        lambda path: {"metadata": metadata, "research": research, "contract_hash": "bundle"},
    )
    monkeypatch.setattr(
        hft.evidence,
        "_shadow",
        lambda *a, **kw: stat | {"incomplete": False, "execution_hash": "shadow"},
    )
    account = Account(500, Costs(0, 0))
    logs = tmp_path / "logs"
    for day in range(days):
        window = calendar[day]
        now = [window["open_ns"]]
        log = EventLog(logs / f"{day:03d}.jsonl", clock=lambda now=now: now[0])
        log.write(
            "start",
            source="broker-paper",
            synthetic=False,
            symbol="AAPL",
            initial_cash=500,
            costs=metadata["costs"],
            risk=metadata["risk"],
            sizing=metadata["sizing"],
            latency_ms=75,
            bar_seconds=5,
            account_state=account.to_state(),
            metadata=metadata | {"contract_hash": "bundle", "account_id": "paper-account"},
        )
        log.write("session_start", event_ns=now[0], calendar=window)
        trade_count = len(account.trades)
        for tick in range(0, 600, 5):
            now[0] = window["open_ns"] + tick * 10**9
            price = 101 if tick in (15, 25, 35, 45) else 100
            event = dict(
                T="q",
                S="AAPL",
                event_ns=now[0],
                arrival_ns=now[0],
                bp=price,
                ap=price,
                bs=100,
                **{"as": 100},
            )
            log.write("market", event_ns=now[0], payload=event)
            if day == 0 and tick == 15:
                # A boundary bar publishes before its arriving quote is applied.
                log.write(
                    "equity",
                    event_ns=now[0],
                    equity=account.mark(100),
                    cash=account.cash,
                    position=account.position,
                )
            if tick in (10, 15, 20, 25, 30, 35, 40, 45):
                qty = Decimal(".1") if tick % 10 == 0 else Decimal("-.1")
                fill = Execution(
                    f"{day}:{tick}", f"order-{day}-{tick}", now[0], qty, Decimal(price), Decimal(0)
                )
                account.apply(fill)
                log.write("fill", event_ns=now[0], execution=asdict(fill))
                if len(account.trades) > trade_count:
                    trade_count += 1
                    log.write("trade", event_ns=now[0], **asdict(account.trades[-1]))
            log.write(
                "equity",
                event_ns=now[0],
                equity=account.mark(price),
                position=account.position,
                cash=account.cash,
            )
        now[0] = window["close_ns"] - 10**9
        log.write(
            "market",
            event_ns=now[0],
            payload=dict(
                T="q",
                S="AAPL",
                event_ns=now[0],
                arrival_ns=now[0],
                bp=100,
                ap=100,
                bs=100,
                **{"as": 100},
            ),
        )
        now[0] = window["close_ns"]
        log.write(
            "session",
            event_ns=now[0],
            date=window["session_id"],
            calendar=window,
            complete=True,
            reconciled=True,
            position=0,
            equity=account.cash,
        )
        log.write(
            "finish",
            complete=True,
            reconciled=True,
            unresolved=False,
            position=0,
            cash=account.cash,
        )
        log.close()
    atomic_json(logs / "calendar.json", {"calendar": calendar})
    return logs, calendar, calendar[days - 1]["close_ns"] + 10**9


def test_recomputed_paper_gate_and_expiry(tmp_path, monkeypatch):
    logs, calendar, now = paper_fixture(tmp_path, monkeypatch)
    result = graduate(tmp_path / "model", logs, now_ns=now, max_notional=10)
    assert result["passed"], result["reasons"]
    assert result["trades"] == 120 and result["sessions"] == 30
    stale = graduate(tmp_path / "model", logs, now_ns=calendar[36]["close_ns"])
    assert "paper-evidence-expired" in stale["reasons"]
    truncated = graduate(
        tmp_path / "model", logs, calendar_records=calendar[:30], now_ns=calendar[36]["close_ns"]
    )
    assert "calendar-does-not-cover-current-time" in truncated["reasons"]


def test_rehashed_invented_profit_is_rejected(tmp_path, monkeypatch):
    from hft.logs import read_log

    logs, _, now = paper_fixture(tmp_path, monkeypatch)
    path = logs / "000.jsonl"
    rows = read_log(path)
    path.unlink()
    clock = [0]
    log = EventLog(path, clock=lambda: clock[0])
    for row in rows:
        clock[0] = row["wall_ns"]
        values = {
            k: v
            for k, v in row.items()
            if k not in ("event", "run_id", "sequence", "wall_ns", "previous", "hash")
        }
        if row["event"] == "trade":
            values["pnl"] = "100000"
        log.write(row["event"], **values)
    log.close()
    result = graduate(tmp_path / "model", logs, now_ns=now)
    assert "invalid-paper-journal" in result["reasons"]


def test_canary_failure_blocks_otherwise_passing_paper(tmp_path, monkeypatch):
    import hft.evidence

    logs, _, now = paper_fixture(tmp_path, monkeypatch)

    def shadow(model, paths, dates, metadata, max_notional):
        return {
            "net_profit": -1 if max_notional else 1,
            "expectancy": 0.1,
            "max_drawdown": 0.01,
            "incomplete": False,
            "execution_hash": "shadow",
        }

    monkeypatch.setattr(hft.evidence, "_shadow", shadow)
    result = graduate(tmp_path / "model", logs, now_ns=now, max_notional=10)
    assert not result["passed"]
    assert result["reasons"] == ["canary-shadow-failed"]
