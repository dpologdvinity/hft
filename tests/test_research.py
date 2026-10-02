import pytest

from hft.research import eligibility, make_folds, temporal_split


def test_frozen_split_exact_dates_and_fold_endpoints():
    ids = [f"{i:03d}" for i in range(120)]
    dev, test = temporal_split(ids)
    assert dev == ids[:90] and test == ids[90:]
    folds = make_folds(dev)
    assert [(len(f.train), len(f.validation)) for f in folds] == [(54, 12), (66, 12), (78, 12)]
    assert all(set(f.train).isdisjoint(f.validation) for f in folds)
    assert set().union(*(set(f.train) | set(f.validation) for f in folds)).isdisjoint(test)
    with pytest.raises(ValueError, match="insufficient"):
        temporal_split(ids[:60])


def test_synthetic_and_small_sample_cannot_graduate():
    report = {
        "policy": {
            "sessions": 30,
            "trades": 1,
            "expectancy": 1,
            "net_profit": 1,
            "profit_factor": "infinite",
            "max_drawdown": 0,
        },
        "stress": {"net_profit": 1},
        "edge": {"intraday_long": {"positive": True}},
    }
    result = eligibility(report, synthetic=True, primary_control="intraday_long")
    assert not result["passed"]
    assert (
        "synthetic-data" in result["reasons"]
        and "insufficient-completed-trades" in result["reasons"]
    )


def test_ema_arithmetic_and_actual_frequency_control():
    import numpy as np

    from hft.data import synthetic_sessions
    from hft.research import ema, evaluate

    assert ema([10, 13, 16], 5).tolist() == pytest.approx([10, 11, 12 + 2 / 3])
    result, observations = evaluate(synthetic_sessions(1, 100, 11), lambda obs: 1)
    assert result["cash"]["net_profit"] == 0
    assert len(result["random"]["runs"]) == 20
    assert result["random"]["matched_fill_frequency"]
    assert all(
        r["session_fills"] == result["policy"]["session_fills"] for r in result["random"]["runs"]
    )
    assert observations.dtype == np.float32 and observations.shape[1] == 17


def test_freeze_before_search_and_insufficient_history(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hft.research import freeze_experiment

    source = tmp_path / "dataset.json"
    source.write_text("{}")
    sessions = [
        SimpleNamespace(session_id=f"{i:03d}", symbol="SPY", manifest={"synthetic": False})
        for i in range(120)
    ]
    monkeypatch.setattr("hft.data.load_dataset", lambda path: sessions)
    result = freeze_experiment(source, {}, tmp_path / "experiment.json")
    assert result["status"] == "frozen" and result["final_test"] == [
        f"{i:03d}" for i in range(90, 120)
    ]
    assert not result["final_test_consumed"]
    with pytest.raises(FileExistsError):
        freeze_experiment(source, {}, tmp_path / "experiment.json")
    monkeypatch.setattr("hft.data.load_dataset", lambda path: sessions[:60])
    assert freeze_experiment(source, {}, tmp_path / "small.json")["status"] == "insufficient-data"


def test_freeze_loaded_real_parquet_metadata(tmp_path, monkeypatch):
    from hft.history import download_sessions
    from hft.research import freeze_experiment

    class RecordedTransport:
        def get(self, url, params):
            if url.endswith("/calendar"):
                return [{"date": "2025-01-02", "open": "09:30", "close": "16:00"}]
            kind = "quotes" if url.endswith("/quotes") else "trades"
            quote = {
                "t": "2025-01-02T14:30:00.123456789Z",
                "bp": 99.99,
                "ap": 100.01,
                "bs": 1,
                "as": 2,
                "c": ["R"],
            }
            trade = {
                "t": "2025-01-02T14:30:00.123456789Z",
                "p": 100.0,
                "s": 1,
                "i": 1,
                "c": ["@"],
                "z": "C",
            }
            return {kind: [quote if kind == "quotes" else trade], "next_page_token": None}

    monkeypatch.setattr("hft.history._check_disk", lambda *args: None)
    manifest = download_sessions(
        "X", "2025-01-02", "2025-01-02", tmp_path / "data", client=RecordedTransport()
    )
    result = freeze_experiment(manifest, {}, tmp_path / "experiment.json")
    assert result["status"] == "insufficient-data"
    assert result["real_executable_data"] and result["session_hashes"]["2025-01-02"]


def test_rollout_carries_peak_risk_and_permanent_halt_between_sessions(monkeypatch):
    from decimal import Decimal
    from types import SimpleNamespace

    import numpy as np

    from hft.research import _rollout
    from hft.risk import RiskGateway

    created = []

    class Env:
        def __init__(self, session, initial_cash, **kwargs):
            self.session = session
            self.initial_cash = initial_cash
            self.account = SimpleNamespace(
                initial_cash=Decimal(str(initial_cash)), completed_trades=(), fills=[]
            )
            self.risk = RiskGateway()
            self.simulation = SimpleNamespace(
                risk=self.risk,
                snapshot=self.snapshot,
                now_ns=session.open_ns,
                equity_curve=[initial_cash],
            )
            created.append(self)

        def snapshot(self):
            return SimpleNamespace(
                session=self.session,
                equity=Decimal(str(self.initial_cash)),
                initial_equity=self.account.initial_cash,
                history=(),
                quote=None,
                position=Decimal(0),
            )

        def reset(self):
            self.risk.observe(self.snapshot(), self.session.open_ns)
            return np.zeros(17, dtype=np.float32), {}

        def step(self, action):
            self.initial_cash -= 3
            self.simulation.equity_curve.extend([self.initial_cash - 17, self.initial_cash])
            self.risk.observe(self.snapshot(), self.session.open_ns + 1)
            return (
                np.zeros(17, dtype=np.float32),
                0.0,
                True,
                False,
                {"equity": self.initial_cash, "incomplete": False},
            )

        def _observation(self):
            return np.zeros(17, dtype=np.float32)

        def close(self):
            pass

    monkeypatch.setattr("hft.env.TradingEnv", Env)
    sessions = [
        SimpleNamespace(
            session_id=str(i), open_ns=(i + 1) * 10**12, close_ns=(i + 1) * 10**12 + 1000 * 10**9
        )
        for i in range(2)
    ]
    result, _ = _rollout(sessions, lambda obs, env: 0, initial_cash=100)
    assert result["max_drawdown"] == pytest.approx(0.23)
    assert created[-1].risk.peak == Decimal(100)
    assert created[-1].risk.permanent_halt


def test_realtime_partial_sessions_never_supply_qualifying_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hft.research import freeze_experiment

    source = tmp_path / "dataset.json"
    source.write_text("{}")
    sessions = [
        SimpleNamespace(
            session_id=f"{i:03d}",
            symbol="SPY",
            manifest={
                "synthetic": False,
                "feed": "iex",
                "provenance": "alpaca-realtime-iex",
                "complete_session": True,
            },
        )
        for i in range(120)
    ]
    sessions[-1].manifest["complete_session"] = False
    monkeypatch.setattr("hft.data.load_dataset", lambda path: sessions)
    result = freeze_experiment(source, {}, tmp_path / "experiment.json")
    assert result["status"] == "frozen" and not result["real_executable_data"]


def test_nonfinite_metric_or_confidence_lower_cannot_pass_gate():
    from hft.research import eligibility

    report = {
        "policy": {
            "sessions": 30,
            "trades": 100,
            "expectancy": 1,
            "net_profit": 100,
            "profit_factor": "infinite",
            "gross_profit": 100,
            "gross_loss": 0,
            "max_drawdown": 0,
        },
        "stress": {"net_profit": 100},
        "edge": {"intraday_long": {"positive": True, "lower": 0.01}},
    }
    assert eligibility(report, synthetic=False, primary_control="intraday_long")["passed"]
    report["edge"]["intraday_long"]["lower"] = float("nan")
    assert not eligibility(report, synthetic=False, primary_control="intraday_long")["passed"]
    report["edge"]["intraday_long"]["lower"] = 0.01
    report["policy"]["gross_profit"] = float("nan")
    assert not eligibility(report, synthetic=False, primary_control="intraday_long")["passed"]


def test_json_infinity_is_not_explicit_infinite_profit_factor():
    report = {
        "policy": {
            "sessions": 30,
            "trades": 100,
            "expectancy": 1,
            "net_profit": 100,
            "profit_factor": float("inf"),
            "gross_profit": 100,
            "gross_loss": 0,
            "max_drawdown": 0,
        },
        "stress": {"net_profit": 100},
        "edge": {"intraday_long": {"positive": True, "lower": 0.01}},
    }
    assert not eligibility(report, synthetic=False, primary_control="intraday_long")["passed"]


def test_profitable_stress_with_unresolved_inventory_fails_eligibility():
    report = {
        "policy": {
            "sessions": 30,
            "trades": 100,
            "expectancy": 1,
            "net_profit": 100,
            "profit_factor": "infinite",
            "gross_profit": 100,
            "gross_loss": 0,
            "max_drawdown": 0,
        },
        "stress": {"net_profit": 100, "incomplete": True},
        "edge": {"intraday_long": {"positive": True, "lower": 0.01}},
    }
    result = eligibility(report, synthetic=False, primary_control="intraday_long")
    assert not result["passed"]
    assert "incomplete-stress-liquidation" in result["reasons"]


@pytest.mark.parametrize("section", ["policy", "stress"])
def test_feed_gap_sessions_cannot_pass_live_eligibility(section):
    report = {
        "policy": {
            "sessions": 30,
            "trades": 100,
            "expectancy": 1,
            "net_profit": 100,
            "profit_factor": "infinite",
            "gross_profit": 100,
            "gross_loss": 0,
            "max_drawdown": 0,
        },
        "stress": {"net_profit": 100},
        "edge": {"intraday_long": {"positive": True, "lower": 0.01}},
    }
    report[section]["feed_gaps"] = 1
    result = eligibility(report, synthetic=False, primary_control="intraday_long")
    assert not result["passed"]
    assert any("runtime" in reason and "feed-gap" in reason for reason in result["reasons"])


def test_rollout_stops_at_unliquidated_session_instead_of_resetting_inventory():
    from dataclasses import replace

    import numpy as np

    from hft.data import synthetic_sessions
    from hft.research import _rollout

    first, second = synthetic_sessions(days=2, bars_per_day=80)
    # Entry can execute, but the final minute has no executable sell depth.
    first = replace(
        first, bid_size=np.where(first.quote_ns >= first.close_ns - 61 * 10**9, 0, first.bid_size)
    )
    result, _ = _rollout([first, second], lambda obs, env: 1)
    assert result["incomplete"]
    assert result["sessions"] == 1
    assert result["dates"] == [first.session_id]
    assert result["unresolved_position"] > 0
    assert result["failure"] == "incomplete-liquidation"


def test_evaluation_reports_incomplete_sessions_without_unpaired_edge():
    from dataclasses import replace

    import numpy as np

    from hft.data import synthetic_sessions
    from hft.research import evaluate

    first, second = synthetic_sessions(days=2, bars_per_day=80)
    first = replace(
        first, bid_size=np.where(first.quote_ns >= first.close_ns - 61 * 10**9, 0, first.bid_size)
    )
    report, _ = evaluate([first, second], lambda obs: 1)
    assert report["policy"]["incomplete"]
    assert report["policy"]["dates"] == [first.session_id]
    assert not report["edge"]["random"]["positive"]


def test_rollout_never_calls_policy_during_gap_warmup():
    from dataclasses import replace

    import numpy as np

    from hft.data import synthetic_sessions
    from hft.research import evaluate

    data = synthetic_sessions(days=1, bars_per_day=160)[0]
    keep_q = (data.quote_ns < data.open_ns + 306 * 10**9) | (
        data.quote_ns >= data.open_ns + 313 * 10**9
    )
    keep_t = (data.trade_ns < data.open_ns + 306 * 10**9) | (
        data.trade_ns >= data.open_ns + 313 * 10**9
    )
    data = replace(
        data,
        **{
            name: getattr(data, name)[keep_q]
            for name in ("quote_ns", "bid", "ask", "bid_size", "ask_size")
        },
        **{name: getattr(data, name)[keep_t] for name in ("trade_ns", "trade_price", "trade_size")},
    )

    def policy(obs):
        assert np.any(obs), "policy received a masked warmup observation"
        return 0

    report, _ = evaluate([data], policy)
    assert report["policy"]["feed_gaps"] == 1
    assert report["policy"]["ineligible_decisions"] > 0

    # A post-gap entry can be matched. Scheduling random starts by callback
    # count would never reach many sampled physical bar offsets after downtime.
    traded, _ = evaluate([data], lambda obs: int(obs[13] <= 0.225))
    assert traded["policy"]["session_fills"] == [2]
    assert traded["random"]["matched_fill_frequency"]


@pytest.mark.parametrize(
    "module,field,value",
    [
        ("hft.features", "FEATURE_VERSION", 4),
        ("hft.data", "AGGREGATION_VERSION", "new-causal-aggregation"),
    ],
)
def test_frozen_identity_changes_with_execution_feature_contract(
    tmp_path, monkeypatch, module, field, value
):
    import importlib
    from types import SimpleNamespace

    from hft.research import freeze_experiment

    source = tmp_path / "dataset.json"
    source.write_text("{}")
    sessions = [
        SimpleNamespace(session_id=f"{i:03d}", symbol="SPY", manifest={"synthetic": False})
        for i in range(120)
    ]
    monkeypatch.setattr("hft.data.load_dataset", lambda path: sessions)
    before = freeze_experiment(source, {}, tmp_path / "before.json")
    monkeypatch.setattr(importlib.import_module(module), field, value)
    after = freeze_experiment(source, {}, tmp_path / "after.json")
    assert before["experiment_hash"] != after["experiment_hash"]
