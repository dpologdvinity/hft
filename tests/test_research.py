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
