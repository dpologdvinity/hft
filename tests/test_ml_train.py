import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet as pq
import pytest

from hft.calendar import SessionWindow
from hft.ml.download import SCHEMA, save_calendar
from hft.ml.evaluate import deflated_sharpe, report
from hft.ml.train import run_variant, variant_id

NS = 1_000_000_000


def _days():
    days, d = [], date(2022, 7, 1)
    while d <= date(2023, 6, 30):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _write(root, symbol, frame, timeframe, year):
    folder = root / timeframe / symbol
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(frame, schema=SCHEMA), folder / f"{year}.parquet")


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml")
    days = _days()
    opens = [
        int(datetime(d.year, d.month, d.day, 13, 30, tzinfo=UTC).timestamp()) * NS for d in days
    ]
    save_calendar(
        root, [SessionWindow(d.isoformat(), o, o + 390 * 60 * NS) for d, o in zip(days, opens)]
    )
    rng = np.random.default_rng(0)
    for symbol in ("SPY", "QQQ", "AAA", "BBB"):
        minutes = 100 * np.exp(np.cumsum(rng.normal(0, 4e-4, (len(days), 390)), axis=None)).reshape(
            len(days), 390
        )
        t = (np.array(opens)[:, None] + np.arange(390) * 60 * NS).ravel()
        vol = rng.integers(100, 1000, minutes.size).astype(float)
        minute = {
            "t": t,
            "o": minutes.ravel(),
            "h": minutes.ravel(),
            "l": minutes.ravel(),
            "c": minutes.ravel(),
            "v": vol,
            "vw": minutes.ravel(),
            "n": np.ones(minutes.size, np.int64),
        }
        daily_t = np.array(
            [int(datetime(d.year, d.month, d.day, 4, tzinfo=UTC).timestamp()) * NS for d in days]
        )
        daily = {
            "t": daily_t,
            "o": minutes[:, 0],
            "h": minutes.max(1),
            "l": minutes.min(1),
            "c": minutes[:, -1],
            "v": vol.reshape(len(days), 390).sum(1),
            "vw": minutes.mean(1),
            "n": np.ones(len(days), np.int64),
        }
        for year in (2022, 2023):
            m = np.array([d.year == year for d in days])
            _write(root, symbol, {k: v[np.repeat(m, 390)] for k, v in minute.items()}, "1Min", year)
            _write(root, symbol, {k: v[m] for k, v in daily.items()}, "1Day", year)
    return root


CONFIG = {
    "symbols": ["AAA", "BBB"],
    "daily": {"k": 2, "gate": False},
    "minute": {"model": "cnn", "epochs": 1, "per_day": 2, "width": 8},
}


def test_a_tiny_variant_trains_and_scores_against_holding(market, tmp_path):
    metrics = run_variant(CONFIG, market, tmp_path)
    assert metrics["split"] == "validation" and metrics["first_day"].startswith("2023-01")
    assert metrics["strategy"]["days"] == metrics["holding"]["days"] > 100
    folder = tmp_path / metrics["variant"]
    assert {p.name for p in folder.iterdir()} >= {
        "config.json",
        "metrics-validation.json",
        "daily.txt",
        "minute.pt",
    }


def test_holding_is_an_equal_weight_buy_at_the_first_open(market, tmp_path):
    metrics = run_variant({**CONFIG, "minute": {"model": "none"}}, market, tmp_path)
    total = metrics["holding"]["total"]
    days = [d for d in _days() if d.year == 2023]
    import pyarrow.parquet as pq2

    values = []
    for symbol in ("AAA", "BBB"):
        table = pq2.read_table(market / "1Day" / symbol / "2023.parquet")
        values.append(table.column("c").to_numpy()[-1] / table.column("o").to_numpy()[0])
    assert len(days) == metrics["holding"]["days"]
    assert total == pytest.approx(np.mean(values) - 1)


def test_variant_ids_ignore_where_a_variant_runs():
    assert variant_id({**CONFIG, "device": "cuda", "threads": 4}) == variant_id(CONFIG)


def test_the_test_split_cannot_be_scored_directly(market, tmp_path):
    with pytest.raises(PermissionError):
        run_variant(CONFIG, market, tmp_path, split="test")


def test_report_and_deflated_sharpe():
    rng = np.random.default_rng(0)
    strategy = rng.normal(0.002, 0.01, 500)
    holding = rng.normal(0.0, 0.01, 500)
    out = report(strategy, holding, strategy, trials=1)
    assert out["beats_holding"] and out["daily_difference_ci95"][0] > 0
    assert not report(holding, strategy, holding)["beats_holding"]
    assert deflated_sharpe(1.0, 1, 500) > deflated_sharpe(1.0, 100, 500)


def test_a_search_logs_every_variant_and_freezes_the_best(market, tmp_path):
    from hft.ml.search import grid, run_search

    configs = [
        c
        for c in grid()
        if c["daily"]["model"] == "lgbm"
        and c["daily"]["target"] == "return"
        and c["minute"]["model"] in ("none", "cnn")
    ]
    configs = [
        {
            **c,
            "daily": {**c["daily"], "gate": False},
            "minute": {**c["minute"], "epochs": 1, "per_day": 2, "width": 8},
        }
        for c in configs[:3]
    ]
    summary = run_search(configs, market, tmp_path, symbols=["AAA", "BBB"], log=lambda _: None)
    rows = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and summary["trials"] == 3
    assert all(r["trials"] == 3 for r in rows)
    assert json.loads((tmp_path / "frozen.json").read_text()) == summary["frozen"]
    assert set(summary["frozen"]) <= {r["variant"] for r in rows}
    assert len(grid()) == 84


def test_selection_ranks_and_the_optional_gate_requires_beating_costs():
    from hft.ml.train import _select

    prediction = np.array([[0.0010, 0.0002, 0.0007]])  # 10, 2 and 7 bp
    valid = np.ones((1, 3), bool)
    costs = np.array([1.5, 1.5, 1.5])  # round trip 5 bp
    trainable = np.array([True, True, True])
    assert _select(prediction, valid, costs, 2, trainable) == [[0, 2]]
    assert _select(prediction, valid, costs, 3, trainable, gate=True) == [[0, 2]]
    assert _select(prediction, valid, costs, 3, trainable, gate=True, threshold_bp=4) == [[0]]


def test_a_range_target_picks_movers_and_trades(market, tmp_path):
    config = {**CONFIG, "daily": {"k": 2, "target": "range"}, "minute": {"model": "none"}}
    metrics = run_variant(config, market, tmp_path)
    assert metrics["trades"] > 100 and metrics["days_with_trades"] > 100


def test_holding_buys_late_listings_at_their_own_first_open(market, tmp_path):
    import shutil

    late = tmp_path / "late"
    shutil.copytree(market, late)
    for timeframe in ("1Day", "1Min"):
        path = late / timeframe / "BBB" / "2023.parquet"
        table = pq.read_table(path)
        cut = int(datetime(2023, 3, 1, tzinfo=UTC).timestamp()) * NS
        pq.write_table(table.filter(pa.compute.greater_equal(table.column("t"), cut)), path)
        (late / timeframe / "BBB" / "2022.parquet").unlink()
    metrics = run_variant({**CONFIG, "minute": {"model": "none"}}, late, tmp_path / "out")
    values = []
    for symbol, first in (("AAA", 0), ("BBB", 1)):  # a listing is tradable from its 2nd day
        table = pq.read_table(late / "1Day" / symbol / "2023.parquet")
        values.append(table.column("c").to_numpy()[-1] / table.column("o").to_numpy()[first])
    assert metrics["holding"]["total"] == pytest.approx(np.mean(values) - 1)


def test_results_are_also_reported_at_three_times_the_costs(market, tmp_path):
    metrics = run_variant({**CONFIG, "minute": {"model": "none"}}, market, tmp_path)
    triple = metrics["at_3x_cost"]
    assert triple["strategy"]["total"] < metrics["strategy"]["total"]
    assert triple["holding"] == metrics["holding"]


def test_the_final_test_runs_once_on_the_saved_frozen_models(market, tmp_path, monkeypatch):
    from hft.ml import datasets, train
    from hft.ml.search import run_final, run_search

    monkeypatch.setitem(datasets.SPLITS, "validation", (date(2023, 1, 1), date(2023, 3, 31)))
    monkeypatch.setitem(datasets.SPLITS, "test", (date(2023, 4, 1), date(2099, 12, 31)))
    configs = [{**CONFIG, "minute": {"model": "none"}}, CONFIG]
    run_search(configs, market, tmp_path / "search", symbols=["AAA", "BBB"], log=lambda _: None)

    def no_training(*args, **kwargs):
        raise AssertionError("frozen models must not be retrained")

    monkeypatch.setattr(train, "_fit_daily", no_training)
    monkeypatch.setattr(train, "_fit_minute", no_training)
    registry = tmp_path / "registry.json"
    final = run_final(
        tmp_path / "search", market, tmp_path / "final", registry=registry, log=lambda _: None
    )
    assert len(final["results"]) == 2 and final["confidence"] == 97.5
    assert all(r["split"] == "test" and r["first_day"] >= "2023-04" for r in final["results"])
    assert json.loads(registry.read_text())["frozen"] == final["frozen"]
    with pytest.raises(RuntimeError, match="already run"):
        run_final(tmp_path / "search", market, tmp_path / "final", registry=registry)
    frozen = final["frozen"][0]
    weights = next((tmp_path / "search" / frozen).glob("daily*"))
    weights.write_bytes(weights.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        run_final(tmp_path / "search", market, tmp_path / "final2", registry=tmp_path / "r2.json")
