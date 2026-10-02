import json

import pytest

from hft.account import Costs
from hft.data import synthetic_data
from hft.features import FEATURE_NAMES
from hft.policy import OnnxPolicy, export_model
from hft.research import evaluate, metrics, paired_bootstrap, walk_forward


def test_walk_forward_is_temporal_and_test_context_has_no_future():
    data = synthetic_data(days=8, bars_per_day=100)
    folds = walk_forward(data, train_days=3, test_days=2)
    assert [(s.train_stop, s.test_start, s.test_stop) for s in folds] == [
        (300, 300, 500),
        (500, 500, 700),
    ]
    for split in folds:
        assert data.rows[split.train_stop - 1].timestamp < data.rows[split.test_start].timestamp
        assert split.context_start == split.test_start - 61
    with pytest.raises(ValueError):
        walk_forward(data, 0, 2)


def test_metrics_use_daily_returns_and_closed_net_trades():
    result = metrics([1000, 1010, 1005, 1020], [5, -2, 3])
    assert result["net_profit"] == 20
    assert result["max_drawdown"] == pytest.approx(5 / 1010)
    assert result["profit_factor"] == 4
    assert result["expectancy"] == 2
    assert result["trades"] == 3
    assert result["sharpe"] > 0
    assert metrics([1000, 1000], [])["profit_factor"] is None
    assert metrics([1000, 1001], [1])["profit_factor"] is None  # no losses, JSON safe


def test_evaluation_costs_and_matched_random_change_frequency():
    data = synthetic_data(days=3, bars_per_day=100)
    result = evaluate(data, lambda obs: 1, costs=Costs(0.01, 0), seed=1)
    assert set(result) == {"policy", "buy_hold", "random", "candlestick", "edge"}
    assert result["policy"]["fills"] == 6
    assert result["policy"]["target_changes"] == 1  # model target, excluding daily flatten
    assert result["random"]["target_changes"] == 1
    assert len(result["policy"]["daily_equity"]) == 3
    assert result["policy"]["trades"] == 3
    assert result["edge"]["random"]["samples"] == 3


def test_paired_bootstrap_detects_constant_edge_reproducibly():
    first = paired_bootstrap([0.02] * 40, [0.01] * 40, seed=4)
    assert first["lower"] == pytest.approx(0.01)
    assert first["positive"] is True
    assert first == paired_bootstrap([0.02] * 40, [0.01] * 40, seed=4)
    assert not paired_bootstrap([0.01], [0])["positive"]  # insufficient evidence


def test_real_ppo_export_runtime_parity_and_metadata_validation(tmp_path):
    from stable_baselines3 import PPO

    from hft.env import TradingEnv

    env = TradingEnv(synthetic_data(days=3, bars_per_day=100), episode_steps=64)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=3, device="cpu", verbose=0)
    model.learn(total_timesteps=128)
    path = tmp_path / "policy.onnx"
    export_model(model, path, symbol="SYNTH", costs=Costs(), synthetic=True, bar_seconds=1)
    runtime = OnnxPolicy(path)
    obs, _ = env.reset()
    assert runtime(obs) == int(model.predict(obs, deterministic=True)[0])
    assert runtime.metadata["feature_names"] == list(FEATURE_NAMES)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["feature_version"] = -1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        OnnxPolicy(path)
