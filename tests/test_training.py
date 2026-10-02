import pytest

from hft.training import select_champion, trial_key, validate_config


def test_config_rejects_unknown_and_nonfinite_and_live():
    for config in ({"mode": "live"}, {"timesteps": float("nan")}, {"n_envs": 0}):
        with pytest.raises(ValueError):
            validate_config(config)
    assert validate_config({})["n_envs"] == 4


def test_validation_only_selection_and_hashed_trials():
    trials = [
        {
            "candidate": 0,
            "seed": 42,
            "validation": {"return": 0.1, "max_drawdown": 0.04, "turnover": 2},
        },
        {
            "candidate": 1,
            "seed": 43,
            "validation": {"return": 0.2, "max_drawdown": 0.06, "turnover": 1},
        },
    ]
    assert select_champion(trials) == (0, 42)
    assert trial_key({"data": "a"}, 0, 42, {}) != trial_key({"data": "b"}, 0, 42, {})


def test_real_fit_changes_parameters_and_reload(tmp_path):
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from hft.data import synthetic_sessions
    from hft.env import TradingEnv
    from hft.training import fit_candidate

    sessions = synthetic_sessions(2, 100, 13)
    env = TradingEnv(sessions)
    torch.set_num_threads(2)
    original = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        seed=7,
        device="cpu",
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )
    before = [p.detach().clone() for p in original.policy.parameters()]
    model = fit_candidate(
        sessions,
        {"learning_rate": 3e-4, "entropy_coefficient": 0},
        7,
        {"timesteps": 128, "n_envs": 1},
    )
    assert model.fit_complete and model.num_timesteps == 256
    assert any(
        not torch.equal(a, b) for a, b in zip(before, model.policy.parameters(), strict=True)
    )
    path = tmp_path / "checkpoint"
    model.save(path)
    reloaded = PPO.load(path, device="cpu")
    obs, _ = env.reset()
    assert np.array_equal(
        model.predict(obs, deterministic=True)[0], reloaded.predict(obs, deterministic=True)[0]
    )


def test_interrupted_trials_resume_and_test_is_consumed_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from hft import training

    sessions = [SimpleNamespace(session_id=str(i)) for i in range(5)]
    calls = []
    evaluated = []

    class Model:
        num_timesteps = 256
        peak_rss_bytes = 1

        def __init__(self, complete):
            self.fit_complete = complete

        def predict(self, obs, deterministic=True):
            return np.array(0), None

        def save(self, path):
            from pathlib import Path

            Path(path).write_bytes(b"checkpoint")

    def fit(data, settings, seed, budget):
        calls.append([s.session_id for s in data])
        return Model(len(calls) != 2)

    def evaluate(data, policy, **kwargs):
        evaluated.append([s.session_id for s in data])
        metric = {
            "return": 0.0,
            "turnover": 0.0,
            "max_drawdown": 0.0,
            "net_profit": 0.0,
            "sessions": len(data),
            "trades": 0,
            "expectancy": 0.0,
            "profit_factor": None,
        }
        return {
            "policy": metric,
            "intraday_long": metric,
            "ema_5_20": metric,
            "random": metric,
            "stress": metric,
            "edge": {"intraday_long": {"positive": False}},
        }, np.zeros((1, 17), dtype=np.float32)

    monkeypatch.setattr(training, "fit_candidate", fit)
    monkeypatch.setattr(training, "evaluate", evaluate)
    monkeypatch.setattr("hft.policy.export_bundle", lambda *a, **kw: None)
    experiment = {
        "config": {"timesteps": 128, "seeds": [42], "n_envs": 1},
        "experiment_hash": "frozen",
        "folds": [
            {"train": ["0"], "validation": ["1"]},
            {"train": ["0", "1"], "validation": ["2"]},
        ],
        "development": ["0", "1", "2", "3"],
        "final_test": ["4"],
        "synthetic": True,
        "symbol": "SYNTH",
        "candidates": 1,
    }
    first = training._search(sessions, experiment, tmp_path, smoke=True)
    assert first["status"] == "incomplete-search" and len(first["trials"]) == 1
    second = training._search(sessions, experiment, tmp_path, resume=True, smoke=True)
    assert second["status"] == "research-only"
    assert calls.count(["0"]) == 1
    assert evaluated.count(["4"]) == 1
    training._search(sessions, experiment, tmp_path, resume=True, smoke=True)
    assert evaluated.count(["4"]) == 1
    checkpoint = next(tmp_path.glob("*.zip"))
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity"):
        training._search(sessions, experiment, tmp_path, resume=True, smoke=True)
