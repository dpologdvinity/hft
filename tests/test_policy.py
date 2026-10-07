import json
import subprocess
import sys

import numpy as np
import pytest

from hft.policy import OnnxPolicy, export_bundle


def test_runtime_import_does_not_import_training_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hft.policy; assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules",
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_real_observation_onnx_bundle_and_tamper(tmp_path):
    from stable_baselines3 import PPO

    from hft.data import synthetic_sessions
    from hft.env import TradingEnv

    env = TradingEnv(synthetic_sessions(3, 100, 1), episode_steps=64)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, device="cpu", seed=3)
    model.learn(total_timesteps=128)
    while not env.done:
        env.step(0)
    obs, _ = env.reset()
    samples = []
    for _ in range(20):
        samples.append(obs.copy())
        obs, _, done, truncated, _ = env.step(0)
        if done or truncated:
            obs, _ = env.reset()
    bundle = tmp_path / "bundle"
    export_bundle(
        model,
        {"symbol": "SYNTH", "synthetic": True, "config": {"initial_cash": 500}},
        bundle,
        observations=np.asarray(samples),
    )
    runtime = OnnxPolicy(bundle)
    for observation in samples:
        assert runtime.predict(observation) == int(
            model.predict(observation, deterministic=True)[0]
        )
    with pytest.raises(ValueError):
        runtime.predict(samples[0].astype(np.float64))
    manifest = json.loads((bundle / "manifest.json").read_text())
    from copy import deepcopy

    from hft.policy import contract_hash, validate_bundle

    legacy = deepcopy(manifest)
    legacy["metadata"]["feature_version"] = 2
    legacy.pop("contract_hash")
    legacy["contract_hash"] = contract_hash(legacy)
    (bundle / "manifest.json").write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="feature"):
        validate_bundle(bundle)
    manifest["metadata"]["costs"]["slippage_bps"] = 0
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract"):
        OnnxPolicy(bundle)


def test_export_requires_explicit_observations_before_reading_anything(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr("hft.data.load_dataset", lambda *a, **kw: loaded.append(a) or [])
    monkeypatch.setattr("hft.research._rollout", lambda *a, **kw: loaded.append(a) or (0, []))
    # A frozen manifest naming final-test sessions must never be opened by export.
    experiment = {
        "symbol": "SYNTH",
        "dataset_manifest": str(tmp_path / "manifest.json"),
        "final_test": ["2026-01-02"],
    }
    checkpoint = tmp_path / "missing-checkpoint.zip"
    bundle = tmp_path / "bundle"
    with pytest.raises(TypeError, match="observations"):
        export_bundle(checkpoint, experiment, bundle)
    for observations in (None, [], np.zeros((1, 3), dtype=np.float32)):
        with pytest.raises(ValueError, match="observations"):
            export_bundle(checkpoint, experiment, bundle, observations=observations)
    assert loaded == []
    assert not bundle.exists()


@pytest.mark.parametrize(
    "section,values",
    [
        ("risk", {"max_daily_loss": 2}),
        ("sizing", {"allocation_fraction": 1}),
        ("risk", {"unknown_setting": 1}),
        ("sizing", {"unknown_setting": 1}),
    ],
)
def test_semantically_invalid_execution_contract_rejected_even_rehashed(tmp_path, section, values):
    from dataclasses import asdict

    from hft.account import Costs
    from hft.features import FEATURE_NAMES, FEATURE_VERSION
    from hft.policy import contract_hash, digest, validate_bundle
    from hft.risk import RiskConfig
    from hft.sizing import SizingConfig

    (tmp_path / "model.onnx").write_bytes(b"weights")
    metadata = {
        "sha256": digest(tmp_path / "model.onnx"),
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "targets": [0, 1],
        "symbol": "SPY",
        "feed": "iex",
        "costs": asdict(Costs()),
        "risk": asdict(RiskConfig()),
        "sizing": asdict(SizingConfig()),
        "latency_ms": 75,
        "bar_seconds": 5,
        "initial_cash": 500,
    }
    metadata[section].update(values)
    metadata["execution_hash"] = contract_hash(
        {
            k: metadata[k]
            for k in ("costs", "risk", "sizing", "latency_ms", "bar_seconds", "symbol", "feed")
        }
    )
    manifest = {"metadata": metadata, "research": {}}
    manifest["contract_hash"] = contract_hash(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_bundle(tmp_path)
