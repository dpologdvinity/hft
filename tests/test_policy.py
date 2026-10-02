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
    manifest["metadata"]["costs"]["slippage_bps"] = 0
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract"):
        OnnxPolicy(bundle)
