"""Atomic ONNX bundles; importing this runtime requires no Torch or SB3."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .account import Costs
from .features import FEATURE_NAMES, FEATURE_VERSION, TARGETS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _export(model, path, samples):
    import onnx
    import onnxruntime as ort
    import torch

    class Actor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            return self.policy(obs, deterministic=True)[0]

    actor = Actor(model.policy.cpu().eval())
    # Legacy exporter is supported by the pinned Torch/SB3 versions. Its
    # deprecation warnings remain visible; runtime parity is mandatory.
    torch.onnx.export(
        actor,
        torch.from_numpy(samples[:1]),
        str(path),
        opset_version=17,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["action"], {"observation": samples})[0].reshape(-1)
    expected, _ = model.predict(samples, deterministic=True)
    if not np.array_equal(actual, expected.reshape(-1)):
        raise ValueError("ONNX action parity failed")


def export_bundle(checkpoint, experiment, output, *, observations=None):
    from stable_baselines3 import PPO

    from .risk import RiskConfig
    from .sizing import SizingConfig

    checkpoint_id = (
        digest(checkpoint)
        if isinstance(checkpoint, (str, Path))
        else experiment.get("selected_checkpoint_id")
        if isinstance(experiment, dict)
        else None
    )
    if isinstance(checkpoint, (str, Path)):
        model = PPO.load(str(checkpoint), device="cpu")
    else:
        model = checkpoint
    if isinstance(experiment, (str, Path)):
        experiment = json.loads(Path(experiment).read_text())
    if model.action_space.n != 2:
        raise ValueError("incompatible two-action model contract")
    if observations is None:
        from .data import load_dataset
        from .research import _rollout

        if not experiment.get("dataset_manifest") or not experiment.get("final_test"):
            raise ValueError("export needs held-out actual observations or frozen dataset manifest")
        held_out = set(experiment["final_test"])
        sessions = [
            s for s in load_dataset(experiment["dataset_manifest"]) if s.session_id in held_out
        ]
        _, observations = _rollout(
            sessions,
            lambda obs, env: int(model.predict(obs, deterministic=True)[0]),
            initial_cash=experiment.get("config", {}).get("initial_cash", 500),
        )
    samples = np.asarray(observations)
    if (
        samples.dtype != np.float32
        or samples.ndim != 2
        or samples.shape[1] != len(FEATURE_NAMES)
        or not len(samples)
        or not np.isfinite(samples).all()
    ):
        raise ValueError("export requires actual finite float32 observations")
    real_count = len(samples)
    # State boundaries and finite fuzz augment actual held-out observations.
    fuzz = np.random.default_rng(4).normal(size=(32, len(FEATURE_NAMES))).astype(np.float32)
    boundary = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    boundary[1, 12] = 1
    boundary[2, 10] = 0.15
    samples = np.concatenate([samples, boundary, fuzz])
    output = Path(output)
    if output.exists():
        raise FileExistsError("immutable bundle already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bundle-", dir=output.parent))
    try:
        path = temporary / "model.onnx"
        _export(model, path, samples)
        metadata = {
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "targets": list(TARGETS),
            "symbol": experiment["symbol"],
            "feed": experiment.get("feed", "iex"),
            "costs": experiment.get("costs", asdict(Costs())),
            "risk": experiment.get("risk", asdict(RiskConfig())),
            "sizing": experiment.get("sizing", asdict(SizingConfig())),
            "latency_ms": 75,
            "bar_seconds": 5,
            "initial_cash": experiment.get("config", {}).get("initial_cash", 500),
            "sha256": digest(path),
            "synthetic_training": experiment.get("synthetic", False),
            "real_executable_data": experiment.get("real_executable_data", False),
            "dataset_hash": experiment.get("dataset_hash"),
            "selected_checkpoint_id": checkpoint_id,
            "session_hashes": experiment.get("session_hashes", {}),
            "research_hash": contract_hash(experiment.get("research", {})),
            "experiment_hash": experiment.get("experiment_hash"),
            "training_sessions": experiment.get("development", []),
            "testing_sessions": experiment.get("final_test", []),
            "parity_samples": len(samples),
            "real_observation_parity_samples": real_count,
            "package_versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "stable-baselines3", "onnx", "onnxruntime")
            },
            "exporter": "torch.onnx.export(dynamo=False, opset=17)",
            "created_at": time.time(),
        }
        metadata["execution_hash"] = contract_hash(
            {
                k: metadata[k]
                for k in ("costs", "risk", "sizing", "latency_ms", "bar_seconds", "symbol", "feed")
            }
        )
        manifest = {"metadata": metadata, "research": experiment.get("research", {})}
        manifest["contract_hash"] = contract_hash(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        runtime = OnnxPolicy(temporary)
        durations = []
        for obs in samples[:128]:
            start = time.perf_counter_ns()
            runtime.predict(obs)
            durations.append((time.perf_counter_ns() - start) / 1e6)
        metadata["inference_ms"] = {
            str(p): float(np.percentile(durations, p)) for p in (50, 95, 99)
        }
        manifest["contract_hash"] = contract_hash(
            {k: v for k, v in manifest.items() if k != "contract_hash"}
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        with (temporary / "manifest.json").open("rb") as f:
            os.fsync(f.fileno())
        with path.open("rb") as f:
            os.fsync(f.fileno())
        os.rename(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_bundle(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("contract_hash") != contract_hash(
        {k: v for k, v in manifest.items() if k != "contract_hash"}
    ):
        raise ValueError("bundle contract digest mismatch")
    metadata = manifest["metadata"]
    if metadata.get("sha256") != digest(path / "model.onnx"):
        raise ValueError("model digest mismatch")
    if (
        metadata.get("feature_version") != FEATURE_VERSION
        or metadata.get("feature_names") != list(FEATURE_NAMES)
        or metadata.get("targets") != [0, 1]
    ):
        raise ValueError("incompatible feature/action contract")
    if metadata.get("execution_hash") != contract_hash(
        {
            k: metadata[k]
            for k in ("costs", "risk", "sizing", "latency_ms", "bar_seconds", "symbol", "feed")
        }
    ):
        raise ValueError("execution contract mismatch")
    from .risk import RiskConfig
    from .sizing import SizingConfig

    try:
        Costs(**metadata["costs"])
        RiskConfig(**metadata["risk"])
        SizingConfig(**metadata["sizing"])
    except TypeError as error:
        raise ValueError("unknown or invalid execution contract fields") from error
    if (
        not isinstance(metadata.get("latency_ms"), (int, float))
        or not np.isfinite(metadata["latency_ms"])
        or metadata["latency_ms"] < 0
    ):
        raise ValueError("invalid execution latency")
    if (
        metadata.get("bar_seconds") != 5
        or not metadata.get("symbol")
        or metadata.get("feed") not in ("iex", "synthetic")
    ):
        raise ValueError("invalid market contract")
    if not np.isfinite(metadata["initial_cash"]) or metadata["initial_cash"] <= 0:
        raise ValueError("invalid strategy capital")
    return manifest


class OnnxPolicy:
    def __init__(self, path, *, symbol=None, feed=None, execution_hash=None):
        import onnxruntime as ort

        path = Path(path)
        self.manifest = validate_bundle(path)
        self.metadata = self.manifest["metadata"]
        for key, value in (("symbol", symbol), ("feed", feed), ("execution_hash", execution_hash)):
            if value is not None and self.metadata[key] != value:
                raise ValueError(f"incompatible {key} contract")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "observation"
            or inputs[0].type != "tensor(float)"
            or inputs[0].shape[1:] != [len(FEATURE_NAMES)]
        ):
            raise ValueError("incompatible ONNX input")

    def predict(self, obs):
        obs = np.asarray(obs)
        if (
            obs.dtype != np.float32
            or obs.shape != (len(FEATURE_NAMES),)
            or not np.isfinite(obs).all()
        ):
            raise ValueError("invalid inference observation")
        output = self.session.run(["action"], {"observation": obs[None]})[0]
        if output.size != 1 or output.dtype.kind not in "iu" or int(output.flat[0]) not in (0, 1):
            raise ValueError("invalid policy action")
        return int(output.flat[0])

    __call__ = predict
