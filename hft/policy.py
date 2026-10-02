"""Deterministic portable policies with a checked observation and execution contract."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .account import Costs
from .features import FEATURE_NAMES, FEATURE_VERSION, TARGETS


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_model(model, path, *, symbol, costs, synthetic, bar_seconds, initial_cash=10_000):
    import onnx
    import onnxruntime as ort
    import torch

    class Actor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            return self.policy(obs, deterministic=True)[0]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    actor = Actor(model.policy.cpu().eval())
    samples = np.random.default_rng(4).normal(size=(32, len(FEATURE_NAMES))).astype(np.float32)
    samples[0] = 0
    samples[1, 13] = 1
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
    exported = session.run(["action"], {"observation": samples})[0].reshape(-1)
    expected, _ = model.predict(samples, deterministic=True)
    if not np.array_equal(exported, expected.reshape(-1)):
        raise ValueError("ONNX action parity failed")
    metadata = {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "targets": list(TARGETS),
        "symbol": symbol,
        "costs": asdict(costs),
        "synthetic_training": bool(synthetic),
        "bar_seconds": bar_seconds,
        "initial_cash": initial_cash,
        "sha256": digest(path),
        "parity_samples": len(samples),
        "created_at": time.time(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


class OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort

        path = Path(path)
        self.metadata = json.loads(path.with_suffix(".json").read_text())
        m = self.metadata
        if (
            m.get("feature_version") != FEATURE_VERSION
            or m.get("feature_names") != list(FEATURE_NAMES)
            or m.get("targets") != list(TARGETS)
        ):
            raise ValueError("incompatible model feature/action contract")
        if m.get("sha256") != digest(path):
            raise ValueError("model digest does not match metadata")
        Costs(**m["costs"])
        if m.get("bar_seconds") not in (1, 5) or not m.get("symbol"):
            raise ValueError("invalid model market contract")
        if not np.isfinite(m["initial_cash"]) or m["initial_cash"] <= 0:
            raise ValueError("invalid model initial cash")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "observation"
            or inputs[0].type != "tensor(float)"
            or inputs[0].shape[1:] != [len(FEATURE_NAMES)]
        ):
            raise ValueError("incompatible ONNX input")

    def __call__(self, obs) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (len(FEATURE_NAMES),) or not np.isfinite(obs).all():
            raise ValueError("invalid inference observation")
        output = self.session.run(["action"], {"observation": obs[None]})[0]
        if (
            output.size != 1
            or output.dtype.kind not in "iu"
            or int(output.flat[0]) not in (0, 1, 2)
        ):
            raise ValueError("invalid policy action")
        return int(output.flat[0])
