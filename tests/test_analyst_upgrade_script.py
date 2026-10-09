import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyst_upgrade.py"


def _module():
    spec = importlib.util.spec_from_file_location("analyst_upgrade", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_intervals_resample_whole_dates():
    module = _module()
    rng = np.random.default_rng(0)
    dates = ["d1"] * 50 + ["d2"] * 50  # two dates only: trades on a date move together
    values = [0.01] * 50 + [-0.01] * 50
    low, high = module.clustered_interval(dates, values, rng)
    assert low < 0 < high  # resampling dates, not trades, keeps the uncertainty
    out = module.summary(["a", "b", "c"], [0.002, 0.003, 0.001], rng)
    assert out["trades"] == 3 and out["mean_bp"] == 20.0 and out["passes"]
