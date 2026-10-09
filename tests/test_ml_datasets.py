import json
from datetime import date

import numpy as np
import pytest

from hft.ml.datasets import SPLITS, load_dataset, split_mask, write_dataset
from hft.ml.features import DailyFrame


def _frame():
    dates = np.arange(np.datetime64("2022-12-28"), np.datetime64("2023-01-06")).astype(
        "datetime64[D]"
    )
    rng = np.random.default_rng(0)
    x = rng.normal(size=(len(dates), 2, 3)).astype(np.float32)
    label = rng.normal(size=(len(dates), 2)).astype(np.float32)
    return DailyFrame(dates, np.array(["NVDA", "AAPL"]), x, label, np.ones((len(dates), 2), bool))


def test_splits_are_disjoint_and_follow_the_spec():
    assert SPLITS["train"] == (date(2016, 1, 1), date(2022, 12, 31))
    assert SPLITS["validation"] == (date(2023, 1, 1), date(2024, 12, 31))
    assert SPLITS["test"][0] == date(2025, 1, 1)
    dates = np.arange(np.datetime64("2016-01-01"), np.datetime64("2026-10-09")).astype(
        "datetime64[D]"
    )
    masks = [split_mask(dates, name) for name in SPLITS]
    assert (sum(m.astype(int) for m in masks) == 1).all()


def test_labels_stay_inside_their_split():
    # A daily label is that session's own open-to-close move, so the last training day's
    # label never uses a validation-day price.
    frame = _frame()
    train = split_mask(frame.dates, "train")
    assert frame.dates[train].max() == np.datetime64("2022-12-31")


def test_a_tampered_dataset_is_refused(tmp_path):
    manifest = write_dataset(_frame(), tmp_path / "daily", feature_names=("a", "b", "c"))
    assert len(manifest["sha256"]) == 64 and manifest["symbols"] == ["NVDA", "AAPL"]
    loaded = load_dataset(tmp_path / "daily")
    np.testing.assert_array_equal(loaded.x, _frame().x)
    data = dict(np.load(tmp_path / "daily" / "arrays.npz"))
    data["label"][0, 0] += 1
    np.savez(tmp_path / "daily" / "arrays.npz", **data)
    with pytest.raises(ValueError, match="fingerprint"):
        load_dataset(tmp_path / "daily")
    assert json.loads((tmp_path / "daily" / "manifest.json").read_text())["features"] == [
        "a",
        "b",
        "c",
    ]
