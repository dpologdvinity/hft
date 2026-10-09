"""Train, validation and test splits, and fingerprinted datasets for training runs.

The test split is evaluated once, after the variant choice is frozen. A dataset is
saved as `arrays.npz` plus `manifest.json` holding the SHA-256 of the arrays file;
loading refuses any copy (for example on Kaggle) whose fingerprint differs.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from .features import DailyFrame

SPLITS = {
    "train": (date(2016, 1, 1), date(2022, 12, 31)),
    "validation": (date(2023, 1, 1), date(2024, 12, 31)),
    "test": (date(2025, 1, 1), date(2099, 12, 31)),
}


def split_mask(dates, name) -> np.ndarray:
    first, last = SPLITS[name]
    dates = np.asarray(dates).astype("datetime64[D]")
    return (dates >= np.datetime64(first)) & (dates <= np.datetime64(last))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(frame: DailyFrame, path, *, feature_names) -> dict:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = path / "arrays.npz"
    np.savez(
        arrays,
        dates=frame.dates.astype("datetime64[D]").astype("int64"),
        symbols=frame.symbols.astype(str),
        x=frame.x,
        label=frame.label,
        valid=frame.valid,
    )
    manifest = {
        "features": list(feature_names),
        "symbols": [str(s) for s in frame.symbols],
        "first_date": str(frame.dates[0]),
        "last_date": str(frame.dates[-1]),
        "splits": {k: [str(a), str(b)] for k, (a, b) in SPLITS.items()},
        "sha256": _sha256(arrays),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_dataset(path) -> DailyFrame:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if _sha256(path / "arrays.npz") != manifest["sha256"]:
        raise ValueError("dataset fingerprint does not match its manifest")
    data = np.load(path / "arrays.npz")
    return DailyFrame(
        data["dates"].astype("datetime64[D]"),
        data["symbols"],
        data["x"],
        data["label"],
        data["valid"],
    )


def data_manifest(root) -> dict:
    """SHA-256 of every bar file and the calendar under a download root."""
    root = Path(root)
    files = sorted([*root.rglob("*.parquet"), root / "calendar.json"])
    return {str(f.relative_to(root)): _sha256(f) for f in files if f.exists()}


def write_data_manifest(root) -> dict:
    manifest = data_manifest(root)
    (Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def data_fingerprint(root) -> str:
    """One hash identifying the data. With a saved manifest (written when the data was
    packaged), any missing, extra or changed file is refused."""
    current = data_manifest(root)
    saved_path = Path(root) / "manifest.json"
    if saved_path.exists() and json.loads(saved_path.read_text()) != current:
        raise ValueError("market data does not match its manifest")
    return hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
