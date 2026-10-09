"""Build the two Kaggle uploads: the downloaded market data and this code.

    .venv/bin/python -m hft.ml.package --data data/ml --out dist/kaggle

Writes `hft-ml-data.zip` (daily and minute bars plus the exchange calendar) and
`hft-ml-code.zip` (the committed source tree from `git archive HEAD`), each with a
SHA-256 in `uploads.json`. Upload both as private Kaggle datasets, attach them to
`notebooks/kaggle_train.ipynb`, turn on a GPU and run all cells.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_data(data, target) -> Path:
    data = Path(data)
    if not (data / "calendar.json").exists():
        raise ValueError(f"{data} has no calendar.json; run hft.ml.download first")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:  # Parquet is compressed
        for path in sorted(data.rglob("*")):
            if path.is_file() and path.suffix in (".parquet", ".json"):
                archive.write(path, Path("hft-ml-data") / path.relative_to(data))
    return target


def package_code(target) -> Path:
    subprocess.run(
        ["git", "archive", "--format=zip", "--prefix=hft-ml-code/", "-o", str(target), "HEAD"],
        cwd=ROOT,
        check=True,
    )
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ml")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "kaggle")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    built = [
        package_data(args.data, args.out / "hft-ml-data.zip"),
        package_code(args.out / "hft-ml-code.zip"),
    ]
    manifest = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in built}
    (args.out / "uploads.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
