import json
import zipfile

import pytest

from hft.ml.package import package_data


def test_data_package_holds_bars_and_calendar_only(tmp_path):
    data = tmp_path / "ml"
    (data / "1Min" / "NVDA").mkdir(parents=True)
    (data / "1Min" / "NVDA" / "2020.parquet").write_bytes(b"x")
    (data / "1Min" / "NVDA" / "2020.parquet.tmp").write_bytes(b"partial")
    (data / "calendar.json").write_text(json.dumps([]))
    target = package_data(data, tmp_path / "data.zip")
    names = zipfile.ZipFile(target).namelist()
    assert sorted(names) == [
        "hft-ml-data/1Min/NVDA/2020.parquet",
        "hft-ml-data/calendar.json",
        "hft-ml-data/manifest.json",  # training checks every file against it
    ]


def test_data_without_a_calendar_is_refused(tmp_path):
    with pytest.raises(ValueError, match="calendar"):
        package_data(tmp_path, tmp_path / "data.zip")


def test_changed_data_is_refused_once_it_has_a_manifest(tmp_path):
    from hft.ml.datasets import data_fingerprint, write_data_manifest

    (tmp_path / "1Day" / "SPY").mkdir(parents=True)
    bars = tmp_path / "1Day" / "SPY" / "2020.parquet"
    bars.write_bytes(b"a")
    (tmp_path / "calendar.json").write_text("[]")
    before = data_fingerprint(tmp_path)  # no manifest yet: just a hash
    write_data_manifest(tmp_path)
    assert data_fingerprint(tmp_path) == before
    bars.write_bytes(b"b")
    with pytest.raises(ValueError, match="manifest"):
        data_fingerprint(tmp_path)
