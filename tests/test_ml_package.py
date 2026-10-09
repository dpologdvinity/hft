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
    assert sorted(names) == ["hft-ml-data/1Min/NVDA/2020.parquet", "hft-ml-data/calendar.json"]


def test_data_without_a_calendar_is_refused(tmp_path):
    with pytest.raises(ValueError, match="calendar"):
        package_data(tmp_path, tmp_path / "data.zip")
