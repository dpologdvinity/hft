import json

import pytest

from hft.logs import EventLog, read_log


def test_journal_roundtrip_and_corruption(tmp_path):
    path = tmp_path / "run.jsonl"
    log = EventLog(path, clock=lambda: 42)
    log.write("start", source="replay")
    log.write("finish", complete=False)
    log.close()
    assert len(read_log(path)) == 2
    with pytest.raises(FileExistsError):
        EventLog(path)
    rows = path.read_text().splitlines()
    row = json.loads(rows[1])
    row["complete"] = True
    path.write_text(rows[0] + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="integrity"):
        read_log(path)


def test_truncated_journal_rejected(tmp_path):
    path = tmp_path / "run.jsonl"
    log = EventLog(path)
    log.write("start")
    log.close()
    path.write_text(path.read_text().rstrip())
    with pytest.raises(ValueError, match="truncated"):
        read_log(path)
