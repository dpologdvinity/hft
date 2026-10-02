import json

from hft.calendar import calendar_from_records, load_calendar, session_at
from hft.feed import parse_timestamp


def test_calendar_holiday_early_close_dst(tmp_path):
    records = [
        {"date": "2025-03-07", "open": "09:30", "close": "16:00"},
        {"date": "2025-03-10", "open": "09:30", "close": "16:00"},
        {"date": "2025-07-03", "open": "09:30", "close": "13:00"},
    ]
    p = tmp_path / "calendar.json"
    p.write_text(json.dumps(records))
    windows = load_calendar(p)
    assert windows[0].open_ns == parse_timestamp("2025-03-07T14:30:00Z")
    assert windows[1].open_ns == parse_timestamp("2025-03-10T13:30:00Z")
    assert windows[2].close_ns == parse_timestamp("2025-07-03T17:00:00Z")
    assert windows[0].next_session_id == "2025-03-10"
    assert session_at(windows, parse_timestamp("2025-07-04T15:00:00Z")) is None
    assert calendar_from_records([]) == []
