"""Exchange sessions from provider records; never infer holidays from weekdays."""

import itertools
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NS = 1_000_000_000
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class SessionWindow:
    session_id: str
    open_ns: int
    close_ns: int
    next_session_id: str | None = None

    def __post_init__(self):
        if not self.session_id or self.open_ns <= 0 or self.close_ns <= self.open_ns:
            raise ValueError("invalid session window")

    def contains(self, timestamp_ns: int) -> bool:
        return self.open_ns <= timestamp_ns < self.close_ns


def calendar_from_records(records: list[dict]) -> list[SessionWindow]:
    windows = []
    for record in records:
        if "open_ns" in record:
            windows.append(SessionWindow(**record))
            continue
        day = record["date"]
        stamps = []
        for key in ("open", "close"):
            dt = datetime.fromisoformat(f"{day}T{record[key]}").replace(tzinfo=NY)
            stamps.append(int(dt.timestamp()) * NS)
        windows.append(SessionWindow(day, *stamps))
    if any(b.open_ns <= a.close_ns for a, b in itertools.pairwise(windows)):
        raise ValueError("calendar must be ordered without overlap")
    return [
        SessionWindow(
            w.session_id,
            w.open_ns,
            w.close_ns,
            windows[i + 1].session_id if i + 1 < len(windows) else w.next_session_id,
        )
        for i, w in enumerate(windows)
    ]


def load_calendar(path: Path) -> list[SessionWindow]:
    value = json.loads(Path(path).read_text())
    return calendar_from_records(value if isinstance(value, list) else value["calendar"])


def session_at(windows: list[SessionWindow], timestamp_ns: int) -> SessionWindow | None:
    return next((w for w in windows if w.contains(timestamp_ns)), None)
