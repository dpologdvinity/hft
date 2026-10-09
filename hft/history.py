"""Free IEX history: allowlisted GET-only transport and resumable raw partitions."""

import json
import os
import re
import shutil
import time
from collections import deque
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pyarrow as pa
import pyarrow.parquet as pq

from .calendar import calendar_from_records
from .data import AGGREGATION_VERSION, NS, atomic_json, checksum
from .feed import CONDITION_SOURCE, parse_timestamp, quote_from_event

DATA_BASE = "https://data.alpaca.markets"
CALENDAR_URL = "https://paper-api.alpaca.markets/v2/calendar"
MIN_FREE_BYTES = 5 * 1024**3
ESTIMATED_SESSION_BYTES = 100 * 1024**2


class HistoryError(RuntimeError):
    pass


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HistoryError("history redirect rejected by endpoint allowlist")


def _transport(url, params, headers):
    request = Request(url + "?" + urlencode(params), headers=headers, method="GET")
    with build_opener(_RejectRedirect()).open(request, timeout=30) as response:
        return json.load(response)


class ReadOnlyClient:
    """No mutable endpoint or arbitrary host can reach the HTTP transport."""

    def __init__(self, key=None, secret=None, *, transport=None, clock=None, sleep=None):
        key = key or os.environ.get("ALPACA_API_KEY")
        secret = secret or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise HistoryError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._transport = transport or _transport
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._requests = deque()

    def _budget(self):
        now = self._clock()
        while self._requests and now - self._requests[0] >= 60:
            self._requests.popleft()
        if len(self._requests) >= 150:
            self._sleep(max(0, 60 - (now - self._requests[0])))
            now = self._clock()
            while self._requests and now - self._requests[0] >= 60:
                self._requests.popleft()
        self._requests.append(now)

    def get(self, url, params):
        parsed = urlsplit(url)
        allowed = url == CALENDAR_URL or (
            parsed.scheme == "https"
            and parsed.netloc == "data.alpaca.markets"
            and (
                re.fullmatch(r"/v2/stocks/[A-Z0-9.-]+/(quotes|trades|bars)(/latest)?", parsed.path)
                or parsed.path == "/v2/stocks/trades/latest"
                or parsed.path == "/v1beta1/news"
            )
            and not parsed.query
            and not parsed.fragment
        )
        if not allowed:
            raise HistoryError("read-only endpoint allowlist rejected request")
        for attempt in range(4):
            self._budget()
            try:
                return self._transport(url, params, self._headers)
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise HistoryError(f"history HTTP {error.code}; access unavailable") from None
                retry = error.headers.get("Retry-After", "") if error.headers else ""
                try:
                    delay = min(60.0, max(0.0, float(retry)))
                except ValueError:
                    delay = 2.0**attempt
                self._sleep(delay)
            except (URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise HistoryError("history GET failed after bounded retries") from None
                self._sleep(2.0**attempt)
        raise HistoryError("unreachable GET failure")


def _symbol(value):
    if not re.fullmatch(r"[A-Z0-9.-]+", value):
        raise ValueError("invalid stock symbol")
    return value


def _utc(ns):
    return datetime.fromtimestamp(ns // NS, UTC).isoformat().replace("+00:00", "Z")


def _calendar(client, start, end):
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("start must precede end")
    # Include the next real session in the calendar, rather than guessing weekdays.
    future = (date.fromisoformat(end) + timedelta(days=14)).isoformat()
    records = client.get(CALENDAR_URL, {"start": start, "end": future})
    windows = calendar_from_records(records)
    return windows, [w for w in windows if start <= w.session_id <= end]


def _pages(client, symbol, window, kind):
    params = {
        "start": _utc(window.open_ns),
        "end": _utc(window.close_ns),
        "feed": "iex",
        "sort": "asc",
        "limit": 10000,
    }
    seen = set()
    while True:
        payload = client.get(f"{DATA_BASE}/v2/stocks/{symbol}/{kind}", params)
        if not isinstance(payload, dict) or kind not in payload:
            raise HistoryError("invalid history response")
        rows = payload[kind]
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            raise HistoryError("invalid history rows")
        yield rows
        token = payload.get("next_page_token")
        if not token:
            return
        if not isinstance(token, str) or token in seen:
            raise HistoryError("pagination token loop")
        seen.add(token)
        params = {**params, "page_token": token}
        if len(seen) > 100_000:
            raise HistoryError("pagination page budget exhausted")


def _check_disk(path, estimated_bytes=0):
    path = Path(path)
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free - estimated_bytes < MIN_FREE_BYTES:
        raise HistoryError(
            "disk guard: preserve at least 5GB free; partial cache remains resumable"
        )


def _schema(kind):
    if kind == "quotes":
        numeric = [
            ("quote_ns", pa.int64()),
            ("bid", pa.float64()),
            ("ask", pa.float64()),
            ("bid_size", pa.float64()),
            ("ask_size", pa.float64()),
            ("raw_bs", pa.float64()),
            ("raw_as", pa.float64()),
        ]
    else:
        numeric = [
            ("trade_ns", pa.int64()),
            ("trade_price", pa.float64()),
            ("trade_size", pa.float64()),
        ]
    return pa.schema(
        numeric
        + [
            ("raw_json", pa.string()),
            ("i", pa.string()),
            ("c", pa.list_(pa.string())),
            ("x", pa.string()),
            ("z", pa.string()),
            ("bx", pa.string()),
            ("ax", pa.string()),
        ]
    )


def _write_partition(client, symbol, window, kind, target):
    temp = target.with_suffix(".parquet.tmp")
    rows_count = 0
    first = None
    last = None
    schema = _schema(kind)
    writer = pq.ParquetWriter(temp, schema, compression="zstd")
    try:
        for page in _pages(client, symbol, window, kind):
            _check_disk(target.parent, max(1024 * 1024, len(page) * 1024))
            columns = {name: [] for name in schema.names}
            for raw in page:
                stamp = parse_timestamp(raw.get("t"))
                if not window.contains(stamp):
                    continue
                if last is not None and stamp < last:
                    raise HistoryError("unordered history pagination")
                first = stamp if first is None else first
                last = stamp
                if kind == "quotes":
                    normalized = {
                        "quote_ns": stamp,
                        "bid": raw["bp"],
                        "ask": raw["ap"],
                        "bid_size": raw["bs"] * 100.0,
                        "ask_size": raw["as"] * 100.0,
                        "raw_bs": raw["bs"],
                        "raw_as": raw["as"],
                    }
                else:
                    normalized = {
                        "trade_ns": stamp,
                        "trade_price": raw["p"],
                        "trade_size": raw["s"],
                    }
                normalized.update(
                    raw_json=json.dumps(raw, sort_keys=True, allow_nan=False),
                    i=str(raw["i"])
                    if "i" in raw
                    else quote_from_event({**raw, "T": "q", "S": symbol}).quote_id
                    if kind == "quotes"
                    else str(stamp) + ":" + str(raw["p"]) + ":" + str(raw["s"]),
                    c=raw.get("c", []),
                    x=raw.get("x"),
                    z=raw.get("z"),
                    bx=raw.get("bx"),
                    ax=raw.get("ax"),
                )
                for name, value in normalized.items():
                    columns[name].append(value)
            table = pa.table(columns, schema=schema)
            writer.write_table(table)
            rows_count += len(table)
        writer.close()
        with temp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, target)
    except BaseException:
        writer.close()
        raise
    return {
        "path": target.name,
        "sha256": checksum(target),
        "rows": rows_count,
        "first_ns": first,
        "last_ns": last,
        "request": {
            "feed": "iex",
            "start": _utc(window.open_ns),
            "end": _utc(window.close_ns),
            "sort": "asc",
            "limit": 10000,
        },
    }


def _valid_partition(root, part):
    try:
        return bool(part) and checksum(root / part["path"]) == part["sha256"]
    except (OSError, KeyError):
        return False


def download_sessions(symbol: str, start: str, end: str, cache_dir: Path, *, client=None) -> Path:
    symbol = _symbol(symbol)
    client = client or ReadOnlyClient()
    windows, selected = _calendar(client, start, end)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    entries = []
    dataset = {
        "schema_version": 2,
        "status": "partial",
        "symbol": symbol,
        "feed": "iex",
        "synthetic": False,
        "calendar": [asdict(w) for w in windows],
        "request": {"start": start, "end": end},
        "aggregation_version": AGGREGATION_VERSION,
        "sessions": entries,
    }
    atomic_json(path, dataset)
    try:
        _check_disk(root, len(selected) * ESTIMATED_SESSION_BYTES)
        for window in selected:
            directory = root / symbol / window.session_id
            directory.mkdir(parents=True, exist_ok=True)
            session_path = directory / "manifest.json"
            old = json.loads(session_path.read_text()) if session_path.exists() else {}
            parts = old.get("partitions", {}) if old.get("feed") == "iex" else {}
            m = {
                "schema_version": 2,
                "status": "partial",
                "symbol": symbol,
                **asdict(window),
                "feed": "iex",
                "synthetic": False,
                "provenance": "alpaca-historical-iex",
                "quote_size_source_units": "round_lots",
                "quote_size_multiplier": 100,
                "arrival_times_available": False,
                "arrival_limitation": "historical events lack observed arrival times",
                "condition_source": CONDITION_SOURCE,
                "aggregation_version": AGGREGATION_VERSION,
                "partitions": parts,
                "calendar": asdict(window),
            }
            atomic_json(session_path, m)
            for kind in ("quotes", "trades"):
                if not _valid_partition(directory, parts.get(kind)):
                    parts[kind] = _write_partition(
                        client, symbol, window, kind, directory / f"{kind}.parquet"
                    )
                    atomic_json(session_path, m)
            m["status"] = "complete"
            m["coverage"] = {
                kind: {
                    "rows": p["rows"],
                    "first_ns": p.get("first_ns"),
                    "last_ns": p.get("last_ns"),
                }
                for kind, p in parts.items()
            }
            atomic_json(session_path, m)
            entries.append(
                {
                    "session_id": window.session_id,
                    "manifest": str(session_path.relative_to(root)),
                    "sha256": checksum(session_path),
                    "synthetic": False,
                }
            )
            atomic_json(path, dataset)
        dataset["status"] = "complete"
        atomic_json(path, dataset)
    except BaseException:
        dataset["status"] = "partial"
        atomic_json(path, dataset)
        raise
    return path


def probe_access(symbol: str, session_date: str, *, client=None) -> dict:
    symbol = _symbol(symbol)
    client = client or ReadOnlyClient()
    result = {
        "symbol": symbol,
        "date": session_date,
        "feed": "iex",
        "synthetic": False,
        "orders_submitted": False,
    }
    try:
        _, sessions = _calendar(client, session_date, session_date)
        if not sessions:
            return {**result, "status": "no_session", "quotes": 0, "trades": 0}
        window = sessions[0]
        result["calendar"] = asdict(window)
        for kind in ("quotes", "trades"):
            count = 0
            first = None
            last = None
            for page in _pages(client, symbol, window, kind):
                for row in page:
                    stamp = parse_timestamp(row["t"])
                    if window.contains(stamp):
                        count += 1
                        first = stamp if first is None else min(first, stamp)
                        last = stamp
            result[kind] = count
            result[f"{kind}_coverage"] = {"first_ns": first, "last_ns": last}
        result["status"] = "available" if result["quotes"] and result["trades"] else "empty_session"
        result["entitlement"] = "iex only; no paid fallback"
        return result
    except HistoryError as error:
        return {**result, "status": "unavailable", "reason": str(error)}
