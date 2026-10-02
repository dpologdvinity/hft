"""Validated immutable session arrays and auditable Parquet partitions."""

import hashlib
import heapq
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

NY = ZoneInfo("America/New_York")
NS = 1_000_000_000
AGGREGATION_VERSION = "causal-5s-v3-boundary-timers-latest-quote-250ms"
MAX_SESSION_BYTES = 2 * 1024**3
MAX_DATASET_BYTES = 6 * 1024**3
MAX_WORKING_BYTES = 8 * 1024**3
NUMERIC_NAMES = (
    "quote_ns",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "trade_ns",
    "trade_price",
    "trade_size",
)


def local_time(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp // NS, NY)


def session_day(timestamp: int) -> str:
    return local_time(timestamp).date().isoformat()


@dataclass(frozen=True, slots=True)
class Quote:
    quote_id: str
    event_ns: int
    arrival_ns: int | None
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal

    def __post_init__(self):
        values = (self.bid, self.ask, self.bid_size, self.ask_size)
        if not all(isinstance(v, Decimal) and v.is_finite() for v in values):
            raise ValueError("quote values must be finite Decimal")
        if self.event_ns <= 0 or self.bid <= 0 or self.ask < self.bid:
            raise ValueError("invalid quote")
        if min(self.bid_size, self.ask_size) < 0:
            raise ValueError("invalid quote depth")

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def timestamp(self) -> int:
        return self.event_ns


@dataclass(frozen=True, slots=True, init=False)
class Bar:
    start_ns: int
    end_ns: int
    quote_ns: int
    session_id: str
    tradable: bool
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    vwap: float

    def __init__(
        self,
        timestamp=None,
        open=None,
        high=None,
        low=None,
        close=None,
        volume=None,
        bid=None,
        ask=None,
        bid_size=None,
        ask_size=None,
        *,
        start_ns=None,
        end_ns=None,
        quote_ns=None,
        session_id=None,
        tradable=True,
        vwap=None,
    ):
        # Legacy timestamps explicitly mean completed end time, with a 1s interval.
        if end_ns is None:
            end_ns = timestamp
        elif timestamp is not None and timestamp != end_ns:
            raise ValueError("timestamp must equal completed end_ns")
        if not isinstance(end_ns, (int, np.integer)) or end_ns <= 0:
            raise ValueError("end_ns must be positive UTC nanoseconds")
        values = {
            "start_ns": end_ns - NS if start_ns is None else start_ns,
            "end_ns": int(end_ns),
            "quote_ns": end_ns - 1 if quote_ns is None else quote_ns,
            "session_id": session_day(end_ns) if session_id is None else session_id,
            "tradable": bool(tradable),
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "vwap": close if vwap is None else vwap,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if self.start_ns >= self.end_ns or self.quote_ns >= self.end_ns:
            raise ValueError("invalid bar time boundaries")
        nums = (open, high, low, close, volume, bid, ask, bid_size, ask_size, self.vwap)
        if not all(v is not None and math.isfinite(v) for v in nums):
            raise ValueError("market values must be finite")
        if min(open, high, low, close, bid, ask, self.vwap) <= 0:
            raise ValueError("prices must be positive")
        if min(volume, bid_size, ask_size) < 0 or bid > ask:
            raise ValueError("invalid sizes or crossed quote")
        if not low <= min(open, close) <= max(open, close) <= high:
            raise ValueError("invalid OHLC")

    @property
    def timestamp(self):
        return self.end_ns

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def quote_age_ns(self):
        return self.end_ns - self.quote_ns


@dataclass(frozen=True, slots=True)
class SessionData:
    symbol: str
    session_id: str
    open_ns: int
    close_ns: int
    quote_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    trade_ns: np.ndarray
    trade_price: np.ndarray
    trade_size: np.ndarray
    manifest: dict

    def __post_init__(self):
        if not self.symbol or self.open_ns <= 0 or self.close_ns <= self.open_ns:
            raise ValueError("invalid session metadata")
        total = sum(len(getattr(self, name)) * 8 for name in NUMERIC_NAMES)
        total += metadata_bytes(self.manifest)
        if total > MAX_SESSION_BYTES:
            raise ValueError("session exceeds memory budget")
        for names in [
            ("quote_ns", "bid", "ask", "bid_size", "ask_size"),
            ("trade_ns", "trade_price", "trade_size"),
        ]:
            n = len(getattr(self, names[0]))
            for name in names:
                raw = np.asarray(getattr(self, name))
                if raw.ndim != 1 or len(raw) != n:
                    raise ValueError("array length mismatch")
                if name.endswith("_ns"):
                    if (
                        raw.dtype.kind not in "iu"
                        or np.any(raw < self.open_ns)
                        or np.any(raw >= self.close_ns)
                    ):
                        raise ValueError("timestamps must be int64 in session")
                    if np.any(raw[1:] < raw[:-1]):
                        raise ValueError("unordered events")
                a = np.ascontiguousarray(
                    raw, dtype=np.int64 if name.endswith("_ns") else np.float64
                ).copy()
                if not np.all(np.isfinite(a)):
                    raise ValueError("nonfinite market data")
                if not name.endswith("_ns") and (
                    np.any(a < 0) or (name in ("bid", "ask", "trade_price") and np.any(a <= 0))
                ):
                    raise ValueError("invalid price or size")
                a.flags.writeable = False
                object.__setattr__(self, name, a)
        if total > MAX_SESSION_BYTES:
            raise ValueError("session exceeds memory budget")
        if np.any(self.bid > self.ask):
            raise ValueError("crossed quotes")
        for kind, n in [("quote", len(self.quote_ns)), ("trade", len(self.trade_ns))]:
            extra = self.manifest.get(f"{kind}_metadata", {})
            for values in extra.values():
                if len(values) != n:
                    raise ValueError("metadata length mismatch")
            ids = extra.get("i", extra.get("quote_id" if kind == "quote" else "trade_id"))
            if ids is not None and (
                (pc.count_distinct(ids).as_py() != len(ids))
                if isinstance(ids, (pa.Array, pa.ChunkedArray))
                else len({str(x) for x in ids}) != len(ids)
            ):
                raise ValueError("duplicate record identity")
        # Duplicate indistinguishable rows are corruption, equal-time distinct ticks are valid.
        for names in [
            ("quote_ns", "bid", "ask", "bid_size", "ask_size"),
            ("trade_ns", "trade_price", "trade_size"),
        ]:
            if len(getattr(self, names[0])) > 1:
                duplicate = np.ones(len(getattr(self, names[0])) - 1, dtype=bool)
                for name in names:
                    a = getattr(self, name)
                    duplicate &= a[1:] == a[:-1]
                kind = "quote" if names[0] == "quote_ns" else "trade"
                extra = self.manifest.get(f"{kind}_metadata", {})
                if np.any(duplicate) and not any(k in extra for k in ("i", "quote_id", "trade_id")):
                    raise ValueError("duplicate events")

    @property
    def synthetic(self):
        return bool(self.manifest.get("synthetic", False))

    @property
    def quote_arrival_ns(self):
        value = self.manifest.get("quote_metadata", {}).get("arrival_ns")
        return None if value is None else np.asarray(value, dtype=np.int64)

    @property
    def trade_arrival_ns(self):
        value = self.manifest.get("trade_metadata", {}).get("arrival_ns")
        return None if value is None else np.asarray(value, dtype=np.int64)

    def iter_events(self):
        """Merge numeric arrays by recorded arrival time, otherwise event time."""

        def stream(kind):
            stamps = self.quote_ns if kind == "q" else self.trade_ns
            extra = self.manifest.get("quote_metadata" if kind == "q" else "trade_metadata", {})
            arrival = extra.get("arrival_ns")
            order = (
                np.argsort(np.asarray(arrival), kind="stable")
                if arrival is not None
                else range(len(stamps))
            )
            for idx in order:
                i = int(idx)
                event = {
                    key: (
                        values[i].as_py()
                        if isinstance(values, (pa.Array, pa.ChunkedArray))
                        else values[i]
                    )
                    for key, values in extra.items()
                }
                if event.get("arrival_ns") is None:
                    event.pop("arrival_ns", None)
                else:
                    event["arrival_ns"] = int(event["arrival_ns"])
                event.update(T=kind, S=self.symbol, event_ns=int(stamps[i]), sizes_in_shares=True)
                if kind == "q":
                    event.update(
                        bp=float(self.bid[i]),
                        ap=float(self.ask[i]),
                        bs=float(self.bid_size[i]),
                        **{"as": float(self.ask_size[i])},
                    )
                else:
                    event.update(p=float(self.trade_price[i]), s=float(self.trade_size[i]))
                yield (int(event.get("arrival_ns", stamps[i])), 0 if kind == "q" else 1, i, event)

        for _, _, _, event in heapq.merge(stream("q"), stream("t")):
            yield event


def build_bars(session: SessionData, seconds: int = 5) -> tuple[Bar, ...]:
    from .calendar import SessionWindow
    from .feed import BarAggregator, historical_timeline

    a = BarAggregator(
        session.symbol,
        seconds,
        SessionWindow(session.session_id, session.open_ns, session.close_ns),
    )
    bars = []
    for event, now in historical_timeline(session, seconds):
        bars.extend(a.advance_to(now) if event is None else a.add(event, now))
    return tuple(bars)


def synthetic_sessions(days=5, bars_per_day=80, seed=42) -> list[SessionData]:
    if days < 1 or bars_per_day < 1 or bars_per_day * 5 > 23_400:
        raise ValueError("invalid fixture size")
    rng = np.random.default_rng(seed)
    result = []
    day = datetime(2025, 1, 6, 9, 30, tzinfo=NY)
    for d in range(days):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        start = int(day.timestamp()) * NS
        n = bars_per_day * 50
        qns = start + np.arange(n, dtype=np.int64) * 100_000_000
        prices = 100 * np.exp(np.cumsum(rng.normal((0.000002, 0, -0.000002)[d % 3], 0.00003, n)))
        ti = np.arange(0, n, 10)
        result.append(
            SessionData(
                "SYNTH",
                day.date().isoformat(),
                start,
                start + bars_per_day * 5 * NS,
                qns,
                prices - 0.01,
                prices + 0.01,
                np.full(n, 100.0),
                np.full(n, 100.0),
                qns[ti],
                prices[ti],
                np.full(len(ti), 10.0),
                {
                    "synthetic": True,
                    "provenance": "synthetic-100ms-fixture",
                    "feed": "synthetic",
                    "schema_version": 2,
                    "arrival_times_available": False,
                    "aggregation_version": AGGREGATION_VERSION,
                },
            )
        )
        day += timedelta(days=1)
    return result


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def save_dataset(sessions, path) -> Path:
    path = Path(path)
    manifest_path = path if path.suffix == ".json" else path / "manifest.json"
    root = manifest_path.parent
    entries = []
    for s in sessions:
        directory = root / s.symbol / s.session_id
        directory.mkdir(parents=True, exist_ok=True)
        parts = {}
        for kind, names in [
            ("quotes", ("quote_ns", "bid", "ask", "bid_size", "ask_size")),
            ("trades", ("trade_ns", "trade_price", "trade_size")),
        ]:
            target = directory / f"{kind}.parquet"
            temp = target.with_suffix(".parquet.tmp")
            columns = {name: getattr(s, name) for name in names}
            # Raw identifier/condition fields remain Arrow columns, not Python tick objects.
            columns.update(
                s.manifest.get("quote_metadata" if kind == "quotes" else "trade_metadata", {})
            )
            pq.write_table(pa.table(columns), temp, compression="zstd")
            os.replace(temp, target)
            parts[kind] = {
                "path": target.name,
                "sha256": checksum(target),
                "rows": len(getattr(s, names[0])),
            }
        metadata = {
            k: v
            for k, v in s.manifest.items()
            if k not in ("quote_metadata", "trade_metadata", "partitions")
        }
        metadata.update(
            schema_version=2,
            symbol=s.symbol,
            session_id=s.session_id,
            open_ns=s.open_ns,
            close_ns=s.close_ns,
            partitions=parts,
            status="complete",
            aggregation_version=AGGREGATION_VERSION,
        )
        session_path = directory / "manifest.json"
        atomic_json(session_path, metadata)
        entries.append(
            {
                "session_id": s.session_id,
                "manifest": str(session_path.relative_to(root)),
                "sha256": checksum(session_path),
                "synthetic": s.synthetic,
            }
        )
    atomic_json(
        manifest_path,
        {
            "schema_version": 2,
            "sessions": entries,
            "status": "complete",
            "synthetic": any(s.synthetic for s in sessions),
            "aggregation_version": AGGREGATION_VERSION,
        },
    )
    return manifest_path


def metadata_bytes(manifest):
    """Count retained Arrow buffers, including raw JSON/conditions/identifiers."""
    total = 0
    for kind in ("quote_metadata", "trade_metadata"):
        for values in manifest.get(kind, {}).values():
            if isinstance(values, (pa.Array, pa.ChunkedArray, np.ndarray)):
                total += values.nbytes
            else:
                # Only small caller fixtures use Python metadata; Arrow is authoritative on disk.
                total += pa.array(values).nbytes
    return total


def session_bytes(session):
    return sum(getattr(session, name).nbytes for name in NUMERIC_NAMES) + metadata_bytes(
        session.manifest
    )


def _session_preflight(path, resident_bytes=0):
    path = Path(path)
    m = json.loads(path.read_text())
    if m.get("status") != "complete":
        raise ValueError("incomplete session")
    numeric = metadata = 0
    for kind, names in [("quotes", NUMERIC_NAMES[:5]), ("trades", NUMERIC_NAMES[5:])]:
        part = m["partitions"][kind]
        target = path.parent / part["path"]
        if checksum(target) != part["sha256"]:
            raise ValueError("partition checksum mismatch")
        file = pq.ParquetFile(target, memory_map=True)
        rows = file.metadata.num_rows
        if rows != part["rows"]:
            raise ValueError("partition row count mismatch")
        for name in names:
            dtype = file.schema_arrow.field(name).type
            if name.endswith("_ns") and dtype != pa.int64():
                raise ValueError("timestamp Parquet columns must be exact int64 nanoseconds")
        if (
            "arrival_ns" in file.schema_arrow.names
            and file.schema_arrow.field("arrival_ns").type != pa.int64()
        ):
            raise ValueError("arrival timestamps must be exact int64 nanoseconds")
        numeric += rows * len(names) * 8
        if numeric > MAX_SESSION_BYTES:
            raise ValueError("session exceeds memory budget")
        # Scan only metadata in bounded Arrow batches before numeric allocation.
        # Encoded Parquet byte sizes alone undercount dictionary-expanded raw JSON.
        extra = [name for name in file.schema_arrow.names if name not in names]
        for batch in file.iter_batches(batch_size=64, columns=extra):
            metadata += batch.nbytes
            retained = numeric + metadata
            if retained > MAX_SESSION_BYTES or resident_bytes + retained > MAX_DATASET_BYTES:
                raise ValueError("dataset/session exceeds resident memory budget")
        # Preallocated arrays coexist with their immutable defensive copies.
        if resident_bytes + 2 * numeric + metadata > MAX_WORKING_BYTES:
            raise ValueError("dataset load exceeds working memory budget")
    retained = numeric + metadata
    if resident_bytes + retained > MAX_DATASET_BYTES:
        raise ValueError("dataset exceeds resident memory budget; load individual sessions")
    return retained


def load_session(manifest_path: Path, *, resident_bytes=0) -> SessionData:
    path = Path(manifest_path)
    _session_preflight(path, resident_bytes)
    m = json.loads(path.read_text())
    if m.get("status") != "complete":
        raise ValueError("incomplete session")
    arrays = {}
    meta = {}
    for kind, names in [
        ("quotes", ("quote_ns", "bid", "ask", "bid_size", "ask_size")),
        ("trades", ("trade_ns", "trade_price", "trade_size")),
    ]:
        part = m["partitions"][kind]
        target = path.parent / part["path"]
        if checksum(target) != part["sha256"]:
            raise ValueError("partition checksum mismatch")
        file = pq.ParquetFile(target, memory_map=True)
        rows = file.metadata.num_rows
        if rows != part["rows"]:
            raise ValueError("partition row count mismatch")
        if rows * len(names) * 8 > MAX_SESSION_BYTES:
            raise ValueError("session exceeds memory budget")
        for name in names:
            arrays[name] = np.empty(rows, dtype=np.int64 if name.endswith("_ns") else np.float64)
        extra = {name: [] for name in file.schema_arrow.names if name not in names}
        offset = 0
        for batch in file.iter_batches(batch_size=65_536):
            n = len(batch)
            for name in names:
                col = batch.column(batch.schema.get_field_index(name))
                if col.null_count:
                    raise ValueError("null market value")
                arrays[name][offset : offset + n] = col.to_numpy(zero_copy_only=False)
            for name, chunks in extra.items():
                col = batch.column(batch.schema.get_field_index(name))
                chunks.append(col)
            offset += n
        if extra:
            meta["quote_metadata" if kind == "quotes" else "trade_metadata"] = {
                name: pa.chunked_array(chunks, type=file.schema_arrow.field(name).type)
                for name, chunks in extra.items()
            }
    m.update(meta)
    return SessionData(
        m["symbol"], m["session_id"], m["open_ns"], m["close_ns"], manifest=m, **arrays
    )


def load_dataset(path) -> list[SessionData]:
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    m = json.loads(path.read_text())
    if "sessions" not in m:
        return [load_session(path)]
    if m.get("status") != "complete":
        raise ValueError("incomplete dataset")
    result = []
    for entry in m["sessions"]:
        target = path.parent / entry["manifest"]
        if checksum(target) != entry["sha256"]:
            raise ValueError("session manifest checksum mismatch")
        resident = sum(session_bytes(session) for session in result)
        _session_preflight(target, resident)
        result.append(load_session(target, resident_bytes=resident))
    if any(b.open_ns <= a.open_ns for a, b in itertools.pairwise(result)):
        raise ValueError("unordered sessions")
    return result


class MarketData:
    def __init__(self, rows, symbol="UNKNOWN", synthetic=False, bar_seconds=1):
        self.rows = tuple(rows)
        if not self.rows or not all(isinstance(row, Bar) for row in self.rows):
            raise ValueError("data must contain validated bars")
        if any(b.timestamp <= a.timestamp for a, b in zip(self.rows, self.rows[1:])):
            raise ValueError("timestamps must be strictly increasing")
        if bar_seconds not in (1, 5):
            raise ValueError("only 1s or 5s bars are supported")
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol is required")
        self.symbol, self.synthetic, self.bar_seconds = symbol, bool(synthetic), bar_seconds

    def __len__(self):
        return len(self.rows)

    def subset(self, start: int, stop: int):
        return MarketData(self.rows[start:stop], self.symbol, self.synthetic, self.bar_seconds)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([asdict(row) for row in self.rows])
        table = table.replace_schema_metadata(
            {
                "symbol": self.symbol,
                "synthetic": str(self.synthetic).lower(),
                "bar_seconds": str(self.bar_seconds),
            }
        )
        pq.write_table(table, path, compression="zstd")

    @classmethod
    def read(cls, path):
        table = pq.read_table(path, memory_map=True)
        required = {f.name for f in fields(Bar)}
        if not required <= set(table.column_names):
            raise ValueError(
                f"missing Parquet columns: {sorted(required - set(table.column_names))}"
            )
        if "timestamp" in table.column_names and pa.types.is_timestamp(
            table.schema.field("timestamp").type
        ):
            table = table.set_column(
                table.schema.get_field_index("timestamp"),
                "timestamp",
                table["timestamp"].cast(pa.timestamp("ns")).cast(pa.int64()),
            )
        metadata = table.schema.metadata or {}
        symbol = metadata.get(b"symbol", b"UNKNOWN").decode()
        synthetic = metadata.get(b"synthetic", b"false") == b"true"
        seconds = int(metadata.get(b"bar_seconds", b"1"))
        return cls(
            [Bar(**row) for row in table.select(sorted(required)).to_pylist()],
            symbol,
            synthetic,
            seconds,
        )


def synthetic_data(days=10, bars_per_day=600, seed=42, bar_seconds=1) -> MarketData:
    """A plumbing fixture, never graduation or profitability evidence."""
    if days < 1 or bars_per_day < 62 or bars_per_day * bar_seconds > 23_400:
        raise ValueError("need positive days and 62+ bars within a regular session")
    rng = np.random.default_rng(seed)
    day = datetime(2025, 1, 6, 9, 30, tzinfo=NY)
    rows, previous = [], 100.0
    for d in range(days):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        for i in range(bars_per_day):
            regime = d % 3
            change = rng.normal((0.00003, 0, -0.00003)[regime], (0.0001, 0.0005, 0.0001)[regime])
            price = previous * math.exp(change)
            high = max(previous, price) * (1 + abs(rng.normal(0, 0.00005)))
            low = min(previous, price) / (1 + abs(rng.normal(0, 0.00005)))
            stamp = int((day + timedelta(seconds=(i + 1) * bar_seconds)).timestamp()) * NS
            rows.append(
                Bar(
                    stamp,
                    previous,
                    high,
                    low,
                    price,
                    float(rng.integers(10, 1000)),
                    price - 0.01,
                    price + 0.01,
                    float(rng.integers(1, 100)),
                    float(rng.integers(1, 100)),
                )
            )
            previous = price
        day += timedelta(days=1)
    return MarketData(rows, "SYNTH", True, bar_seconds)


def augment(data: MarketData, rng) -> MarketData:
    """Invert every price coherently; perturb positive volumes without future fitting."""
    inverted = rng.random() < 0.5
    anchor = data.rows[0].close ** 2
    rows = []
    for b in data.rows:
        noise = rng.lognormal(0, 0.1, 3)
        if inverted:
            b = Bar(
                b.timestamp,
                anchor / b.open,
                anchor / b.low,
                anchor / b.high,
                anchor / b.close,
                b.volume,
                anchor / b.ask,
                anchor / b.bid,
                b.ask_size,
                b.bid_size,
            )
        rows.append(
            Bar(
                b.timestamp,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume * noise[0],
                b.bid,
                b.ask,
                b.bid_size * noise[1],
                b.ask_size * noise[2],
            )
        )
    return MarketData(rows, data.symbol, data.synthetic, data.bar_seconds)
