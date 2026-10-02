"""Bounded raw recording and transactional preparation with honest coverage."""

import asyncio
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .calendar import calendar_from_records, session_at
from .data import NS, atomic_json, build_bars, checksum, load_dataset
from .feed import alpaca_events, event_timestamp, quote_from_event
from .history import _check_disk, _schema
from .runtime import fetch_calendar

MAX_SORT_WORKING_BYTES = 2 * 1024**3
BUFFER_ROWS = 1000


async def record(symbol, output, duration=None, stream=None):
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("invalid duration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"raw-{time.time_ns()}.jsonl"
    calendar = await asyncio.to_thread(fetch_calendar)
    atomic_json(output / "calendar.json", {"calendar": calendar, "provenance": "alpaca-calendar"})
    start = time.monotonic()
    count = 0
    iterator = (stream or alpaca_events(symbol)).__aiter__()
    try:
        with path.open("x", buffering=1) as file:
            while duration is None or time.monotonic() - start < duration:
                _check_disk(output, 1024 * 1024)
                timeout = min(5, duration - (time.monotonic() - start)) if duration else 5
                try:
                    event = await asyncio.wait_for(
                        iterator.__anext__(), timeout=max(0.001, timeout)
                    )
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    if duration and time.monotonic() - start >= duration:
                        break
                    raise RuntimeError("recording feed timeout") from None
                file.write(json.dumps(event, allow_nan=False) + "\n")
                count += 1
    finally:
        if hasattr(iterator, "aclose"):
            await iterator.aclose()
    return {"recording": str(path), "events": count, "prepare_required": True}


def _sort_preflight(path):
    """Count decoded Arrow bytes before full-table sorting or its copies."""
    file = pq.ParquetFile(path)
    decoded = 0
    for batch in file.iter_batches(batch_size=64):
        decoded += batch.nbytes
        if 3 * decoded + file.metadata.num_rows * 8 > MAX_SORT_WORKING_BYTES:
            raise ValueError("recorded session exceeds sorting memory budget")
    return decoded


def _summary(path):
    sessions = load_dataset(path)
    return {
        "dataset": str(path),
        "sessions": len(sessions),
        "complete_sessions": sum(s.manifest.get("complete_session", False) for s in sessions),
        "bars": sum(len(build_bars(s)) for s in sessions),
    }


def prepare_recordings(source):
    root = Path(source)
    if root.is_file() and root.name == "manifest.json":
        return _summary(root)
    calendar_path = root / "calendar.json"
    windows = calendar_from_records(json.loads(calendar_path.read_text())["calendar"])
    raw_paths = sorted(root.glob("raw-*.jsonl"))
    if not raw_paths:
        raise ValueError("no raw recording files")
    inputs = {
        "calendar": checksum(calendar_path),
        "files": {p.name: checksum(p) for p in raw_paths},
    }
    run_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    prepared = root / "prepared"
    dataset = prepared / "manifest.json"
    if dataset.exists():
        old = json.loads(dataset.read_text())
        if old.get("recording_hash") == run_hash:
            return _summary(dataset)
    _check_disk(root, sum(p.stat().st_size for p in raw_paths) * 3)
    # Each preparation owns an immutable input-hashed generation. A failed rerun
    # cannot invalidate partitions referenced by the previously published dataset.
    generation = prepared / run_hash
    writers = {}
    buffers = {}
    counts = {}
    first = {}
    last = {}
    quality = {}
    quote_state = {}
    session_symbols = set()

    def flush(key):
        if buffers[key]:
            _check_disk(root, 1024 * 1024)
            writers[key].write_table(pa.Table.from_pylist(buffers[key], schema=writers[key].schema))
            buffers[key].clear()

    try:
        for raw_path in raw_paths:
            with raw_path.open() as file:
                for line in file:
                    if not line.endswith("\n"):
                        raise ValueError("truncated raw recording")
                    raw = json.loads(line)
                    kind = raw.get("T")
                    if kind not in ("q", "t", "c", "x"):
                        continue
                    stamp = event_timestamp(raw)
                    arrival = raw.get("arrival_ns")
                    if not isinstance(arrival, int) or isinstance(arrival, bool):
                        raise ValueError("recorded events require integer arrival_ns")  # noqa: TRY004 - recording validation contract
                    window = session_at(windows, stamp)
                    if not window:
                        continue
                    symbol = raw["S"]
                    sid = (symbol, window.session_id)
                    session_symbols.add(sid)
                    if len({s for s, _ in session_symbols}) > 1:
                        raise ValueError("prepare one stock per recording directory")
                    qcounts = quality.setdefault(sid, Counter())
                    if kind in ("c", "x"):
                        qcounts["corrections" if kind == "c" else "cancellations"] += 1
                        continue
                    if not window.contains(arrival):
                        qcounts["arrival_outside_session"] += 1
                    if arrival - stamp > 250_000_000 or stamp - arrival > 250_000_000:
                        qcounts["late_or_future_events"] += 1
                    if kind == "q":
                        state = quote_state.setdefault(
                            sid,
                            {
                                "event": None,
                                "arrival": None,
                                "first_arrival": arrival,
                                "last_arrival": arrival,
                            },
                        )
                        if state["event"] is not None:
                            if stamp < state["event"]:
                                qcounts["reordered_quotes"] += 1
                            elif stamp - state["event"] > 5 * NS:
                                qcounts["quote_event_gaps"] += 1
                            if arrival < state["arrival"]:
                                qcounts["arrival_reversals"] += 1
                            elif arrival - state["arrival"] > 5 * NS:
                                qcounts["quote_arrival_gaps"] += 1
                        state["event"] = max(stamp, state["event"] or stamp)
                        state["arrival"] = max(arrival, state["arrival"] or arrival)
                        state["first_arrival"] = min(arrival, state["first_arrival"])
                        state["last_arrival"] = max(arrival, state["last_arrival"])
                    key = (*sid, "quotes" if kind == "q" else "trades")
                    directory = generation / symbol / window.session_id
                    directory.mkdir(parents=True, exist_ok=True)
                    if key not in writers:
                        schema = _schema(key[2]).append(pa.field("arrival_ns", pa.int64()))
                        writers[key] = pq.ParquetWriter(
                            directory / f"{key[2]}.parquet.tmp", schema, compression="zstd"
                        )
                        buffers[key] = []
                        counts[key] = 0
                        first[key] = stamp
                    first[key] = min(first[key], stamp)
                    last[key] = max(last.get(key, stamp), stamp)
                    counts[key] += 1
                    row = {
                        "raw_json": json.dumps(raw, allow_nan=False),
                        "i": str(raw["i"]) if "i" in raw else None,
                        "c": raw.get("c", []),
                        "x": raw.get("x"),
                        "z": raw.get("z"),
                        "bx": raw.get("bx"),
                        "ax": raw.get("ax"),
                    }
                    if kind == "q":
                        q = quote_from_event(raw, arrival)
                        row["i"] = q.quote_id
                        row.update(
                            quote_ns=stamp,
                            bid=float(q.bid),
                            ask=float(q.ask),
                            bid_size=float(q.bid_size),
                            ask_size=float(q.ask_size),
                            raw_bs=raw["bs"],
                            raw_as=raw["as"],
                            arrival_ns=arrival,
                        )
                    else:
                        row["i"] = str(raw.get("i", f"{stamp}:{raw['p']}:{raw['s']}"))
                        row.update(
                            trade_ns=stamp,
                            trade_price=raw["p"],
                            trade_size=raw["s"],
                            arrival_ns=arrival,
                        )
                    buffers[key].append(row)
                    if len(buffers[key]) >= BUFFER_ROWS:
                        flush(key)
        for key, writer in writers.items():
            flush(key)
            writer.close()
        entries = []
        for window in windows:
            matching = [symbol for symbol, date in session_symbols if date == window.session_id]
            if not matching:
                continue
            symbol = matching[0]
            directory = generation / symbol / window.session_id
            parts = {}
            for kind in ("quotes", "trades"):
                key = (symbol, window.session_id, kind)
                if key not in writers:
                    # Empty trades are valid; no fictional volume is generated.
                    pq.write_table(
                        pa.Table.from_pylist(
                            [], schema=_schema(kind).append(pa.field("arrival_ns", pa.int64()))
                        ),
                        directory / f"{kind}.parquet.tmp",
                    )
                    counts[key] = 0
                    first[key] = last[key] = None
                path = directory / f"{kind}.parquet"
                temporary = path.with_suffix(".parquet.tmp")
                _check_disk(root, max(1024 * 1024, temporary.stat().st_size * 2))
                _sort_preflight(temporary)
                table = pq.read_table(temporary)
                table = table.sort_by(
                    [("quote_ns" if kind == "quotes" else "trade_ns", "ascending")]
                )
                pq.write_table(table, temporary, compression="zstd")
                del table
                temporary.replace(path)
                parts[kind] = {
                    "path": path.name,
                    "sha256": checksum(path),
                    "rows": counts[key],
                    "first_ns": first[key],
                    "last_ns": last[key],
                }
            sid = (symbol, window.session_id)
            state = quote_state.get(sid)
            qcounts = quality[sid]
            boundaries = bool(
                state
                and parts["quotes"]["first_ns"] <= window.open_ns + 5 * NS
                and parts["quotes"]["last_ns"] >= window.close_ns - 2 * NS
                and state["first_arrival"] <= window.open_ns + 5 * NS
                and state["last_arrival"] >= window.close_ns - 2 * NS
            )
            full = boundaries and not any(qcounts.values())
            manifest = {
                "schema_version": 2,
                "status": "complete",
                "symbol": symbol,
                **asdict(window),
                "feed": "iex",
                "synthetic": False,
                "provenance": "alpaca-realtime-iex",
                "complete_session": full,
                "arrival_times_available": True,
                "partitions": parts,
                "quality": dict(qcounts),
                "coverage": {
                    "boundaries_covered": boundaries,
                    "maximum_allowed_quote_gap_ns": 5 * NS,
                },
                "recording_hash": run_hash,
            }
            atomic_json(directory / "manifest.json", manifest)
            entries.append(
                {
                    "session_id": window.session_id,
                    "manifest": str((directory / "manifest.json").relative_to(prepared)),
                    "sha256": checksum(directory / "manifest.json"),
                    "synthetic": False,
                }
            )
        candidate = generation / "dataset.json"
        data = {
            "schema_version": 2,
            "status": "complete",
            "feed": "iex",
            "synthetic": False,
            "symbol": next(iter(session_symbols))[0] if session_symbols else None,
            "calendar": [asdict(w) for w in windows],
            "sessions": entries,
            "recording_hash": run_hash,
            "recording_inputs": inputs,
        }
        # Validate a candidate manifest before replacing the published dataset.
        atomic_json(
            candidate,
            {
                **data,
                "sessions": [
                    {**entry, "manifest": str(Path(entry["manifest"]).relative_to(run_hash))}
                    for entry in entries
                ],
            },
        )
        load_dataset(candidate)
        atomic_json(dataset, data)
        return _summary(dataset)
    finally:
        for writer in writers.values():
            writer.close()
