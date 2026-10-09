"""Download Alpaca's free news headlines for the ML universe, resumably.

Every article tagged with any universe symbol is stored as Parquet per month under
`<root>/news/`, with its first-publication time (`created_at`), headline, summary,
source and tagged symbols. Article bodies are not fetched. A month is finished once
it was requested through its last day; the latest month is always refreshed.

    .venv/bin/python -m hft.ml.news
"""

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..feed import parse_timestamp
from ..history import DATA_BASE, HistoryError, ReadOnlyClient
from .universe import DATA_START, UNIVERSE

SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("created_ns", pa.int64()),
        ("updated_ns", pa.int64()),
        ("headline", pa.string()),
        ("summary", pa.string()),
        ("source", pa.string()),
        ("symbols", pa.list_(pa.string())),
    ]
)
MAX_PAGES = 20_000


def _month_end(day):
    following = (day.replace(day=1) + timedelta(days=32)).replace(day=1)
    return following - timedelta(days=1)


def _iso(day):
    return datetime(day.year, day.month, day.day, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch(client, symbols, first, last):
    params = {
        "symbols": ",".join(symbols),
        "start": _iso(first),
        "end": _iso(last + timedelta(days=1)),
        "limit": 50,
        "sort": "asc",
        "include_content": "false",
    }
    rows, seen = [], set()
    for _ in range(MAX_PAGES):
        payload = client.get(f"{DATA_BASE}/v1beta1/news", params)
        if not isinstance(payload, dict):
            raise TypeError("invalid news response")
        rows += payload.get("news") or []
        token = payload.get("next_page_token")
        if not token:
            return rows
        if token in seen:
            raise RuntimeError("news pagination token loop")
        seen.add(token)
        params = {**params, "page_token": token}
    raise RuntimeError("news page budget exhausted")


def _table(rows):
    unique = {}
    for row in rows:
        unique.setdefault(int(row["id"]), row)
    ordered = sorted(unique.values(), key=lambda r: (r["created_at"], int(r["id"])))
    return pa.table(
        {
            "id": [int(r["id"]) for r in ordered],
            "created_ns": [parse_timestamp(r["created_at"]) for r in ordered],
            "updated_ns": [
                parse_timestamp(r.get("updated_at") or r["created_at"]) for r in ordered
            ],
            "headline": [r.get("headline") or "" for r in ordered],
            "summary": [r.get("summary") or "" for r in ordered],
            "source": [r.get("source") or "" for r in ordered],
            "symbols": [list(r.get("symbols") or []) for r in ordered],
        },
        schema=SCHEMA,
    )


def download_news(symbols, start, end, root, *, client=None) -> list[Path]:
    client = client or ReadOnlyClient()
    folder = Path(root) / "news"
    folder.mkdir(parents=True, exist_ok=True)
    paths, month = [], start.replace(day=1)
    while month <= end:
        path = folder / f"{month:%Y-%m}.parquet"
        marker = path.with_suffix(".json")
        last = min(_month_end(month), end)
        finished = path.exists() and marker.exists()
        finished = finished and json.loads(marker.read_text())["through"] >= f"{_month_end(month)}"
        if not finished:
            rows = _fetch(client, symbols, max(start, month), last)
            temporary = path.with_suffix(".parquet.tmp")
            pq.write_table(_table(rows), temporary)
            temporary.replace(path)
            marker.write_text(json.dumps({"through": last.isoformat()}) + "\n")
        paths.append(path)
        month = _month_end(month) + timedelta(days=1)
    return paths


def read_news(root) -> pa.Table:
    files = sorted((Path(root) / "news").glob("*.parquet"))
    if not files:
        return SCHEMA.empty_table()
    table = pa.concat_tables([pq.read_table(f, schema=SCHEMA) for f in files])
    return table.sort_by([("created_ns", "ascending"), ("id", "ascending")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/ml"))
    parser.add_argument("--start", type=date.fromisoformat, default=DATA_START)
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date())
    args = parser.parse_args(argv)
    from ..cli import _read_credentials

    _read_credentials()
    client = ReadOnlyClient()
    symbols = [s.ticker for s in UNIVERSE]
    month = args.start.replace(day=1)
    while month <= args.end:
        last = min(_month_end(month), args.end)
        path = None
        for attempt in range(6):
            try:
                path = download_news(
                    symbols, max(args.start, month), last, args.root, client=client
                )[0]
                break
            except HistoryError as error:
                if "429" not in str(error) or attempt == 5:
                    raise
                time.sleep(60)  # the rate limit is shared with other programs on the key
        print(f"{month:%Y-%m} {pq.read_metadata(path).num_rows:>6} articles", flush=True)
        month = last + timedelta(days=1)
    if (args.root / "manifest.json").exists():
        from .datasets import write_data_manifest

        write_data_manifest(args.root)  # news is part of the data: record the new version
    return 0


if __name__ == "__main__":
    sys.exit(main())
