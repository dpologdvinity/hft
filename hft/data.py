"""Validated single-symbol market rows and Parquet storage."""

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NY = ZoneInfo('America/New_York')
NS = 1_000_000_000


def local_time(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp / NS, NY)


def session_day(timestamp: int) -> str:
    return local_time(timestamp).date().isoformat()


@dataclass(frozen=True, slots=True)
class Bar:
    timestamp: int  # completed row / quote time, UTC nanoseconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    def __post_init__(self):
        if not isinstance(self.timestamp, (int, np.integer)) or self.timestamp <= 0:
            raise ValueError('timestamp must be positive UTC nanoseconds')
        values = [getattr(self, f.name) for f in fields(self)[1:]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('market values must be finite')
        if min(self.open, self.high, self.low, self.close, self.bid, self.ask) <= 0:
            raise ValueError('prices must be positive')
        if min(self.volume, self.bid_size, self.ask_size) < 0:
            raise ValueError('volumes must be nonnegative')
        if self.bid > self.ask:
            raise ValueError('crossed quote')
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError('invalid OHLC candle')

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class MarketData:
    def __init__(self, rows, symbol='UNKNOWN', synthetic=False, bar_seconds=1):
        self.rows = tuple(rows)
        if not self.rows or not all(isinstance(row, Bar) for row in self.rows):
            raise ValueError('data must contain validated bars')
        if any(b.timestamp <= a.timestamp for a, b in zip(self.rows, self.rows[1:])):
            raise ValueError('timestamps must be strictly increasing')
        if bar_seconds not in (1, 5):
            raise ValueError('only 1s or 5s bars are supported')
        if not symbol or not isinstance(symbol, str):
            raise ValueError('symbol is required')
        self.symbol, self.synthetic, self.bar_seconds = symbol, bool(synthetic), bar_seconds

    def __len__(self):
        return len(self.rows)

    def subset(self, start: int, stop: int):
        return MarketData(self.rows[start:stop], self.symbol, self.synthetic, self.bar_seconds)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([asdict(row) for row in self.rows])
        table = table.replace_schema_metadata({
            'symbol': self.symbol, 'synthetic': str(self.synthetic).lower(),
            'bar_seconds': str(self.bar_seconds),
        })
        pq.write_table(table, path, compression='zstd')

    @classmethod
    def read(cls, path):
        table = pq.read_table(path, memory_map=True)
        required = {f.name for f in fields(Bar)}
        if not required <= set(table.column_names):
            raise ValueError(f'missing Parquet columns: {sorted(required - set(table.column_names))}')
        if pa.types.is_timestamp(table.schema.field('timestamp').type):
            table = table.set_column(table.schema.get_field_index('timestamp'), 'timestamp',
                                     table['timestamp'].cast(pa.timestamp('ns')).cast(pa.int64()))
        metadata = table.schema.metadata or {}
        symbol = metadata.get(b'symbol', b'UNKNOWN').decode()
        synthetic = metadata.get(b'synthetic', b'false') == b'true'
        seconds = int(metadata.get(b'bar_seconds', b'1'))
        return cls([Bar(**row) for row in table.select(sorted(required)).to_pylist()],
                   symbol, synthetic, seconds)


def synthetic_data(days=10, bars_per_day=600, seed=42, bar_seconds=1) -> MarketData:
    """A plumbing fixture, never graduation or profitability evidence."""
    if days < 1 or bars_per_day < 62 or bars_per_day * bar_seconds > 23_400:
        raise ValueError('need positive days and 62+ bars within a regular session')
    rng = np.random.default_rng(seed)
    day = datetime(2025, 1, 6, 9, 30, tzinfo=NY)
    rows, previous = [], 100.0
    for d in range(days):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        for i in range(bars_per_day):
            regime = d % 3
            change = rng.normal((.00003, 0, -.00003)[regime], (.0001, .0005, .0001)[regime])
            price = previous * math.exp(change)
            high = max(previous, price) * (1 + abs(rng.normal(0, .00005)))
            low = min(previous, price) / (1 + abs(rng.normal(0, .00005)))
            stamp = int((day + timedelta(seconds=(i + 1) * bar_seconds)).timestamp()) * NS
            rows.append(Bar(stamp, previous, high, low, price, float(rng.integers(10, 1000)),
                            price - .01, price + .01, float(rng.integers(1, 100)),
                            float(rng.integers(1, 100))))
            previous = price
        day += timedelta(days=1)
    return MarketData(rows, 'SYNTH', True, bar_seconds)


def augment(data: MarketData, rng) -> MarketData:
    """Invert every price coherently; perturb positive volumes without future fitting."""
    inverted = rng.random() < .5
    anchor = data.rows[0].close ** 2
    rows = []
    for b in data.rows:
        noise = rng.lognormal(0, .1, 3)
        if inverted:
            b = Bar(b.timestamp, anchor / b.open, anchor / b.low, anchor / b.high,
                    anchor / b.close, b.volume, anchor / b.ask, anchor / b.bid,
                    b.ask_size, b.bid_size)
        rows.append(Bar(b.timestamp, b.open, b.high, b.low, b.close, b.volume * noise[0],
                        b.bid, b.ask, b.bid_size * noise[1], b.ask_size * noise[2]))
    return MarketData(rows, data.symbol, data.synthetic, data.bar_seconds)
