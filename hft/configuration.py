"""Frozen model contracts own runtime defaults; overrides may only tighten canary size."""

import math
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .account import Costs
from .logs import canonical_hash
from .risk import RiskConfig
from .sizing import SizingConfig


@dataclass(frozen=True)
class RuntimeConfig:
    symbol: str
    feed: str
    capital: float
    costs: Costs
    risk: RiskConfig
    sizing: SizingConfig
    latency_ms: float = 75
    bar_seconds: int = 5

    @property
    def execution_hash(self):
        return canonical_hash(
            {
                k: asdict(self)[k]
                for k in ("costs", "risk", "sizing", "latency_ms", "bar_seconds", "symbol", "feed")
            }
        )


def runtime_config(metadata, path=None, capital=None, max_notional=None):
    base = RuntimeConfig(
        metadata["symbol"],
        metadata["feed"],
        metadata["initial_cash"],
        Costs(**metadata["costs"]),
        RiskConfig(**metadata["risk"]),
        SizingConfig(**metadata["sizing"]),
        metadata["latency_ms"],
        metadata["bar_seconds"],
    )
    if path:
        requested = tomllib.loads(Path(path).read_text())
        accepted = {
            "symbol",
            "feed",
            "capital",
            "latency_ms",
            "bar_seconds",
            "costs",
            "risk",
            "sizing",
        }
        if set(requested) - accepted:
            raise ValueError("unknown runtime configuration fields")
        current = asdict(base)
        if any(v != current[k] for k, v in requested.items()):
            raise ValueError("configuration differs from frozen model; retrain and revalidate")
    if capital is not None and (
        not math.isfinite(capital) or capital <= 0 or capital != base.capital
    ):
        raise ValueError("capital must match the evaluated model allocation")
    if max_notional is not None:
        if not math.isfinite(max_notional) or not 0 < max_notional <= 10:
            raise ValueError("live canary notional must be in (0,10]")
        base = replace(
            base,
            sizing=replace(
                base.sizing, max_entry_notional=min(base.sizing.max_entry_notional, max_notional)
            ),
        )
    return base
