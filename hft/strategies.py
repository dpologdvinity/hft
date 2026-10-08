"""Decision rules that work on any stock without training."""

from dataclasses import dataclass
from pathlib import Path

RULE_STRATEGIES = frozenset({"ema-crossover", "hold-day"})
STRATEGY_VERSION = "1"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    version: str = STRATEGY_VERSION
    bundle: Path | None = None

    @property
    def is_rule(self):
        return self.name in RULE_STRATEGIES


def parse_strategy(text: str) -> StrategySpec:
    if text in RULE_STRATEGIES:
        return StrategySpec(text)
    if text.startswith("model:") and len(text) > len("model:"):
        return StrategySpec("model", bundle=Path(text.removeprefix("model:")))
    raise ValueError(f"unknown strategy {text!r}; use ema-crossover, hold-day or model:<bundle>")


def rule_action(name: str, history) -> int:
    """Flat (0) or long (1) from completed bars; same rules as the research controls."""
    if name == "hold-day":
        return 1
    if name == "ema-crossover":
        from .research import ema

        closes = [b.close for b in history]
        return int(ema(closes, 5)[-1] > ema(closes, 20)[-1])
    raise ValueError(f"unknown rule strategy {name!r}")
