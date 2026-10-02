"""Runtime behavior remains identical except for a stricter live entry ceiling."""

from dataclasses import asdict

import pytest

from hft.account import Costs
from hft.configuration import runtime_config
from hft.risk import RiskConfig
from hft.sizing import SizingConfig


def metadata():
    return {
        "symbol": "AAPL",
        "feed": "iex",
        "initial_cash": 500.0,
        "costs": asdict(Costs()),
        "risk": asdict(RiskConfig()),
        "sizing": asdict(SizingConfig()),
        "latency_ms": 75,
        "bar_seconds": 5,
    }


@pytest.mark.parametrize(
    "config",
    [
        "latency_ms = 250\n",
        'symbol = "MSFT"\n',
        'feed = "sip"\n',
        "bar_seconds = 1\n",
        'mode = "live"\n',
        "[costs]\ncommission = 0.0\nslippage_bps = 0.0\n",
        "[risk]\nmax_daily_loss = 0.5\n",
        "[sizing]\nallocation_fraction = 1.0\n",
    ],
)
def test_frozen_behavior_override_rejected(tmp_path, config):
    path = tmp_path / "runtime.toml"
    path.write_text(config)
    with pytest.raises(ValueError):
        runtime_config(metadata(), path)


def test_canary_only_tightens_absolute_ceiling():
    manifest = metadata()
    base = runtime_config(manifest)
    canary = runtime_config(manifest, max_notional=10)
    assert canary.sizing.max_entry_notional == 10
    assert canary.execution_hash != base.execution_hash
    assert canary.costs == base.costs and canary.risk == base.risk
    assert canary.capital == base.capital and canary.latency_ms == base.latency_ms
    assert canary.sizing.allocation_fraction == base.sizing.allocation_fraction
    manifest["sizing"]["max_entry_notional"] = 2
    assert runtime_config(manifest, max_notional=10).sizing.max_entry_notional == 2
    for ceiling in (0, 11, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            runtime_config(metadata(), max_notional=ceiling)
    with pytest.raises(ValueError, match="capital"):
        runtime_config(metadata(), capital=1000)
