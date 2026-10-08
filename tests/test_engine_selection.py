import builtins

import pytest

from hft import market_engine
from hft.market_engine import PyMarketEngine, make_market_engine


def _without_native(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hftcore":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_auto_falls_back_to_python_without_the_module(monkeypatch):
    monkeypatch.delenv("HFT_ENGINE", raising=False)
    _without_native(monkeypatch)
    assert isinstance(make_market_engine("X"), PyMarketEngine)
    with pytest.raises(ImportError, match="pip install ./cpp"):
        make_market_engine("X", implementation="cpp")


def test_environment_override_and_validation(monkeypatch):
    monkeypatch.setenv("HFT_ENGINE", "python")
    assert make_market_engine("X").implementation == "python"
    monkeypatch.setenv("HFT_ENGINE", "fpga")
    with pytest.raises(ValueError, match="fpga"):
        make_market_engine("X")


def test_auto_prefers_the_native_engine_when_installed(monkeypatch):
    pytest.importorskip("hftcore")
    monkeypatch.delenv("HFT_ENGINE", raising=False)
    assert make_market_engine("X", "hold-day").implementation == "cpp"
    monkeypatch.setattr(market_engine, "NATIVE_VERSION", "0.0.0")  # unproven version
    assert make_market_engine("X").implementation == "python"
