"""The native engine module; required in CI via HFT_REQUIRE_HFTCORE=1."""

import importlib
import os

import pytest


def _hftcore():
    try:
        return importlib.import_module("hftcore")
    except ImportError:
        if os.environ.get("HFT_REQUIRE_HFTCORE") == "1":
            raise
        pytest.skip("hftcore is not installed")


def test_version():
    assert _hftcore().version() == "0.1.0"
