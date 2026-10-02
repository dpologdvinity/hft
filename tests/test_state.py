import json

import pytest

from hft.account import Account
from hft.state import AccountStateStore


def test_state_integrity_and_account_identity(tmp_path):
    a = Account(500)
    a.execute(".2", 100, ".001", 1)
    store = AccountStateStore("fixture", "paper", tmp_path / "first")
    store.save({"account": a.to_state(), "risk": {"halted": "daily_loss"}})
    restored = Account.from_state(store.load()["account"])
    assert restored.cash == a.cash and restored.position == a.position
    restored.apply(a.fills[0])
    assert restored.position == a.position
    other = AccountStateStore("fixture", "paper", tmp_path / "different")
    with store:
        with pytest.raises(RuntimeError), other:
            pass
        with AccountStateStore("other", "paper", tmp_path):
            pass
    store.path.write_text(json.dumps({"state": {}, "sha256": "bad"}))
    with pytest.raises(RuntimeError):
        store.load()


def test_failed_replace_preserves_old_state(tmp_path, monkeypatch):
    store = AccountStateStore("crash", "paper", tmp_path)
    store.save({"generation": 1})

    def crash(*args):
        raise OSError("crash")

    monkeypatch.setattr("hft.state.os.replace", crash)
    with pytest.raises(OSError):
        store.save({"generation": 2})
    assert AccountStateStore("crash", "paper", tmp_path).load() == {"generation": 1}
    with pytest.raises(RuntimeError):
        store.save({"generation": 3})
