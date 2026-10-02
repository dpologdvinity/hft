"""Atomic durable state and fixed account-keyed process ownership."""

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_STATE_ROOT = Path(__file__).resolve().parents[1] / ".state"


class AccountStateStore:
    def __init__(self, account_id, mode="paper", root=None):
        if not account_id or not mode:
            raise ValueError("account identity is required")
        self.key = hashlib.sha256(f"{mode}:{account_id}".encode()).hexdigest()
        self.root = Path(root) if root is not None else DEFAULT_STATE_ROOT
        self.path = self.root / f"{self.key}.json"
        self.lock_path = DEFAULT_STATE_ROOT / f"{self.key}.lock"
        self._lock = None
        self.failed = False

    def load(self):
        if self.failed:
            raise RuntimeError("state store failed")
        if not self.path.exists():
            return None
        try:
            envelope = json.loads(self.path.read_text())
            body = json.dumps(
                envelope["state"], sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            if hashlib.sha256(body.encode()).hexdigest() != envelope["sha256"]:
                raise ValueError("state checksum mismatch")
            return envelope["state"]
        except (ValueError, KeyError, TypeError) as exc:
            self.failed = True
            raise RuntimeError("corrupt account state") from exc

    def save(self, state):
        if self.failed:
            raise RuntimeError("state store failed")
        temp = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            body = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
            envelope = {"state": state, "sha256": hashlib.sha256(body.encode()).hexdigest()}
            with tempfile.NamedTemporaryFile("w", dir=self.root, delete=False) as f:
                temp = f.name
                json.dump(envelope, f, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, self.path)
            fd = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            self.failed = True
            if temp and os.path.exists(temp):
                os.unlink(temp)
            raise

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.lock_path, "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise RuntimeError("account already owned") from None
        return self

    def __exit__(self, *args):
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None


account_owner = AccountStateStore
AccountOwner = AccountStateStore
