"""Explicit Alpaca paper/live routing with actual-fill accounting and reconciliation."""

import fcntl
import json
import math
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .account import Account
from .data import NS
from .feed import parse_timestamp


class AlpacaClient:
    def __init__(self, mode="broker-paper"):
        if mode not in ("broker-paper", "live"):
            raise ValueError("broker mode must be broker-paper or live")
        self.mode = mode
        self.base = (
            "https://api.alpaca.markets" if mode == "live" else "https://paper-api.alpaca.markets"
        )
        self.key, self.secret = (
            os.environ.get("ALPACA_API_KEY"),
            os.environ.get("ALPACA_SECRET_KEY"),
        )
        if not self.key or not self.secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

    def request(self, method, path, body=None):
        if not path.startswith("/v2/") or ".." in path:
            raise ValueError("invalid broker path")
        request = Request(
            self.base + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=5) as response:
                content = response.read()
                return json.loads(content) if content else None
        except HTTPError as error:
            raise RuntimeError(f"broker {method} request rejected: HTTP {error.code}") from None
        except (URLError, TimeoutError):
            # A timed-out POST has an unknown outcome: do not blindly retry it.
            raise RuntimeError(
                f"broker {method} outcome unknown; reconcile before restarting"
            ) from None


@dataclass
class BrokerOrder:
    id: str
    client_id: str
    target: int
    direction: int
    submitted: int
    signal_mid: float
    filled: float = 0
    notional: float = 0


class AlpacaBroker:
    TERMINAL = {"filled", "canceled", "expired", "rejected"}

    def __init__(
        self,
        client,
        symbol,
        risk,
        costs,
        *,
        state_path,
        initial_cash,
        enable_live=False,
        evidence=None,
        clock=time.time_ns,
    ):
        if client.mode == "live" and not (
            enable_live and evidence and evidence.get("passed") is True
        ):
            raise ValueError("live mode requires explicit activation and passing paper evidence")
        if not symbol.isascii() or not symbol.replace(".", "").isalpha():
            raise ValueError("invalid stock symbol")
        self.client, self.symbol, self.risk, self.clock = client, symbol, risk, clock
        self.pending, self.last_rejection, self.last_signal_mid = None, None, None
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.state_path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("another trader owns this account state") from None
        try:
            snapshot = client.request("GET", "/v2/account")
            positions = client.request("GET", "/v2/positions")
            orders = client.request("GET", "/v2/orders?status=open&limit=500")
            if (
                snapshot["status"] != "ACTIVE"
                or snapshot.get("trading_blocked")
                or snapshot.get("account_blocked")
            ):
                raise ValueError("broker account is not active for trading")
            if positions or orders:
                raise ValueError(
                    "dedicated broker account must start flat with no open orders; reconcile first"
                )
            cash = float(snapshot["cash"])
            if not math.isfinite(cash) or abs(cash - initial_cash) > initial_cash * 0.01:
                raise ValueError("broker cash must be within 1% of model training cash")
            self.account = Account(initial_cash, costs)
            self.account.cash = cash
            self.account_id = snapshot["id"]
            asset = client.request("GET", f"/v2/assets/{symbol}")
            if not asset.get("tradable"):
                raise ValueError("symbol is not tradable")
            self.shortable = asset.get("shortable", False)
            if self.state_path.exists():
                saved = json.loads(self.state_path.read_text())
                if saved["account_id"] != self.account_id or saved["mode"] != client.mode:
                    raise ValueError("risk state belongs to a different broker account")
                risk.day, risk.day_start, risk.peak = (
                    saved["day"],
                    saved["day_start"],
                    saved["peak"],
                )
                risk.halted, risk.permanent_halt = saved["halted"], saved["permanent_halt"]
                risk.order_times = deque(saved["order_times"])
            else:
                # Prior-close equity prevents a restart from resetting the daily-loss reference.
                from .data import session_day

                risk.day = session_day(clock())
                risk.day_start = float(snapshot["last_equity"])
                risk.peak = max(initial_cash, cash, risk.day_start)
            self._save()
        except BaseException:
            self.close()
            raise

    def _save(self):
        state = {
            "account_id": self.account_id,
            "mode": self.client.mode,
            "day": self.risk.day,
            "day_start": self.risk.day_start,
            "peak": self.risk.peak,
            "halted": self.risk.halted,
            "permanent_halt": self.risk.permanent_halt,
            "order_times": list(self.risk.order_times),
            "pending": self.pending.client_id if self.pending else None,
        }
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, allow_nan=False))
        temp.replace(self.state_path)

    def submit(self, target, quote, now, *, history=(), emergency=False):
        if self.pending or abs(target - self.account.position) < 1e-9:
            return None
        now = self.clock()
        reason = self.risk.check(
            target, quote, self.account, now, history=history, count_rate=not emergency
        )
        if reason and not (emergency and target == 0):
            self.last_rejection = reason
            self._save()
            return reason
        if target < 0 and not self.shortable:
            return "symbol_not_shortable"
        if not self.client.request("GET", "/v2/clock")["is_open"]:
            return "market_closed"
        # Re-check after network work so a quote cannot become stale while checking the clock.
        reason = self.risk.check(
            target, quote, self.account, self.clock(), history=history, count_rate=not emergency
        )
        if reason and not (emergency and target == 0):
            return reason
        delta = target - self.account.position
        direction = 1 if delta > 0 else -1
        client_id = "hft-" + uuid.uuid4().hex
        price = quote.ask if direction > 0 else quote.bid
        price *= 1 + direction * self.account.costs.slippage_bps / 10_000
        # Round toward a bounded execution price, respecting stock tick precision.
        price = math.ceil(price * 100) / 100 if direction > 0 else math.floor(price * 100) / 100
        body = {
            "symbol": self.symbol,
            "qty": f"{abs(delta):g}",
            "side": "buy" if direction > 0 else "sell",
            "type": "limit",
            "time_in_force": "day",
            "limit_price": f"{price:.2f}",
            "client_order_id": client_id,
        }
        if emergency:
            body.pop("limit_price")
            body["type"] = "market"
        self.pending = BrokerOrder("", client_id, target, direction, now, quote.mid)
        self.risk.record_order(now)
        self._save()  # unknown POST outcomes remain explicitly recorded
        order = self.client.request("POST", "/v2/orders", body)
        self.pending.id = order["id"]
        self._save()
        return None

    def _poll(self):
        if not self.pending:
            return None
        order = self.pending
        if not order.id:
            raise RuntimeError("unresolved order submission; account requires reconciliation")
        response = self.client.request("GET", f"/v2/orders/{order.id}")
        quantity = float(response["filled_qty"])
        notional = quantity * float(response["filled_avg_price"] or 0)
        if quantity < order.filled or not math.isfinite(quantity) or not math.isfinite(notional):
            raise RuntimeError("invalid cumulative broker execution")
        fill = None
        if quantity > order.filled:
            delta = quantity - order.filled
            price = (notional - order.notional) / delta
            timestamp = (
                parse_timestamp(response["filled_at"])
                if response.get("filled_at")
                else self.clock()
            )
            fill = self.account.execute(
                order.direction * delta, price, delta * self.account.costs.commission, timestamp
            )
            order.filled, order.notional = quantity, notional
            self.last_signal_mid = order.signal_mid
        if response["status"] in self.TERMINAL:
            self.pending = None
            if response["status"] == "rejected":
                self.last_rejection = "broker_rejected"
        elif self.clock() - order.submitted > 2 * NS:
            self.client.request("DELETE", f"/v2/orders/{order.id}")
            # Still poll until cancellation is terminal; an in-flight fill may race cancellation.
        self._save()
        return fill

    def _reconcile(self):
        positions = self.client.request("GET", "/v2/positions")
        if any(p["symbol"] != self.symbol for p in positions):
            raise RuntimeError("unexpected position in dedicated account")
        position = sum(float(p["qty"]) for p in positions)
        if not math.isfinite(position) or abs(position - self.account.position) > 1e-6:
            raise RuntimeError(
                "broker position differs from fill ledger; reconcile before continuing"
            )
        snapshot = self.client.request("GET", "/v2/account")
        cash = float(snapshot["cash"])
        if not math.isfinite(cash):
            raise RuntimeError("invalid broker cash")
        # Configured fees conservatively approximate broker regulatory fees.
        tolerance = 1 + sum(f.fee for f in self.account.fills)
        if abs(cash - self.account.cash) > tolerance:
            raise RuntimeError("broker cash differs from fill ledger; reconcile before continuing")

    def on_quote(self, quote, now, *, history=()):
        fill = self._poll()
        self.risk.observe(quote, self.account)
        if not self.pending:
            self._reconcile()
        if self.risk.halted:
            if self.pending:
                self.client.request("DELETE", f"/v2/orders/{self.pending.id}")
            elif self.account.position:
                self.submit(0, quote, now, emergency=True)
        self._save()
        return fill

    def flatten(self, quote):
        """Cancel, reconcile final fills, then close the actual remaining position."""
        deadline = time.monotonic() + 20
        if self.pending and self.pending.id:
            self.client.request("DELETE", f"/v2/orders/{self.pending.id}")
        fills = []
        while self.pending and time.monotonic() < deadline:
            fill = self._poll()
            if fill:
                fills.append(fill)
            if self.pending:
                time.sleep(0.1)
        if self.pending:
            raise RuntimeError(
                "order cancellation unresolved; position requires manual reconciliation"
            )
        if self.account.position:
            reason = self.submit(0, quote, self.clock(), emergency=True)
            if reason:
                raise RuntimeError(f"emergency liquidation rejected: {reason}")
            while self.pending and time.monotonic() < deadline:
                fill = self._poll()
                if fill:
                    fills.append(fill)
                if self.pending:
                    time.sleep(0.1)
        if self.account.position or self.pending:
            raise RuntimeError("liquidation unresolved; broker position may remain open")
        self._reconcile()
        return fills  # caller logs each partial execution

    def close(self):
        if getattr(self, "lock", None) and not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
