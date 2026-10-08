"""Serialized Alpaca order lifecycle; persist IDs before POST, book actual fills only."""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .account import Account, Execution, decimal
from .feed import parse_timestamp
from .logs import canonical_hash, json_value
from .risk import RiskDecision
from .sizing import OrderIntent
from .state import AccountStateStore

NS = 1_000_000_000


# Statuses Alpaca uses to refuse an order outright (invalid, insufficient funds).
REJECTION_STATUSES = frozenset({400, 403, 422})


# Alpaca books each fill's cash to the cent while the ledger keeps exact decimals, so
# the two drift by up to half a cent per fill.
CASH_TOLERANCE = Decimal(".01")
ROUNDING_PER_FILL = Decimal(".005")


def cash_tolerance(fills):
    return CASH_TOLERANCE + ROUNDING_PER_FILL * fills


class BrokerIntegrityError(RuntimeError):
    """A broker report that contradicts the ledger; retrying cannot fix it."""


class BrokerHTTPError(RuntimeError):
    """An HTTP error response; the status lets callers tell refusals from unknowns."""

    def __init__(self, status, method):
        super().__init__(f"broker {method} HTTP {status}; reconcile outcome")
        self.status = status


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("broker redirect refused")


class AlpacaClient:
    def __init__(self, mode="broker-paper"):
        if mode not in ("broker-paper", "live"):
            raise ValueError("invalid broker mode")
        self.mode = mode
        self.base = (
            "https://api.alpaca.markets" if mode == "live" else "https://paper-api.alpaca.markets"
        )
        prefix = "ALPACA_LIVE_" if mode == "live" else "ALPACA_PAPER_"
        self.key = os.environ.get(prefix + "API_KEY")
        self.secret = os.environ.get(prefix + "SECRET_KEY")
        if not self.key or not self.secret:
            raise ValueError(f"{prefix}API_KEY and {prefix}SECRET_KEY required")
        self.transport = build_opener(_NoRedirect())

    def request(self, method, path, body=None):
        if (
            method not in ("GET", "POST", "DELETE")
            or not path.startswith("/v2/")
            or ".." in path
            or "#" in path
        ):
            raise ValueError("invalid broker request")
        request = Request(
            self.base + path,
            method=method,
            data=json.dumps(body, allow_nan=False).encode() if body is not None else None,
            headers={
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "Content-Type": "application/json",
            },
        )
        try:
            with self.transport.open(request, timeout=5) as response:
                content = response.read()
                return json.loads(content) if content else None
        except HTTPError as error:
            raise BrokerHTTPError(error.code, method) from None
        except (URLError, TimeoutError, ConnectionError):
            raise RuntimeError(f"broker {method} outcome unknown; reconcile before retry") from None


@dataclass
class BrokerOrder:
    intent: OrderIntent
    id: str | None = None
    filled: Decimal = Decimal(0)
    notional: Decimal = Decimal(0)
    fee: Decimal = Decimal(0)
    unknown: bool = False
    cancel_requested: bool = False

    def state(self):
        return json_value(asdict(self))

    @classmethod
    def restore(cls, state):
        intent = state["intent"]
        intent = OrderIntent(
            **{
                **intent,
                "quantity": decimal(intent["quantity"]),
                "limit_price": decimal(intent["limit_price"]),
            }
        )
        return cls(
            intent,
            state["id"],
            decimal(state["filled"]),
            decimal(state["notional"]),
            decimal(state["fee"]),
            state["unknown"],
            state["cancel_requested"],
        )


class AlpacaBroker:
    TERMINAL = frozenset({"filled", "canceled", "expired", "rejected"})

    def __init__(
        self,
        client,
        symbol,
        risk,
        costs,
        *,
        state_path=None,
        initial_cash=500,
        enable_live=False,
        evidence=None,
        model_path=None,
        logs_path=None,
        max_live_notional=None,
        contract_hash=None,
        clock=time.time_ns,
    ):
        if client.mode == "live":
            if not enable_live or evidence is not None or not model_path or not logs_path:
                raise ValueError(
                    "live requires explicit activation and independently verified evidence paths"
                )
            from .evidence import graduate
            from .runtime import fetch_calendar

            checked = graduate(
                model_path,
                logs_path,
                now_ns=clock(),
                max_notional=max_live_notional,
                calendar_records=fetch_calendar(client),
            )
            if not checked["passed"]:
                raise ValueError("live evidence failed: " + ", ".join(checked["reasons"]))
            if max_live_notional is None or not 0 < max_live_notional <= 10:
                raise ValueError("live requires canary ceiling <= $10")
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
            raise ValueError("invalid stock symbol")
        self.client, self.symbol, self.risk, self.clock = client, symbol, risk, clock
        self.contract_hash = contract_hash or canonical_hash(
            {
                "symbol": symbol,
                "costs": asdict(costs),
                "risk": asdict(risk.config),
                "capital": initial_cash,
            }
        )
        self.max_live_notional = max_live_notional
        self.pending = None
        self.rejections = 0
        self.recovered_fills = ()
        self.last_error = None
        self.live_sessions = 0
        self.account_snapshot = client.request("GET", "/v2/account")
        self.account_id = self.account_snapshot["id"]
        # One fixed account lock regardless of user-selected log/state paths.
        self.store = AccountStateStore(self.account_id, client.mode, root=state_path)
        self.store.__enter__()
        try:
            self._active_account(self.account_snapshot)
            asset = client.request("GET", f"/v2/assets/{symbol}")
            if (
                not asset.get("tradable")
                or not asset.get("fractionable")
                or asset.get("class", asset.get("asset_class")) != "us_equity"
            ):
                raise ValueError("stock must be tradable and fractionable")
            state = self.store.load()
            if state:
                if state["contract_hash"] != self.contract_hash or state["symbol"] != symbol:
                    raise ValueError("saved account belongs to another execution contract")
                self.account = Account.from_state(state["account"])
                if self.account.initial_cash != decimal(initial_cash):
                    raise ValueError("capital differs from saved allocation")
                self.risk.restore(state["risk"])
                self.remote_initial_cash = decimal(state["remote_initial_cash"])
                self.live_sessions = state.get("live_sessions", 0)
                if client.mode == "live" and self.live_sessions >= 10:
                    raise ValueError("live canary review required after ten sessions")
                self.pending = BrokerOrder.restore(state["pending"]) if state["pending"] else None
            else:
                if client.request("GET", "/v2/positions") or client.request(
                    "GET", "/v2/orders?status=open&limit=500"
                ):
                    raise ValueError("dedicated account must start flat with no open orders")
                free = min(
                    decimal(self.account_snapshot["cash"]),
                    decimal(
                        self.account_snapshot.get(
                            "non_marginable_buying_power", self.account_snapshot["cash"]
                        )
                    ),
                )
                if decimal(initial_cash) > free:
                    raise ValueError("strategy capital exceeds verified non-borrowed cash")
                self.account = Account(initial_cash, costs)
                self.remote_initial_cash = decimal(self.account_snapshot["cash"])
            self.refresh()
            if self.pending:
                self.recovered_fills = self.poll()
            self.reconcile()
            self._save()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _active_account(snapshot):
        if (
            snapshot.get("status") != "ACTIVE"
            or snapshot.get("trading_blocked")
            or snapshot.get("account_blocked")
        ):
            raise ValueError("broker account is blocked")
        if not decimal(snapshot["cash"]).is_finite():
            raise ValueError("invalid broker cash")

    def _save(self):
        self.store.save(
            {
                "account_id": self.account_id,
                "mode": self.client.mode,
                "symbol": self.symbol,
                "contract_hash": self.contract_hash,
                "account": self.account.to_state(),
                "risk": self.risk.to_state(),
                "remote_initial_cash": str(self.remote_initial_cash),
                "live_sessions": self.live_sessions,
                "pending": self.pending.state() if self.pending else None,
            }
        )

    def refresh(self):
        self.account_snapshot = self.client.request("GET", "/v2/account")
        self._active_account(self.account_snapshot)
        self.market_clock = self.client.request("GET", "/v2/clock")
        self.refreshed_ns = self.clock()

    def submit(self, intent, snapshot, now_ns=None):
        now = self.clock() if now_ns is None else now_ns
        if self.pending:
            return RiskDecision(False, "order_outstanding")
        if (
            not intent.emergency
            and now - intent.created_ns >= self.risk.config.order_expiry_seconds * NS
        ):
            return RiskDecision(False, "expired_intent")
        if now - self.refreshed_ns > 5 * NS:
            return RiskDecision(False, "stale_broker_snapshot")
        if not self.market_clock.get("is_open"):
            return RiskDecision(False, "market_closed")
        if intent.symbol != self.symbol:
            return RiskDecision(False, "symbol_mismatch")
        snap = snapshot() if callable(snapshot) else snapshot
        emergency = False
        if intent.emergency and intent.side == "sell":
            positions = self.client.request("GET", "/v2/positions")
            qty = sum(
                (decimal(p["qty"]) for p in positions if p["symbol"] == self.symbol), Decimal(0)
            )
            if qty != self.account.position or not 0 < intent.quantity <= qty:
                return RiskDecision(False, "inventory_mismatch")
            emergency = True
        decision = RiskDecision(True) if emergency else self.risk.evaluate(intent, snap, now)
        if not decision.allowed:
            return decision
        if intent.side == "buy":
            free = min(
                decimal(self.account_snapshot["cash"]),
                decimal(
                    self.account_snapshot.get(
                        "non_marginable_buying_power", self.account_snapshot["cash"]
                    )
                ),
            )
            if intent.quantity * intent.limit_price > free:
                return RiskDecision(False, "broker_cash_limit")
            if self.client.mode == "live" and intent.quantity * intent.limit_price > decimal(
                self.max_live_notional
            ):
                return RiskDecision(False, "canary_limit")
        step = Decimal(".01") if intent.limit_price >= 1 else Decimal(".0001")
        price = decimal(intent.limit_price).quantize(
            step, rounding=ROUND_DOWN if intent.side == "buy" else ROUND_UP
        )
        if price <= 0:
            return RiskDecision(False, "invalid_limit")
        intent = replace(intent, client_order_id="hft-" + intent.client_order_id, limit_price=price)
        self.pending = BrokerOrder(intent)
        if intent.side == "buy":
            self.risk.record_order(now)
        self._save()  # persist intent and unique ID BEFORE any network mutation
        body = {
            "symbol": self.symbol,
            "qty": str(intent.quantity),
            "side": intent.side,
            "type": "market" if intent.emergency and intent.side == "sell" else "limit",
            "time_in_force": "day",
            "client_order_id": intent.client_order_id,
            "extended_hours": False,
        }
        if body["type"] == "limit":
            body["limit_price"] = str(price)
        # Recheck immediately after durable writes and any blocking position lookup.
        now = self.clock()
        current = snapshot() if callable(snapshot) else snapshot
        final = (
            RiskDecision(True)
            if emergency
            else self.risk.evaluate(intent, current, now, count_rate=False)
        )
        if (
            now - self.refreshed_ns > 5 * NS
            or not final.allowed
            or not emergency
            and now - intent.created_ns >= self.risk.config.order_expiry_seconds * NS
        ):
            self.pending = None
            self._save()
            return RiskDecision(False, final.reason or "stale_broker_snapshot")
        return self._send(intent, body)

    def _send(self, intent, body):
        """POST a durably saved pending order; resolve an unknown outcome by client id."""
        try:
            result = self.client.request("POST", "/v2/orders", body)
            self.pending.id = result["id"]
            self._save()
            # Fills on the acknowledgement are consumed by poll once, not thrown away.
        except (RuntimeError, OSError, KeyError) as error:
            self.pending.unknown = True
            self._save()
            if self._confirmed_rejection(error, intent):
                self.pending = None
                self.rejections += 1
                self._save()
                return RiskDecision(False, f"broker_rejected:{error.status}")
            try:
                found = self.client.request(
                    "GET",
                    "/v2/orders:by_client_order_id?"
                    + urlencode({"client_order_id": intent.client_order_id}),
                )
                if not found or not found.get("id"):
                    raise RuntimeError("unknown submitted order")
                self.pending.id = found["id"]
                self.pending.unknown = False
                self._save()
            except (RuntimeError, OSError, KeyError):
                raise RuntimeError(
                    "submitted order outcome unknown; ID saved; do not retry"
                ) from None
        return RiskDecision(True)

    def _confirmed_rejection(self, error, intent):
        # Only a refusal status plus "no such order" proves nothing reached the book.
        if getattr(error, "status", None) not in REJECTION_STATUSES:
            return False
        try:
            self.client.request(
                "GET",
                "/v2/orders:by_client_order_id?"
                + urlencode({"client_order_id": intent.client_order_id}),
            )
        except BrokerHTTPError as lookup:
            return lookup.status == 404
        except (RuntimeError, OSError):
            return False
        return False

    def apply_update(self, row, *, allow_old=False):
        if self.pending is None:
            return ()
        order = self.pending
        intent = order.intent
        if (
            row.get("client_order_id") != intent.client_order_id
            or row.get("symbol") != self.symbol
            or row.get("side") != intent.side
        ):
            raise BrokerIntegrityError("broker order identity mismatch")
        filled = decimal(row["filled_qty"])
        if not filled.is_finite() or filled < 0 or filled > intent.quantity:
            raise BrokerIntegrityError("broker overfill or invalid quantity")
        if filled < order.filled:
            if allow_old:
                return ()
            raise BrokerIntegrityError("broker cumulative fill moved backward")
        executions = ()
        if filled > order.filled:
            average = decimal(row["filled_avg_price"])
            total = filled * average
            increment = filled - order.filled
            price = (total - order.notional) / increment
            commission = decimal(row.get("commission") or 0)
            fee = commission - order.fee
            if price <= 0 or not price.is_finite() or fee < 0:
                raise BrokerIntegrityError("invalid cumulative execution price/fee")
            stamp = row.get("filled_at") or row.get("updated_at")
            timestamp = parse_timestamp(stamp) if stamp else self.clock()
            execution = Execution(
                f"{intent.client_order_id}:{filled}",
                intent.client_order_id,
                timestamp,
                increment if intent.side == "buy" else -increment,
                price,
                fee,
            )
            self.account.apply(execution)
            order.filled, order.notional, order.fee = filled, total, commission
            executions = (execution,)
        if row.get("status") in self.TERMINAL:
            self.pending = None
        elif self.pending:
            self.pending.cancel_requested = False
        self._save()
        return executions

    def poll(self):
        if self.pending is None:
            return ()
        if not self.pending.id:
            row = self.client.request(
                "GET",
                "/v2/orders:by_client_order_id?"
                + urlencode({"client_order_id": self.pending.intent.client_order_id}),
            )
            if not row or not row.get("id"):
                raise RuntimeError("unknown order needs manual reconciliation")
            self.pending.id = row["id"]
            self.pending.unknown = False
            self._save()
        row = self.client.request("GET", f"/v2/orders/{self.pending.id}")
        return self.apply_update(row)

    def cancel(self):
        if self.pending and self.pending.id and not self.pending.cancel_requested:
            self.pending.cancel_requested = True
            self._save()
            try:
                self.client.request("DELETE", f"/v2/orders/{self.pending.id}")
            except (RuntimeError, OSError):
                self.pending.cancel_requested = False
                self._save()
                raise
            # HTTP cancellation ack does not confirm final status or fill quantity.

    def reconcile(self):
        positions = self.client.request("GET", "/v2/positions")
        if any(p["symbol"] != self.symbol or decimal(p["qty"]) < 0 for p in positions):
            raise RuntimeError("unrelated or short broker position")
        quantity = sum((decimal(p["qty"]) for p in positions), Decimal(0))
        if quantity != self.account.position:
            raise RuntimeError("broker inventory mismatch")
        orders = self.client.request("GET", "/v2/orders?status=open&limit=500")
        if any(
            not self.pending or o.get("client_order_id") != self.pending.intent.client_order_id
            for o in orders
        ):
            raise RuntimeError("unrelated broker order")
        snapshot = self.client.request("GET", "/v2/account")
        self._active_account(snapshot)
        expected = self.remote_initial_cash + self.account.cash - self.account.initial_cash
        if abs(decimal(snapshot["cash"]) - expected) > cash_tolerance(len(self.account.fills)):
            raise RuntimeError(
                "broker cash/fee/deposit mismatch; entries halted for reconciliation"
            )
        self.account_snapshot = snapshot
        return True

    def close(self):
        if getattr(self, "store", None):
            self.store.__exit__()


async def trade_updates(client):
    """Dedicated stream, independently consumed; REST polling remains a recovery fallback."""
    from websockets.asyncio.client import connect

    url = (
        "wss://api.alpaca.markets/stream"
        if client.mode == "live"
        else "wss://paper-api.alpaca.markets/stream"
    )
    async with connect(url, open_timeout=10, close_timeout=2, max_queue=16) as socket:
        await socket.send(
            json.dumps({"action": "auth", "key": client.key, "secret": client.secret})
        )
        response = json.loads(await socket.recv())
        if response.get("data", {}).get("status") != "authorized":
            raise RuntimeError("trade updates authorization failed")
        await socket.send(json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}}))
        async for raw in socket:
            response = json.loads(raw)
            if response.get("stream") == "trade_updates":
                yield response["data"]
