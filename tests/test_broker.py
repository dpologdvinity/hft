from dataclasses import replace
from decimal import Decimal

import pytest

from hft.account import Costs
from hft.broker import AlpacaBroker
from hft.calendar import SessionWindow
from hft.data import NS, Quote
from hft.risk import RiskGateway, Snapshot
from hft.sizing import SizingConfig, make_intent

NOW = 1_735_830_000 * NS


class Exchange:
    mode = "broker-paper"

    def __init__(self):
        self.cash = Decimal(1000)
        self.qty = Decimal(0)
        self.order = None
        self.posts = 0
        self.timeout = False

    def request(self, method, path, body=None):
        if path == "/v2/account":
            return {
                "id": "fixture",
                "status": "ACTIVE",
                "cash": str(self.cash),
                "non_marginable_buying_power": str(self.cash),
            }
        if path == "/v2/clock":
            return {"is_open": True}
        if path == "/v2/assets/AAPL":
            return {"tradable": True, "fractionable": True, "asset_class": "us_equity"}
        if path == "/v2/positions":
            return [{"symbol": "AAPL", "qty": str(self.qty)}] if self.qty else []
        if path.startswith("/v2/orders?"):
            return (
                [self.order]
                if self.order and self.order["status"] not in ("filled", "canceled")
                else []
            )
        if method == "POST":
            self.posts += 1
            self.order = {
                **body,
                "id": "order",
                "filled_qty": "0",
                "filled_avg_price": None,
                "status": "new",
            }
            if self.timeout:
                raise RuntimeError("timeout")
            return self.order
        if ":by_client_order_id?" in path:
            return self.order
        if method == "DELETE":
            return None  # cancel ack is NOT terminal
        if path == "/v2/orders/order":
            return self.order
        raise AssertionError((method, path, body))


def setup(tmp_path, exchange=None, **kwargs):
    exchange = exchange or Exchange()
    broker = AlpacaBroker(
        exchange,
        "AAPL",
        RiskGateway(),
        Costs(0, 0),
        state_path=tmp_path,
        initial_cash=500,
        clock=lambda: NOW,
        **kwargs,
    )
    q = Quote("q", NOW, NOW, Decimal("99.99"), Decimal("100.01"), Decimal(10), Decimal(10))
    session = SessionWindow("2025-01-02", NOW - NS, NOW + 1000 * NS)
    snap = lambda: Snapshot(
        q,
        session,
        broker.account.position,
        broker.account.mark(q.mid),
        broker.account.initial_cash,
        broker.account.cash,
        (),
    )
    intent = make_intent(
        1, broker.account, q, SizingConfig(slippage_bps=0, fee_per_share=0), NOW, symbol="AAPL"
    )
    return exchange, broker, snap, intent


def test_partial_cumulative_fills_and_cancel_ack(tmp_path):
    exchange, broker, snap, intent = setup(tmp_path)
    try:
        assert broker.submit(intent, snap, NOW).allowed
        assert broker.account.position == 0
        exchange.order.update(filled_qty=".2", filled_avg_price="100", status="partially_filled")
        exchange.qty = Decimal(".2")
        exchange.cash = Decimal(980)
        fills = broker.poll()
        assert fills[0].signed_quantity == Decimal(".2")
        assert broker.poll() == ()
        broker.cancel()
        assert broker.pending is not None
        exchange.order.update(
            filled_qty=str(intent.quantity), filled_avg_price="100.01", status="filled"
        )
        exchange.qty = intent.quantity
        exchange.cash = Decimal(1000) - intent.quantity * Decimal("100.01")
        broker.poll()
        assert broker.pending is None
        assert broker.account.position == intent.quantity
        broker.reconcile()
    finally:
        broker.close()


def test_unknown_post_lookup_does_not_retry_and_restart_books_once(tmp_path):
    exchange = Exchange()
    exchange.timeout = True
    exchange, broker, snap, intent = setup(tmp_path, exchange)
    assert broker.submit(intent, snap, NOW).allowed
    assert exchange.posts == 1 and broker.pending
    broker.close()
    exchange.order.update(filled_qty=str(intent.quantity), filled_avg_price="100", status="filled")
    exchange.qty = intent.quantity
    exchange.cash = Decimal(1000) - intent.quantity * 100
    _, recovered, _, _ = setup(tmp_path, exchange)
    try:
        assert recovered.account.position == intent.quantity
        assert recovered.poll() == ()
        assert exchange.posts == 1
    finally:
        recovered.close()


def test_same_account_lock_even_different_state_path(tmp_path):
    exchange, first, _, _ = setup(tmp_path / "one")
    try:
        with pytest.raises(RuntimeError, match="owned"):
            setup(tmp_path / "two", exchange)
    finally:
        first.close()


def test_reject_other_positions_and_forged_live_evidence(tmp_path):
    exchange = Exchange()
    exchange.qty = Decimal(1)
    with pytest.raises(ValueError, match="flat"):
        setup(tmp_path, exchange)
    exchange = Exchange()
    exchange.mode = "live"
    with pytest.raises(ValueError, match="live"):
        setup(tmp_path, exchange, enable_live=True, evidence={"passed": True})


def test_stale_snapshot_at_send_and_limit_precision(tmp_path):
    exchange, broker, snap, intent = setup(tmp_path)
    try:
        stale = lambda: replace(snap(), quote=replace(snap().quote, event_ns=NOW - 3 * NS))
        assert not broker.submit(intent, stale, NOW).allowed
        assert exchange.posts == 0
        assert broker.submit(intent, snap, NOW).allowed
        assert Decimal(exchange.order["limit_price"]) <= intent.limit_price
        assert exchange.order["type"] == "limit"
    finally:
        broker.close()


def test_failed_cancel_can_retry_and_emergency_bypasses_stale_quote(tmp_path):
    exchange, broker, snap, intent = setup(tmp_path)
    try:
        broker.submit(intent, snap, NOW)
        original = exchange.request
        fail = [True]

        def request(method, path, body=None):
            if method == "DELETE" and fail.pop():
                raise RuntimeError("cancel timeout")
            return original(method, path, body)

        exchange.request = request
        with pytest.raises(RuntimeError, match="timeout"):
            broker.cancel()
        assert broker.pending.cancel_requested is False
        exchange.request = original
        broker.cancel()
        assert broker.pending is not None
        exchange.order["status"] = "canceled"
        broker.poll()
        exchange.qty = broker.account.position = Decimal(".1")
        stale = lambda: replace(snap(), quote=replace(snap().quote, event_ns=NOW - 10 * NS))
        exit_intent = replace(intent, side="sell", quantity=Decimal(".1"), emergency=True)
        assert broker.submit(exit_intent, stale, NOW).allowed
        assert exchange.order["type"] == "market"
    finally:
        broker.close()


def test_unknown_post_remains_durable_and_blocks_second_submission(tmp_path):
    exchange = Exchange()
    exchange.timeout = True
    exchange, broker, snap, intent = setup(tmp_path, exchange)
    original = exchange.request

    def request(method, path, body=None):
        if ":by_client_order_id?" in path:
            raise RuntimeError("lookup unavailable")
        return original(method, path, body)

    exchange.request = request
    try:
        with pytest.raises(RuntimeError, match="outcome unknown"):
            broker.submit(intent, snap, NOW)
        assert broker.store.load()["pending"]["unknown"] is True
        assert not broker.submit(intent, snap, NOW).allowed
        assert exchange.posts == 1
    finally:
        broker.close()
    with pytest.raises(RuntimeError, match="lookup unavailable"):
        setup(tmp_path, exchange)
    assert exchange.posts == 1
