import pytest

from hft.account import Costs
from hft.broker import AlpacaBroker
from hft.data import NS, Bar
from hft.risk import RiskGateway


class Exchange:
    """Transport fixture; the broker's order lifecycle and ledger stay real."""

    mode = "broker-paper"

    def __init__(self):
        self.qty = 0
        self.cash = 1000.0
        self.order = None

    def request(self, method, path, body=None):
        if path == "/v2/account":
            return {
                "id": "fixture",
                "status": "ACTIVE",
                "cash": str(self.cash),
                "equity": str(self.cash + self.qty * 100),
                "last_equity": "1000",
                "trading_blocked": False,
                "account_blocked": False,
            }
        if path == "/v2/positions":
            return [{"symbol": "AAPL", "qty": str(self.qty)}] if self.qty else []
        if path.startswith("/v2/orders?"):
            return []
        if path == "/v2/assets/AAPL":
            return {"tradable": True, "shortable": True}
        if path == "/v2/clock":
            return {"is_open": True}
        if method == "POST" and path == "/v2/orders":
            assert body["symbol"] == "AAPL" and body["qty"] == "1"
            assert body["type"] == "limit" and body["time_in_force"] == "day"
            assert body["client_order_id"].startswith("hft-")
            self.order = {
                "id": "order1",
                "status": "new",
                "filled_qty": "0",
                "filled_avg_price": None,
            }
            return self.order
        if path == "/v2/orders/order1":
            if method == "DELETE":
                self.order = dict(self.order, status="canceled")
            return self.order
        raise AssertionError((method, path, body))


def quote(ts=1_735_830_000 * NS):
    return Bar(ts, 100, 100, 100, 100, 10, 99.99, 100.01, 10, 10)


def test_acknowledgement_is_not_a_fill_and_partial_execution_reconciles(tmp_path):
    exchange = Exchange()
    broker = AlpacaBroker(
        exchange,
        "AAPL",
        RiskGateway(),
        Costs(0, 0),
        state_path=tmp_path / "state.json",
        initial_cash=1000,
        clock=lambda: quote().timestamp,
    )
    q = quote()
    assert broker.submit(1, q, q.timestamp) is None
    assert broker.account.position == 0
    assert broker.pending is not None
    exchange.qty, exchange.cash = 0.5, 949.995
    exchange.order = dict(
        exchange.order, status="partially_filled", filled_qty=".5", filled_avg_price="100.01"
    )
    fill = broker.on_quote(q, q.timestamp)
    assert fill.quantity == 0.5
    assert broker.account.position == 0.5
    assert broker.pending is not None
    exchange.qty, exchange.cash = 1, 899.99
    exchange.order = dict(
        exchange.order, status="filled", filled_qty="1", filled_avg_price="100.01"
    )
    fill = broker.on_quote(q, q.timestamp)
    assert fill.quantity == 0.5
    assert broker.pending is None
    assert broker.account.position == 1
    assert broker.on_quote(q, q.timestamp) is None
    broker.close()


def test_broker_rejects_shared_account_and_duplicate_process(tmp_path):
    exchange = Exchange()
    exchange.qty = 1
    with pytest.raises(ValueError, match="flat"):
        AlpacaBroker(
            exchange,
            "AAPL",
            RiskGateway(),
            Costs(),
            state_path=tmp_path / "bad.json",
            initial_cash=1000,
        )
    exchange.qty = 0
    first = AlpacaBroker(
        exchange,
        "AAPL",
        RiskGateway(),
        Costs(),
        state_path=tmp_path / "state.json",
        initial_cash=1000,
    )
    with pytest.raises(ValueError, match="another"):
        AlpacaBroker(
            exchange,
            "AAPL",
            RiskGateway(),
            Costs(),
            state_path=tmp_path / "state.json",
            initial_cash=1000,
        )
    first.close()


def test_live_requires_explicit_activation_and_passing_evidence(tmp_path):
    exchange = Exchange()
    exchange.mode = "live"
    with pytest.raises(ValueError, match="live"):
        AlpacaBroker(
            exchange,
            "AAPL",
            RiskGateway(),
            Costs(),
            state_path=tmp_path / "state.json",
            initial_cash=1000,
        )
