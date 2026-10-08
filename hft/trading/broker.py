"""One Alpaca paper account trading several stocks, one order book per stock.

Each `SymbolBook` reuses `AlpacaBroker`'s tested order path (durable intent and
client order id before POST, lookup by the same id, cumulative fills, refusal
handling) without its single-stock constructor. `PortfolioBroker` owns the
account lock, saved state, startup checks and a reconciliation covering every
configured stock plus the account cash.
"""

from decimal import Decimal

from ..account import Account, decimal
from ..broker import AlpacaBroker, BrokerOrder
from ..risk import RiskDecision
from ..state import AccountStateStore

MAX_REJECTIONS_PER_SESSION = 3


class SymbolBook(AlpacaBroker):
    """Per-stock order state; account-level work is delegated to the portfolio."""

    def __init__(self, portfolio, symbol, account, risk):
        self.portfolio, self.client, self.symbol = portfolio, portfolio.client, symbol
        self.account, self.risk, self.clock = account, risk, portfolio.clock
        self.pending = None
        self.rejections = 0
        self.max_live_notional = None
        self.recovered_fills = ()
        self.last_error = None
        self.live_sessions = 0

    @property
    def refreshed_ns(self):
        return self.portfolio.refreshed_ns

    @property
    def market_clock(self):
        return self.portfolio.market_clock

    @property
    def account_snapshot(self):
        return self.portfolio.account_snapshot

    def _save(self):
        self.portfolio.save()

    def refresh(self):
        self.portfolio.refresh()

    def reconcile(self):
        return self.portfolio.reconcile()

    def submit(self, intent, snapshot, now_ns=None):
        if self.rejections >= MAX_REJECTIONS_PER_SESSION and not intent.emergency:
            return RiskDecision(False, "rejection_limit")
        return super().submit(intent, snapshot, now_ns)

    def close(self):
        """The portfolio owns the account lock."""


class PortfolioBroker:
    def __init__(
        self, client, budgets, costs, risk_factory, *, run_dir, identity, clock, book_class=None
    ):
        if client.mode != "broker-paper":
            raise ValueError("multi-stock trading runs on the paper account only")
        if not budgets:
            raise ValueError("at least one stock is required")
        self.client, self.clock, self.identity = client, clock, identity
        book_class = book_class or SymbolBook
        self.budgets = {s: decimal(b) for s, b in budgets.items()}
        self.account_snapshot = client.request("GET", "/v2/account")
        self.account_id = self.account_snapshot["id"]
        # Same lock key as broker-paper: one bot per paper account, whatever the run name.
        self.store = AccountStateStore(self.account_id, client.mode, root=run_dir)
        self.store.__enter__()
        try:
            AlpacaBroker._active_account(self.account_snapshot)
            for symbol in self.budgets:
                asset = client.request("GET", f"/v2/assets/{symbol}")
                if (
                    asset.get("status", "active") != "active"
                    or not asset.get("tradable")
                    or not asset.get("fractionable")
                    or asset.get("class", asset.get("asset_class")) != "us_equity"
                ):
                    raise ValueError(f"{symbol} must be an active, tradable, fractionable stock")
            state = self.store.load()
            self.books = {}
            # Run-level state owned by the runner (e.g. the account guard), saved with books.
            self.extra = dict(state.get("extra", {})) if state else {}
            if state:
                if state["identity"] != identity:
                    raise ValueError("run identity changed; start a new run name")
                self.remote_initial_cash = decimal(state["remote_initial_cash"])
                for symbol, saved in state["books"].items():
                    book = book_class(
                        self, symbol, Account.from_state(saved["account"]), risk_factory(symbol)
                    )
                    book.risk.restore(saved["risk"])
                    book.pending = (
                        BrokerOrder.restore(saved["pending"]) if saved["pending"] else None
                    )
                    self.books[symbol] = book
            else:
                if client.request("GET", "/v2/positions") or client.request(
                    "GET", "/v2/orders?status=open&limit=500"
                ):
                    raise ValueError(
                        "paper account must be flat with no open orders to start a run"
                    )
                free = min(
                    decimal(self.account_snapshot["cash"]),
                    decimal(
                        self.account_snapshot.get(
                            "non_marginable_buying_power", self.account_snapshot["cash"]
                        )
                    ),
                )
                if sum(self.budgets.values(), Decimal(0)) > free:
                    raise ValueError("stock budgets exceed the paper account's non-borrowed cash")
                self.remote_initial_cash = decimal(self.account_snapshot["cash"])
                for symbol, budget in self.budgets.items():
                    self.books[symbol] = book_class(
                        self, symbol, Account(budget, costs), risk_factory(symbol)
                    )
            self.refresh()
            self.recovered_fills = self.poll()
            self.reconcile()
            self.save()
        except BaseException:
            self.close()
            raise

    def book(self, symbol) -> SymbolBook:
        return self.books[symbol]

    def refresh(self):
        self.account_snapshot = self.client.request("GET", "/v2/account")
        AlpacaBroker._active_account(self.account_snapshot)
        self.market_clock = self.client.request("GET", "/v2/clock")
        self.refreshed_ns = self.clock()

    def start_session(self):
        for book in self.books.values():
            book.rejections = 0

    def poll(self):
        """Fills from every book; one failing book cannot hide another book's fills."""
        fills, self.poll_errors = [], {}
        for symbol, book in self.books.items():
            try:
                fills += [(symbol, e) for e in book.poll()]
            except (RuntimeError, OSError) as error:
                self.poll_errors[symbol] = str(error)
        return fills

    def cancel_all(self):
        """Request every cancel, then raise the first failure."""
        failure = None
        for book in self.books.values():
            try:
                book.cancel()
            except (RuntimeError, OSError) as error:
                failure = failure or error
        if failure:
            raise failure

    def reconcile(self):
        positions = self.client.request("GET", "/v2/positions")
        held = {}
        for p in positions:
            if p["symbol"] not in self.books or decimal(p["qty"]) < 0:
                raise RuntimeError(f"unexpected broker position: {p['symbol']}")
            held[p["symbol"]] = decimal(p["qty"])
        for symbol, book in self.books.items():
            if held.get(symbol, Decimal(0)) != book.account.position:
                raise RuntimeError(f"broker inventory mismatch: {symbol}")
        pending = {b.pending.intent.client_order_id: s for s, b in self.books.items() if b.pending}
        for order in self.client.request("GET", "/v2/orders?status=open&limit=500"):
            if pending.get(order.get("client_order_id")) != order.get("symbol"):
                raise RuntimeError(f"unexpected broker order: {order.get('symbol')}")
        snapshot = self.client.request("GET", "/v2/account")
        AlpacaBroker._active_account(snapshot)
        expected = self.remote_initial_cash + sum(
            (b.account.cash - b.account.initial_cash for b in self.books.values()), Decimal(0)
        )
        if abs(decimal(snapshot["cash"]) - expected) > Decimal(".01"):
            raise RuntimeError("broker cash mismatch; trading stopped for reconciliation")
        self.account_snapshot = snapshot
        return True

    def save(self):
        self.store.save(
            {
                "account_id": self.account_id,
                "mode": self.client.mode,
                "identity": self.identity,
                "remote_initial_cash": str(self.remote_initial_cash),
                "extra": getattr(self, "extra", {}),
                "books": {
                    s: {
                        "account": b.account.to_state(),
                        "risk": b.risk.to_state(),
                        "pending": b.pending.state() if b.pending else None,
                    }
                    for s, b in getattr(self, "books", {}).items()
                },
            }
        )

    def close(self):
        if getattr(self, "store", None):
            self.store.__exit__()
