"""Exact long-only ledger; execution acknowledgements never change inventory."""

import math
from dataclasses import dataclass
from decimal import Decimal


def decimal(value):
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True)
class Costs:
    commission: float = 0.005
    slippage_bps: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.commission, self.slippage_bps)):
            raise ValueError("invalid costs")


@dataclass(frozen=True)
class Execution:
    execution_id: str
    client_order_id: str
    timestamp_ns: int
    signed_quantity: Decimal
    price: Decimal
    fee: Decimal

    @property
    def quantity(self):
        return self.signed_quantity

    @property
    def timestamp(self):
        return self.timestamp_ns


@dataclass(frozen=True)
class CompletedTrade:
    entry_ns: int
    exit_ns: int
    pnl: Decimal
    opening_notional: Decimal
    execution_ids: tuple[str, ...]

    @property
    def opened(self):
        return self.entry_ns

    @property
    def closed(self):
        return self.exit_ns


class Account:
    def __init__(self, initial_cash=500, costs=None):
        self.initial_cash = decimal(initial_cash)
        if not self.initial_cash.is_finite() or self.initial_cash <= 0:
            raise ValueError("initial cash must be positive and finite")
        self.cash = self.initial_cash
        self.session_start_equity = self.initial_cash
        self.costs = costs or Costs()
        # Only ledgers that mirror a broker set this: a confirmed fill is a fact even when
        # a market order cost more than the cash it was sized for.
        self.allow_overdraft = False
        self.position = Decimal(0)
        self.entry_price = Decimal(0)
        self.entry_fees = Decimal(0)
        self.realized_pnl = Decimal(0)
        self.opened = 0
        self.fills = []
        self.cash_flows = []
        self.income = []
        self._seen = {}
        self._trades = []
        self._roundtrip_pnl = Decimal(0)
        self._opening_notional = Decimal(0)
        self._execution_ids = []

    @property
    def completed_trades(self):
        return tuple(self._trades)

    @property
    def trades(self):
        return self.completed_trades

    @property
    def closed_pnl(self):
        return [float(t.pnl) for t in self._trades]

    @property
    def last_fill_ns(self):
        return self.fills[-1].timestamp_ns if self.fills else None

    def mark(self, price):
        price = decimal(price)
        if not price.is_finite() or price <= 0:
            raise ValueError("invalid mark")
        return self.cash + self.position * price

    equity = mark

    def unrealized(self, price):
        return self.position * (decimal(price) - self.entry_price) - self.entry_fees

    def apply(self, execution):
        old = self._seen.get(execution.execution_id)
        if old is not None:
            if old != execution:
                raise ValueError("inconsistent duplicate execution")
            return
        q, p, f = map(decimal, (execution.signed_quantity, execution.price, execution.fee))
        if (
            not execution.execution_id
            or not all(v.is_finite() for v in (q, p, f))
            or q == 0
            or p <= 0
            or f < 0
        ):
            raise ValueError("invalid execution")
        overdrawn = self.cash - q * p - f < 0 and not self.allow_overdraft
        if self.position + q < 0 or overdrawn:
            raise ValueError("insufficient inventory or cash")
        if q > 0:
            if self.position == 0:
                self.opened = execution.timestamp_ns
                self._roundtrip_pnl = Decimal(0)
                self._opening_notional = Decimal(0)
                self._execution_ids = []
            self.entry_price = (self.position * self.entry_price + q * p) / (self.position + q)
            self.entry_fees += f
            self._opening_notional += q * p
        else:
            opening_fee = self.entry_fees * (-q) / self.position
            pnl = (-q) * (p - self.entry_price) - opening_fee - f
            self.realized_pnl += pnl
            self._roundtrip_pnl += pnl
            self.entry_fees -= opening_fee
        self.cash -= q * p + f
        self.position += q
        self._execution_ids.append(execution.execution_id)
        self._seen[execution.execution_id] = execution
        self.fills.append(execution)
        if self.position == 0:
            self._trades.append(
                CompletedTrade(
                    self.opened,
                    execution.timestamp_ns,
                    self._roundtrip_pnl,
                    self._opening_notional,
                    tuple(self._execution_ids),
                )
            )
            self.entry_price = self.entry_fees = Decimal(0)

    def record_cash_flow(self, amount, timestamp_ns):
        amount = decimal(amount)
        if (
            not amount.is_finite()
            or amount == 0
            or self.cash + amount < 0
            or self.initial_cash + amount <= 0
        ):
            raise ValueError("invalid cash flow")
        self.cash += amount
        self.initial_cash += amount
        self.session_start_equity += amount
        self.cash_flows.append({"amount": str(amount), "timestamp_ns": int(timestamp_ns)})

    def record_income(self, amount, timestamp_ns, source):
        """Cash the broker paid on a holding (e.g. a dividend): profit, not a deposit."""
        amount = decimal(amount)
        if not amount.is_finite():
            raise ValueError("invalid income")
        self.cash += amount
        self.realized_pnl += amount
        self.income.append(
            {"amount": str(amount), "timestamp_ns": int(timestamp_ns), "source": str(source)}
        )

    def execute(self, delta, price, fee, timestamp):
        execution = Execution(
            f"local-{len(self.fills)}",
            "local",
            int(timestamp),
            decimal(delta),
            decimal(price),
            decimal(fee),
        )
        self.apply(execution)
        return execution

    def to_state(self):
        from dataclasses import asdict

        def strings(row):
            return {k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(row).items()}

        return {
            "initial_cash": str(self.initial_cash),
            "session_start_equity": str(self.session_start_equity),
            "cash": str(self.cash),
            "position": str(self.position),
            "entry_price": str(self.entry_price),
            "entry_fees": str(self.entry_fees),
            "realized_pnl": str(self.realized_pnl),
            "opened": self.opened,
            "roundtrip_pnl": str(self._roundtrip_pnl),
            "opening_notional": str(self._opening_notional),
            "execution_ids": self._execution_ids,
            "fills": [strings(f) for f in self.fills],
            "trades": [strings(t) for t in self._trades],
            "costs": asdict(self.costs),
            "cash_flows": self.cash_flows,
            "income": self.income,
        }

    @classmethod
    def from_state(cls, state):
        account = cls(state["initial_cash"], Costs(**state["costs"]))
        for field in (
            "session_start_equity",
            "cash",
            "position",
            "entry_price",
            "entry_fees",
            "realized_pnl",
        ):
            setattr(account, field, decimal(state[field]))
        account.cash_flows = list(state.get("cash_flows", []))
        account.income = list(state.get("income", []))
        account.opened = state["opened"]
        account._roundtrip_pnl = decimal(state["roundtrip_pnl"])
        account._opening_notional = decimal(state["opening_notional"])
        account._execution_ids = list(state["execution_ids"])
        account.fills = [
            Execution(
                **{
                    **f,
                    "signed_quantity": decimal(f["signed_quantity"]),
                    "price": decimal(f["price"]),
                    "fee": decimal(f["fee"]),
                }
            )
            for f in state["fills"]
        ]
        account._seen = {f.execution_id: f for f in account.fills}
        account._trades = [
            CompletedTrade(
                **{
                    **t,
                    "pnl": decimal(t["pnl"]),
                    "opening_notional": decimal(t["opening_notional"]),
                    "execution_ids": tuple(t["execution_ids"]),
                }
            )
            for t in state["trades"]
        ]
        if (
            account.cash < 0
            or account.position < 0
            or any(
                not getattr(account, k).is_finite()
                for k in ("cash", "position", "entry_price", "entry_fees")
            )
        ):
            raise ValueError("invalid restored ledger")
        return account
