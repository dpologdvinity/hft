"""Cash, inventory, commissions and closed-trade accounting shared by all modes."""

from dataclasses import dataclass
import math

from .data import Bar


@dataclass(frozen=True)
class Costs:
    commission: float = .005  # dollars / share
    slippage_bps: float = 1.0  # adverse to bid/ask

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.commission, self.slippage_bps)):
            raise ValueError('costs must be finite and nonnegative')


@dataclass(frozen=True)
class Fill:
    timestamp: int
    quantity: float  # signed shares
    price: float
    fee: float


@dataclass(frozen=True)
class Trade:
    opened: int
    closed: int
    pnl: float


class Account:
    def __init__(self, initial_cash=10_000.0, costs=None):
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError('initial cash must be finite and positive')
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.costs = costs or Costs()
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_fees = 0.0
        self.opened = 0
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []

    @property
    def closed_pnl(self):
        return [trade.pnl for trade in self.trades]

    def equity(self, price):
        return self.cash + self.position * price

    def unrealized(self, price):
        return self.position * (price - self.entry_price) - self.entry_fees

    def target(self, target: int, quote: Bar):
        if isinstance(target, bool) or target not in (-1, 0, 1):
            raise ValueError('target must be -1, 0, or 1')
        delta = target - self.position
        if abs(delta) < 1e-9:
            return None
        price = (quote.ask if delta > 0 else quote.bid)
        price *= 1 + math.copysign(self.costs.slippage_bps / 10_000, delta)
        return self.execute(delta, price, abs(delta) * self.costs.commission, quote.timestamp)

    def execute(self, delta, price, fee, timestamp) -> Fill:
        """Book an actual (possibly partial) execution, never an order acknowledgement."""
        if not all(math.isfinite(v) for v in (delta, price, fee)) or delta == 0 or price <= 0 or fee < 0:
            raise ValueError('invalid execution')
        old = self.position
        closing = min(abs(old), abs(delta)) if old * delta < 0 else 0
        opening = abs(delta) - closing
        if closing:
            old_fee = self.entry_fees * closing / abs(old)
            close_fee = fee * closing / abs(delta)
            pnl = math.copysign(closing, old) * (price - self.entry_price) - old_fee - close_fee
            self.trades.append(Trade(self.opened, int(timestamp), pnl))
            self.entry_fees -= old_fee
        new = old + delta
        if opening:
            remainder = abs(old) - closing
            self.entry_price = (self.entry_price * remainder + price * opening) / (remainder + opening)
            self.entry_fees += fee * opening / abs(delta)
            if remainder == 0:
                self.opened = int(timestamp)
        if abs(new) < 1e-9:
            new, self.entry_price, self.entry_fees = 0.0, 0.0, 0.0
        self.cash -= delta * price + fee
        self.position = new
        fill = Fill(int(timestamp), float(delta), float(price), float(fee))
        self.fills.append(fill)
        return fill
