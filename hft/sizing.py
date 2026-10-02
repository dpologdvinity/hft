"""Conservative fractional target transitions and explicit marketable limits."""

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from uuid import uuid4

from .account import decimal


@dataclass(frozen=True)
class SizingConfig:
    allocation_fraction: float = 0.10
    max_entry_notional: float = 500.0
    minimum_notional: float = 1.0
    quantity_precision: str = "0.000001"
    price_precision: str = "0.000001"
    fee_per_share: float = 0.005
    slippage_bps: float = 1.0

    def __post_init__(self):
        values = [
            decimal(v)
            for v in (
                self.allocation_fraction,
                self.max_entry_notional,
                self.minimum_notional,
                self.quantity_precision,
                self.price_precision,
                self.fee_per_share,
                self.slippage_bps,
            )
        ]
        if (
            any(not v.is_finite() or v < 0 for v in values)
            or not 0 < values[0] <= Decimal(".1")
            or min(values[1:5]) <= 0
        ):
            raise ValueError("invalid sizing configuration")


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    limit_price: Decimal
    created_ns: int
    emergency: bool = False


def make_intent(action, account, quote, limits, now_ns, *, symbol="SPY", emergency=False):
    if isinstance(action, bool) or action not in (0, 1):
        raise ValueError("action must be flat or long")
    if action == 1 and account.position > 0 or action == 0 and account.position == 0:
        return None
    buy = action == 1
    slip = decimal(limits.slippage_bps) / 10000
    price = (quote.ask * (1 + slip) if buy else quote.bid * (1 - slip)).quantize(
        decimal(limits.price_precision), rounding=ROUND_DOWN if buy else ROUND_UP
    )
    if buy:
        budget = min(
            account.session_start_equity * decimal(limits.allocation_fraction),
            decimal(limits.max_entry_notional),
            account.cash,
        )
        quantity = (budget / (price + decimal(limits.fee_per_share))).quantize(
            decimal(limits.quantity_precision), rounding=ROUND_DOWN
        )
        if quantity <= 0 or quantity * price < decimal(limits.minimum_notional):
            return None
    else:
        quantity = account.position
    return OrderIntent(
        uuid4().hex, symbol, "buy" if buy else "sell", quantity, price, int(now_ns), emergency
    )
