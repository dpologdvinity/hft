"""Overnight drift on the paper account: buy near the close, sell at the next open.

Stocks have earned most of their return between the close and the next open,
but on free data the spread paid to trade at those times eats the effect
(docs/multiday-results.md). Auction orders avoid the spread, so this runner trades
through the auctions wherever Alpaca allows it and is judged forward on paper:

- Entry, 15:45-15:49 ET: a market-on-close order (`time_in_force=cls`) for whole
  shares when they use at least 90% of the stock's cash. Otherwise, 15:57-15:59 ET:
  a fractional market order. Alpaca accepts fractional quantities only for day
  orders, and on-close orders are refused after 15:50.
- Exit, 09:00-09:27:30 ET: a market-on-open order (`opg`) for whole-share
  positions, or a day market order for fractional ones (Alpaca fills market orders
  received before 09:28 at the official opening price). Any position still held
  after the open is sold with a day market order.

Positions are held overnight on purpose (owner decision, 2026-10-08). Each stock
keeps its own ledger and order book on one paper account, reusing the
`PortfolioBroker` startup checks, durable order state and reconciliation. A run
that stops overnight keeps its positions; restarting it sells them at the next
open. Entries stop for the day after the account-wide daily loss limit, and for
good after the drawdown limit; exits are never blocked.
"""

import asyncio
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from uuid import uuid4

from ..account import decimal
from ..broker import BrokerOrder
from ..logs import EventLog
from ..risk import RiskDecision
from ..sizing import OrderIntent
from ..strategies import StrategySpec
from .broker import MAX_REJECTIONS_PER_SESSION, SymbolBook
from .budget import AccountGuard

NS = 1_000_000_000
MINUTE = 60 * NS
AUCTION_ENTRY = (15 * MINUTE, 10 * MINUTE + 30 * NS)  # before the close
FRACTIONAL_ENTRY = (3 * MINUTE, 1 * MINUTE)  # before the close
PRE_OPEN_EXIT = (30 * MINUTE, 2 * MINUTE + 30 * NS)  # before the open
WHOLE_SHARE_USE = Decimal("0.9")  # whole shares must use this much of the cash
HEADROOM = Decimal("0.99")  # market orders can fill above the reference price
QUANTITY_STEP = Decimal("0.000001")
MINIMUM_NOTIONAL = Decimal(1)


@dataclass(frozen=True)
class OvernightConfig:
    name: str
    symbols: tuple[str, ...]
    budgets: dict
    strategy: StrategySpec
    log_dir: Path
    run_dir: Path
    daily_loss: float = 0.02
    max_drawdown: float = 0.05


class OvernightBook(SymbolBook):
    """A stock's book that sends scheduled market orders, including auction orders."""

    def submit_order(self, side, quantity, time_in_force, reference_price, now_ns):
        quantity, price = decimal(quantity), decimal(reference_price)
        if self.pending:
            return RiskDecision(False, "order_outstanding")
        if self.rejections >= MAX_REJECTIONS_PER_SESSION and side == "buy":
            return RiskDecision(False, "rejection_limit")
        if now_ns - self.refreshed_ns > 5 * NS:
            return RiskDecision(False, "stale_broker_snapshot")
        if side not in ("buy", "sell") or time_in_force not in ("day", "cls", "opg"):
            return RiskDecision(False, "invalid_intent")
        if not quantity.is_finite() or quantity <= 0 or not price.is_finite() or price <= 0:
            return RiskDecision(False, "invalid_intent")
        if time_in_force != "day" and quantity != quantity.to_integral_value():
            return RiskDecision(False, "fractional_auction_order")
        if side == "sell" and quantity > self.account.position:
            return RiskDecision(False, "inventory_limit")
        if side == "buy":
            if self.risk.halted:
                return RiskDecision(False, self.risk.halted)
            if self.account.position > 0:
                return RiskDecision(False, "already_long")
            if quantity * price > self.account.cash:
                return RiskDecision(False, "cash_limit")
            snapshot = self.account_snapshot
            free = min(
                decimal(snapshot["cash"]),
                decimal(snapshot.get("non_marginable_buying_power", snapshot["cash"])),
            )
            if quantity * price > free:
                return RiskDecision(False, "broker_cash_limit")
        intent = OrderIntent("hft-" + uuid4().hex, self.symbol, side, quantity, price, now_ns)
        self.pending = BrokerOrder(intent)
        if side == "buy":
            self.risk.record_order(now_ns)
        self._save()  # persist intent and unique ID BEFORE any network mutation
        body = {
            "symbol": self.symbol,
            "qty": str(quantity),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
            "client_order_id": intent.client_order_id,
            "extended_hours": False,
        }
        return self._send(intent, body)


def last_trade_prices(symbols):
    """Latest IEX trade price per stock (read-only market data)."""
    from ..history import DATA_BASE, ReadOnlyClient

    client = ReadOnlyClient()
    prices = {}
    for symbol in symbols:
        payload = client.get(f"{DATA_BASE}/v2/stocks/{symbol}/trades/latest", {"feed": "iex"})
        prices[symbol] = decimal(str(payload["trade"]["p"]))
    return prices


class OvernightRunner:
    def __init__(
        self,
        config,
        broker,
        windows,
        *,
        prices=last_trade_prices,
        clock=time.time_ns,
        sleep=asyncio.sleep,
        calendar_source=None,
    ):
        if not config.strategy.is_scheduled:
            raise ValueError("the overnight runner trades the overnight-drift strategy")
        self.config, self.broker, self.windows = config, broker, windows
        self.prices, self.clock, self.sleep = prices, clock, sleep
        self.calendar_source, self.calendar_ns = calendar_source, clock()
        self.started_ns = clock()
        self.guard = AccountGuard(
            sum((decimal(b) for b in config.budgets.values()), Decimal(0)),
            config.daily_loss,
            config.max_drawdown,
        )
        self.attempted = set()  # (symbol, session_id, step) already tried
        self.session_id = None
        self.last_prices = {}
        self.logs = {}
        for symbol in config.symbols:
            log = EventLog(config.log_dir / f"{symbol}-{self.started_ns}.jsonl", clock=clock)
            book = broker.book(symbol)
            log.write(
                "start",
                event_ns=self.started_ns,
                symbol=symbol,
                source="trade-paper",
                synthetic=False,
                initial_cash=book.account.initial_cash,
                position=book.account.position,
                metadata={
                    "run": config.name,
                    "strategy": config.strategy.name,
                    "strategy_version": config.strategy.version,
                    "account_id": broker.account_id,
                },
            )
            self.logs[symbol] = log
        self.trades_seen = {s: len(broker.book(s).account.trades) for s in config.symbols}

    @staticmethod
    def _near(window, now):
        """Within the order windows, plus margin for fills and retries after the open."""
        return (
            window.open_ns - PRE_OPEN_EXIT[0] - MINUTE <= now < window.open_ns + 20 * MINUTE
            or window.close_ns - AUCTION_ENTRY[0] - MINUTE <= now < window.close_ns + 5 * MINUTE
        )

    def _window(self, now):
        """The next session that has not closed yet."""
        return next((w for w in self.windows if w.close_ns > now), None)

    def _equity(self, prices):
        total = Decimal(0)
        for symbol in self.config.symbols:
            account = self.broker.book(symbol).account
            price = prices.get(symbol)
            total += account.mark(price) if price is not None else account.cash
        return total

    def _record(self, symbol, execution):
        book, log = self.broker.book(symbol), self.logs[symbol]
        log.write("fill", event_ns=execution.timestamp, execution=execution)
        trades = book.account.trades
        for trade in trades[self.trades_seen[symbol] :]:
            log.write(
                "trade",
                event_ns=trade.closed,
                entry_ns=trade.opened,
                exit_ns=trade.closed,
                pnl=trade.pnl,
                opening_notional=trade.opening_notional,
            )
        self.trades_seen[symbol] = len(trades)
        price = execution.price
        log.write(
            "equity",
            event_ns=execution.timestamp,
            equity=book.account.mark(price),
            cash=book.account.cash,
            position=book.account.position,
        )

    def _submit(self, symbol, side, quantity, tif, price, now, step, window):
        book = self.broker.book(symbol)
        self.attempted.add((symbol, window.session_id, step))
        if self.clock() - self.broker.refreshed_ns > 4 * NS:
            self.broker.refresh()
        decision = book.submit_order(side, quantity, tif, price, self.clock())
        self.logs[symbol].write(
            "order",
            event_ns=now,
            side=side,
            quantity=quantity,
            time_in_force=tif,
            reference_price=price,
            allowed=decision.allowed,
            reason=decision.reason,
            step=step,
        )
        return decision

    def _exits(self, window, now):
        pre_open = window.open_ns - PRE_OPEN_EXIT[0] <= now < window.open_ns - PRE_OPEN_EXIT[1]
        session = window.open_ns <= now < window.close_ns - AUCTION_ENTRY[0]
        for symbol in self.config.symbols:
            book = self.broker.book(symbol)
            position = book.account.position
            if position <= 0 or book.pending:
                continue
            if pre_open and (symbol, window.session_id, "exit") not in self.attempted:
                whole = position == position.to_integral_value()
                price = self.last_prices.get(symbol) or book.account.entry_price
                self._submit(
                    symbol, "sell", position, "opg" if whole else "day", price, now, "exit", window
                )
            elif session:
                # Retry at most once a minute: an unfilled opening order, or a restart.
                step = f"exit-{(now - window.open_ns) // MINUTE}"
                if (symbol, window.session_id, step) not in self.attempted:
                    price = self._price(symbol)
                    if price is not None:
                        self._submit(symbol, "sell", position, "day", price, now, step, window)

    def _price(self, symbol):
        try:
            self.last_prices.update(self.prices([symbol]))
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            self.logs[symbol].write("quality", event_ns=self.clock(), reason=f"price:{error}")
            return None
        return self.last_prices.get(symbol)

    def _entries(self, window, now):
        auction = window.close_ns - AUCTION_ENTRY[0] <= now < window.close_ns - AUCTION_ENTRY[1]
        fractional = (
            window.close_ns - FRACTIONAL_ENTRY[0] <= now < window.close_ns - FRACTIONAL_ENTRY[1]
        )
        if not (auction or fractional):
            return
        key = (None, window.session_id, "guard")
        if key not in self.attempted:
            self.attempted.add(key)
            try:
                self.last_prices.update(self.prices(list(self.config.symbols)))
            except (RuntimeError, OSError, KeyError, ValueError):
                pass
            equity = self._equity(self.last_prices)
            reason = self.guard.observe(equity)  # one overnight cycle since the last entry
            self.guard.start_session(equity)
            for symbol in self.config.symbols:
                book = self.broker.book(symbol)
                book.risk.halted = reason or self.guard.latched
                book.risk.permanent_halt |= self.guard.latched is not None
                if reason:
                    self.logs[symbol].write("halt", event_ns=now, reason=reason)
                price = self.last_prices.get(symbol)
                self.logs[symbol].write(
                    "equity",
                    event_ns=now,
                    equity=book.account.mark(price) if price else book.account.cash,
                    cash=book.account.cash,
                    position=book.account.position,
                )
        for symbol in self.config.symbols:
            book = self.broker.book(symbol)
            if book.account.position > 0 or book.pending or book.risk.halted:
                continue
            price = self.last_prices.get(symbol)
            if price is None:
                continue
            cash = book.account.cash * HEADROOM
            whole = (cash / price).to_integral_value(rounding=ROUND_DOWN)
            use_auction = whole >= 1 and whole * price >= book.account.cash * WHOLE_SHARE_USE
            if (
                auction
                and use_auction
                and (symbol, window.session_id, "entry") not in self.attempted
            ):
                self._submit(symbol, "buy", whole, "cls", price, now, "entry", window)
            elif fractional and (symbol, window.session_id, "entry-late") not in self.attempted:
                # Fractional stocks enter here; so does a whole-share stock whose on-close
                # order failed or was never sent (a pending on-close order is skipped above).
                price = self._price(symbol) or price
                quantity = (book.account.cash * HEADROOM / price).quantize(
                    QUANTITY_STEP, rounding=ROUND_DOWN
                )
                if quantity * price >= MINIMUM_NOTIONAL:
                    self._submit(symbol, "buy", quantity, "day", price, now, "entry-late", window)

    async def run(self, *, until_ns=None):
        for symbol, execution in self.broker.recovered_fills:
            self._record(symbol, execution)
        last_poll = last_refresh = 0
        try:
            while until_ns is None or self.clock() < until_ns:
                now = self.clock()
                if now - last_refresh >= MINUTE:
                    # Poll first so fills that already happened are in the ledger.
                    for symbol, execution in await asyncio.to_thread(self.broker.poll):
                        self._record(symbol, execution)
                    await asyncio.to_thread(self.broker.refresh)
                    await asyncio.to_thread(self.broker.reconcile)
                    self.broker.save()
                    last_refresh = now
                window = self._window(now)
                if window and window.session_id != self.session_id:
                    self.session_id = window.session_id
                    self.broker.start_session()
                if window:
                    self._exits(window, self.clock())
                    self._entries(window, self.clock())
                busy = any(self.broker.book(s).pending for s in self.config.symbols)
                if now - last_poll >= (5 if busy else 60) * NS:
                    for symbol, execution in await asyncio.to_thread(self.broker.poll):
                        self._record(symbol, execution)
                    last_poll = now
                if (
                    self.calendar_source
                    and (window is None or now < window.open_ns - PRE_OPEN_EXIT[0])
                    and now - self.calendar_ns > 6 * 3600 * NS
                ):
                    self.windows = await asyncio.to_thread(self.calendar_source)
                    self.calendar_ns = now
                await self.sleep(1.0 if window and self._near(window, now) else 10.0)
        finally:
            await self.shutdown()
        return self.summary()

    async def shutdown(self):
        """Cancel open orders; positions stay and are sold at the next open after a restart."""
        try:
            await asyncio.to_thread(self.broker.cancel_all)
            for _ in range(10):
                fills = await asyncio.to_thread(self.broker.poll)
                for symbol, execution in fills:
                    self._record(symbol, execution)
                if not any(self.broker.book(s).pending for s in self.config.symbols):
                    break
                await self.sleep(0.5)
        finally:
            self.broker.save()
            for symbol, log in self.logs.items():
                log.write(
                    "finish",
                    event_ns=self.clock(),
                    position=self.broker.book(symbol).account.position,
                )
                log.close()

    def summary(self):
        return {
            symbol: {
                "position": str(self.broker.book(symbol).account.position),
                "cash": str(self.broker.book(symbol).account.cash),
                "trades": len(self.broker.book(symbol).account.trades),
                "halted": self.broker.book(symbol).risk.halted,
            }
            for symbol in self.config.symbols
        }
