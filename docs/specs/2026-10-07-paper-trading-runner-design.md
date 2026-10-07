# Paper trading runner: user-chosen stocks and strategies

Date: 2026-10-07. Status: approved design, pending implementation plan.

## Goal

Let the owner run the trading system on Alpaca **paper** (simulated money) today,
on one or more stocks of their choosing, with a selectable strategy, every trading
day until stopped, and watch it in the existing dashboard. No research-qualified
model is required to paper trade.

## Decisions (owner, 2026-10-07)

| Topic | Decision |
| --- | --- |
| Money | Paper only. Real money stays behind the existing research qualification and 30-session paper graduation; the `live` command and its gates are unchanged. |
| Decision source | Built-in rule strategies that work on any stock now; a trained model bundle can be selected for the stock it was trained on. |
| Capital | A dollar budget per stock, given on the command line. |
| Controls | Start/stop from the terminal; the dashboard shows the bot read-only. |
| Schedule | One command runs every trading day until stopped. |
| Delivery | Staged: one stock first (stage 1), then several stocks on one paper account (stage 2). |
| Acceptance | Short real paper-account sessions during market hours are authorized for acceptance testing. |

## Non-goals

- Real-money trading changes, or any bypass of graduation for live trading.
- Finding a profitable strategy (separate research track). Rule strategies are
  baselines; their paper results never count as graduation evidence.
- Dashboard controls (the dashboard stays read-only), several accounts per run,
  short selling, leverage, overnight positions, or a different decision cadence.

## Command interface

```bash
hft trade --paper --symbols NVDA=200 AAPL=100 --strategy ema-crossover [--name RUN]
          [--daily-loss 0.02] [--max-drawdown 0.05]
hft trade --paper --symbols NVDA=300 --strategy hold-day
hft trade --paper --symbols NVDA=200 --strategy model:artifacts/nvda/search/bundle
hft trade --status [--name RUN]
```

- `--paper` is required. `--live` (or omitting `--paper`) is refused with:
  "real money unlocks after research qualification and 30-session paper graduation".
- `--symbols SYMBOL=DOLLARS ...`: 1 symbol in stage 1, 1-30 symbols in stage 2
  (Alpaca's free real-time stream limit). Symbols are validated against Alpaca's
  asset list (active, tradable, fractionable). Duplicate symbols are rejected.
- `--name`: run name (default derived from symbols and strategy). State and
  journals are namespaced by run name.
- Ctrl-C (SIGINT/SIGTERM) performs the controlled shutdown described below.
- `--status` prints, per stock: status, position, average price, last price,
  unrealized and realized profit today and in total, open order, halt reason.

## Strategies (`hft/strategies.py`, new)

One interface: `decide(observation, bars) -> 0 | 1` (flat or long), plus `name`,
`version` and an identity string recorded in run identity, state and journals.
The runtime calls it on every completed 5-second bar once the existing 61-bar
warmup is satisfied; risk checks still decide whether an order is allowed.

| Strategy | Rule | Notes |
| --- | --- | --- |
| `ema-crossover` | Long while EMA(5) > EMA(20) of the last 61 bar closes | Same rule as the research `ema_5_20` control (`hft/research.py:446-450`), sharing `research.ema`. |
| `hold-day` | Always long once eligible | Same as the research intraday-long control; buys after warmup, exits at the pre-close window. |
| `model:<bundle>` | `OnnxPolicy.predict(observation)` | Bundle is validated as today; refused unless its symbol equals the (single) configured stock. |

## Budgets and risk

Paper `trade` runs use explicit dollar budgets as their exposure guard. The
fraction-based `RiskConfig` / `SizingConfig` defaults and validation (10% entry,
15% maintenance exposure) are **unchanged** for bundles, research and live.

- Per stock: an entry may invest at most the stock's remaining budget
  (budget minus current position value), limited by available cash.
- Account: total invested across stocks never exceeds the sum of budgets, and the
  sum of budgets must not exceed the paper account's non-borrowed cash at start.
- Daily loss (default 2%) applies per stock (of its budget) and to the account (of
  the total budget): new entries stop for that stock / all stocks until the next
  session; holdings exit normally or at the pre-close window.
- Drawdown (default 5%) per stock and account latches entries for the rest of
  the run. There is no bypass flag; resuming trading needs a new run name, which
  in turn requires a flat account.
- Kept unchanged: 5 orders/minute (per stock), 2-second quote freshness, future-quote
  rejection, volatility breaker, pre-close exit window (60 s), 5-second order
  expiry, marketable limits at ask x (1 + 1 bp) / bid x (1 - 1 bp), fractional
  precision, $1 minimum notional, flat by close every day, no shorting or leverage.
- `RiskGateway` gains an optional absolute entry cap used only by budget runs;
  its default behavior is unchanged and covered by existing tests.

## Runtime changes

Stage 1 (one stock):

1. **Feed silence no longer kills the run.** Today more than 5 s without events
   raises a fatal error (`hft/runtime.py:213-215`). Instead: log a `gap` row, reset
   warmup, block new entries for that stock; keep holdings; exits still use fresh
   quotes. Matches the simulator's gap treatment.
2. **Reconnect.** Websocket errors trigger reconnect with exponential backoff
   (1 s to 60 s); entries are blocked while disconnected. If still disconnected
   at the pre-close window, holdings are sold with the existing emergency market
   reduction based on broker positions.
3. **Daily supervisor.** Uses Alpaca's calendar (holidays, early closes): wait for
   the open, run the session, flatten in the pre-close window, write the session
   row, sleep until the next session, repeat until stopped.
4. **Rejected orders.** Today a rejected POST leaves `pending` set forever and
   blocks startup (`hft/broker.py:65-66, 332-349, 404-410`). After a rejection the
   broker is queried by client order ID; if no order exists, `pending` is cleared
   and an `order_rejected` row is journaled. Three rejections for one stock in a
   session stop entries for that stock until the next session. Timeouts keep the
   existing lookup-by-same-ID behavior (never a blind resubmission).
5. **Run identity and state.** Built from symbols and budgets, strategy identity,
   risk/sizing/cost settings, latency (75 ms), bar width (5 s) and feed (`iex`).
   State lives under `.state/trade/<run>/`; an account-level lock allows one bot
   per paper account. Resuming refuses a changed identity. A fresh run requires a
   flat account with no open orders (existing check).

Stage 2 (several stocks):

6. One websocket subscription for all configured stocks; events are routed by
   symbol to a per-stock engine (bar aggregator, features, strategy, latest quote,
   pending order). The single process-wide `latest` quote (`hft/runtime.py:39-43`)
   becomes per stock so orders are priced from the right quote.
7. One account ledger: shared cash, one position per stock; per-stock and account
   risk as above.
8. Broker reconciliation covers every configured stock (positions and open orders
   per symbol) plus the existing account cash formula (`hft/broker.py:429-450`).
   Anything unexpected still fails closed.

## Shutdown, failures and recovery

| Situation | Behavior |
| --- | --- |
| Ctrl-C / SIGTERM | Stop entries, cancel open orders, wait for final fills, sell holdings if the market is open, reconcile, write a clean finish row. |
| Feed quiet > 5 s | Gap row, warmup reset, entries blocked for that stock. |
| Disconnect | Reconnect with backoff; emergency sell at pre-close if still down. |
| Rejected order | Confirm no broker record, clear, journal; 3 per stock per session stops that stock for the day. |
| Unknown order after timeout | Lookup by the same client order ID (unchanged). |
| Reconciliation mismatch | Fail closed: stop trading, keep state and journals for inspection (unchanged). |
| Crash or reboot | Rerun the same command; it reconciles before trading. |
| Event backlog > 1 s | Existing `runtime_gap` behavior: drop, reset history, halt entries for the session. |

## Journals, status and dashboard

- Journals: `logs/trade/<run>/session-<date>.jsonl`, using the existing hash-chained
  log format, with `source: "trade-paper"`, the strategy identity and per-row
  symbol. `evidence.graduate` only accepts `broker-paper` journals, so these rows
  can never be counted as graduation evidence.
- `hft trade --status` reads state and the latest journal; no network calls.
- Dashboard: a read-only `/api/trading` endpoint and a panel on the Paper trading
  view showing each run and stock (status: waiting for open, warming up, gap,
  trading, halted, flat; position; open order; fills; profit), built from state
  and journals only. Host/Origin/path validation and read-only behavior remain.

## Testing and acceptance

Automated (synthetic and faked broker/stream; no credentials):

- Strategies: crossover signals on constructed bars, hold-day, model symbol refusal.
- Run identity and resume refusal; `--live` refused; argument validation.
- Budget sizing and risk: per-stock caps, account total, daily loss and drawdown
  per stock and account; existing fraction-based tests unchanged.
- Runtime: silence handling (no crash, gap row, entries blocked), reconnect with
  backoff, pre-close emergency sell while disconnected.
- Broker: rejected-order clearing and the three-rejection stop; multi-symbol
  reconciliation (stage 2), including fail-closed on a foreign position.
- Supervisor: multiple sessions with a fake clock and calendar, early close, holiday.
- End to end: fake stream and broker, two stocks over two sessions (stage 2).
- Existing 218 tests, Ruff, frontend tests and build stay green; CI.

Acceptance on the owner's paper account during market hours (authorized):

- Stage 1: `hft trade --paper --symbols <one stock>=<budget> --strategy ema-crossover`
  runs a session, journals decisions, places orders when signaled, is flat before
  the close, reconciles cleanly, and the dashboard shows it.
- Stage 2: the same with three or more stocks on one paper account, with budgets
  enforced and clean reconciliation.

## Known limitations

- Stocks with frequent IEX gaps rarely complete the 61-bar warmup, so strategies
  act less often on them; the dashboard shows the gap status.
- Paper fills come from Alpaca's paper simulator, which differs from real execution.
- Rule strategies are baselines with no demonstrated edge; paper profit or loss
  from them says nothing about real-money readiness.
- The free real-time stream is IEX only and allows up to 30 symbols.
