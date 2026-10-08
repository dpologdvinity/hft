# Paper trading runner with a C++ decision engine

Date: 2026-10-07. Status: approved design (revised), pending implementation plan.

## Goal

Let the owner run the trading system on Alpaca **paper** (simulated money) on one
or more stocks of their choosing, with a selectable strategy, every trading day
until stopped, and watch it in the existing dashboard. Market-data handling and
trading decisions run in a new low-latency **C++ engine** that is also used by
backtests, so live and historical behavior share one implementation.

## Decisions (owner, 2026-10-07)

| Topic | Decision |
| --- | --- |
| Money | Paper only. Real money stays behind the existing research qualification and 30-session paper graduation; the `live` command and its gates are unchanged. |
| Decision source | Built-in rule strategies that work on any stock now; a trained model bundle can be selected for the stock it was trained on. |
| Capital | A dollar budget per stock, given on the command line. |
| Controls | Start/stop from the terminal; the dashboard shows the bot read-only. |
| Schedule | One command runs every trading day until stopped. |
| C++ scope | The engine that ingests market data and decides: message parsing, per-stock event ordering, 5 s bars, gap/warmup handling, the 17 v3 features and rule strategies. |
| Python scope | Network connection, budget sizing, risk gateway, Decimal ledger, broker orders and reconciliation, ONNX model inference, CLI, dashboard. |
| Delivery | Multi-stock from the start; C++ engine and Python runner built in parallel against a frozen interface, then integrated. |
| Acceptance | Short real paper-account sessions during market hours are authorized for acceptance testing. |

## Non-goals

- Real-money changes, or any bypass of graduation for live trading.
- Finding a profitable strategy (separate research track). Rule strategies are
  baselines; their paper results never count as graduation evidence.
- Changing the decision cadence (stays 5 s; faster cadence raises the measured cost
  hurdle) or the v3 feature contract (the C++ engine must reproduce it exactly).
- Moving networking, money accounting, risk or broker code to C++: their latency is
  dominated by the internet round trip to the broker, and they are tested
  safety code.
- Dashboard controls, several accounts per run, shorting, leverage, overnight holds.

## Architecture

```
Alpaca IEX websocket -- Python: connect/reconnect --> raw JSON frames
                                                          |
historical Parquet --> +------------- hftcore (C++) -------v-------------------+
  (backtests)          | parse -> per stock: dedup, arrival order -> 5 s bars  |
                       | -> gap/warmup -> 17 features -> strategy -> decision  |
                       +---------------------------+---------------------------+
                                                   v  Decision{symbol, bar, observation, action, status}
     Python: budget sizing -> risk gateway -> Decimal ledger -> Alpaca orders -> reconciliation
```

"Fast" is defined by what this system controls and can measure: tick-to-decision
latency inside the engine and backtest throughput. Order latency is bounded by the
internet round trip to a retail broker (tens to hundreds of milliseconds) and the
free single-exchange feed; no code change removes that.

## C++ engine (`cpp/`, Python module `hftcore`)

Tooling: C++20, CMake, pybind11, scikit-build-core packaging as a separate
installable project under `cpp/` (the `hft` package keeps its current build). Pinned
third-party code: simdjson (message parsing) and Catch2 (C++ unit tests), fetched by
CMake at fixed versions with recorded hashes. `-Wall -Wextra -Werror`.

Components, each with its own header, implementation and Catch2 tests:

| Unit | Responsibility |
| --- | --- |
| `Event` | Plain struct: kind (quote/trade), symbol id, event and arrival nanoseconds, prices/sizes, trade-eligibility flag, 128-bit identity hash. |
| `MessageParser` | Alpaca stream frames (arrays of `q`, `t`, control messages) to `Event`s with simdjson; rejects malformed or future-dated input exactly like `hft/feed.py`. |
| `BarAggregator` | Exact port of `hft.feed.BarAggregator`: causal 5 s boundary timers, latest-quote rule (250 ms), trade condition/tape filtering, duplicate suppression. |
| `GapTracker` | 5 s silence = gap: reset the 61-bar history, block entries; matches the simulator. |
| `FeatureEngine` | Exact float32 port of the 17 v3 features in `hft/features.py`; account-dependent inputs (position, cash, fills, order rate) are passed in by Python per decision. |
| `Strategy` | `ema-crossover` (EMA 5/20 of the bar history, as `hft.research` control) and `hold-day`; `model` strategies return the observation for Python ONNX inference. |
| `Engine` | Symbol table and per-stock state in contiguous storage; `on_frame(raw)`, `on_quote/on_trade` (numeric), `advance_to(now_ns)`; returns decisions. No heap allocation per event on the hot path after warmup. |
| `replay` | Batch API over numeric arrays for backtests: bars, ready times and gap times, replacing the Python loop in `Simulation.__init__` when the module is available. |

Correctness gate (non-negotiable): the C++ path must produce **identical** bars, gap
times, eligibility, observations (bitwise float32) and decisions to the Python
reference on synthetic fixtures and on real development sessions (MCD and NVDA
probe days), and identical always-long rollout results. The Python implementation
remains the reference and the fallback when `hftcore` is not installed. Any
difference is a bug to fix, never a relabeled contract.

Performance targets (to be measured and published, not promised):
replay throughput of several million events per second per core; tick-to-decision
p99 in the tens of microseconds; NVDA 2026-06-05 simulation build reduced from
47.7 s by at least an order of magnitude. Benchmarks: a C++ micro-benchmark binary
and the existing `benchmarks/session_pipeline.py` comparing both paths with the same
fingerprints.

Quality: CI builds the module, runs Catch2 tests, Python parity tests, and a
separate AddressSanitizer/UndefinedBehaviorSanitizer job.

## Python runner

### Command interface

```bash
hft trade --paper --symbols NVDA=200 AAPL=100 --strategy ema-crossover [--name RUN]
          [--daily-loss 0.02] [--max-drawdown 0.05]
hft trade --paper --symbols NVDA=300 --strategy hold-day
hft trade --paper --symbols NVDA=200 --strategy model:artifacts/nvda/search/bundle
hft trade --status [--name RUN]
```

- `--paper` is required. `--live` (or omitting `--paper`) is refused with:
  "real money unlocks after research qualification and 30-session paper graduation".
- `--symbols SYMBOL=DOLLARS ...`: 1-30 symbols (Alpaca's free real-time stream
  limit), validated against Alpaca's asset list (active, tradable, fractionable);
  duplicates rejected. A `model:` strategy requires exactly one symbol, equal to the
  bundle's symbol.
- `--name`: run name (default derived from symbols and strategy); state and journals
  are namespaced by run name.
- Ctrl-C (SIGINT/SIGTERM) performs the controlled shutdown below.
- `--status` prints per stock: status, position, average price, last price,
  unrealized and realized profit (today and total), open order, halt reason.

### Budgets and risk

Paper `trade` runs use explicit dollar budgets as their exposure guard. The
fraction-based `RiskConfig` / `SizingConfig` defaults and validation (10% entry,
15% maintenance exposure) are **unchanged** for bundles, research and live.

- Per stock: an entry may invest at most the stock's remaining budget (budget minus
  current position value), limited by available cash.
- Account: total invested never exceeds the sum of budgets, and the sum of budgets
  must not exceed the paper account's non-borrowed cash at start.
- Daily loss (default 2%) per stock (of its budget) and for the account (of the total
  budget): entries stop for that stock / all stocks until the next session, and the
  existing risk gateway also sells current holdings (its unchanged, stricter behavior).
- Drawdown (default 5%) per stock and account latches entries for the rest of the
  run. No bypass flag; trading again needs a new run name, which requires a flat
  account.
- Unchanged: 5 orders/minute per stock, 2 s quote freshness, future-quote rejection,
  volatility breaker, 60 s pre-close exit window, 5 s order expiry, marketable limits
  at ask x (1 + 1 bp) / bid x (1 - 1 bp), fractional precision, $1 minimum notional,
  flat by close daily, no shorting or leverage.
- `RiskGateway` gains an optional absolute entry cap used only by budget runs; its
  default behavior is unchanged and covered by existing tests.

### Runtime

1. One websocket subscription for all configured stocks; raw frames go to the engine
   (C++ when installed, Python reference otherwise) through one interface.
2. Per-stock latest quote, pending order and position; one ledger with shared cash.
3. Feed silence > 5 s for a stock: gap row, warmup reset, entries blocked for that
   stock; holdings kept; exits use fresh quotes. (Today this is fatal:
   `hft/runtime.py:213-215`.)
4. Reconnect with exponential backoff (1 s to 60 s), entries blocked while
   disconnected; emergency market reduction at the pre-close window if still down.
5. Daily supervisor on Alpaca's calendar: wait for open, trade, flatten in the
   pre-close window, write the session row, sleep, repeat until stopped.
6. Rejected orders: after a rejection, query by client order ID; if no order exists,
   clear `pending` and journal `order_rejected` (today a rejection blocks forever:
   `hft/broker.py:65-66, 332-349, 404-410`). Three rejections for one stock in a
   session stop that stock until the next session. Timeouts keep lookup-by-same-ID.
7. Reconciliation covers every configured stock (positions, open orders) plus the
   existing account cash formula (`hft/broker.py:429-450`); anything unexpected fails
   closed.
8. Run identity from symbols and budgets, strategy identity, engine implementation
   and version, risk/sizing/cost settings, latency (75 ms), bar width (5 s), feed.
   State under `.state/trade/<run>/`; one bot per paper account (account lock);
   resume refuses a changed identity; a fresh run requires a flat account.

### Shutdown, failures and recovery

| Situation | Behavior |
| --- | --- |
| Ctrl-C / SIGTERM | Stop entries, cancel open orders, wait for final fills, sell holdings if the market is open, reconcile, write a clean finish row. |
| Feed quiet > 5 s | Gap row, warmup reset, entries blocked for that stock. |
| Disconnect | Reconnect with backoff; emergency sell at pre-close if still down. |
| Rejected order | Confirm no broker record, clear, journal; 3 per stock per session stops that stock for the day. |
| Unknown order after timeout | Lookup by the same client order ID (unchanged). |
| Reconciliation mismatch | Fail closed: stop trading, keep state and journals (unchanged). |
| Crash or reboot | Rerun the same command; it reconciles before trading. |
| Event backlog > 1 s | Drop stale events, journal one gap, restart warmup from fresh bars (entries blocked until ready). Amended 2026-10-08: the first paper session lost NVDA for the whole day to a single stall at the open under the original halt-for-the-session rule. |

### Journals, status and dashboard

- Journals: `logs/trade/<run>/<SYMBOL>-<run start ns>.jsonl` (one hash-chained file
  per stock and process start, so a resumed run never overwrites an old journal),
  `source: "trade-paper"`, strategy and engine identity. Trade runs journal bars,
  decisions, orders, fills and per-bar equity but not raw market rows, which would
  reach millions of rows per liquid stock per day.
  `evidence.graduate` only accepts `broker-paper`, so these never count as evidence.
- `hft trade --status` reads state and journals only (no network).
- Dashboard: read-only `/api/trading` endpoint and a Paper trading panel per run and
  stock (waiting for open, warming up, gap, trading, halted, flat; position; open
  order; fills; profit). Host/Origin/path validation and read-only behavior remain.

## Delivery

1. Freeze the engine interface (Python protocol + C++ header + shared test vectors).
2. In parallel, in separate git worktrees:
   - Track A, C++ engine: units above, Catch2 tests, parity tests, benchmarks, CI.
   - Track B, Python runner: everything in "Python runner", driving the Python
     reference engine through the frozen interface.
3. Integrate: the runner uses `hftcore` when installed and parity tests pass;
   `Simulation` uses `replay`; publish benchmarks.
4. Acceptance on the paper account (authorized): one session with three or more
   stocks, decisions journaled, orders when signaled, flat before close, clean
   reconciliation, dashboard shows it.

## Testing

- C++: Catch2 unit tests per unit; sanitizer CI job.
- Parity (pytest): C++ vs Python on fixtures and real development sessions, as in
  the correctness gate.
- Python: strategies, run identity/resume refusal, `--live` refusal, argument
  validation; budget sizing and per-stock/account limits (existing fraction-based
  tests unchanged); silence handling, reconnect, pre-close emergency sell; rejected
  order clearing and the three-rejection stop; multi-stock reconciliation including
  fail-closed on a foreign position; supervisor over several sessions with a fake
  clock and calendar (early close, holiday); end to end with fake stream and broker,
  two stocks over two sessions.
- Existing Python tests, Ruff, frontend tests and build stay green; CI.

## Known limitations

- Stocks with frequent IEX gaps rarely complete the 61-bar warmup, so strategies act
  less often on them; the dashboard shows the gap status.
- Paper fills come from Alpaca's paper simulator, which differs from real execution.
- Rule strategies have no demonstrated edge; their paper results say nothing about
  real-money readiness.
- The free real-time stream is IEX only and allows up to 30 symbols.
