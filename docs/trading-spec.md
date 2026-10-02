# AI stock trader: agreed requirements and technical design

Date: 2026-10-02. Requirements and technical design for the execution, risk,
research and graduation contracts implemented in `hft/`.

## 1. What the user wants

- An AI stock trader that learns by repeatedly simulating historical markets.
- It should improve its trading decisions to maximize profit after costs.
- Training need not run continuously or use real money.
- It must support dry runs and eventually real stock trading.
- No cryptocurrency, paid data, paid hosting, or subscription services.
- Potential starting capital is $500–$5,000. Profits can increase the capital.
- Choose a practical starting speed. User is not requiring exchange colocated
  infrastructure or a native execution language.
- Deliver a detailed, concrete implementation plan for a cheaper model.

The laptop has 32 GB physical RAM. `lscpu` reports an Intel i7-1265U,
6 cores / 12 logical CPUs. This Linux VM exposes approximately 15 GB RAM;
budget against the VM rather than the physical laptop. GPU is not assumed.

## 2. Decisions and alternatives

Use Alpaca's free Basic account and IEX market data, local Parquet storage,
Gymnasium and Stable-Baselines3 PPO, and CPU ONNX inference. Evaluate one
liquid stock initially; AAPL is a development example, not an investment
recommendation. The universe is explicit and must not be chosen after
looking at held-out profits.

Start with completed **5-second bars** and event-level quotes for execution.
Inference every five seconds does not mean placing an order every five
seconds. Trades occur only when a target change survives cost and risk checks.
Do not describe this version as institutional microsecond HFT.

PPO is the first learning algorithm because actions can be discrete and a
historical simulator can generate its own training trajectories. A cost-aware
rule strategy is a required control. A supervised forecasting model would
be a reasonable later alternative if PPO does not beat those controls;
do not implement a second learning framework in the initial scope.

Initial planning default: **long-only**, fractional shares, no borrowing and
no leverage. The optional short-selling question is unanswered; it does not
block implementing this stated default. If the user later chooses shorts,
write a separate extension plan: fractional shorting is not supported by
the selected broker, so do not silently expand this plan to short positions.

Four modes have distinct permissions and evidence:

| Mode | Inputs | Orders | Purpose |
|---|---|---|---|
| `replay` | Historical data | Local simulation | Tests and research |
| `dry-run` | Live IEX stream | Local simulation only | Real-time policy/execution validation |
| `broker-paper` | Live IEX stream | Alpaca paper endpoint | Actual broker order lifecycle validation |
| `live` | Live IEX stream | Alpaca live endpoint | Explicitly activated, gated real-money trading |

There is no automatic transition to live mode. Passing evidence makes a
model eligible; only an explicit live command starts real trading. A running
model is immutable. Retraining produces a challenger, never changes weights
inside a real-money process.

## 3. What “free” actually permits

The free Basic plan provides IEX real-time stock data and historical API
access with limits. IEX is one exchange, not the consolidated national market.
Use `feed=iex` explicitly in training downloads and real-time subscriptions.
Do not train on SIP volume/spread features then run on IEX without a separately
validated contract. No purchased feed may be enabled automatically.

Before large implementation or downloads, probe historical **trades and
quotes**, not just minute OHLCV. Verify availability, date coverage, and
entitlement on the user's free account. Five-second execution research needs
actual quotes and trades, not subdivided minute candles. Cache raw data by
session, request paginated results, and cap requests at 150/minute to leave
headroom under the currently documented 200/minute Basic limit.

If adequate quotes are unavailable, finish the engineering tests and live
recorder, report `insufficient_execution_data`, and collect free live IEX
quotes. Do not fabricate spreads or claim a tradable edge from Yahoo daily
prices, estimated quotes, or synthetic samples. Minute-only exploratory
research must be labeled separately and cannot pass live graduation.

Current broker documentation describes dynamic intraday margin rules,
replacing the old PDT-count rule. Recheck these rules and the actual account
flags before live integration; do not hardcode a remembered $25,000 rule.
Use non-borrowed buying power, broker account status and available cash as
hard constraints. If an eventual cash-account provider is introduced,
implement its settlement rules separately; do not apply cash T+1 accounting
to Alpaca's margin account as though they were the same account type.

Sources, checked on 2026-10-02:

- [Alpaca data plans](https://docs.alpaca.markets/us/docs/about-market-data-api)
- [Historical feeds](https://docs.alpaca.markets/us/docs/historical-stock-data-1)
- [Real-time quote/trade schemas](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data)
- [Fractional orders](https://docs.alpaca.markets/us/docs/fractional-trading)
- [Trading account fields](https://docs.alpaca.markets/us/docs/account-plans)
- [Intraday margin rules](https://docs.alpaca.markets/us/docs/the-intraday-margin-rule)
- [FINRA intraday trading](https://www.finra.org/investors/insights/frequent-intraday-trading)
- [Paper simulator limitations](https://docs.alpaca.markets/us/docs/paper-trading)

## 4. Architecture and ownership

```text
historical download / live recorder
          |
          v
validated trades + quotes + exchange calendar -> shared bar builder
          |                                          |
          v                                          v
event-driven historical simulator              rolling live state
          |                                          |
       Gym Env                                      ONNX
          |                                          |
PPO candidate search -> validation -> frozen model    target
          |                                          |
    untouched test                                   v
          |                                  sizing + risk gateway
    export + manifest                                |
          |                           local paper / broker-paper / live
          +------------------------------------------|
                                                     v
                                     fills + ledger + logs + reports
```

Use a single package, a CLI and local files. No database server, message bus,
Kubernetes, microservices, generalized provider framework, or elaborate web
app. Add a small read-only status report after execution works. Alpaca's
dashboard can show broker-paper orders without building a second UI.

`data.py` owns validation/storage; `feed.py` owns aggregation;
`features.py` owns the observation contract; `account.py` owns the ledger;
`execution.py` owns historical/local-paper fill rules; `risk.py` owns safety;
`env.py` wraps historical execution for learning; `research.py` owns temporal
search/evaluation; `policy.py` owns model artifacts; `paper.py` owns the
event loop/logs; `broker.py` owns remote order state; `evidence.py` owns
graduation. Shared functions must not be copied between these layers.

## 5. Time and data contract

Persist UTC nanoseconds as int64. Persist event time and, for live recordings,
arrival time separately. Trading sessions come from the broker calendar,
including holidays and early closes; never assume every weekday closes at
16:00. Runtime scheduling uses monotonic clocks; quote freshness uses UTC
wall time. Nanosecond timestamps must not be parsed through float seconds.

Raw quote columns: `event_ns`, `bid`, `ask`, `bid_size`, `ask_size`, exchange
identifiers, conditions and stable record identity. Raw trade columns:
`event_ns`, `trade_id`, `price`, `size`, conditions and correction identity.
Prices/size arrays are float64; volumes are shares. Provider quote-size units
must be converted explicitly (Alpaca stock quote sizes are reported in lots).
Keep original units in the raw cache for audit; feature imbalance is a ratio.

Five-second intervals are half-open `[start_ns, end_ns)`. A completed bar
contains OHLC from eligible trades, volume from those trades, trailing trade
VWAP, and the latest eligible quote strictly before `end_ns`. Include
`quote_ns`, `quote_age_ns`, `tradable`, and `session_id`. Never use a future
quote to populate the completed bar. A quote exactly at `end_ns` belongs to
the next interval and may subsequently execute an order, subject to latency.

If a bar has no trades, carry the last trade price, set volume to zero, and
do not invent volume. Quotes stay as-of quotes, with explicit age. If no
fresh quote exists, the bar is not tradable. If prices/context are missing
at session start, warm up; do not fill missing history using future values.
Do not retroactively change observations after a late correction. Record
corrections and apply the same causal correction policy to historical and
live processing. A bounded reorder policy must preserve when information
was actually available; any simulation that lacks arrival times declares
that limitation and receives execution stress tests.

Use session-partitioned Parquet and contiguous NumPy arrays. Materializing
a year of ticks into Python dictionaries/dataclasses is prohibited. Bar
views for an individual event are fine; large raw sequences remain arrays.
Hash dataset partitions, metadata, aggregation version and calendar into a
dataset manifest. Unadjusted raw quotes/trades are executable prices; avoid
mixing adjusted OHLC with unadjusted quotes. Reset per-session features and
stay flat overnight to avoid unsupported split/dividend position accounting.

## 6. Observation and actions

Feature schema v3 has 17 float32 values in this exact order:

1. Log close returns over 1, 5, 15, and 60 completed bars (four values).
2. Signed candle body/range, upper wick/range, lower wick/range.
3. Relative spread `(ask-bid)/mid`, quote-size imbalance, trailing 60-bar
   trade-volume-weighted price deviation.
4. Inventory as held notional / strategy equity, unrealized net PnL /
   strategy equity, free non-borrowed cash / strategy equity.
5. Remaining regular-session fraction using the actual calendar.
6. Seconds since last strategy fill / 60, clipped to `[0,1]`.
7. Remaining order slots / configured slots in the rolling minute.
8. Quote freshness flag (1 fresh, 0 stale/unavailable).

Reset rolling price context each exchange session. Require 61 completed bars
before the first signal (305 seconds at five-second resolution). Training
and runtime use the identical function and warmup. No whole-dataset fitted
normalization, return clipping based on future statistics, or future labels
in observations. Reject nonfinite features.

The v3 contract bounds history to the most recent 61 contiguous, causally
published bars. Five-second timer boundaries publish replay bars even without a
new payload; a timer precedes a same-time arrival. Observed feed gaps reset warmup
and cancel pending simulation orders. Such sessions cannot qualify because the
streaming runtime stops after five seconds without a symbol event. Development
preflight checks this before fitting and never opens reserved final-test partitions.
Historical arrival latency remains unknown when the source did not record it.

Discrete actions: `0 = flat`, `1 = hold/enter long`. On a flat-to-long
transition, size to at most 10% of session-start reconciled strategy equity. On `1` while
already long, **hold the existing quantity** rather than rebalance every
five seconds. On `0`, close the actual held quantity. Round fractional
share quantity down to broker precision; ensure notional >= broker minimum
and fees cannot take cash negative. If the capital/asset/order type cannot
support the proposed quantity, return a structured rejection, not an
alternate larger order. No short action exists in v1.

Capital grows only from reconciled strategy profits or explicit deposits.
Size new positions from session-start reconciled equity with the same 10%
fraction; changing equity does not increase leverage or risk percentages.
First real-money canary uses a separate $10 maximum position notional until
ten completed live sessions and execution review; then the user can raise
the absolute ceiling. The $500–$5,000 is capital, not an instruction to
expose all of it at once. Fractional-share sizing must work at both endpoints.

## 7. Execution, accounting, and reward

At completed bar `t`, produce an action from data available then. Schedule
the resulting order for `decision_time + simulated_latency`. Fill only at
the first eligible fresh quote at or after that due time, never against the
decision's historical bar close. Replay every intervening quote to the
next decision. Buys use ask plus adverse slippage; sells use bid minus
adverse slippage. Respect displayed size, remaining quantity and expiry.
Local simulation limits round buys up and sells down at the configured price
precision, allowing at most one precision unit beyond the slippage boundary
to keep unchanged off-grid quotes executable. Actual simulated fill prices
remain bid/ask plus the exact adverse slippage, and sizing reserves cash using
the rounded buy limit. Remote broker limits retain the separately specified
buy-down/sell-up tick rounding so an actual submitted limit never exceeds its
configured maximum fill bound.

Do not claim queue-position realism for passive orders; v1 attempts
marketable limit orders and treats non-crossing orders as unfilled.

Base local assumptions: 75ms latency, 1bp adverse slippage beyond bid/ask,
and a conservative $0.005/share fee reserve. These are configurable stress
assumptions, not claims of the broker's actual commission schedule. Stress
replay uses 250ms latency, 3bp slippage and twice the fee reserve. A broker
fill ledger records actual price and separately reserves/reconciles fees.

Equity = cash + signed shares * current eligible mark. Track fills, cash,
open weighted entry basis, opening fees, realized close PnL, and fees. A
partial sell allocates the opening fee pro rata. Accumulate partial closes
into one completed flat-to-flat trade; do not inflate statistical sample
size with each partial execution. Acknowledging an order does not change
inventory or cash. Use decimal arithmetic for broker quantity/price/order
strings; simulations may use float64 with explicit monetary tolerances.

Reward uses net equity change, scaled by initial strategy capital, plus an
incremental drawdown penalty:

`reward = 10000 * (delta_equity - 0.1 * positive_drawdown_increase) / initial_equity`

Spread, slippage and fees already affect equity; do not subtract them again.
Do not optimize an in-sample Sharpe reward or add an arbitrary holding
reward just to make the agent trade. The flat policy is a valid outcome.
Episode termination and forced liquidation pay costs. Never mark an open
position liquidated on a stale quote or a mere remote acknowledgement.

## 8. Learning and proof of an edge

Training runs offline on the laptop. PPO learns policy weights from repeated
randomized historical episodes. Candidate search also compares explicit
learning rate/entropy configurations. This is numerical machine learning,
not an LLM paid to decide each trade.

Freeze the last 25% of valid sessions (minimum 30) as final test data:
`n_test = max(30, ceil(0.25 * n_sessions))`. Keep
those dates and their dataset hash in a manifest before tuning. In the
first 75%, create three expanding folds: train on the first 60% of that
development period, validate on successive non-overlapping thirds of its
remaining 40%. Initial training contains `floor(0.60 * n_development)`
sessions; split the remainder into three chronological validation blocks
using integer division, putting residual sessions into the last block.
Require at least 30 initial training sessions and 5 in every validation block;
otherwise return insufficient-history. Use the same dates for all candidates. Do not train episodes
across validation/test boundaries. No test-day context is required because
each session warms up independently.

Initial smoke budget: one candidate, one seed, one fold, 2,048 training
steps. Research budget: three candidate configurations, two seeds, three
folds, 100,000 steps per fit, and a two-hour hard wall-clock ceiling. Save
completed trials and resume them by manifest rather than repeating them.
If fewer trials fit within the ceiling, report partial search; do not mark
the incomplete experiment eligible. CPU PyTorch uses two threads and four
vector environments initially; cap parent plus workers below 8 GB measured
resident memory. Begin with DummyVecEnv; switch to subprocesses only after
a benchmark demonstrates a benefit and memory remains bounded.

Candidate configurations `(learning_rate, entropy_coefficient)` are
`(3e-4,0.0)`, `(1e-4,0.01)`, `(5e-4,0.01)`. PPO MLP actor/critic each use
two 64-unit layers, `n_steps=256`, `batch_size=64`, `gamma=0.99`,
`gae_lambda=0.95`, `clip_range=0.2`. Reward/risk/execution assumptions are
held constant across candidates. Record all settings and seeds.

Rank by median validation net return across folds/seeds, subject to <5%
intraday peak drawdown in every fold. Break ties by lower turnover. Select
the configuration and seed using development-only cross-validation results,
then refit that seed on all development data. Never choose the seed using
final-test performance. Evaluate the single selected artifact once on final test.
If it fails, keep it in research; a new experiment must reserve later unseen
test dates rather than repeatedly mining the same test until it passes.

Controls: cash/no trade, always-long intraday at the same exposure (flat
overnight), seeded random entry/exit opportunities matched to actual policy
trade frequency, and a fixed EMA(5)/EMA(20) crossover using the same risk,
cash and execution costs. Repeat the random benchmark for 20 seeds. Report
the ordinary long-term stock return separately as context, not as an
intraday risk-matched benchmark.

Metrics: net profit/return, daily-return Sharpe, intraday maximum drawdown,
completed trades, net expectancy, profit factor, turnover, rejects,
unfilled orders, latency distribution and stress results. Use paired daily
return block bootstrap (2,000 samples, block length approximately sqrt(days)).
The test reports comparisons to each of three active controls; use Bonferroni
family alpha=0.05 across those three comparisons (bootstrap quantiles
0.05/(2*3) and 1-0.05/(2*3)). Random's paired daily control return is the mean
across its 20 seeds. Every metric includes sample counts.
Profit factor with wins and no losses is represented explicitly as
`infinite`, not JSON Infinity/NaN and not an automatic small-sample pass.

Paper eligibility: real executable data, >=30 test sessions, >=100 completed
trades, positive net expectancy, positive net profit, profit factor >=1.3,
drawdown <5%, positive stress profit, and a positive lower confidence bound
on excess daily returns against the strongest development-selected active
control. These are filters, not a promise of future profitability. If an
honest strategy cannot pass, the software is still complete and correctly
refuses graduation; the implementer must not loosen gates to produce a win.

## 9. Risk, persistence and runtime

- Entry allocation: at most 10% of session-start equity. Maintenance gross
  exposure cap: 15% of current strategy equity; no leverage and no shorting.
  The maintenance cushion lets ordinary gains occur without rebalancing
  every five seconds. Exceeding the hard maintenance cap requests flattening.
- Max order submissions: 5 per rolling minute. Unchanged targets submit none.
- Daily loss: halt at 1% of session-start strategy equity, inclusive of fees.
- Peak drawdown: latch halt at 5% of strategy equity peak; explicit operator
  reset is required after review, not a date change or process restart.
- Quote freshness: <=2 seconds; future timestamps >250ms fail closed.
- One outstanding order for the symbol; reconcile cancellation before a
  replacement. Partial fills remain protected and reflected in exposure.
- ATR breaker: completed ATR(14) above trailing prior ATR mean + 3 standard
  deviations, using at least 60 prior bars; pause entries for 60 seconds.
- Explicit optional news blackout file: block entries during listed UTC
  intervals. No paid news API or claim that the file covers all news.
- Disallow new entries during final 60 seconds of actual exchange session;
  flatten while the market is still open and reconcile final fills.
- Protect exits from normal entry rate/exposure limits. Quote staleness can
  stop local simulated fills; a remote emergency exit uses broker-reported
  position and must track the actual execution, even when local data failed.
- Network loss, system sleep, order uncertainty, account mismatch and stale
  data cancel queued entries and halt. Reconnect data collection freely;
  resume trading only after reconciliation and fresh session warmup.

One local owner per broker account, with file locking independent of log
directory. Store account-keyed state under a fixed project state root;
use a hash of mode/account ID, not the secret API key. Persist loss baselines,
peak, halt status, rolling order times, model/config identity, last fill IDs
and outstanding client order IDs with atomic replacement and fsync.

Maintain quote intake independently of potentially slow REST work. Use
bounded queues and a single serialized order manager. Cache account/clock
state with explicit age bounds; periodic reconciliation and trade updates
refresh it. Recheck risk against the latest fresh quote immediately before
the actual network send. Do not put several blocking HTTP requests in each
quote callback. Deduplicate trade updates by execution identity and book
incremental fills from cumulative quantity/notional when needed.

Timed-out POST is an unknown outcome. Query by recorded `client_order_id`;
never blindly resend a new order ID. Cancel acknowledgement is not terminal
cancellation. If liquidation cannot finish, preserve unresolved state,
write an alert, and exit nonzero; never log a clean/flat session.

## 10. Paper evidence and live activation

Dry-run logs and broker-paper logs must record actual wall-clock observation
times, model hash, feature schema, feed, symbol, config/cost hashes, mode,
session calendar, decisions, fill lifecycle, closed trades and equity. Replay
and synthetic data can never create qualifying real-time sessions.

Live eligibility requires the research gate plus >=30 consecutive complete
exchange sessions of the same frozen model/config in real-time broker-paper
mode, >=100 completed trades, positive expectancy, profit factor >=1.3,
drawdown <5%, and no unresolved orders or accounting mismatches. Also run the
local conservative shadow fill simulator on the same recorded live stream;
its net profit must be positive. Coverage is measured against the calendar
after warmup, not inferred from file names or the number of log records.
Measure different volatility/trend conditions and report their coverage;
do not fabricate mandatory regimes on a quiet month to pass a gate.

Evidence reports list every failed requirement, exact inputs and hashes.
Do not accept a caller-supplied `{'passed': true}` dictionary as authority.
The live CLI recomputes evidence from trusted local logs and manifests,
checks identity and age (most recent qualifying paper session no more than
five exchange sessions old), and then requires explicit `--enable-live` and
`--max-live-notional`. No training or smoke command can instantiate a live
client. Separate paper/live credentials and endpoint allowlists.

The only permitted activation-time behavior override is a **stricter**
absolute entry-notional ceiling for the canary. Recompute a full shadow
replay at that ceiling using the frozen model and the qualifying recorded
streams; it must retain positive net expectancy/profit and drawdown <5%.
Record both config hashes and this ceiling-specific result in the evidence.
It is not enough to assume a smaller order preserves model behavior. Any
larger cap or other behavior change requires new validated evidence.

Building or testing the software never sends orders, creates or funds
accounts, buys services, or starts live deployment.
Credentialed paper tests and elapsed real-time runs are separate operator
actions; the plan must state which checks are offline fixtures versus real
broker integration.

## 11. Deliverables and stopping point

Ship an installable CLI, reproducible tests, free data acquisition/recording,
causal simulation, resumable offline candidate search, frozen test report,
ONNX export/parity, replay/dry-run/broker-paper/live execution code, risk and
recovery, evidence reports, and an operator runbook. A smoke workflow must
run without credentials or network once dependencies are installed.

Optional later work: multiple symbols, shorting, passive market making,
native runtime, GPU training, automatic scheduling, and a web dashboard.
Do not build these before the initial acceptance path passes. Do not claim
the model is profitable or live-ready merely because the program runs.
