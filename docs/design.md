# Trading research and paper execution

The supplied `PROMPT.md` is the engineering brief. Build an end-to-end,
single-symbol US-equity research system, with discrete flat/long/short
one-share targets and a Python ONNX execution runtime. No profitability or
sub-millisecond latency is assumed. No real-money orders are submitted by
this implementation: promotion is evidence, not permission to deploy.

## Data and accounting invariants

Parquet rows contain UTC nanosecond timestamps, OHLCV, bid/ask, and quoted
bid/ask sizes. Reject unsorted, duplicate, nonfinite, negative-volume,
nonpositive-price, crossed-quote, and invalid-candle data. Features use only
completed rows, with 60-row warmup, a trailing VWAP and fixed transforms.
The same feature builder and account ledger serve training and execution.
No fitted normalization can leak held-out information.

Target changes buy at ask and sell at bid, with adverse slippage and
per-share commission. Repeating a target creates no fill. Reversal closes
the prior trade before opening the next; fees belong to the appropriate
trade. Equity changes already include trading costs, so reward deducts
them once, plus optional drawdown and downside penalties. Episodes execute
at the following row, liquidate at boundaries, and never execute against
the row that generated the signal.

## Research

Gymnasium owns the environment contract; SB3 PPO owns optimization and
vectorization. Random episode starts and optional coherent price inversion
and volume jitter provide training augmentation. Expanding walk-forward
splits train only on earlier days, with preceding warmup used solely as
evaluation context. Benchmarks share the execution costs and sessions:
long, seeded random targets at the policy's target-change frequency, and a
candle rule. Report daily Sharpe, drawdown, net expectancy, profit factor,
and paired daily-return bootstrap intervals. Never claim an edge from a
single synthetic smoke run.

Export deterministic PPO actions to ONNX with feature/account/cost metadata
and an artifact digest. Verify action parity before accepting the export.
Runtime rejects incompatible metadata, dimensions, hashes, and actions.

## Paper execution and risk

Replay uses validated rows. Alpaca WebSocket quotes and trades build
completed 1-second or 5-second bars; trade volume is real trade volume,
not quoted size. Live data gaps and disconnects halt the process rather
than silently continue. Credentials come only from environment variables.
The default local broker simulates a 75ms minimum delay and fills against
the next eligible quote. Revalidate risk at fill time, deduplicate targets,
and cancel pending increases on a halt. Session-close liquidation and
emergency risk reductions bypass the normal order-rate limit.

Risk owns daily loss, peak drawdown, exposure, leverage, stale quotes,
order rate, ATR spikes, and explicitly supplied news blackout intervals.
Logs record decisions, rejections, fills, equity, slippage and halts as
append-only JSONL. A localhost read-only dashboard displays those logs.
Paper account connectivity can be verified without sending real orders.

Graduation requires at least 30 consecutive exchange sessions, coverage of
three reported regimes, strictly positive net closed-trade expectancy,
profit factor >= 1.3 and drawdown < 5%. Missing evidence fails closed.
Live routing, automatic allocation scaling and live slippage auditing
remain unavailable until real paper evidence and broker integration tests
exist; research code must not pretend to have completed those phases.

## Acceptance

Run the accounting, causality, split, paper-latency, risk and evidence
tests; exercise Gymnasium validation; generate synthetic Parquet; train a
small PPO model; evaluate, export and replay it; inspect its logs and
dashboard. Commit each coherent completed component. Real historical
data, broker credentials, profitable results and 30 days of elapsed paper
trading are external requirements, not artifacts a code build can supply.
