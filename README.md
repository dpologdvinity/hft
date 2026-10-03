# Local AI stock trader

This is a CPU-compatible stock research and trading system. It trains PPO on
historical quote/trade events, evaluates unseen sessions, exports an immutable
ONNX policy, and supports replay, local dry runs and Alpaca broker paper trading.
It makes decisions every **5 seconds**. A laptop and free IEX data are a practical
starting point for automated intraday trading; this does not provide exchange
colocation or submillisecond high-frequency execution.

The current
real-data experiment is blocked by development coverage; no model has qualified.

Start with the offline engineering check:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.txt
python -m pip install --no-deps -e .
python -m hft smoke --output artifacts/my-smoke
```

The smoke command performs actual PPO training, ONNX export/parity checks and
quote-event replay. Its generated data is explicitly synthetic. Its report says
`research-only`, and it cannot qualify a policy for streaming or live trading.
Use a new output directory for each smoke run.

For inference-only installation, use `requirements-runtime.txt`. The runtime
imports neither Torch nor Stable-Baselines3. `constraints.txt` records the package
versions used during verification; the training requirements select CPU Torch.
Python 3.12 or later is required. Training defaults to two CPU threads and four
in-process environments, with a two-hour cumulative search budget and an 8GB
process memory ceiling. Data has separate resident and transient memory guards.

## Local dashboard

```bash
npm --prefix frontend ci
npm --prefix frontend run build
python -m hft dashboard
```

Open **http://127.0.0.1:8765**. Overview, Research and Paper trading show local
dataset history, recorded training outcomes, data-quality diagnostics and verified
paper journals. Refresh runs automatically every 30 seconds. Failed reads retain
the last snapshot and label it stale. The dashboard binds to your computer only,
keeps credentials on the server, and has no order or activation controls.

Market history displays the last reported trade from each development session;
it is not a strategy profit chart. Reserved final-test prices are never read by
the dashboard. Empty results stay empty until actual evidence exists. See
[frontend setup](frontend/README.md) for development commands.

## Get free stock data

Create an Alpaca paper account yourself and provide market-data credentials in
local environment variables. Keep keys out of Git and logs:

```bash
export ALPACA_API_KEY='your-market-data-key'
export ALPACA_SECRET_KEY='your-market-data-secret'
python -m hft data-probe --symbol AAPL --session 2026-09-28
python -m hft download --symbol AAPL --start 2026-03-01 --end 2026-09-30 --output data/aapl
python -m hft prepare --input data/aapl/manifest.json
```

These commands only read data/calendar endpoints. No paid feed fallback exists.
Free IEX is one exchange's feed; the software does not pretend it is consolidated
NBBO or a Level 2 order book. The probe distinguishes holidays, empty coverage
and denied historical access. Historical entitlement must be checked on your own
account. Downloads paginate, checkpoint checksummed session partitions and resume
by rerunning the same command. They preserve a 5GB disk reserve.

If past quote/trade data is unavailable, collect real events locally instead:

```bash
python -m hft record --symbol AAPL --output data/recordings/aapl --duration 23400
python -m hft prepare --input data/recordings/aapl
```

Start before the actual exchange opening and keep the computer awake. Recording
preserves arrival timestamps. Partial sessions remain labelled partial for
eligibility. A recording directory holds one stock. Missing history produces an
insufficient-data report, never invented historical quotes. Calendar records
handle holidays, DST and early closes.

## Train and compare

```bash
python -m hft experiment --dataset data/aapl/manifest.json --config config/research.toml --output artifacts/aapl/experiment.json
python -m hft data-quality --experiment artifacts/aapl/experiment.json
python -m hft train --experiment artifacts/aapl/experiment.json
python -m hft train --experiment artifacts/aapl/experiment.json --resume
python -m hft report --experiment artifacts/aapl/experiment.json
```

Edit `config/research.toml` **before freezing an experiment**, including
`initial_cash` if your strategy allocation will differ from $500. The search
compares three PPO configurations, two seeds and three expanding validation
folds. It refits the selected configuration on development data and consumes the
reserved final test once. Real final-test dates are registered in `.state/` so
another output filename does not silently allow retuning on the same dates.
After a failed final test, obtain later unseen sessions for another final test.
Interrupted final evaluation may require fresh test dates; it fails conservatively.

The development-only data-quality check uses the shared simulation to count
warmup coverage and gaps. Training runs this check automatically before fitting.
Any event gap over the runtime's five-second limit blocks the experiment; fewer
than 61 completed bars or unverified data provenance also fail. The check stops
at the first unusable session and records the number checked in `diagnostics.json`.
It does not evaluate the final test. Setup, preflight and fitting all count toward
the cumulative training time budget. Resuming a completed final report preserves it.

The current local MCD dataset has 82 real IEX sessions, with 52 development and
30 reserved final-test sessions. Its feature-v3 experiment is
`artifacts/mcd-v3/experiment.json`. The first development session fails coverage:
936 event gaps, 42 eligible warmup decisions and 4,577 ineligible decisions.
Training is blocked before fitting, and profitability remains unproven. Use data
with sufficient continuity for these execution rules; extra training cannot fix
missing market coverage.

Reports include cash, risk-matched intraday long, EMA 5/20, twenty random controls,
transaction costs, completed flat-to-flat trades, quote-level drawdown, stress
execution and a corrected paired block bootstrap. A separate full-stock return
is context rather than a risk-matched control. Failure to match actual random
execution frequency prevents eligibility. Too little history is an explicit
`insufficient-data` result. Passing engineering tests does not establish an edge.

The resulting `artifacts/aapl/search/bundle/` contains `model.onnx` and a hashed
manifest with the feature, execution, dataset and research contracts. Old single
ONNX files and the previous three-action schema are rejected.
Feature-v2 bundles and older frozen experiments are also incompatible with the
current feature-v3 and causal boundary-timer contract. Freeze a new experiment
and retrain; do not relabel an old model or delete final-test reservations.

## Replay and dry runs

```bash
python -m hft replay --model artifacts/aapl/search/bundle --dataset data/aapl/manifest.json
python -m hft dry-run --model artifacts/aapl/search/bundle --duration 3600
python -m hft status --logs logs --state .state
```

Replay and dry-run use the same long-only fractional sizing, ledger, features,
risk gateway and displayed-liquidity execution simulator as training. Fills need
a genuine eligible future quote after 75ms latency; they cannot use a bar close or
invent liquidity. Defaults allocate 10% of session-start strategy equity, reserve
fees, cap maintenance exposure at 15%, enforce daily loss at 1% and persistent
peak drawdown at 5%, and limit entries to five per minute. Repeated long signals
hold the existing quantity. The strategy can grow its allocation with earnings
within its frozen hard notional ceiling.

The last minute of the actual session is reserved for closing. Warmup requires
61 completed bars each session. Stale/future quotes, excessive volatility, sleep,
backlogs and unresolved orders prevent entries. A stop without a subsequent
executable local quote records remaining inventory; it cannot fabricate a fill.
Local account and risk state survives restart. Simulated outstanding orders are
cancelled on restart; broker orders are reconciled.

Replay publishes bars at causal five-second timer boundaries. Training and
runtime keep the same bounded 61-bar history and reset warmup on feed gaps.
Truncated training episodes use genuine subsequent quotes for bounded liquidation;
unresolved positions block reset rather than disappearing. Incomplete normal,
stress or primary-control rollouts cannot qualify. Historical data without arrival
timestamps still cannot establish actual network latency, and boundary-timer
replay idealizes the streaming loop's polling delay.

## Broker paper trading and graduation

Broker paper trading submits real requests to the **paper endpoint only**. Use a
dedicated account with no unrelated positions/orders, with strategy capital no
larger than verified non-borrowed cash:

```bash
export ALPACA_PAPER_API_KEY='your-paper-trading-key'
export ALPACA_PAPER_SECRET_KEY='your-paper-trading-secret'
python -m hft broker-paper --model artifacts/aapl/search/bundle --capital 500
python -m hft graduate --model artifacts/aapl/search/bundle --logs logs/broker-paper --output artifacts/aapl/graduation.json
```

Keep market-data credentials configured too; paper credentials are accepted as
their fallback. Runtime credentials for paper and live order endpoints are
separate. No account is created, funded or reset by this program.

Quote intake uses a bounded independent task while REST order operations run
serially in a worker thread. Trade updates provide primary notifications and REST
polling recovers missed updates. An acknowledgement is not a fill. Client order
IDs and pending state are durable before POST; a timeout triggers lookup using
the same ID, never a second blind submission. A cancellation acknowledgement does
not establish a terminal status. Fresh broker inventory permits an emergency
market reduction when local data has failed. Unexpected positions, orders, cash,
fees or funding changes stop trading and disqualify that session.

The gate recomputes results from hash-checked journals and their executions,
checks the frozen research result, and requires at least 30 consecutive complete
real broker-paper sessions, 100 completed trades, positive profit/expectancy,
profit factor >=1.3 and drawdown <5%. It replays recorded events under worse costs
and latency, and checks the stricter live canary separately. Evidence expires
five actual exchange sessions later. Replay, dry runs, synthetic sessions,
incomplete coverage, changed settings and a caller-supplied `passed` value cannot
satisfy the live constructor. Hash chains detect corruption, not a malicious
owner rewriting the whole application or its local files.

## Live activation and operations

The build and tests never submit live orders. The operator must separately
configure `ALPACA_LIVE_API_KEY` and `ALPACA_LIVE_SECRET_KEY`, provide a frozen real
bundle, and explicitly activate the command. After passing graduation:

```bash
python -m hft live --model artifacts/aapl/search/bundle --logs logs/broker-paper --enable-live --capital 500 --max-live-notional 10
```

This path rechecks evidence and the current broker calendar before trading. The
first live allocation is capped at $10 per entry and stops after ten completed
live sessions for operator review. Raising the ceiling is not implemented as an
automatic response to profits. Changing capital, feed, features, costs or risk
settings requires a newly evaluated bundle; the only supported execution override
is a stricter canary cap with its own replay.

Use Ctrl-C for controlled shutdown. Remote shutdown cancels outstanding orders,
waits for final fills, closes verified remaining inventory while the market is
open, and reconciles. Unresolved shutdown returns an error and leaves durable IDs
and truthful logs. Inspect the broker's positions/open orders before restarting.
Do not delete `.state/` to bypass an unknown order, a drawdown latch or a failed
reconciliation. A second process cannot own the same account by choosing a new
log or state path.

Actual broker regulatory fees not exposed on an order, external transfers and
corporate actions currently require operator reconciliation: cash differences
beyond one cent halt entries rather than being silently treated as trading PnL.
Fractional limit order acceptance and fee/settlement behavior still need a real
paper-account integration run. Broker rules and entitlements are checked through
account/API responses; the code does not embed an obsolete universal $25,000
intraday-trading assumption.

See [the specification](docs/trading-spec.md), [implementation plan](docs/trading-implementation-plan.md)
and [verification record](docs/progress.md) for contracts and proof. Run local
verification with `python -m pytest -q`, `ruff check hft tests` and
`ruff format --check hft tests`. Actual free-data access, profitable historical
results and elapsed paper/live sessions remain external gates.
