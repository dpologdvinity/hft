# Reusing stock-analyzer in this project

October 2, 2026. Feasibility assessment for reusing a separate pattern-analysis
project as a baseline. Integration is not implemented by this assessment.

## Recommendation

Yes: reuse selected analysis components from a separate local pattern-analysis project (`stock-analyzer`) for stronger
development baselines and clearer diagnostic breakdowns. Start with one causal
long-only pattern baseline evaluated through HFT's existing quote simulator.
Pattern inputs for PPO are a later experiment, justified by development evidence
and feature ablations.

Comprehensive development diagnostics and memory measurement remain the first
project. The analyzer does not resolve the current MCD feed-gap preflight failure
or establish a profitable model. Pattern logic still needs usable data.

## Inspected source and evidence

The analyzer has two tools: `stock.py` scores current ticker suitability from
fundamentals/indicators; `candlebench` measures candlestick signals on cached
history. Its last committed revision was `c156cc8`. The working tree also has
ongoing changes, including untracked `candlebench/quotes.py`. A future extraction
must record the exact source snapshot, preserve user edits, and carry relevant
tests and parameter defaults. A sibling checkout is not a portable dependency.

Focused offline analyzer tests ran with HFT's existing environment:
`test_patterns.py`, `test_context.py`, `test_metrics.py`, `test_windows.py`,
and `test_quotes.py`: **177 passed in 9.73 seconds**. These include future-bar
independence tests for every detector. This proves tested source behavior,
not a full integration or historical profitability. No network requests,
broker orders, background jobs or analyzer edits were performed.

## Valuable pieces

| Source in stock-analyzer | HFT use | Boundary |
| --- | --- | --- |
| `candlebench/patterns/context.py:56` (`trend_series`), `:91` (`geometry`) | Causal trend and OHLCV geometry for an experimental baseline. | Operate on HFT's completed published bars and contiguous decision history. |
| `candlebench/patterns/__init__.py:107` (`detect`) and pattern detectors | Explicit pattern signals with centrally applied history/prior-trend gates. | Preserve trend measurement before the pattern starts; long-only actions. |
| `candlebench/trades.py:175` (`breakdown`), `:150` (`time_bucket`) | Results grouped by time of day, exit reason and eventually symbol. | Adapt to HFT's verified trades and dollar P&L; HFT equity/drawdown remain authoritative. |
| `candlebench/quotes.py:70` (`half_spread_bps`), `:121` (`build_table`) | Sampled spread/depth diagnostics and calibration comparisons. | Bucket medians are estimates, not executable quotes or guaranteed liquidity. |
| Saved-run comparison UI | Ideas for comparing research runs and drilling into actual trades. | Add small views to the existing read-only HFT API after useful research; retain its security boundary. |
| `stock.py` | Possible current research shortlist using liquidity/spread indicators. | Present-day scores/fundamentals cannot serve as historical point-in-time observations. |

HFT already has cash, intraday-long, EMA(5,20), matched-random comparisons,
session-based paired bootstrap, funded account/risk state and quote-event fills.
Add a testable signal or explanation; retain these existing evaluation capabilities.

## Three integration approaches

1. **Experimental baseline and diagnostics — recommended.** Extract a reviewed,
   pinned pure NumPy subset with its tests. Route its decisions through
   `hft.research._rollout`. This tests whether it adds information under actual
   HFT costs without changing the PPO observation.
2. **Pattern features for PPO — later.** Add selected flags only after development
   evidence motivates them. Version the observation schema, freeze a new experiment,
   retrain/export, prove historical/runtime parity, and compare with/without features.
3. **Depend on the entire analyzer package — higher integration cost.** This brings
   pandas/yfinance, overlapping acquisition/evaluation code, and different bar-fill
   and short-trade assumptions. A whole-application merge is unnecessary for the pilot.

## Small first pilot after data readiness

Concrete source interfaces:

```text
geometry(arrays, trend_lookback, trend_min_slope)  # NumPy OHLCV arrays
patterns.get("bullish_engulfing")                # PatternSpec
patterns.detect(spec, geometry, thresholds)      # gated boolean array
```

Thresholds live in `candlebench.config.Thresholds`. Geometry and detectors use
NumPy/stdlib; the pilot can reuse that subset without adding the full package.
The candidate is one bullish-engulfing rule with prior downtrend context. Define
and select holding/exit behavior on development data before freezing evaluation.
Bearish signals can mean flat/exit; they must never introduce shorting.
This is a candidate baseline, not an assertion that engulfing has an edge.

Add the experimental rule beside the existing EMA policy in `hft.research.evaluate`.
Its internal `(observation, env)` callback can access `env.history` completed bars.
The current 17 features do not contain enough OHLC history to reconstruct these
patterns. Do not read future session arrays to generate the current decision.
Keep actual quote matching, freshness, latency, fees, size, partial fills,
liquidation, loss limits and account/risk carry in the existing executor.

## Incompatibilities and leakage risks

- **Bars/timing:** analyzer `ticks.resample` sorts by event time and drops empty
  intervals. It does not model HFT arrival order or boundary publication. HFT
  can publish zero-volume carry bars and reset history after gaps. Generate signals
  from HFT bars; do not substitute sparse analyzer bars as uninterrupted history.
  Current detector code rejects zero-range shapes, despite stale README commentary.
- **Execution:** analyzer `engine.simulate` enters at next OHLC open, resolves
  stops/targets from bar ranges, and sizes from a fixed risk allowance. It is not
  HFT's funded quote-depth engine. Its sampled quote table does not change that.
- **Contemporaneous cost knowledge:** analyzer `runner._estimate_spreads` uses
  an entire session's narrowest eligible bars. That estimate would leak future
  information if used as a current feature. Fit any calibration on allowed training
  data only, with separate provenance for observed versus estimated spread.
- **Stock scores:** `stock.py` uses present Yahoo data and statements.
  `Analysis.win_rate` is the fraction of heuristic checks passing, not a measured
  trade win probability. Historical use needs timestamped point-in-time snapshots.
- **Universe:** the current hardcoded liquid list is not a historical universe;
  using it for past selection introduces survivorship bias. It also includes ETFs.
- **Evaluation:** analyzer chronological windows summarize sampled periods;
  they are not trained expanding-window folds or an untouched final-test contract.
  Its individual-trade bootstrap can treat correlated trades as independent;
  `EDGE` can be assigned without a control. HFT gates remain authoritative.
- **Accounting:** summed per-trade returns and cumulative R are not funded portfolio
  returns. Analyzer cumulative-R drawdown omits the initial-zero peak and can
  miss the first loss. Group actual HFT trades while retaining HFT equity/drawdown.
- **Reserved dates:** inspect analyzer cache/report provenance for overlap before
  using its leaderboards for selection. Use only HFT development dates. If reserved
  data informed selection, it cannot still count as unseen evidence; obtain unseen
  final dates rather than reset reservations or move the same dates to another path.

## Acceptance and sequence

1. Complete the existing development-readiness plan and select usable data.
2. Pin the reviewed analyzer subset, exact defaults and tests. Write a bounded
   baseline design that specifies entry, holding and exit behavior.
3. Prove future-bar independence, prior-trend gates, session boundaries, gap resets,
   pending orders, stale-quote handling and long-only behavior in HFT. Confirm the
   detector runs on the same completed history in replay and runtime.
4. Evaluate through HFT chronological development folds and existing controls.
   Report net returns, uncertainty by session, trade counts, turnover, drawdown
   and stressed execution. Declare new comparisons before final evaluation and
   version affected contracts rather than relabeling old artifacts.
5. Add time-of-day/exit breakdowns from verified HFT trades. Consider PPO features
   only if development comparison and ablation evidence justify a new experiment.

Stop at a reproducible baseline comparison. This proposal does not automatically
consume final tests, enable paid services, merge applications or submit orders.
