# Development research readiness: measured results

Measured October 3, 2026 (UTC), using the existing local CPU environment. All
numbers below describe development diagnostics, not training or profitability.
**No model has qualified.** The MCD v3 search remains blocked before fitting.

## Reproduction and identity

```bash
.venv/bin/python -m hft data-quality --experiment artifacts/mcd-v3/experiment.json --all-development
```

- Frozen experiment hash: `ad1a09f1d8a1bd60ce1f3b3928dcf5f2600f2267be975848e67d8afff1d5cfc4`.
- Dataset index SHA-256: `cacce9dc91dc89802d7797e2d650eab2f40cf086290c08b35f3b8fa9888ac4c9`.
- Feature schema: v3; aggregation: `causal-5s-v3-boundary-timers-latest-quote-250ms`.
- Output: `artifacts/mcd-v3/diagnostics.json`, atomically published on success.
  Measured output SHA-256: `de38fb24ee3e0927a91594ae8500f89824d4de476682e941f924481a4c73fe29`.
- Prior one-session report saved locally as
  `artifacts/mcd-v3/diagnostics-before-all-development.json`. Generated artifacts
  are ignored by Git; these recorded findings survive a later default preflight
  replacing the full report with a correctly labelled prefix.
- External monotonic elapsed time: **318.651 seconds** (5m 18.65s), exit code 0.
  Other local verification activity overlapped part of the run; this is observed
  elapsed time, not an isolated throughput benchmark or a PPO search estimate.

Only the 52 declared development session manifests/partitions were loaded, from
**2026-06-04 through 2026-08-18**. The 30 reserved final sessions were not inspected.
Before/after SHA-256 checks matched for `experiment.json`, the dataset index,
`search/research.json` and `search/budget.json`. `final_test_consumed` stayed false;
`.state/research-final-tests.json` was absent before and after. No reservation was
created, reset or consumed. No fitting, download, broker order or paid service ran.

## Coverage and usability

| Measurement | Result |
| --- | ---: |
| Inspected / declared development sessions | 52 / 52 |
| Coverage complete | true |
| Sessions with no diagnostic reasons | 0 |
| Sessions with `runtime-feed-gap` | 52 |
| Sessions also with `insufficient-decision-warmup` | 10 |
| Eligible decisions | 2,054 |
| Ineligible decisions | 238,106 |
| Counted decisions | 240,160 |
| Aggregate eligible fraction | 0.855263% |
| Feed gaps | 57,049 |

The result remains `insufficient-data`, `live_comparable=false` and
`paper_eligible=false`. Reasons overlap: the ten zero-eligible sessions are a
subset of the 52 gap sessions. Per-session eligible fractions range from 0% to
4.914484% (July 27). The aggregate counts equal the sums of the inspected rows.

Eligibility here measures the simulator's contiguous decision-history warmup,
not permission to place an order. It excludes the initial 61-bar decision prefix
and does not waive quote freshness, loss/risk halts or execution checks. Complete
inspection means all development days were accounted for; it does not mean any
day passed or that historical timestamps prove live feed health.

## Memory measurements

| Retained buffer accounting | Sum across inspected sessions | MiB |
| --- | ---: | ---: |
| Numeric arrays | 95,774,000 bytes | 91.34 |
| Metadata columns | 595,218,153 bytes | 567.64 |
| Total | 690,992,153 bytes | 658.98 |

Sessions were released before loading the next. These sums are **not simultaneous
resident memory**. Metadata contributes 86.14% of the accounted bytes. Quote and
trade raw JSON together contribute 331,477,278 bytes, or 47.97% of the total.
The existing accounting uses referenced Arrow/NumPy `.nbytes`; it excludes Python
objects, timeline indices and transient copies and does not deduplicate shared
buffers. The small Python-list fixture fallback uses Arrow sizing.

Largest retained sessions:

| Development date | Numeric bytes | Metadata bytes | Retained bytes |
| --- | ---: | ---: | ---: |
| 2026-07-27 | 5,083,472 | 31,426,501 | 36,509,973 |
| 2026-08-05 | 3,088,904 | 19,151,276 | 22,240,180 |
| 2026-08-03 | 2,908,584 | 18,097,903 | 21,006,487 |
| 2026-07-15 | 2,715,344 | 16,913,301 | 19,628,645 |
| 2026-08-17 | 2,661,960 | 16,472,818 | 19,134,778 |

Leading metadata columns, summed across inspected sessions:

| Column | Bytes |
| --- | ---: |
| `quote_metadata.raw_json` | 298,063,367 |
| `quote_metadata.i` | 151,527,256 |
| `trade_metadata.raw_json` | 33,413,911 |
| `quote_metadata.c` | 20,055,078 |
| `quote_metadata.raw_as` | 18,105,305 |
| `quote_metadata.raw_bs` | 18,105,305 |
| `quote_metadata.ax` | 11,141,710 |

Per-session retained totals range from 5,544,392 to 36,509,973 bytes. The largest
sampled **cumulative process peak RSS** is **309,493,760 bytes (295.16 MiB,
0.288 GiB)**, below the 8 GiB target for this diagnostic workload. Linux
`ru_maxrss` is converted from KiB to bytes. A row's RSS includes earlier sessions
and any earlier process allocations; it is not that day's memory allocation.
Neither these buffer sums nor this sequential RSS proves the eager loader,
prepared-simulation caches or a full PPO search fit in memory.

## Next project recommendation — proposal

Prioritize a **bounded development-only free-data suitability probe** using the
existing acquisition and diagnostic modules. The measured blocker is continuity:
all 52 MCD days fail the existing five-second contract. Additional PPO steps or a
pattern policy cannot repair that. Diagnostic memory is comfortably within the
target, so a loader rewrite is not the first remedy for this dataset.

In that separate project, select a small declared set of development dates for a
denser stock, review existing partial AAPL archive provenance first, then use only
the existing free read-only feed if acquisition is needed. Preserve final dates
and cross-experiment reservations. Record the same session reasons/coverage,
retained-column bytes, RSS and elapsed time; preserve the current gap, 61-bar
warmup and two-second quote-freshness guards. Stop after a bounded sample and a
truthful go/no-go result. Do not broaden acquisition or training automatically.
If denser data also fails, return to the roadmap's evidence-based feed-health or
slower-strategy designs as a separately versioned contract, rather than raising
a timeout to pass a gate. Historical silence alone cannot distinguish a quiet
instrument from loss of feed.

Raw JSON is the leading measured memory candidate for a future execution-view
projection if denser data creates pressure. Preserve raw archives and checksums,
trace field consumers and prove identical events/execution before any projection.
No projection or feed-health change is implemented here.

After completing diagnostics, reviewed [stock-analyzer integration](stock-analyzer-integration.md).
Keep its pinned causal NumPy bullish-engulfing/prior-downtrend baseline as a
**separate follow-up after data readiness**. Use HFT completed history, quote
execution, chronological development folds and controls; pin source/defaults,
prove future-bar independence and define exits before evaluation. No sibling
repository edits, detector extraction, PPO features or baseline comparison were
performed in this project. The analyzer's OHLC executor and heuristic verdicts
cannot replace HFT accounting or qualification.

## Verification

- Focused red/green regressions cover full/default traversal, reserved-manifest
  isolation, per-row reasons, checksum/identity/path/index validation, deadlines,
  previous-report preservation, zero denominators, provenance, metadata sizing,
  unchanged events, session buffer release, RSS conversion and CLI dispatch/help.
- Full Python rerun: **188 passed**, two existing ONNX exporter deprecation
  warnings. The first sandbox run hit denied loopback socket creation; that
  dashboard test passed with socket access. A stalled command wrapper required
  a full rerun with loopback access and faulthandler reporting; no application
  change was needed for it.
- Ruff check, Ruff format check (50 files), dependency consistency and
  `git diff --check` pass. Pip emitted a sandbox cache-permission warning;
  dependency consistency passed.
- Frontend unchanged; no frontend checks or browser run required by this plan.

Remaining blockers: usable development data under the current execution contract,
then development baseline/model evidence, a genuinely reserved final evaluation
and elapsed paper evidence. None is established by these diagnostics.
