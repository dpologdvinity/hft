# Free-data suitability probe — October 4, 2026

The declared development-only sample contains AAPL, MSFT and NVDA on June 4–5, 2026. The existing GET-only IEX downloader and unchanged v3 simulator inspected all six sessions. No fitting, final-test price inspection, reservation, broker order or paid service ran.

## Results

| Stock | Date | Quotes | Trades | Feed gaps | Warmup-eligible decisions | Retained MiB | Process peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AAPL | 2026-06-04 | 1,112,754 | 24,761 | 3 | 96.04% | 308.83 | 871.89 |
| AAPL | 2026-06-05 | 1,493,820 | 38,959 | 0 | 100.00% | 415.19 | 1088.67 |
| MSFT | 2026-06-04 | 75,147 | 18,408 | 396 | 13.23% | 23.56 | 1088.67 |
| MSFT | 2026-06-05 | 78,467 | 17,504 | 279 | 21.22% | 24.24 | 1088.67 |
| NVDA | 2026-06-04 | 2,829,418 | 50,595 | 0 | 100.00% | 788.45 | 1736.29 |
| NVDA | 2026-06-05 | 5,103,471 | 76,525 | 0 | 100.00% | 1420.28 | 2989.46 |

Both NVDA sessions pass development coverage; AAPL June 5 passes, June 4 fails with three gaps. Both MSFT sessions fail. These two dates do not establish broader coverage, predictive edge, executable profitability or paper eligibility. Eligibility here counts uninterrupted feature warmup, not successful entry/fill checks. Historical events lack observed arrival timestamps.

Process peak RSS is cumulative across the sequential probe, not a per-session allocation. NVDA June 5 retains approximately 1.39 GiB; 86.17% is metadata. A full eager training archive would exceed the existing resident limit after only a few comparable sessions. Do not extrapolate sample suitability into a qualified training dataset.

## Provenance and reproduction

The original partial AAPL archive remains unchanged: June 4 is complete; June 5 has an unfinished temporary quote partition. Its archive manifests carry the historical v2 aggregation label. The complete June 4 raw files were checksum-verified and copied into a separate cache; the existing downloader reused those bytes while writing fresh acquisition manifests for the current diagnostic contract. No old experiment or bundle was relabelled.

Declared plan and one-off measurement script live in ignored `artifacts/free-data-probe-2026-10-04/`; downloads live in ignored `data/free-data-probe-2026-10-04/`. The script writes diagnostic-only `probe.json` descriptors with no final dates or folds, not training experiments. Do not pass them to `train`. Acquire another sample with the existing `download` command and inspect only explicitly declared development dates. All chosen symbol/feed/date identities were checked against existing frozen final-test sets and the global registry before acquisition.

The measurement process ended after publishing all six diagnostic rows, before its final summary update. A separate verification accounted for all six rows and checked every protected checksum against the declared plan: original AAPL files, MCD index, existing experiments, MCD diagnostics/search/budget and absent final-test registry remain unchanged.

| Stock | Acquisition seconds | Diagnostic seconds | Dataset index SHA-256 | Diagnostics SHA-256 |
| --- | ---: | ---: | --- | --- |
| AAPL | 71.798 | 125.684 | `fc3779ac9f43d582073c4ad95021e854e0610ebc6cbc6f0cc664ab0e093d9db7` | `a7c155a5b1795f63b5006ca036afa988df73e5dc06defa7cfc8b49ded33616e3` |
| MSFT | 9.697 | 8.424 | `15d510fd8b34f6c3a42704e3eda01f44fd2e4dbedd156e7590a9171c5af9a96f` | `b5bd1a444a67d24e6fdb7ca26e581770caaec94b83a1e7b7bc8321613d525e45` |
| NVDA | 377.108 | 436.238 | `50e6c3b22129d576dad6f605a56a95269ba77a39aada055ca687e9e339519043` | `bee71c178b0ad0b806dc9f20df7b4dc865a4547b8191b047ae6f731e445f097a` |

Declared plan SHA-256: `28bf8faf2965b743bdf7e1af25d0211021d6e28bd460723c455662e46554a060`.

## Research loading correction

Inspection found that experiment freezing and search setup eagerly opened final-test prices before reservation. Freezing now validates checked session manifests without opening Parquet. Search setup loads only development sessions; final evaluation loads its prices only after the global reservation and persisted consumed marker. Session selection, archive checksums, phase order, frozen hashes, partition validation and resident limits remain enforced. Existing process peak RSS is now included in the transient loading guard, covering retained model/simulation allocations; this conservative high-water mark can reject loads after memory was released.

A metadata-only freeze of the actual 82-session MCD archive took 0.161 seconds and peaked at 68.16 MiB. A file-open spy confirmed zero market partitions opened. It reproduced the existing v3 experiment hash exactly: `ad1a09f1d8a1bd60ce1f3b3928dcf5f2600f2267be975848e67d8afff1d5cfc4`. This is a freeze measurement, not full-search memory proof.

## Next bounded project

Prioritize a measured execution view that projects proven-unused archive fields out of memory while retaining raw Parquet/checksums and all identifiers, conditions, exchange, tape and arrival consumers. Then establish bounded session/prepared-simulation ownership and measure throughput/RSS before acquiring a broad NVDA development archive. Extend the sample across predeclared development dates before deciding training readiness; do not select only passing dates, weaken the five-second rule, or use reserved final prices.

Alpaca documentation confirms IEX is single-exchange coverage; consolidated historical data is not equivalent to the free live IEX execution feed. See the [market-data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq).
