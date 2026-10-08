# Paper acceptance results

The first sessions of `hft trade --paper` against a real Alpaca paper account. They
test the plumbing: market data, decisions, orders, fills, ledgers, reconciliation,
shutdown, and the overnight schedule. Paper fills are simulated by Alpaca. These are
rule-strategy runs, which never count as graduation evidence.

## 2026-10-08: intraday runs (ema-crossover, NVDA and AAPL, $100 each)

| Run | Time (ET) | Round trips | Net | Ended by |
| --- | --- | ---: | ---: | --- |
| `acceptance` | 09:30-09:35 | 1 | +$0.03 | stopped to deploy fix 1 |
| `acceptance-2` | 09:36-09:47 | 4 | -$0.01 | bug 2 (cash mismatch) |
| `acceptance-3` | 09:52-15:40 | 180 | +$0.90 | planned stop; flattened on SIGTERM |

`acceptance-3` made 90 round trips per stock, about +0.26 bp (AAPL) and +0.75 bp
(NVDA) per trade, with 27 and 30 winners. The spread guard refused 32 entries, and
the volatility breaker 43. One day of rule trading says nothing about an edge. The
quote-level replay of this strategy lost about 4-5 bp per round trip
([replay results](replay-results.md)), so Alpaca's simulated paper fills look
more generous than the project's simulator; paper profits should not be read as
achievable live.

## Bugs found and fixed the same day

Each fix has a regression test that fails on the old code.

1. **One stall halted a stock for the day.** At 09:31:07, during the opening burst,
   130 NVDA events arrived more than 1 s late. The backlog rule from the original
   design halted NVDA for the session. Now stale events are dropped and the stock
   warms up again from fresh bars (15fd6b2).
2. **Cent rounding stopped the run.** Alpaca books each fill's cash to the cent; the
   ledger keeps exact decimals. After eight fractional fills they differed by
   $0.012 against a fixed $0.01 tolerance. Reconciliation now allows half a cent per
   fill since its last clean check and re-anchors (d1e1a0b).
3. **REST latency aged market events.** `acceptance-3` restarted warmup 37 times in
   six hours at low machine load: events waited in the queue while the loop made
   about 0.5 s of sequential REST calls. Queued events are now routed after every
   broker call (7d6e255). The run itself kept the old code.
4. **Paper expires limit-on-close orders.** The overnight run's three limit-on-close
   entries (BAC, NVDA, PFE) were accepted and then expired unfilled at the close,
   although the close was below each limit. Whole-share entries now use
   market-on-close orders sized as if filled 2% above the 15:45 price (b82d767).
5. **Status after a restart.** The newest journal of a restarted run has no equity
   row yet, so status showed holdings as near-total losses. It now reads older
   journals and never shows a holding at zero (1438cc7).

## 2026-10-08: first overnight entry (overnight-drift, 10 stocks, $500 each)

Started 15:41 ET. Seven fractional marketable limit buys at 15:57 filled within
cents of the reference trade (AAPL 340.55, AMZN 254.22, GOOGL 348.09, JPM 331.64,
KO 87.74, MSFT 522.85, XOM 168.38). The three on-close orders expired (bug 4), so
BAC, NVDA and PFE hold nothing tonight. The run was restarted with the fixes and
resumed its saved state and positions. The opening exits are next.
