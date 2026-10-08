# Rule-strategy replay results

`python -m hft trade --replay DAY` runs each stock through the paper bot's engine,
strategy, budget sizing and risk gateway, with quote-level fills simulated on recorded
IEX quotes (75 ms latency, $0.005/share, 1 bp slippage). These are single-day
replays: they show how the bot behaves, not whether a strategy has an edge, and they
never count as research or graduation evidence.

## 2026-10-07, ema-crossover, $100 per stock

| Stock | Decisions | Round trips | Winners | Net | Stock moved | Stopped by |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NVDA | 4,620 | 54 | 9 | -2.00% | -0.12% | daily loss at 12:15 ET |
| AAPL | 3,585 | 37 | 6 | -2.02% | -0.13% | daily loss at 11:16 ET |

Both stocks were flat, so the losses are friction, not direction. The 5/20-bar EMA
crossover flips about every 50 seconds (median hold 50 s). Each round trip loses
3.7 bp (NVDA) or 5.5 bp (AAPL) on average, about the cost of crossing the spread and
paying slippage. The 2% daily-loss limit halted each stock as designed. This matches
the cost-hurdle analysis: crossing the spread on 5-second to 30-minute horizons needs
predictive power far beyond realistic signals. A tradable rule needs much longer
holds or far fewer trades.

## Wide-spread entry guard (MSFT, 2026-06-05)

On MSFT, IEX's own book was thin (26-71 bp spreads). It showed a bid of 406.56 while
the market traded near 428. Without a spread limit, the bot bought at 428.79. Marked
at the IEX quote, the position showed a large loss, which tripped the loss limits.
The bot then sold into the 406.56 bid 21 seconds later and stayed halted for the day.
Budget runs now skip entries while the spread exceeds `--max-spread-bps` (default 10).
Exits are never blocked.

| MSFT 2026-06-05 | Round trips | Entries refused for spread | Net | Stock moved |
| --- | ---: | ---: | ---: | ---: |
| No spread limit (1000 bp) | 1 | 0 | -5.19% | -2.63% |
| 10 bp limit (default) | 3 | 54 | -2.22% | -2.63% |

With the limit, the fake-quote entry never happened. The remaining three trades were
ordinary losses on a falling day, and the daily-loss limit stopped the stock at 13:31 ET.
A held position can still be marked against a thin book. Liquid IEX names (NVDA, AAPL
near 2 bp) remain the recommended choice.
