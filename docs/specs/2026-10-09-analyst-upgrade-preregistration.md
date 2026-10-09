# Pre-registration: day-trading pre-open analyst upgrades

Written and committed before any validation (2023-2024) or test (2025-2026) result
for this rule was computed.

## Where the idea came from

A diagnostic on the training years only (2016-2022, 112,519 symbol-days of the 68
trainable symbols, [news features](2026-10-09-news-features-design.md)) found that on
days with a focused pre-open article matching an analyst upgrade or price-target
raise, the stock's open-to-close move averaged +10.4 bp (standard error 3.7 bp,
3,360 days), against +0.4 bp on days without news. Downgrades averaged -5.6 bp
(standard error 5.4 bp). Earnings beats averaged -18.6 bp (the gap fades), so they
are not part of the rule.

## Rule

- Each session, a symbol is a candidate when its pre-open news window (previous
  close to 09:25 ET) holds net analyst-up events: `news_analyst_net > 0`, from focused
  articles (at most three symbols tagged).
- Buy at the second minute's open (09:31), sell at the open of the last minute
  (five minutes before the close), with the ML day trader's fill simulator: real
  later bars only, each side paying the symbol's half-spread (at least half a cent)
  plus 1 bp slippage.
- No other filter, no tuning. Long only.

## Evaluation

- Validation: sessions 2023-01-01 to 2024-12-31. The validation years were already
  used for 84 ML variants; this rule adds one trial.
- **Primary:** mean net return per trade, with a 95% bootstrap interval resampling
  dates (all trades of a date together). Pass if the lower bound is above zero.
- **Secondary (reported, not the pass bar):** a portfolio with equal slots of 1/5 of
  capital per candidate (at most five per day, by most analyst-up events, ties by
  symbol order), cash otherwise, against holding the same symbols. The rule is in
  cash most days, so it is not expected to beat holding on its own; the per-trade
  edge is what would make it a building block.
- Also reported: results at 3x costs, the downgrade mirror (would shorting have
  paid; reported only, long only stays), and trades per year.
- If, and only if, the primary passes on validation, the same rule is run once on
  the test years (2025-01-01 onward) with the same bar, and recorded either way.
