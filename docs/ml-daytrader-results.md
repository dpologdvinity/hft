# ML day trader results

Validation of the [ML day trader](specs/2026-10-08-ml-daytrader-design.md): 84 variants
trained on 2016-2022 and scored on 2023-2024 by simulated trading with costs, against
holding the same 68 symbols (equal slots, late listings bought on their first
tradable day). Run on a Kaggle T4 GPU in about 20 minutes (code b8f72b6, data
fingerprint 54cce01d9385). The test years (2025-2026) were not touched.

## Result: no variant beats holding

Holding the 68 symbols returned **+97.2%** over the two validation years. The best
variants:

| Daily model | Target | Picks | Minute model | Total | Per trade | 3x costs | Edge vs holding (bp/day, 95% interval) |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| Trees | range | 3 | hold from open | +8.6% | +4.1 bp | -48.8% | -10.0 (-26.0 to +6.0) |
| Network | return, gated | 5 | hold from open | +13.6% | +4.1 bp | -33.5% | -10.9 (-20.3 to -1.4) |
| Network | return | 5 | hold from open | +12.6% | +3.1 bp | -42.5% | -11.0 (-20.6 to -1.5) |
| Network | range | 5 | hold from open | +5.7% | +2.7 bp | -49.3% | -11.4 (-23.9 to +1.1) |
| Network | range | 3 | hold from open | +1.2% | +2.6 bp | -52.3% | -11.6 (-27.7 to +4.5) |

The best variant's deflated Sharpe ratio (the probability that its true Sharpe is
above zero after 84 tries) is 0.03.

What the 84 variants show:

- **Daily picks held through the day earn about the trading cost.** The top
  variants made +3 to +4 bp per trade at the assumed costs, roughly the stocks'
  average intraday drift in a rising market. At three times the costs every one of
  them loses 33-53%. None is close to holding, which also earns the overnight
  moves the day trader gives up.
- **The minute models mostly learned to do nothing.** The GRU and transformer
  networks at a 15-minute horizon, and the transformers at 30 minutes, never
  predicted a gain larger than the round-trip cost, so 36 variants made no trades
  at all. The GRU at 30 minutes traded rarely (1 to 161 trades in two years).
- **When the minute models traded often, they lost.** The convolutional networks
  traded up to about 1,300 times; most lost 1-8 bp per trade, and the few with a
  small gain traded rarely. On days picked for their expected range they lost the
  most: -12% to -22%.

## Decision

No variant is frozen for the final test. All five candidates are below holding,
four of them with intervals entirely below zero, so spending the one-time test on
them would prove nothing. The test years stay unseen for a better idea.

This repeats the lesson of every earlier experiment ([project history](project-history.md)):
on these liquid stocks, the move a model can predict from prices alone at minute and
daily horizons is smaller than the cost of trading it.
