# Overnight and multi-day rule results

Evaluation of the [pre-registered rules](specs/2026-10-08-multiday-preregistration.md)
on 46 stocks' split-adjusted 1-minute bars. Values are basis points per holding
period, equal-weight across the stocks held. Net figures charge the measured NBBO
half-spread for each side's time bucket, 1 bp slippage and $0.005 per share per
side. Intervals are bootstrap intervals over holding periods. The decision uses
the Bonferroni-adjusted 97.5% interval, because two rules are tested.

```bash
.venv/bin/python scripts/multiday_rules.py --bars ~/git/stock-analyzer/.cache/bars \
    --split development --output artifacts/multiday/development.json
```

## Development: periods starting 2024-10-03 to 2025-10-02

No reserved final-test session falls in the data.

| Rule | Periods | Mean before costs | Mean net | Net 97.5% interval | Hit rate (net) |
| --- | ---: | ---: | ---: | --- | ---: |
| Overnight drift (per night) | 250 | +4.81 | -4.15 | -15.48 to +6.81 | 0.53 |
| Control: same stocks, 09:30 to 15:58 | 250 | +6.50 | -2.46 | -17.31 to +14.62 | 0.48 |
| Weekly reversal, 5 losers (per week) | 49 | +53.02 | +46.31 | -85.35 to +170.77 | 0.61 |
| Control: all 46 stocks (per week) | 49 | +51.88 | +45.48 | -47.62 to +129.52 | 0.57 |
| Weekly reversal minus control | 49 | +1.13 | +0.84 | -65.78 to +67.17 | 0.45 |

The reference ETFs earned +2.15 bp per night before costs and -0.60 bp after.

Reading: in this year the stocks earned slightly less overnight than during the
day, so overnight drift did not appear and the rule lost about the round-trip
cost (about 9 bp) each night. The weekly losers did no better than the whole
universe; the reversal rule's +46 bp per week is the market's rise, not an edge.
Neither rule meets its pre-registered bar on development data.

## Holdout: periods starting 2025-10-03 to 2026-10-01 (run once)

This year was already used once, for the intraday momentum test (see the
pre-registration). No reserved final-test session falls in the data.

| Rule | Periods | Mean before costs | Mean net | Net 97.5% interval | Hit rate (net) |
| --- | ---: | ---: | ---: | --- | ---: |
| Overnight drift (per night) | 250 | +9.09 | +0.37 | -7.04 to +7.89 | 0.53 |
| Control: same stocks, 09:30 to 15:58 | 250 | +1.17 | -7.56 | -17.04 to +1.50 | 0.45 |
| Weekly reversal, 5 losers (per week) | 50 | +73.66 | +66.99 | -62.55 to +205.49 | 0.56 |
| Control: all 46 stocks (per week) | 50 | +48.17 | +41.90 | -13.35 to +98.37 | 0.66 |
| Weekly reversal minus control | 50 | +25.48 | +25.09 | -89.04 to +149.87 | 0.52 |

The reference ETFs earned +7.09 bp per night before costs and +4.39 bp after
(97.5% interval -4.53 to +13.18).

## Decision

Both rules fail their pre-registered criteria and are not added to the engine.

- Overnight drift showed up before costs in the holdout year: +9.09 bp per night,
  with a 97.5% interval of +1.86 to +16.54. It did not show up in the development
  year (+4.81, interval -6.11 to +15.90). After paying the spread at the close and
  the wider spread at the open, the holdout net is +0.37 bp per night, with an
  interval straddling zero. The effect is about the size of the cost.
- Weekly reversal beat the universe by +25 bp per week in the holdout and +0.84 bp
  in development. The weekly interval is roughly ±115 bp, so one year of 50
  non-overlapping weeks on 46 stocks cannot detect an effect of this size. That is
  a power limit, not evidence of an edge.

Across the EMA replays and three pre-registered tests (intraday momentum,
overnight drift and weekly reversal), no long-only rule on these liquid stocks clears the
round-trip cost with statistical confidence. The closest is overnight drift,
whose gross size roughly equals the cost. Lowering the cost would matter more
than finding another signal: the 09:30 open-bucket half-spread (median 2.67 bp)
is the largest single charge, and an opening-auction order would avoid it, but
that needs execution modeling this project does not have yet.
