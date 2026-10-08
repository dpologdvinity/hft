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
