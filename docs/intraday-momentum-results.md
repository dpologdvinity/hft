# Intraday momentum results

Evaluation of the [pre-registered rule](specs/2026-10-08-intraday-momentum-preregistration.md):
buy at 15:30 ET when the 09:30-10:00 return is positive, sell at the 15:58 bar
close. Net figures charge the measured NBBO close half-spread on both sides, 1 bp
slippage per side and $0.005 per share per side (about 6.5 bp per round trip for
the stocks). Means are basis points per trade, averaged by date. Intervals are
95% bootstrap intervals over dates.

```bash
.venv/bin/python scripts/intraday_momentum.py --bars ~/git/stock-analyzer/.cache/bars \
    --split development --output artifacts/intraday-momentum/development.json
```

## Development: 2024-10-03 to 2025-10-02

46 stocks, 250 dates. No reserved final-test session falls in the data.

| Rule (stocks) | Trades | Mean net | 95% interval | Hit rate |
| --- | ---: | ---: | --- | ---: |
| Momentum, before costs | 5,687 | +2.05 | -1.27 to +5.53 | 0.52 |
| Momentum | 5,687 | -4.42 | -7.71 to -1.12 | 0.44 |
| Every day (control) | 11,339 | -5.03 | -8.20 to -1.86 | 0.43 |
| First half hour negative | 5,652 | -5.48 | -8.79 to -2.20 | 0.43 |
| Momentum incl. overnight (reference) | 5,929 | -5.99 | -9.09 to -3.02 | 0.42 |
| Momentum at 3x cost | 5,687 | -17.36 | -20.66 to -14.09 | 0.28 |

The reference ETFs (SPY, QQQ, IWM, DIA; 496 momentum trades) show the same
shape: +2.54 bp before costs, -0.12 bp after (interval -3.23 to +2.83).

Reading: a positive first half hour shifts the last half hour by about 1 bp
relative to a negative one, in the published direction but far smaller than the
round-trip cost and not distinguishable from zero. Net of costs the rule loses
money with an interval entirely below zero. It does not meet the pre-registered
bar on development data.
