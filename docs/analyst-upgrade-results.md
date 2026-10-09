# Analyst-upgrade day trade results

Evaluation of the [pre-registered rule](specs/2026-10-09-analyst-upgrade-preregistration.md):
buy at 09:31 every symbol whose focused pre-open news holds net analyst-up events
(upgrades, price-target raises), sell five minutes before the close, with the ML
day trader's fill simulator and costs.

```bash
.venv/bin/python scripts/analyst_upgrade.py --split validation --output artifacts/analyst-upgrade/validation.json
```

## Validation: 2023-01-03 to 2024-12-31 (68 symbols)

| | Trades | Mean net per trade | 95% interval (dates resampled) | Hit rate |
| --- | ---: | ---: | --- | ---: |
| Upgrade day trade | 1,532 | +4.7 bp | -8.1 to +17.9 | 0.51 |
| Same, at 3x costs | 1,532 | -6.7 bp | -19.2 to +6.1 | 0.48 |
| Downgrade mirror (short, reported only) | 552 | -1.8 bp | -20.0 to +16.7 | 0.50 |

Five-slot portfolio: +13.0% over the two years, against +97.2% for holding the same
symbols. About 770 qualifying trades a year.

## Decision

Fails the pre-registered bar: the per-trade mean is positive (+4.7 bp, about half the
+10.4 bp seen in the training years) but its interval includes zero, and it turns
negative at 3x costs. The test years are not used. The rule is not added to the bot.

The training-years effect shrank on new data, as small effects found by looking
usually do, and two years of upgrades are too few to tell a real +5 bp edge from
zero. Downgrades showed nothing a short seller could use either.
