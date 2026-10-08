# Overnight drift with auction execution, 2016-2024

Evaluation of the [pre-registered test](specs/2026-10-08-overnight-auction-preregistration.md)
on Alpaca daily SIP bars (split- and dividend-adjusted). Each night holds every
stock from the close to the next open, equal weight. Net figures charge 1 bp per
side for auction impact plus $0.005 per share per side: about 4.4 bp per round trip
for the stocks and 2.5 bp for the ETFs. Values are basis points per night with 95%
bootstrap intervals over nights. No reserved final-test session falls in the period.

```bash
.venv/bin/python scripts/overnight_auction.py --output artifacts/overnight-auction.json
```

| 2,201 nights, 2016-01-04 to 2024-10-01 | Before costs | Net | Net 95% interval | Net 3x cost |
| --- | ---: | ---: | --- | ---: |
| 46 stocks, overnight | +5.88 (+2.42 to +9.37) | +1.43 | -2.14 to +4.89 | -7.46 |
| 46 stocks, open to close (control) | +2.64 (-1.07 to +6.50) | -1.80 | -5.53 to +1.96 | -10.69 |
| SPY, QQQ, IWM, DIA, overnight | +4.54 (+1.33 to +7.68) | +2.05 | -1.18 to +5.28 | -2.93 |
| SPY, QQQ, IWM, DIA, open to close | +1.49 (-2.27 to +5.18) | -1.00 | -4.71 to +2.70 | -5.98 |

Net overnight return of the stocks by year: 2016 -4.06, 2017 +3.35, 2018 +1.02,
2019 +2.78, 2020 +7.66, 2021 +7.10, 2022 -9.38, 2023 -0.84, 2024 (to October) +6.40.

## Decision

Fails the pre-registered criterion: the stocks' net mean of +1.43 bp per night has
a 95% interval from -2.14 to +4.89, which includes zero.

The effect itself is real. Before costs, overnight holding earned +5.88 bp per night
on the stocks and +4.54 bp on the ETFs, with both intervals above zero, and more
than holding the same stocks during the day. The ETFs have no survivorship bias, so
the stock result is not just a product of picking winners. The problem is size:
about 4 bp of costs per round trip, assumed here, removes most of it.

That cost assumption is the uncertain part. Alpaca charges no commission, so the
$0.005 per share is conservative, and the 1 bp per side for auction impact is a
guess. Paper trading cannot settle it: paper fills are simulated by the broker
rather than matched in the real auctions. The `overnight-drift` paper run keeps
going to prove the order flow (auction orders accepted, filled, reconciled and
held overnight safely) and to show the simulated fill prices, not as evidence of a
profitable strategy. Measuring real auction costs would need tiny live orders,
which stay locked behind research qualification and paper graduation.
