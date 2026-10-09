# Project history

A running record of what this project set out to do, every experiment it ran, what
each one found, and how the goals and ideas changed as a result. Newest entries are
added at the end of each section. Detailed numbers live in the linked result pages.

## How the goals changed

| Date | Goal | Why it changed |
| --- | --- | --- |
| 2026-10-02 | Learn to trade one stock from historical simulations with a reinforcement-learning agent (PPO, neural network) on 5-second bars, prove it honestly on held-out data, then graduate through paper trading to a small real-money canary. | Starting point. |
| 2026-10-04 | Fix the data before training: the free 5-second data was too gappy to train on. | Readiness checks failed (see experiment 1). |
| 2026-10-07 | Be able to trade now: a bot that paper-trades any chosen stocks every day with per-stock budgets, using simple rule strategies until a model qualifies. Real money stays locked behind research qualification and 30 paper sessions. | The owner wanted a working trading bot, not only a research platform. |
| 2026-10-07 | Make the market-data and decision core fast: a C++20 engine bit-identical to the Python reference. | Speed matters for a trading bot, and C++ is the owner's strongest language. |
| 2026-10-08 | Find a strategy that is profitable after costs; allow positions to be held overnight. | Every same-day rule lost money to trading costs; the owner approved overnight holds. |
| 2026-10-08 | Profit is now the priority. Success is defined as beating simply holding the same stocks, after costs, on data the model never trained on. Next idea under design: train a neural network on a GPU (Kaggle), trying many strategies in parallel and combining them. | The owner wants the bot to make money, and a trained model has not yet had a fair chance on good data. |
| 2026-10-08 | Build an ML day trader: a daily model picks which stocks to trade, a minute model times entries and exits, flat by the close; prediction models first, reinforcement learning as a second track; many variants trained in parallel on a free Kaggle GPU and combined. Universe grows to 69 symbols (adds DELL, NBIS, LRCX, LITE, SNDK, PLTR, CRWV, ADI, AMAT, ANET, CMI, STX, TSM, WDC, SPCX and the VLUE, PDBC, SCHD, VOO ETFs). A news-reading research agent is deferred. ([design](specs/2026-10-08-ml-daytrader-design.md)) | The owner wants a bot that trades like a day trader and learns from data; the overfitting demonstration set the bar: beat holding on unseen data. |

## Experiments

| # | Date | Idea | What was done | Result | Decision |
| --- | --- | --- | --- | --- | --- |
| 1 | 10-03 | Train the PPO agent on McDonald's (MCD) 5-second bars | Checked all 82 free IEX sessions before fitting | All 52 development days fail the 5-second continuity check (57,049 feed gaps) | Training blocked; free 5-second data unusable for MCD ([readiness](research-readiness-results.md)) |
| 2 | 10-07 | Maybe busier stocks have clean enough free data | Suitability sweep over liquid stocks | Only very liquid names (NVDA, INTC, NFLX) passed, and only on some days | Free 5-second data is marginal even for the busiest stocks ([suitability](free-data-suitability-results.md)) |
| 3 | 10-07 | How strong must a signal be to beat costs? | Compared typical price moves with round-trip costs at 5 s to 30 min | A predictor would need a correlation of 0.1-0.7 with future moves just to break even; realistic signals reach 0.02-0.05 | Short-horizon trading that crosses the spread is not plausibly profitable |
| 4 | 10-07 | EMA crossover (a classic day-trader rule) | Replayed full days through the paper bot with quote-level fills | June 5: NVDA -1.97%, AAPL -2.87%, MSFT -5.19%. Oct 7: NVDA -2.00%, AAPL -2.02% on flat days; about 4-5 bp lost per round trip | Rule churns and pays the spread ([replay](replay-results.md)) |
| 5 | 10-07 | Thin IEX books show fake prices | Added a maximum-spread guard for entries | MSFT June 5 loss cut from 5.19% to 2.22% | Kept as a default safety rule |
| 6 | 10-08 | Intraday momentum (first half hour predicts last half hour; published 2018) | Pre-registered; 46 stocks, 2 years of 1-minute bars | Development -4.42 bp per trade, holdout -7.50 bp after costs | Fails ([results](intraday-momentum-results.md)) |
| 7 | 10-08 | Hold overnight instead (most stock returns come overnight) | Pre-registered; 2 years, costs at the quoted spread | Holdout +9.09 bp per night before costs, +0.37 after | Fails: the effect is about the size of the cost ([results](multiday-results.md)) |
| 8 | 10-08 | Weekly reversal (last week's losers rebound) | Pre-registered; same data | Beat the universe by +0.84 bp/week (development) and +25 bp/week (holdout), but the uncertainty is about ±115 bp | Fails: one year cannot tell ([results](multiday-results.md)) |
| 9 | 10-08 | Overnight drift through auction orders (no spread) | Pre-registered; 2,201 nights of untouched 2016-2024 daily data | +5.88 bp per night before costs (clearly above zero, ETFs too); +1.43 after about 4 bp of assumed costs, not clearly above zero | Fails narrowly; real auction costs are the open question ([results](overnight-auction-results.md)) |
| 10 | 10-08 | Run the bot on the real paper account | Three intraday sessions, NVDA and AAPL | 185 round trips, about +$0.92; five bugs found and fixed the same day | Plumbing proven; paper fills look more generous than the project's simulator ([paper](paper-acceptance-results.md)) |
| 11 | 10-08 | Overnight drift, forward on paper | Started a daily paper run on 10 stocks x $500 | First entries filled; three auction orders expired (paper broker behavior), fixed to market-on-close | Running; collects a forward track record |
| 12 | 10-08 | "Train the neural network until it stops losing money" | Trained three 2-layer networks (256 units) to maximize profit after a generous 1 bp cost, using 20 lagged returns plus candle body and gap features; first on coin-flip prices with no pattern, then on real daily bars of 46 stocks (train 2016-2021, test 2022-2024) ([script](../scripts/overfitting_demo.py)) | Coin flips: training profit +5,218% (sum of per-period returns), fresh coin flips -50%. Real stocks: training +549% against +120% for holding; test years +5.4% (three combined: +9.6%) against +24.6% for holding | Training profit says nothing about the future: the network memorizes noise. Any model must be judged on untouched data against holding |
| 13 | 10-08 | ML day trader, first run on real data | Built the ML pipeline (download, causal daily and minute features, fill simulator, LightGBM and networks, search, Kaggle packaging); smoke-ran 4 variants on 14 stocks, validation 2023-2024 | The daily return model found nothing (validation correlation -0.015, one tree), so it never cleared costs. Ranked without the cost gate: holding the top-3 return picks from the open made +15.0% (+3.2 bp per trade; with a one-tree model the picks mostly reflect how ties sort, not skill), range picks -19.9%, the CNN minute model -0.1% to -10.1%; holding the same 14 stocks made +81.4% | Plumbing works; no variant is close to holding. The full 84-variant search on all 68 stocks runs on a Kaggle GPU next ([design](specs/2026-10-08-ml-daytrader-design.md)) |

Related evidence from the owner's other project, candlebench: 20 classic candlestick
patterns on 50 stocks (about 80,000 trades) showed no edge over random entry, even
before costs.

## What has been learned

- Trading costs decide everything at short horizons. Every same-day rule tested lost
  roughly its round-trip cost.
- The overnight effect is the one measurable edge so far, and it is about as large as
  the cost of trading it.
- Simply holding the stocks earned more than any strategy tested, because it pays no
  trading costs. Any strategy must beat that, on data it never saw.
- Live paper trading finds bugs that tests miss: in the first session, an opening
  burst, cent rounding, REST latency and auction-order handling.
