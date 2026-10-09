"""The symbols the ML day trader may learn from and trade.

The original 46 large stocks and 4 index ETFs come from the candlebench study; the
rest were added by the owner on 2026-10-08. Many of the additions rose sharply in
2025-2026, which flatters any long strategy, so results are always compared with
holding the same symbols. Costs for the added stocks assume wider spreads.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

DATA_START = date(2016, 1, 4)  # first session of Alpaca's free SIP history

ORIGINAL_STOCKS = [
    "AAPL",
    "ABBV",
    "ADBE",
    "AMD",
    "AMZN",
    "AVGO",
    "BA",
    "BAC",
    "C",
    "CAT",
    "COST",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "F",
    "GE",
    "GOOGL",
    "GS",
    "HD",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "LLY",
    "MA",
    "META",
    "MRK",
    "MS",
    "MSFT",
    "MU",
    "NFLX",
    "NVDA",
    "ORCL",
    "OXY",
    "PEP",
    "PFE",
    "QCOM",
    "SLB",
    "TSLA",
    "TXN",
    "UNH",
    "V",
    "WFC",
    "WMT",
    "XOM",
]
ADDED_STOCKS = [
    "DELL",
    "NBIS",
    "LRCX",
    "LITE",
    "SNDK",
    "PLTR",
    "CRWV",
    "ADI",
    "AMAT",
    "ANET",
    "CMI",
    "STX",
    "TSM",
    "WDC",
    "SPCX",
]
ETFS = ["SPY", "QQQ", "IWM", "DIA", "VLUE", "PDBC", "SCHD", "VOO"]

FIRST_DATES = {
    "NBIS": date(2024, 10, 21),  # earlier bars under this ticker belong to Yandex
    "DELL": date(2018, 12, 28),
    "PLTR": date(2020, 9, 30),
    "SNDK": date(2025, 2, 13),
    "CRWV": date(2025, 3, 28),
    "SPCX": date(2026, 6, 12),
}
TRADE_ONLY = {"SPCX"}  # too little history to train or score on yet


@dataclass(frozen=True)
class Symbol:
    ticker: str
    kind: Literal["stock", "etf"]
    first_date: date
    trade_only: bool
    half_spread_bps: float


def _symbol(ticker, kind, half_spread_bps):
    return Symbol(
        ticker,
        kind,
        FIRST_DATES.get(ticker, DATA_START),
        ticker in TRADE_ONLY,
        half_spread_bps,
    )


UNIVERSE = tuple(
    [_symbol(t, "stock", 1.5) for t in ORIGINAL_STOCKS]
    + [_symbol(t, "stock", 4.0) for t in ADDED_STOCKS]
    + [_symbol(t, "etf", 1.5) for t in ETFS]
)


def tradable(day: date) -> list[Symbol]:
    """Symbols with usable history on `day`."""
    return [s for s in UNIVERSE if s.first_date <= day]
