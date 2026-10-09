from datetime import date

from hft.ml.universe import UNIVERSE, tradable


def _by_ticker():
    return {s.ticker: s for s in UNIVERSE}


def test_universe_has_69_unique_symbols_with_8_etfs():
    tickers = [s.ticker for s in UNIVERSE]
    assert len(tickers) == 69 and len(set(tickers)) == 69
    etfs = {s.ticker for s in UNIVERSE if s.kind == "etf"}
    assert etfs == {"SPY", "QQQ", "IWM", "DIA", "VLUE", "PDBC", "SCHD", "VOO"}


def test_short_histories_and_trade_only_symbols():
    by = _by_ticker()
    assert by["NBIS"].first_date == date(2024, 10, 21)  # earlier bars belong to Yandex
    assert by["DELL"].first_date == date(2018, 12, 28)
    assert by["PLTR"].first_date == date(2020, 9, 30)
    assert by["SNDK"].first_date == date(2025, 2, 13)
    assert by["CRWV"].first_date == date(2025, 3, 28)
    assert by["SPCX"].trade_only and not by["NVDA"].trade_only


def test_costs_are_wider_for_the_added_stocks():
    by = _by_ticker()
    assert by["AAPL"].half_spread_bps == 1.5 and by["SPY"].half_spread_bps == 1.5
    assert by["LITE"].half_spread_bps == 4.0 and by["SPCX"].half_spread_bps == 4.0


def test_tradable_respects_first_dates():
    names = {s.ticker for s in tradable(date(2020, 1, 2))}
    assert "PLTR" not in names and "NVDA" in names and "DELL" in names
