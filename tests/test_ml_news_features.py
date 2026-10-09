from datetime import datetime

import numpy as np
import pyarrow as pa
import pytest

from hft.calendar import SessionWindow
from hft.ml.news import SCHEMA
from hft.ml.news_features import NEWS_FEATURES, classify, news_features

NS = 1_000_000_000


def _ns(text):
    return int(datetime.fromisoformat(text).timestamp()) * NS


SESSIONS = [
    SessionWindow("2023-01-03", _ns("2023-01-03T09:30:00-05:00"), _ns("2023-01-03T16:00:00-05:00")),
    SessionWindow("2023-01-04", _ns("2023-01-04T09:30:00-05:00"), _ns("2023-01-04T16:00:00-05:00")),
    SessionWindow("2023-01-05", _ns("2023-01-05T09:30:00-05:00"), _ns("2023-01-05T16:00:00-05:00")),
]


def _news(rows):
    return pa.table(
        {
            "id": list(range(len(rows))),
            "created_ns": [_ns(t) for t, _, _ in rows],
            "updated_ns": [_ns(t) for t, _, _ in rows],
            "headline": [h for _, h, _ in rows],
            "summary": [""] * len(rows),
            "source": ["benzinga"] * len(rows),
            "symbols": [list(s) for _, _, s in rows],
        },
        schema=SCHEMA,
    )


@pytest.mark.parametrize(
    "headline, up, down",
    [
        ("Morgan Stanley Upgrades Nvidia to Overweight", 1, 0),
        ("Goldman Sachs Maintains Buy on Apple, Raises Price Target to $250", 1, 0),
        ("Nvidia Q3 EPS $0.81 Beats $0.75 Estimate, Sales $35.08B Beat $33.12B", 1, 0),
        ("Barclays Downgrades Intel to Underweight, Lowers Price Target to $20", 0, 1),
        ("Ford Q2 Adj. EPS $0.47 Misses $0.68 Estimate", 0, 1),
        ("Apple Lowers Q4 Guidance", 0, 1),
        ("Pfizer Prices $5B Stock Offering", 0, 1),
        ("FDA Approves Lilly's Weight-Loss Drug", 1, 0),
        ("Microsoft Announces $60B Share Buyback", 1, 0),
        ("Apple Unveils New iPhone", 0, 0),
    ],
)
def test_headlines_are_classified_by_fixed_patterns(headline, up, down):
    event = classify(headline)
    assert (event["up"], event["down"]) == (up, down)


def test_only_news_before_the_cutoff_counts_for_a_session():
    news = _news(
        [
            (
                "2023-01-03T20:00:00-05:00",
                "Morgan Stanley Upgrades Nvidia to Overweight",
                ("NVDA",),
            ),
            ("2023-01-04T09:20:00-05:00", "Nvidia Q3 EPS Beats Estimate", ("NVDA",)),
            ("2023-01-04T09:26:00-05:00", "Barclays Downgrades Nvidia", ("NVDA",)),  # too late
            ("2023-01-04T11:00:00-05:00", "Nvidia Lowers Guidance", ("NVDA",)),  # during the day
            (
                "2023-01-03T22:00:00-05:00",
                "Market roundup: Apple, Nvidia, Intel, AMD gain",
                ("AAPL", "NVDA", "INTC", "AMD"),
            ),  # not focused
        ]
    )
    dates = np.array(["2023-01-03", "2023-01-04", "2023-01-05"], dtype="datetime64[D]")
    x = news_features(news, ["NVDA", "AAPL"], dates, SESSIONS)
    f = {name: i for i, name in enumerate(NEWS_FEATURES)}
    day2, nvda = 1, 0
    assert x[day2, nvda, f["news_count"]] == 3
    assert x[day2, nvda, f["news_focused"]] == 2
    assert x[day2, nvda, f["news_up"]] == 2 and x[day2, nvda, f["news_down"]] == 0
    assert x[day2, nvda, f["news_analyst_net"]] == 1 and x[day2, nvda, f["news_earnings_net"]] == 1
    assert x[day2, 1, f["news_count"]] == 1 and x[day2, 1, f["news_up"]] == 0
    assert x[day2, nvda, f["news_market_net"]] == 2
    # The 11:00 guidance cut belongs to no pre-open window; the 09:26 downgrade neither.
    assert x[2, nvda, f["news_down"]] == 0
    assert x[2, nvda, f["news_attention_5d"]] == 3  # earlier sessions' article counts


def test_sentiment_features_average_focused_pre_open_scores():
    from hft.ml.news_features import SENTIMENT_FEATURES, sentiment_features

    news = _news(
        [
            ("2023-01-03T20:00:00-05:00", "Nvidia soars", ("NVDA",)),
            ("2023-01-04T08:00:00-05:00", "Nvidia slips", ("NVDA",)),
            ("2023-01-04T09:40:00-05:00", "Nvidia late news", ("NVDA",)),  # after the cutoff
            (
                "2023-01-04T07:00:00-05:00",
                "Roundup",
                ("AAPL", "NVDA", "INTC", "AMD"),
            ),  # not focused
        ]
    )
    scores = {0: 0.8, 1: -0.2, 2: -0.9, 3: 0.5}
    dates = np.array(["2023-01-04"], dtype="datetime64[D]")
    x = sentiment_features(news, scores, ["NVDA", "AAPL"], dates, SESSIONS)
    f = {name: i for i, name in enumerate(SENTIMENT_FEATURES)}
    assert x[0, 0, f["news_sent_mean"]] == pytest.approx(0.3)
    assert x[0, 0, f["news_sent_max"]] == pytest.approx(0.8)
    assert x[0, 0, f["news_sent_min"]] == pytest.approx(-0.2)
    assert x[0, 1, f["news_sent_mean"]] == 0  # AAPL only in the roundup
    assert x[0, 1, f["news_market_sent"]] == pytest.approx(0.3)  # focused articles, all symbols
