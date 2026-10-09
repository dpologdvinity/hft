"""News features for the daily model, from headlines matched against fixed patterns.

No language model reads the news here, so no knowledge from after an article's
publication can leak in. An article counts toward session d only if it was first
published between the previous session's close and five minutes before d's open, the
last moment the daily model decides. Events come only from focused articles (at most
three symbols tagged), because market roundups name many stocks without news about
any of them.
"""

import re

import numpy as np

CUTOFF_NS = 5 * 60 * 1_000_000_000  # decisions are made five minutes before the open
FOCUSED_SYMBOLS = 3
ATTENTION_SESSIONS = 5

PATTERNS = {
    ("analyst", 1): [
        r"\bupgrades?\b",
        r"\braises?\s+(?:its\s+)?(?:price\s+target|pt)\b",
        r"\bprice\s+target\s+raised\b",
        r"\binitiates?\s+coverage\b.*\b(?:buy|outperform|overweight)\b",
    ],
    ("analyst", -1): [
        r"\bdowngrades?\b",
        r"\b(?:lowers?|cuts?)\s+(?:its\s+)?(?:price\s+target|pt)\b",
        r"\bprice\s+target\s+(?:cut|lowered)\b",
        r"\binitiates?\s+coverage\b.*\b(?:sell|underperform|underweight)\b",
    ],
    ("earnings", 1): [
        r"\bbeats?\b",
        r"\btops?\b.*\b(?:estimates?|expectations?)\b",
        r"\braises?\s+(?:fy\S*\s+|q\d\s+|full[- ]year\s+|\d{4}\s+)?(?:guidance|outlook|forecast)\b",
    ],
    ("earnings", -1): [
        r"\bmiss(?:es)?\b",
        r"\b(?:lowers?|cuts?)\s+(?:fy\S*\s+|q\d\s+|full[- ]year\s+|\d{4}\s+)?(?:guidance|outlook|forecast)\b",
        r"\bguidance\s+below\b",
    ],
    ("other", 1): [
        r"\bfda\s+approv",
        r"\b(?:buyback|repurchase)\b",
        r"\braises?\s+(?:quarterly\s+)?dividend\b",
        r"\bto\s+be\s+acquired\b",
    ],
    ("other", -1): [
        r"\bcomplete\s+response\s+letter\b",
        r"\bfda\s+rejects?\b",
        r"\boffering\b",
        r"\b(?:lawsuit|sues|sued|investigation|probe|subpoena)\b",
        r"\brecalls?\b",
    ],
}
COMPILED = {key: [re.compile(p, re.IGNORECASE) for p in group] for key, group in PATTERNS.items()}

NEWS_FEATURES = (
    "news_count",
    "news_focused",
    "news_up",
    "news_down",
    "news_net",
    "news_analyst_net",
    "news_earnings_net",
    "news_attention_5d",
    "news_market_net",
)


def classify(headline: str) -> dict:
    """Direction flags (up, down) and net analyst and earnings signals of one headline."""
    hits = {key: any(p.search(headline) for p in group) for key, group in COMPILED.items()}
    return {
        "up": int(any(hit for (_, sign), hit in hits.items() if sign > 0)),
        "down": int(any(hit for (_, sign), hit in hits.items() if sign < 0)),
        "analyst": int(hits[("analyst", 1)]) - int(hits[("analyst", -1)]),
        "earnings": int(hits[("earnings", 1)]) - int(hits[("earnings", -1)]),
    }


def _windows(sessions):
    """[start, end) of each session's pre-open news window."""
    opens = np.array([w.open_ns for w in sessions], dtype=np.int64)
    closes = np.array([w.close_ns for w in sessions], dtype=np.int64)
    starts = np.concatenate([[opens[0] - 18 * 3600 * 1_000_000_000], closes[:-1]])
    return starts, opens - CUTOFF_NS


def news_features(news, symbols, dates, sessions) -> np.ndarray:
    """(D, S, F) float32 news features for the frame's dates and symbols."""
    index = {s: i for i, s in enumerate(symbols)}
    session_of = {np.datetime64(w.session_id, "D"): k for k, w in enumerate(sessions)}
    starts, ends = _windows(sessions)
    per = np.zeros((len(sessions), len(symbols), 7), dtype=np.float32)  # count..earnings
    market = np.zeros(len(sessions), dtype=np.float32)
    created = news.column("created_ns").to_numpy()
    window = np.searchsorted(starts, created, side="right") - 1
    inside = (window >= 0) & (created < ends[np.clip(window, 0, None)])
    headlines = news.column("headline").to_pylist()
    tagged = news.column("symbols").to_pylist()
    for row in np.flatnonzero(inside):
        k, names = window[row], tagged[row] or []
        focused = len(names) <= FOCUSED_SYMBOLS
        event = classify(headlines[row]) if focused else None
        if event:
            market[k] += event["up"] - event["down"]
        for name in names:
            s = index.get(name)
            if s is None:
                continue
            per[k, s, 0] += 1
            if event:
                per[k, s, 1] += 1
                per[k, s, 2] += event["up"]
                per[k, s, 3] += event["down"]
                per[k, s, 4] += event["up"] - event["down"]
                per[k, s, 5] += event["analyst"]
                per[k, s, 6] += event["earnings"]
    counts = per[:, :, 0]
    cumulative = np.concatenate([np.zeros((1, len(symbols))), np.cumsum(counts, axis=0)])
    lagged = np.arange(len(sessions))
    attention = cumulative[lagged] - cumulative[np.clip(lagged - ATTENTION_SESSIONS, 0, None)]
    full = np.concatenate(
        [per, attention[..., None], np.broadcast_to(market[:, None, None], counts.shape + (1,))],
        axis=-1,
    ).astype(np.float32)
    out = np.zeros((len(dates), len(symbols), len(NEWS_FEATURES)), dtype=np.float32)
    for d, day in enumerate(np.asarray(dates).astype("datetime64[D]")):
        k = session_of.get(day)
        if k is not None:
            out[d] = full[k]
    return out


SENTIMENT_FEATURES = ("news_sent_mean", "news_sent_max", "news_sent_min", "news_market_sent")


def sentiment_features(news, scores, symbols, dates, sessions) -> np.ndarray:
    """(D, S, 4) features from per-article sentiment scores (positive minus negative
    probability, for example from FinBERT), over the same pre-open windows and focused
    articles as `news_features`. Symbol-days without a scored article get zeros."""
    index = {s: i for i, s in enumerate(symbols)}
    session_of = {np.datetime64(w.session_id, "D"): k for k, w in enumerate(sessions)}
    starts, ends = _windows(sessions)
    total = np.zeros((len(sessions), len(symbols)), dtype=np.float64)
    count = np.zeros_like(total)
    high = np.full_like(total, -np.inf)
    low = np.full_like(total, np.inf)
    market_total = np.zeros(len(sessions))
    market_count = np.zeros(len(sessions))
    created = news.column("created_ns").to_numpy()
    ids = news.column("id").to_numpy()
    window = np.searchsorted(starts, created, side="right") - 1
    inside = (window >= 0) & (created < ends[np.clip(window, 0, None)])
    tagged = news.column("symbols").to_pylist()
    for row in np.flatnonzero(inside):
        names, score = tagged[row] or [], scores.get(int(ids[row]))
        if score is None or len(names) > FOCUSED_SYMBOLS:
            continue
        k = window[row]
        market_total[k] += score
        market_count[k] += 1
        for name in names:
            s = index.get(name)
            if s is None:
                continue
            total[k, s] += score
            count[k, s] += 1
            high[k, s] = max(high[k, s], score)
            low[k, s] = min(low[k, s], score)
    seen = count > 0
    full = np.stack(
        [
            np.where(seen, total / np.maximum(count, 1), 0.0),
            np.where(seen, high, 0.0),
            np.where(seen, low, 0.0),
            np.broadcast_to((market_total / np.maximum(market_count, 1))[:, None], total.shape),
        ],
        axis=-1,
    ).astype(np.float32)
    out = np.zeros((len(dates), len(symbols), len(SENTIMENT_FEATURES)), dtype=np.float32)
    for d, day in enumerate(np.asarray(dates).astype("datetime64[D]")):
        k = session_of.get(day)
        if k is not None:
            out[d] = full[k]
    return out
