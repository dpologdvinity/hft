"""Shared event timelines for comparing market-engine implementations.

Each vector is (name, session, symbol, timeline) where timeline items are
(event, now_ns) for an arrival or (None, now_ns) for a bar-boundary timer.
"""

from hft.calendar import SessionWindow
from hft.data import NS
from hft.feed import CLOSE_EXCLUDED

OPEN = 1_736_173_800 * NS  # 2025-01-06 14:30 UTC
SESSION = SessionWindow("2025-01-06", OPEN, OPEN + 3600 * NS)
MS = 1_000_000


def quote(t, bid=100.0, ask=100.02, bs=100.0, as_=200.0, arrival=None, **extra):
    event = {"T": "q", "S": "X", "event_ns": t, "bp": bid, "ap": ask, "bs": bs, "as": as_}
    event.update(sizes_in_shares=True, **extra)
    return event, t if arrival is None else arrival


def trade(t, price=100.01, size=10.0, arrival=None, **extra):
    event = {"T": "t", "S": "X", "event_ns": t, "p": price, "s": size, **extra}
    return event, t if arrival is None else arrival


def timer(t):
    return None, t


def _normal():
    items = []
    for k in range(120):
        t = OPEN + k * 500 * MS
        items.append(quote(t, 100 + k * 0.01, 100.02 + k * 0.01, i=f"q{k}"))
        if k % 2:
            items.append(trade(t + MS, 100.01 + k * 0.01, 5.0 + k, i=f"t{k}"))
        if (k + 1) % 10 == 0:
            items.append(timer(OPEN + (k + 1) // 10 * 5 * NS))
    return items


def _same_time_timer():
    boundary = OPEN + 5 * NS
    return [
        quote(OPEN + NS, i="a"),
        trade(OPEN + 2 * NS, i="b"),
        timer(boundary),
        quote(boundary, 101, 101.02, i="c"),
        trade(boundary, 101.01, i="d"),
        timer(boundary + 5 * NS),
    ]


def _duplicates():
    return [
        quote(OPEN + NS, i="same"),
        quote(OPEN + NS, i="same"),
        quote(OPEN + 2 * NS),
        quote(OPEN + 2 * NS),
        trade(OPEN + 3 * NS),
        trade(OPEN + 3 * NS),
        trade(OPEN + 3 * NS, quote_id="via-quote-id"),
        trade(OPEN + 3 * NS, quote_id="via-quote-id"),
        timer(OPEN + 5 * NS),
        quote(OPEN + 6 * NS, i="same"),  # identities reset each bar
        timer(OPEN + 10 * NS),
    ]


def _late_future_outside():
    return [
        quote(OPEN + NS, i="q1"),
        trade(OPEN + 2 * NS, i="t1"),
        timer(OPEN + 5 * NS),
        trade(OPEN + 4 * NS, arrival=OPEN + 6 * NS, i="late-closed"),
        trade(OPEN + 6 * NS, arrival=OPEN + 6 * NS + 300 * MS, i="tolerance"),
        trade(OPEN + 7 * NS + 300 * MS, arrival=OPEN + 7 * NS, i="future"),
        trade(OPEN + 7 * NS + 200 * MS, arrival=OPEN + 7 * NS, i="near-future"),
        quote(OPEN - NS, arrival=OPEN + 7 * NS, i="before-open"),
        timer(OPEN + 10 * NS),
    ]


def _corrections_and_clock():
    return [
        quote(OPEN + NS, i="q1"),
        trade(OPEN + 2 * NS, i="t1"),
        ({"T": "c", "S": "X", "event_ns": OPEN + 3 * NS}, OPEN + 3 * NS),
        ({"T": "x", "S": "X", "event_ns": OPEN + 6 * NS}, OPEN + 6 * NS),
        timer(OPEN + 5 * NS),  # clock reversal: already advanced to 6 s
        ({"T": "q", "S": "OTHER", "event_ns": OPEN + 7 * NS}, OPEN + 7 * NS),
        ({"T": "b", "S": "X", "event_ns": OPEN + 7 * NS}, OPEN + 7 * NS),
        timer(OPEN + 10 * NS),
    ]


def _filtered_trades():
    items = [quote(OPEN + NS, i="q")]
    t = OPEN + NS
    for code in sorted(CLOSE_EXCLUDED) + ["W", "B", "@", "F"]:
        for tape in ("C", "A", None):
            t += MS
            extra = {"c": [code]}
            if tape is not None:
                extra["z"] = tape
            items.append(trade(t, 100 + (t - OPEN) / NS / 1000, i=f"{code}{tape}", **extra))
    items.append(timer(OPEN + 5 * NS))
    return items


def _missing_context_and_gap():
    return [
        trade(OPEN + NS, i="t-before-quote"),
        timer(OPEN + 5 * NS),
        quote(OPEN + 6 * NS, i="q1"),
        timer(OPEN + 10 * NS),
        trade(OPEN + 17 * NS, i="after-6s-silence"),
        timer(OPEN + 20 * NS),
    ]


def _lots_strings_and_text_times():
    return [
        (
            {"T": "q", "S": "X", "t": "2025-01-06T14:30:01.25Z", "bp": "99.99", "ap": 100.01}
            | {"bs": 0.07, "as": 3, "i": 1},
            OPEN + 1300 * MS,
        ),
        (
            {"T": "t", "S": "X", "t": "2025-01-06T09:30:02.5-05:00", "p": "100", "s": 7},
            OPEN + 2600 * MS,
        ),
        timer(OPEN + 5 * NS),
    ]


VECTORS = {
    "normal": _normal,
    "same_time_timer": _same_time_timer,
    "duplicates": _duplicates,
    "late_future_outside": _late_future_outside,
    "corrections_and_clock": _corrections_and_clock,
    "filtered_trades": _filtered_trades,
    "missing_context_and_gap": _missing_context_and_gap,
    "lots_strings_and_text_times": _lots_strings_and_text_times,
}


def engine_vectors():
    return [(name, SESSION, "X", build()) for name, build in VECTORS.items()]
