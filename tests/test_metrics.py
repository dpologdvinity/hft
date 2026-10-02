import json

import pytest

from hft.metrics import metrics, paired_bootstrap


def test_intraday_marks_and_json_safe_profit_factor():
    result = metrics([100, 101], [1], [100, 90, 101])
    assert result["max_drawdown"] == pytest.approx(0.1)
    assert result["profit_factor"] == "infinite"
    assert result["sessions"] == 1
    json.dumps(result, allow_nan=False)
    assert metrics([100, 100], [])["profit_factor"] is None


def test_family_corrected_bootstrap_and_small_samples():
    result = paired_bootstrap([0.02] * 40, [0.01] * 40, seed=4)
    assert result["lower"] == pytest.approx(0.01)
    assert result["quantiles"] == pytest.approx([0.05 / 6, 1 - 0.05 / 6])
    assert result["resamples"] == 2000
    assert result == paired_bootstrap([0.02] * 40, [0.01] * 40, seed=4)
    assert not paired_bootstrap([0.01], [0])["positive"]
    assert not paired_bootstrap([0] * 40, [0.01] * 40)["positive"]


def test_streaming_equity_summary_matches_full_curve():
    from hft.metrics import EquitySummary

    first = EquitySummary(500)
    for value in [510, 495, 505]:
        first.add(value)
    second = EquitySummary(505)
    for value in [490, 520, 500]:
        second.add(value)
    first.combine(second)
    expected = metrics([500, 505, 500], [], [500, 510, 495, 505, 505, 490, 520, 500])
    assert first.drawdown == expected["max_drawdown"]
    assert first.marks == 8
