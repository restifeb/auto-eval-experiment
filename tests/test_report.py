"""Tests for the Auto RR report helpers.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

import pytest

from auto_eval.report import confusion, summarize, sweep_thresholds


def make_record(manual_rr: str, auto_rr: str, score: float | None = None) -> dict[str, object]:
    """Build a flattened row with the fields the report helpers read."""
    return {
        "manual_rr": manual_rr,
        "auto_rr": auto_rr,
        "input_tokens": 1000,
        "output_tokens": 100,
        "completeness.score": score,
    }


def test_confusion_with_both_directions():
    """
    Condition:
    One row each of TP, FN, TN, and FP.

    Expected:
    Every count is 1, and TPR, TNR, and agreement are 0.5.
    """
    records = [
        make_record("PASS", "PASS"),
        make_record("PASS", "FAIL"),
        make_record("FAIL", "FAIL"),
        make_record("FAIL", "PASS"),
    ]

    assert confusion(records) == {"TP": 1, "FN": 1, "TN": 1, "FP": 1, "TPR": 0.5, "TNR": 0.5, "agreement": 0.5}


def test_confusion_without_failed_rows():
    """
    Condition:
    Every manual rating is `PASS`.

    Expected:
    TNR is None because there is no negative row to measure it on.
    """
    counts = confusion([make_record("PASS", "PASS"), make_record("PASS", "FAIL")])

    assert counts["TNR"] is None
    assert counts["TPR"] == 0.5


@pytest.mark.parametrize(
    ("judge_model", "expected_cost"),
    [("openrouter/google/gemini-3.1-flash-lite", 0.001), ("google/gemini-3.1-flash-lite", None)],
)
def test_summarize_with_judge_price(judge_model, expected_cost):
    """
    Condition:
    Two labeled rows judged by a model with or without a known price.

    Expected:
    The cost follows the price table and is None for an unknown judge.
    """
    summary = summarize([make_record("PASS", "PASS"), make_record("FAIL", "FAIL")], judge_model)

    assert summary["cost_usd"] == expected_cost
    assert summary["agreement"] == 1.0


def test_summarize_without_manual_rating():
    """
    Condition:
    The rows have no manual rating.

    Expected:
    The summary reports the Auto RR pass rate and leaves out the comparison.
    """
    summary = summarize([make_record("", "PASS"), make_record("", "FAIL")], "openrouter/google/gemini-3.1-flash-lite")

    assert summary["auto_pass_rate"] == 0.5
    assert "TPR" not in summary


def test_sweep_thresholds_with_one_failed_row():
    """
    Condition:
    The failed row scores lower than both passed rows.

    Expected:
    A threshold above the failed row's score keeps TPR at 1.0 and reaches TNR 1.0.
    """
    records = [make_record("PASS", "PASS", 1.0), make_record("PASS", "PASS", 0.5), make_record("FAIL", "FAIL", 0.0)]

    sweep = sweep_thresholds(records, "completeness")

    assert [entry["threshold"] for entry in sweep] == [0.0, 0.5, 1.0]
    assert (sweep[1]["TPR"], sweep[1]["TNR"]) == (1.0, 1.0)
