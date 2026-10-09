"""Tests for the shared report helpers.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

import pytest
from gllm_evals.types import LLMTestCase

from auto_eval.shared.report import confusion, judge_errors, summarize, sweep_thresholds, verdict

PROVIDER_ERROR = "ProviderAuthError: [OpenRouterLMInvoker] Authorization failed due to issue with API Key."


def make_record(manual_rr: str, auto_rr: str, score: float | None = None) -> dict[str, object]:
    """Build a verdict with the fields the report helpers read."""
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
    The rows have no manual rating, and one judge call failed.

    Expected:
    The summary reports the Auto RR pass rate and the failed judge call, and leaves out the comparison.
    """
    records = [make_record("", "PASS"), {**make_record("", "FAIL"), "judge_error": True}]

    summary = summarize(records, "openrouter/google/gemini-3.1-flash-lite")

    assert (summary["auto_pass_rate"], summary["judge_errors"]) == (0.5, 1)
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


@pytest.mark.parametrize(
    ("explanation", "failed_call"),
    [
        ("The answer names both contracts in the expected response.", False),
        (PROVIDER_ERROR, True),
        ("Malformed attachment target. Use a named attachment ID or a supported URI.", True),
    ],
)
def test_judge_errors_with_explanation(explanation, failed_call):
    """
    Condition:
    Groundedness has a real explanation, a provider error, or the attachment validator's message.

    Expected:
    Only the provider error and the validator's message count as failed judge calls, with the message kept.
    """
    output = {"groundedness": {"score": 0.0, "success": False, "explanation": explanation}}

    errors = judge_errors(output, ("groundedness",))

    assert errors == ({"groundedness": explanation} if failed_call else {})


def test_verdict_with_label():
    """
    Condition:
    A case the DD team passed fails completeness, and the judge used tokens on two metrics.

    Expected:
    The verdict holds the manual and automatic ratings, the summed tokens, and each metric's score.
    """
    case = LLMTestCase(input="Berapa jumlah nilai kontrak Sponsorship tahun 2025?", label="TRUE")
    output = {
        "aggregate_success": False,
        "evaluator_token_usage": {"completeness": {"input_tokens": 900, "output_tokens": 40}, "redundancy": {}},
        "completeness": {"score": 0.5, "success": False, "explanation": "The total is missing."},
    }

    record = verdict(case, output, ("completeness", "redundancy"))

    assert (record["manual_rr"], record["auto_rr"], record["judge_error"]) == ("PASS", "FAIL", False)
    assert (record["input_tokens"], record["output_tokens"]) == (900, 40)
    assert (record["completeness.score"], record["redundancy.score"]) == (0.5, None)
