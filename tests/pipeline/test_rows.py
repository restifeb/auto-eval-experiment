"""Tests for the pipeline Auto RR rows.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

from gllm_evals.types import LLMTestCase

from auto_eval.pipeline.rows import to_record
from auto_eval.shared.project import Sample

METRICS = ("completeness", "groundedness", "redundancy")


def test_to_record_with_failed_groundedness():
    """
    Condition:
    A row the DD team passed fails groundedness, and redundancy did not run.

    Expected:
    Auto RR is `FAIL` and disagrees with the manual rating, groundedness is the failed metric, the judge's tokens are
    summed, and redundancy is marked as not evaluated.
    """
    case = LLMTestCase(
        input="Kapan proklamasi kemerdekaan Indonesia dibacakan?",
        actual_output="Proklamasi dibacakan pada **17 Agustus 1945**.",
        retrieved_context=["Proklamasi kemerdekaan Indonesia dibacakan pada 17 Agustus 1945 di Jakarta."],
        row_no=1,
        issue_category="",
        label="TRUE",
    )
    output = {
        "aggregate_success": False,
        "aggregate_score": 0.5,
        "evaluator_token_usage": {"completeness": {"input_tokens": 900}, "groundedness": {"input_tokens": 1100}},
        "completeness": {"score": 1.0, "success": True, "explanation": "The date matches the expected response."},
        "groundedness": {"score": 0.0, "success": False, "explanation": "The bold date is not in the context."},
    }

    record = to_record(Sample({}, case), output, METRICS)

    assert (record["manual_rr"], record["auto_rr"], record["agree"]) == ("PASS", "FAIL", False)
    assert record["failed_metrics"] == "groundedness"
    assert (record["context_chars"], record["input_tokens"]) == (75, 2000)
    assert record["redundancy.explanation"] == "not evaluated"
