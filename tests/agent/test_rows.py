"""Tests for filling the agent evaluation template.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

from gllm_evals.types import LLMTestCase

from auto_eval.agent.rows import TEMPLATE_METRIC_COLUMNS, to_record
from auto_eval.shared.project import Sample

METRICS = tuple(TEMPLATE_METRIC_COLUMNS)
PROVIDER_ERROR = "ProviderAuthError: [OpenRouterLMInvoker] Authorization failed due to issue with API Key."
EVALUATION_COLUMNS = [
    column
    for metric_column in TEMPLATE_METRIC_COLUMNS.values()
    for column in (metric_column, f"{metric_column}_explanation")
] + ["auto_rr", "possible_issues", "evaluation_error", "agent_error"]


def make_sample() -> Sample:
    """Build an exported run whose evaluation columns are still empty."""
    row = {
        "Index": "1",
        "Question": "Berapa banyak kontrak milik Langit Senja di tahun 2025?",
        "Answer": "Ada **2 kontrak Langit Senja** yang dimulai pada tahun 2025.",
        "Manual_Eval": "Pass",
        **dict.fromkeys(EVALUATION_COLUMNS, ""),
    }
    case = LLMTestCase(input=row["Question"], actual_output=row["Answer"], agent_error="", label="TRUE")
    return Sample(row, case)


def make_output(failed: dict[str, str] | None = None) -> dict[str, object]:
    """Build a dumped evaluator output where the given metrics fail with the given explanations."""
    failed = failed or {}
    output: dict[str, object] = {"aggregate_success": not failed}
    for metric in METRICS:
        explanation = failed.get(metric, f"The answer satisfies {metric}.")
        output[metric] = {
            "score": 0.0 if metric in failed else 1.0,
            "success": metric not in failed,
            "explanation": explanation,
        }
    return output


def test_to_record_with_every_metric_passing():
    """
    Condition:
    Every metric passes.

    Expected:
    Auto RR is `PASS`, the issue and error columns stay empty, and the columns keep the export order.
    """
    sample = make_sample()

    record = to_record(sample, make_output(), METRICS)

    assert list(record) == list(sample.row)
    assert (record["auto_rr"], record["possible_issues"], record["evaluation_error"]) == ("PASS", "", "")
    assert record["geval_refusal_alignment"] == 1.0
    assert record["geval_completeness_explanation"] == "The answer satisfies completeness."


def test_to_record_with_failed_metric():
    """
    Condition:
    Completeness fails on a real verdict.

    Expected:
    Auto RR is `FAIL`, and completeness is the possible issue.
    """
    output = make_output({"completeness": "The answer leaves out both contract titles."})

    record = to_record(make_sample(), output, METRICS)

    assert (record["auto_rr"], record["possible_issues"]) == ("FAIL", "completeness")
    assert record["geval_completeness"] == 0.0


def test_to_record_with_judge_error():
    """
    Condition:
    The groundedness judge call failed with a provider error.

    Expected:
    The error goes to `evaluation_error` and is not listed as a possible issue.
    """
    record = to_record(make_sample(), make_output({"groundedness": PROVIDER_ERROR}), METRICS)

    assert record["auto_rr"] == "FAIL"
    assert record["possible_issues"] == ""
    assert record["evaluation_error"] == f"groundedness: {PROVIDER_ERROR}"


def test_to_record_with_metric_outside_template():
    """
    Condition:
    The evaluator also ran a metric the template does not list.

    Expected:
    Its columns use the metric name and are added after the export's columns.
    """
    output = {**make_output(), "deepeval_answer_relevancy": {"score": 0.9, "success": True, "explanation": "On topic."}}

    record = to_record(make_sample(), output, (*METRICS, "deepeval_answer_relevancy"))

    assert list(record)[-2:] == ["deepeval_answer_relevancy", "deepeval_answer_relevancy_explanation"]
