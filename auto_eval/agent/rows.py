"""Fill the evaluation columns of the agent export template from a verdict.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/calibrate-the-evals
"""

from typing import Any

from auto_eval.shared.project import Sample
from auto_eval.shared.report import judge_errors

TEMPLATE_METRIC_COLUMNS = {
    "completeness": "geval_completeness",
    "groundedness": "geval_groundedness",
    "redundancy": "geval_redundancy",
    "language_consistency": "geval_language_consistency",
    "refusal_alignment": "geval_refusal_alignment",
}


def metric_column(metric: str) -> str:
    """Name the column of a metric's score.

    Args:
        metric (str): The metric name.

    Returns:
        str: The template column, such as `geval_completeness`, or the metric name for a metric the template does
            not list.
    """
    return TEMPLATE_METRIC_COLUMNS.get(metric, metric)


def to_record(sample: Sample, output: dict[str, Any], metrics: tuple[str, ...]) -> dict[str, object]:
    """Copy the exported row and fill its evaluation columns.

    Each metric fills its score column and `<column>_explanation`. `auto_rr` is `PASS` when every metric passes.
    `possible_issues` lists the metrics that failed and stays empty on a pass, as in the platform's generation
    results. A metric whose judge call failed is reported in `evaluation_error` instead of `possible_issues`.

    Args:
        sample (Sample): The exported row and its test case.
        output (dict[str, Any]): The dumped evaluator output.
        metrics (tuple[str, ...]): Names of the metrics the evaluator ran.

    Returns:
        dict[str, object]: The row with its columns in export order. Columns of metrics the template does not list
            are added at the end.
    """
    errors = judge_errors(output, metrics)
    auto_pass = bool(output.get("aggregate_success"))
    evaluation: dict[str, object] = {}
    for metric in metrics:
        result = output.get(metric) or {}
        evaluation[metric_column(metric)] = result.get("score")
        evaluation[f"{metric_column(metric)}_explanation"] = result.get("explanation", "not evaluated")
    failed = [m for m in metrics if (output.get(m) or {}).get("success") is False and m not in errors]
    evaluation["auto_rr"] = "PASS" if auto_pass else "FAIL"
    evaluation["possible_issues"] = "" if auto_pass else ", ".join(failed)
    evaluation["evaluation_error"] = "; ".join(f"{metric}: {error}" for metric, error in errors.items())
    evaluation["agent_error"] = getattr(sample.case, "agent_error", "")
    return {**sample.row, **evaluation}
