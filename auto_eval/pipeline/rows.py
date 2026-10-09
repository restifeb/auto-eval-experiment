"""Turn a pipeline verdict into an Auto RR row.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/calibrate-the-evals
"""

from typing import Any

from auto_eval.shared.project import Sample
from auto_eval.shared.report import manual_rr, token_usage


def to_record(sample: Sample, output: dict[str, Any], metrics: tuple[str, ...]) -> dict[str, object]:
    """Flatten one evaluator output into the Auto RR columns.

    Args:
        sample (Sample): The evaluated row and its test case.
        output (dict[str, Any]): The dumped evaluator output.
        metrics (tuple[str, ...]): Names of the metrics the evaluator ran.

    Returns:
        dict[str, object]: Manual and automatic ratings, failed metrics, and per-metric results. The manual columns
            are empty when the case has no label.
    """
    case = sample.case
    manual = manual_rr(case)
    auto_pass = bool(output.get("aggregate_success"))
    input_tokens, output_tokens = token_usage(output)
    context = case.retrieved_context
    record: dict[str, object] = {
        "no": case.row_no,
        "question": case.input,
        "manual_rr": manual,
        "auto_rr": "PASS" if auto_pass else "FAIL",
        "agree": (manual == "PASS") == auto_pass if manual else "",
        "issue_category": getattr(case, "issue_category", ""),
        "aggregate_score": output.get("aggregate_score"),
        "failed_metrics": ", ".join(m for m in metrics if (output.get(m) or {}).get("success") is False),
        "context_chars": sum(map(len, context)) if isinstance(context, list) else len(context or ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    for metric in metrics:
        result = output.get(metric) or {}
        record[f"{metric}.score"] = result.get("score")
        record[f"{metric}.explanation"] = result.get("explanation", "not evaluated")
    return record
