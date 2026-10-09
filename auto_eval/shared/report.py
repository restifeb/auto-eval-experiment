"""Read verdicts from evaluator outputs, compare Auto RR with the manual rating, and summarize a run.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/calibrate-the-evals
"""

import csv
import re
from pathlib import Path
from typing import Any

from gllm_evals.types import LLMTestCase

from auto_eval.shared.judge import PRICE_PER_MILLION_TOKENS

_JUDGE_ERROR = re.compile(r"^(\w+Error: |Malformed attachment target)")


def dump_output(output: object) -> dict[str, Any]:
    """Return an evaluator output as a dict.

    Args:
        output (object): The evaluator output.

    Returns:
        dict[str, Any]: The output's fields, with one entry per metric.
    """
    return output.model_dump() if hasattr(output, "model_dump") else dict(output)


def manual_rr(case: LLMTestCase) -> str:
    """Read the manual rating of a case.

    Args:
        case (LLMTestCase): The case, with `label` set to `TRUE` or `FALSE` when it was rated.

    Returns:
        str: `PASS` or `FAIL`, or an empty string when the case has no label.
    """
    label = getattr(case, "label", None)
    if not label:
        return ""
    return "PASS" if label == "TRUE" else "FAIL"


def token_usage(output: dict[str, Any]) -> tuple[int, int]:
    """Sum the judge's tokens over every metric of one evaluator output.

    Args:
        output (dict[str, Any]): The dumped evaluator output.

    Returns:
        tuple[int, int]: The input and output tokens.
    """
    usage = (output.get("evaluator_token_usage") or {}).values()
    return sum(item.get("input_tokens", 0) for item in usage), sum(item.get("output_tokens", 0) for item in usage)


def judge_errors(output: dict[str, Any], metrics: tuple[str, ...]) -> dict[str, str]:
    """Find the metrics whose judge call failed instead of giving a verdict.

    gllm-evals records a failed judge call as a failing metric whose explanation starts with the exception name, or,
    for a rejected attachment placeholder, with "Malformed attachment target".

    Args:
        output (dict[str, Any]): The dumped evaluator output.
        metrics (tuple[str, ...]): Names of the metrics the evaluator ran.

    Returns:
        dict[str, str]: The recorded error of each failed judge call, keyed by metric.
    """
    explanations = {metric: str((output.get(metric) or {}).get("explanation") or "") for metric in metrics}
    return {metric: text for metric, text in explanations.items() if _JUDGE_ERROR.match(text)}


def verdict(case: LLMTestCase, output: dict[str, Any], metrics: tuple[str, ...]) -> dict[str, object]:
    """Reduce one evaluator output to the fields the summary and the threshold sweep read.

    Args:
        case (LLMTestCase): The evaluated case.
        output (dict[str, Any]): The dumped evaluator output.
        metrics (tuple[str, ...]): Names of the metrics the evaluator ran.

    Returns:
        dict[str, object]: The manual and automatic ratings, the judge's tokens, whether a judge call failed, and
            each metric's score.
    """
    input_tokens, output_tokens = token_usage(output)
    return {
        "manual_rr": manual_rr(case),
        "auto_rr": "PASS" if output.get("aggregate_success") else "FAIL",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "judge_error": bool(judge_errors(output, metrics)),
        **{f"{metric}.score": (output.get(metric) or {}).get("score") for metric in metrics},
    }


def confusion(records: list[dict[str, object]]) -> dict[str, object]:
    """Count how Auto RR agrees with manual RR.

    Args:
        records (list[dict[str, object]]): Rows with `manual_rr` and `auto_rr`.

    Returns:
        dict[str, object]: TP, FN, TN, FP, TPR, TNR, and agreement, with manual `PASS` as the positive class.
    """
    tp = sum(1 for r in records if r["manual_rr"] == "PASS" and r["auto_rr"] == "PASS")
    fn = sum(1 for r in records if r["manual_rr"] == "PASS" and r["auto_rr"] == "FAIL")
    tn = sum(1 for r in records if r["manual_rr"] == "FAIL" and r["auto_rr"] == "FAIL")
    fp = sum(1 for r in records if r["manual_rr"] == "FAIL" and r["auto_rr"] == "PASS")
    return {
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "TPR": round(tp / (tp + fn), 3) if tp + fn else None,
        "TNR": round(tn / (tn + fp), 3) if tn + fp else None,
        "agreement": round((tp + tn) / len(records), 3),
    }


def sweep_thresholds(records: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    """Recompute TPR and TNR with each observed score of one metric as its threshold.

    Args:
        records (list[dict[str, object]]): Rows with `manual_rr` and `<metric>.score`.
        metric (str): The metric whose score alone decides Auto RR.

    Returns:
        list[dict[str, object]]: One entry per threshold, from the lowest observed score to the highest.
    """
    column = f"{metric}.score"
    thresholds = sorted({r[column] for r in records if r[column] is not None})
    return [
        {
            "threshold": round(threshold, 3),
            **confusion([{**r, "auto_rr": "PASS" if (r[column] or 0) >= threshold else "FAIL"} for r in records]),
        }
        for threshold in thresholds
    ]


def summarize(records: list[dict[str, object]], judge_model: str) -> dict[str, object]:
    """Summarize Auto RR, compare it with manual RR when every row is rated, and price the judge calls.

    Args:
        records (list[dict[str, object]]): Verdicts with `manual_rr`, `auto_rr`, and token counts.
        judge_model (str): The judge model id, used to look up its price.

    Returns:
        dict[str, object]: Pass rates, failed judge calls, confusion counts, TPR, TNR, agreement, tokens, and cost.
            The comparison is left out when a row has no manual rating, and the cost is `None` for a judge without
            a known price.
    """
    n = len(records)
    auto_passes = sum(1 for r in records if r["auto_rr"] == "PASS")
    summary: dict[str, object] = {
        "n": n,
        "auto_pass_rate": round(auto_passes / n, 3),
        "judge_errors": sum(1 for r in records if r.get("judge_error")),
    }
    if all(r["manual_rr"] for r in records):
        counts = confusion(records)
        summary.update(counts, manual_pass_rate=round((counts["TP"] + counts["FN"]) / n, 3))
    tokens = {"input": sum(r["input_tokens"] for r in records), "output": sum(r["output_tokens"] for r in records)}
    price = PRICE_PER_MILLION_TOKENS.get(judge_model)
    summary["judge_tokens"] = tokens
    summary["cost_usd"] = (
        round((tokens["input"] * price[0] + tokens["output"] * price[1]) / 1_000_000, 3) if price else None
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries as a CSV file, with every key that appears in any row as a column.

    Args:
        path (Path): The output file.
        rows (list[dict[str, object]]): The rows to write. A row without some column leaves it empty.
    """
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
