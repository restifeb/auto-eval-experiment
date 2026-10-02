"""Turn evaluator outputs into Auto RR rows and compare them with the manual rating.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/calibrate-the-evals
"""

import csv
from pathlib import Path

from gllm_evals.types import LLMTestCase

from auto_eval.judge import PRICE_PER_MILLION_TOKENS


def to_record(case: LLMTestCase, output: object, metrics: tuple[str, ...]) -> dict[str, object]:
    """Flatten one evaluator output into the Auto RR columns.

    Args:
        case (LLMTestCase): The evaluated case.
        output (object): The evaluator output.
        metrics (tuple[str, ...]): Names of the metrics the evaluator ran.

    Returns:
        dict[str, object]: Manual and automatic ratings, failed metrics, and per-metric results. The manual columns
            are empty when the case has no label.
    """
    dumped = output.model_dump() if hasattr(output, "model_dump") else dict(output)
    usage = dumped.get("evaluator_token_usage") or {}
    auto_pass = bool(dumped.get("aggregate_success"))
    label = getattr(case, "label", None)
    manual_rr = ""
    if label:
        manual_rr = "PASS" if label == "TRUE" else "FAIL"
    record: dict[str, object] = {
        "no": case.row_no,
        "question": case.input,
        "manual_rr": manual_rr,
        "auto_rr": "PASS" if auto_pass else "FAIL",
        "agree": (manual_rr == "PASS") == auto_pass if manual_rr else "",
        "issue_category": getattr(case, "issue_category", ""),
        "aggregate_score": dumped.get("aggregate_score"),
        "failed_metrics": ", ".join(m for m in metrics if (dumped.get(m) or {}).get("success") is False),
        "context_chars": sum(map(len, case.retrieved_context))
        if isinstance(case.retrieved_context, list)
        else len(case.retrieved_context or ""),
        "input_tokens": sum(item.get("input_tokens", 0) for item in usage.values()),
        "output_tokens": sum(item.get("output_tokens", 0) for item in usage.values()),
    }
    for metric in metrics:
        result = dumped.get(metric) or {}
        record[f"{metric}.score"] = result.get("score")
        record[f"{metric}.explanation"] = result.get("explanation", "not evaluated")
    return record


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
        records (list[dict[str, object]]): Flattened rows.
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
    """Summarize Auto RR, compare it with manual RR when every row is labeled, and price the judge calls.

    Args:
        records (list[dict[str, object]]): Flattened rows.
        judge_model (str): The judge model id, used to look up its price.

    Returns:
        dict[str, object]: Pass rates, confusion counts, TPR, TNR, agreement, tokens, and cost. The comparison is
            left out when a row has no manual rating, and the cost is `None` for a judge without a known price.
    """
    n = len(records)
    auto_passes = sum(1 for r in records if r["auto_rr"] == "PASS")
    summary: dict[str, object] = {"n": n, "auto_pass_rate": round(auto_passes / n, 3)}
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
