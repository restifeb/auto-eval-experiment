"""Run one preset on a set of test cases and write its rows and summary.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/evaluate-suites
    [2] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/run-aggregators
"""

import json
from pathlib import Path

from gllm_evals import EvalSuite, evaluate_suites
from gllm_evals.aggregation import true_negative_rate, true_positive_rate
from gllm_evals.experiment_tracker import CSVExperimentTracker
from gllm_evals.types import LLMTestCase

from auto_eval.judge import build_judge
from auto_eval.presets import Preset
from auto_eval.report import summarize, sweep_thresholds, to_record, write_csv


async def run_evaluation(
    cases: list[LLMTestCase],
    preset: Preset,
    judge_model: str,
    output_prefix: Path,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate the cases and write `<output_prefix>_rows.csv` and `<output_prefix>_summary.json`.

    TPR and TNR are computed only when every case has a manual rating. The tracker writes to a `tracker` folder next
    to the outputs, with the prefix's file name as the dataset name.

    Args:
        cases (list[LLMTestCase]): The test cases.
        preset (Preset): The evaluator setup.
        judge_model (str): The judge model id.
        output_prefix (Path): The output folder plus the run name, such as `results/01_baseline_geval`.
        metadata (dict[str, object] | None, optional): Extra fields placed first in the summary. Defaults to None.

    Returns:
        dict[str, object]: The run summary.
    """
    evaluator = preset.build_evaluator(build_judge(judge_model))
    metrics = tuple(metric.name for metric in evaluator.metrics)
    labeled = all(getattr(case, "label", None) for case in cases)
    tracker = CSVExperimentTracker(project_name="auto-eval", output_dir=str(output_prefix.parent / "tracker"))
    result = await evaluate_suites(
        suites=[EvalSuite(name="auto_eval", data=cases, evaluators=[evaluator])],
        experiment_tracker=tracker,
        run_aggregators=[true_positive_rate, true_negative_rate] if labeled else None,
        dataset_name=output_prefix.name,
    )
    rows = result.suites["auto_eval"].results
    records = [to_record(case, next(iter(row[0].values())), metrics) for case, row in zip(cases, rows, strict=True)]
    write_csv(output_prefix.with_name(f"{output_prefix.name}_rows.csv"), records)
    summary = {
        **(metadata or {}),
        "preset": preset.name,
        "judge_model": judge_model,
        "clean_context": preset.clean_context,
        **summarize(records, judge_model),
        "library_aggregators": result.run_aggregators_result,
    }
    if labeled and preset.gate_metric:
        summary["threshold_sweep"] = sweep_thresholds(records, preset.gate_metric)
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
