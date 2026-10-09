"""Run one preset on a project's samples and save the run's rows and summary.

A run writes `rows.csv` and `summary.json` to its own folder, such as
`results/agent/test_case_agent_evaluation_default/`, and the gllm-evals tracker writes to a `tracker` folder next to it.

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

from auto_eval.shared.judge import build_judge
from auto_eval.shared.project import Preset, Project, Sample
from auto_eval.shared.report import dump_output, summarize, sweep_thresholds, verdict, write_csv

ROWS_FILE = "rows.csv"
SUMMARY_FILE = "summary.json"
TRACKER_DIR = "tracker"


async def run_evaluation(  # noqa: PLR0913
    project: Project,
    preset: Preset,
    samples: list[Sample],
    judge_model: str,
    output_dir: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate the samples and write `rows.csv` and `summary.json` to `output_dir`.

    TPR and TNR are computed only when every case has a manual rating. The tracker uses the folder's name as the
    dataset name.

    Args:
        project (Project): The project, which builds the output rows.
        preset (Preset): The evaluator setup.
        samples (list[Sample]): The rows and their test cases.
        judge_model (str): The judge model id.
        output_dir (Path): The run's folder.
        metadata (dict[str, object] | None, optional): Extra fields placed first in the summary. Defaults to None.

    Returns:
        dict[str, object]: The run summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = preset.build_evaluator(build_judge(judge_model))
    metrics = tuple(metric.name for metric in evaluator.metrics)
    cases = [sample.case for sample in samples]
    labeled = all(getattr(case, "label", None) for case in cases)
    tracker = CSVExperimentTracker(
        project_name=f"auto-eval-{project.name}", output_dir=str(output_dir.parent / TRACKER_DIR)
    )
    result = await evaluate_suites(
        suites=[EvalSuite(name="auto_eval", data=cases, evaluators=[evaluator])],
        experiment_tracker=tracker,
        run_aggregators=[true_positive_rate, true_negative_rate] if labeled else None,
        dataset_name=output_dir.name,
    )
    outputs = [dump_output(next(iter(row[0].values()))) for row in result.suites["auto_eval"].results]
    records = [project.to_record(sample, output, metrics) for sample, output in zip(samples, outputs, strict=True)]
    write_csv(output_dir / ROWS_FILE, records)
    verdicts = [verdict(case, output, metrics) for case, output in zip(cases, outputs, strict=True)]
    summary = {
        **(metadata or {}),
        "project": project.name,
        "preset": preset.name,
        "judge_model": judge_model,
        "clean_context": preset.clean_context,
        **summarize(verdicts, judge_model),
        "library_aggregators": result.run_aggregators_result,
    }
    if labeled and preset.gate_metric:
        summary["threshold_sweep"] = sweep_thresholds(verdicts, preset.gate_metric)
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
