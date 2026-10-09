"""Calibrate the pipeline auto eval against the DD labels and validate the result on a second batch.

Steps 01 to 09 run on the 30-row calibration batch: the GitBook default, a rerun, three diagnoses of the groundedness
failures, two calibrations, and answer relevancy. Steps 11 to 14 repeat the setups of steps 01, 06, 07, and 08
unchanged on the 40-row validation batch. Steps 10 and 15 compare the steps without calling the judge, and the
workbook collects every CSV in one file. The datasets are read from `data/pipeline/`, and every step writes to its
own folder in `results/pipeline_calibration/`.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/evals-lifecycle
    [2] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/calibrate-the-evals
"""

import argparse
import asyncio
import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from auto_eval.pipeline import PIPELINE
from auto_eval.pipeline.presets import PRESETS, build_default
from auto_eval.shared.judge import DEFAULT_JUDGE
from auto_eval.shared.project import Preset
from auto_eval.shared.report import confusion, write_csv
from auto_eval.shared.runner import ROWS_FILE, SUMMARY_FILE, run_evaluation

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pipeline"
RESULTS_DIR = ROOT / "results" / "pipeline_calibration"
DATASETS = {
    "calibration": DATA_DIR / "test_auto_eval_intern.csv",
    "validation": DATA_DIR / "auto_rr_results_datatest_tambahan.csv",
}
FLASH_JUDGE = "openrouter/google/gemini-3-flash-preview"
STRICT_THRESHOLD = 1.0
LENIENT_THRESHOLD = 0.5
STUDY_PRESETS = {
    **PRESETS,
    "default_clean_context": Preset(
        "default_clean_context",
        "The default evaluator on cleaned, chunked contexts.",
        build_default,
        clean_context=True,
    ),
}
WORKBOOK_NAME = "auto_rr_all_steps.xlsx"
CALIBRATION_ANALYSIS_FILES = ("overview", "verdicts_by_step", "whatif")
VALIDATION_ANALYSIS_FILES = ("overview", "verdicts_by_step", "conditions", "whatif")
_SHORT_TAB_NAMES = {
    "05_diagnose_judge_gemini3flash_clean_context": "05_diagnose_gemini3flash_clean",
    "07_calibrate_completeness_first_dd_guidance": "07_calibrate_dd_guidance",
    "09_add_metric_answer_relevancy_gemini3flash": "09_add_relevancy_gemini3flash",
    "13_validate_completeness_first_dd_guidance": "13_validate_dd_guidance",
}
_MAX_COLUMN_WIDTH = 60


@dataclass(frozen=True)
class Step:
    """One run in the study.

    Attributes:
        name (str): The step label, used as the prefix of its output files and as the tracker dataset name.
        purpose (str): What the step checks.
        preset (str): A key of `STUDY_PRESETS`.
        dataset (str): A key of `DATASETS`. Defaults to "calibration".
        judge_model (str): The judge model id. Defaults to `DEFAULT_JUDGE`.
        source (str | None): The calibration step whose setup a validation step repeats. Defaults to None.
    """

    name: str
    purpose: str
    preset: str
    dataset: str = "calibration"
    judge_model: str = DEFAULT_JUDGE
    source: str | None = None


CALIBRATION_STEPS = (
    Step("01_baseline_geval", "GitBook standard GEval with the platform's judge settings", "default"),
    Step("02_baseline_geval_rerun", "Step 01 again, to check judge variance", "default"),
    Step("03_diagnose_clean_context", "Groundedness with cleaned contexts", "default_clean_context"),
    Step("04_diagnose_judge_gemini3flash", "Groundedness with a stronger judge", "default", judge_model=FLASH_JUDGE),
    Step(
        "05_diagnose_judge_gemini3flash_clean_context",
        "Stronger judge and cleaned contexts together",
        "default_clean_context",
        judge_model=FLASH_JUDGE,
    ),
    Step(
        "06_calibrate_completeness_first",
        "Completeness decides the verdict, as in DD's first step",
        "completeness_first",
    ),
    Step(
        "07_calibrate_completeness_first_dd_guidance",
        "Step 06 plus DD's standard in the completeness prompt",
        "dd_calibrated",
    ),
    Step("08_add_metric_answer_relevancy", "Answer relevancy, a metric beyond GEval", "answer_relevancy"),
    Step(
        "09_add_metric_answer_relevancy_gemini3flash",
        "Answer relevancy with a stronger judge",
        "answer_relevancy",
        judge_model=FLASH_JUDGE,
    ),
)


def _validation_step(name: str, source: str) -> Step:
    """Repeat a calibration step's setup, unchanged, on the validation batch.

    Args:
        name (str): The validation step label.
        source (str): The calibration step to repeat.

    Returns:
        Step: The validation step.
    """
    calibration_step = next(step for step in CALIBRATION_STEPS if step.name == source)
    changes = {"name": name, "purpose": f"Step {source[:2]} on the validation batch", "dataset": "validation"}
    return Step(**{**asdict(calibration_step), **changes, "source": source})


VALIDATION_STEPS = (
    _validation_step("11_validate_baseline_geval", "01_baseline_geval"),
    _validation_step("12_validate_completeness_first", "06_calibrate_completeness_first"),
    _validation_step("13_validate_completeness_first_dd_guidance", "07_calibrate_completeness_first_dd_guidance"),
    _validation_step("14_validate_answer_relevancy", "08_add_metric_answer_relevancy"),
)
STEPS = CALIBRATION_STEPS + VALIDATION_STEPS


def analysis_path(label: str, name: str) -> Path:
    """Locate one file of an analysis step.

    Args:
        label (str): The analysis step number, `10` or `15`.
        name (str): The file name without `.csv`, such as `overview`.

    Returns:
        Path: The file in the step's `<label>_analysis` folder.
    """
    return RESULTS_DIR / f"{label}_analysis" / f"{name}.csv"


def write_analysis(label: str, name: str, rows: list[dict[str, object]]) -> None:
    """Write one file of an analysis step, creating its folder.

    Args:
        label (str): The analysis step number, `10` or `15`.
        name (str): The file name without `.csv`.
        rows (list[dict[str, object]]): The rows to write.
    """
    path = analysis_path(label, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, rows)


async def run_step(step: Step) -> dict[str, object]:
    """Run one step and write `rows.csv` and `summary.json` to its folder.

    Args:
        step (Step): The step configuration.

    Returns:
        dict[str, object]: The step summary.
    """
    preset = STUDY_PRESETS[step.preset]
    samples = PIPELINE.load(DATASETS[step.dataset], preset, None)
    metadata = {"step": asdict(step)}
    return await run_evaluation(PIPELINE, preset, samples, step.judge_model, RESULTS_DIR / step.name, metadata=metadata)


def load_rows(step_name: str) -> dict[str, dict[str, str]]:
    """Read a step's rows keyed by row number.

    Args:
        step_name (str): The step label.

    Returns:
        dict[str, dict[str, str]]: The rows of the step's `rows.csv`.
    """
    with (RESULTS_DIR / step_name / ROWS_FILE).open(encoding="utf-8", newline="") as file:
        return {row["no"]: row for row in csv.DictReader(file)}


def score(row: dict[str, str], metric: str) -> float:
    """Read a metric score from a saved row, treating a missing score as 0.

    Args:
        row (dict[str, str]): A row from a rows file.
        metric (str): The metric name.

    Returns:
        float: The score.
    """
    value = row.get(f"{metric}.score", "")
    return float(value) if value not in ("", "None") else 0.0


def geval_pass(row: dict[str, str], completeness: float, groundedness: float | None) -> bool:
    """Apply the GEval AND gate to saved scores with the given thresholds.

    Args:
        row (dict[str, str]): A row from a GEval rows file.
        completeness (float): The completeness threshold.
        groundedness (float | None): The groundedness threshold, or `None` to skip groundedness.

    Returns:
        bool: Whether every checked metric passes. Redundancy passes at 0.5 or lower.
    """
    if score(row, "completeness") < completeness or score(row, "redundancy") > LENIENT_THRESHOLD:
        return False
    return groundedness is None or score(row, "groundedness") >= groundedness


def relevance_rules(calibrated_step: str, relevance_step: str) -> list[tuple[str, str, Callable[[str], bool]]]:
    """Combine a calibrated step with answer relevancy, as an extra gate and as a fallback.

    Args:
        calibrated_step (str): The step whose verdicts are the base.
        relevance_step (str): The answer relevancy step on the same batch.

    Returns:
        list[tuple[str, str, Callable[[str], bool]]]: The analysis name, the rule, and a pass check per row number.
    """
    base = load_rows(calibrated_step)
    relevance = load_rows(relevance_step)
    label = f"step {calibrated_step[:2]}"
    group = f"relevance_with_step_{calibrated_step[:2]}"

    def calibrated(n: str) -> bool:
        return base[n]["auto_rr"] == "PASS"

    def relevant(n: str) -> bool:
        return score(relevance[n], "deepeval_answer_relevancy") >= LENIENT_THRESHOLD

    return [
        (group, label, calibrated),
        (group, f"{label}, and relevance must pass", lambda n: calibrated(n) and relevant(n)),
        (group, f"{label}, or relevance passes", lambda n: calibrated(n) or relevant(n)),
    ]


def whatif_rules() -> list[tuple[str, str, Callable[[str], bool]]]:
    """Define the rules that step 10 applies to saved calibration scores.

    Returns:
        list[tuple[str, str, Callable[[str], bool]]]: The analysis name, the rule, and a pass check per row number.
    """
    base = load_rows("01_baseline_geval")
    relevance = load_rows("08_add_metric_answer_relevancy")

    def complete(n: str) -> bool:
        return score(base[n], "completeness") >= STRICT_THRESHOLD

    def grounded(n: str) -> bool:
        return score(base[n], "groundedness") >= STRICT_THRESHOLD

    def relevant(n: str) -> bool:
        return score(relevance[n], "deepeval_answer_relevancy") >= LENIENT_THRESHOLD

    def has_context(n: str) -> bool:
        return int(base[n]["context_chars"]) > 0

    return [
        ("dd_step_readings", "all metrics must pass (AND gate)", lambda n: base[n]["auto_rr"] == "PASS"),
        ("dd_step_readings", "step 1 decides", complete),
        ("dd_step_readings", "step 1, else pass if grounded", lambda n: complete(n) or grounded(n)),
        (
            "dd_step_readings",
            "step 1, else grounded, else pass if relevant",
            lambda n: complete(n) or grounded(n) or relevant(n),
        ),
        (
            "geval_thresholds",
            "platform default",
            lambda n: geval_pass(base[n], STRICT_THRESHOLD, STRICT_THRESHOLD),
        ),
        (
            "geval_thresholds",
            "skip groundedness when the context is empty",
            lambda n: geval_pass(base[n], STRICT_THRESHOLD, STRICT_THRESHOLD if has_context(n) else None),
        ),
        (
            "geval_thresholds",
            "above, and groundedness passes at 0.5",
            lambda n: geval_pass(base[n], STRICT_THRESHOLD, LENIENT_THRESHOLD if has_context(n) else None),
        ),
        (
            "geval_thresholds",
            "above, and completeness passes at 0.5",
            lambda n: geval_pass(base[n], LENIENT_THRESHOLD, LENIENT_THRESHOLD if has_context(n) else None),
        ),
        *relevance_rules("07_calibrate_completeness_first_dd_guidance", "08_add_metric_answer_relevancy"),
    ]


def misses(records: list[dict[str, object]], manual_rr: str) -> str:
    """Join the row numbers where Auto RR differs from the given manual verdict.

    Args:
        records (list[dict[str, object]]): Rows with `no`, `manual_rr`, and `auto_rr`.
        manual_rr (str): `PASS` to list false negatives, `FAIL` to list false positives.

    Returns:
        str: The row numbers, separated by spaces.
    """
    return " ".join(str(r["no"]) for r in records if r["manual_rr"] == manual_rr and r["auto_rr"] != manual_rr)


def overview_rows(steps: tuple[Step, ...]) -> list[dict[str, object]]:
    """Summarize each step on one line.

    Args:
        steps (tuple[Step, ...]): The steps to summarize.

    Returns:
        list[dict[str, object]]: The step setup, confusion counts, TPR, TNR, agreement, and cost per step.
    """
    keys = ("TP", "FN", "TN", "FP", "TPR", "TNR", "agreement", "cost_usd")
    summaries = [json.loads((RESULTS_DIR / s.name / SUMMARY_FILE).read_text(encoding="utf-8")) for s in steps]
    return [{**summary["step"], **{key: summary[key] for key in keys}} for summary in summaries]


def verdict_rows(steps: tuple[Step, ...]) -> list[dict[str, object]]:
    """List each row of a batch with the manual verdict and the Auto RR of every step.

    Args:
        steps (tuple[Step, ...]): Steps that ran on the same batch.

    Returns:
        list[dict[str, object]]: One row per test case, in row order.
    """
    rows_by_step = {s.name: load_rows(s.name) for s in steps}
    base = rows_by_step[steps[0].name]
    return [
        {
            "no": n,
            "question": row["question"],
            "manual_rr": row["manual_rr"],
            "issue_category": row["issue_category"],
            **{name: rows[n]["auto_rr"] for name, rows in rows_by_step.items()},
        }
        for n, row in sorted(base.items(), key=lambda item: int(item[0]))
    ]


def rule_rows(
    rules: list[tuple[str, str, Callable[[str], bool]]], base: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    """Apply each rule to the rows of a batch and compare the result with the manual verdicts.

    Args:
        rules (list[tuple[str, str, Callable[[str], bool]]]): The analysis name, the rule, and a pass check.
        base (dict[str, dict[str, str]]): Any step's rows on the batch, used for row numbers and manual verdicts.

    Returns:
        list[dict[str, object]]: Confusion counts and missed rows per rule.
    """
    output = []
    for analysis, rule, passes in rules:
        records = [
            {"no": n, "manual_rr": row["manual_rr"], "auto_rr": "PASS" if passes(n) else "FAIL"}
            for n, row in base.items()
        ]
        output.append(
            {
                "analysis": analysis,
                "rule": rule,
                **confusion(records),
                "fn_rows": misses(records, "PASS"),
                "fp_rows": misses(records, "FAIL"),
            }
        )
    return output


def condition_rows() -> list[dict[str, object]]:
    """Measure each validation step on the validation batch alone and together with its calibration step.

    The two conditions follow the Calibration Validation Set check of the Calibrate the Evals page. Row numbers are
    prefixed with `cal-` or `val-` because both batches start at 1.

    Returns:
        list[dict[str, object]]: One row per validation step and condition.
    """
    output = []
    for step in VALIDATION_STEPS:
        calibration = [{**r, "no": f"cal-{r['no']}"} for r in load_rows(step.source).values()]
        validation = [{**r, "no": f"val-{r['no']}"} for r in load_rows(step.name).values()]
        conditions = (
            ("calibration batch", calibration),
            ("validation batch", validation),
            ("both batches", calibration + validation),
        )
        for condition, records in conditions:
            output.append(
                {
                    "step": step.name,
                    "source": step.source,
                    "condition": condition,
                    "rows": len(records),
                    **confusion(records),
                    "fn_rows": misses(records, "PASS"),
                    "fp_rows": misses(records, "FAIL"),
                }
            )
    return output


def write_calibration_analysis() -> None:
    """Write the step 10 files for the calibration batch."""
    write_analysis("10", "overview", overview_rows(CALIBRATION_STEPS))
    write_analysis("10", "verdicts_by_step", verdict_rows(CALIBRATION_STEPS))
    write_analysis("10", "whatif", rule_rows(whatif_rules(), load_rows("01_baseline_geval")))


def write_validation_analysis() -> None:
    """Write the step 15 files for the validation batch."""
    write_analysis("15", "overview", overview_rows(VALIDATION_STEPS))
    write_analysis("15", "verdicts_by_step", verdict_rows(VALIDATION_STEPS))
    write_analysis("15", "conditions", condition_rows())
    rules = relevance_rules("13_validate_completeness_first_dd_guidance", "14_validate_answer_relevancy")
    write_analysis("15", "whatif", rule_rows(rules, load_rows("11_validate_baseline_geval")))


def write_workbook() -> None:
    """Combine the step CSVs into one workbook in `results/pipeline_calibration/`, one tab per file in step order.

    Files that do not exist yet are skipped. Excel caps tab names at 31 characters, so longer step names use
    `_SHORT_TAB_NAMES`.
    """
    sources = [(s.name, RESULTS_DIR / s.name / ROWS_FILE) for s in CALIBRATION_STEPS]
    sources += [(f"10_{name}", analysis_path("10", name)) for name in CALIBRATION_ANALYSIS_FILES]
    sources += [(s.name, RESULTS_DIR / s.name / ROWS_FILE) for s in VALIDATION_STEPS]
    sources += [(f"15_{name}", analysis_path("15", name)) for name in VALIDATION_ANALYSIS_FILES]
    with pd.ExcelWriter(RESULTS_DIR / WORKBOOK_NAME, engine="openpyxl") as writer:
        for name, path in sources:
            if not path.exists():
                continue
            tab = _SHORT_TAB_NAMES.get(name, name)
            pd.read_csv(path).to_excel(writer, sheet_name=tab, index=False)
            sheet = writer.sheets[tab]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                longest = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
                sheet.column_dimensions[column[0].column_letter].width = min(longest + 2, _MAX_COLUMN_WIDTH)


async def main(selected: list[str], analysis_only: bool) -> None:
    """Run the selected steps in order, then write each analysis step whose inputs exist, and the workbook.

    Args:
        selected (list[str]): Step prefixes to run, such as `01`. Empty runs every step.
        analysis_only (bool): Whether to skip the judge calls and only rewrite the analysis files and the workbook.
    """
    load_dotenv()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not analysis_only:
        for step in STEPS:
            if selected and not any(step.name.startswith(prefix) for prefix in selected):
                continue
            summary = await run_step(step)
            print(f"{step.name}: TPR {summary['TPR']}, TNR {summary['TNR']}, agreement {summary['agreement']}")
    analyses = (
        ("10", [s.name for s in CALIBRATION_STEPS], write_calibration_analysis, CALIBRATION_ANALYSIS_FILES),
        (
            "15",
            [s.name for s in VALIDATION_STEPS] + [s.source for s in VALIDATION_STEPS],
            write_validation_analysis,
            VALIDATION_ANALYSIS_FILES,
        ),
    )
    for label, required, write, files in analyses:
        missing = [name for name in required if not (RESULTS_DIR / name / ROWS_FILE).exists()]
        if missing:
            print(f"Skipped step {label}, missing results for: {', '.join(missing)}")
            continue
        write()
        print(f"{label}_analysis: wrote {', '.join(f'{name}.csv' for name in files)}")
    write_workbook()
    print(f"Wrote {WORKBOOK_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline calibration study steps in order.")
    parser.add_argument("--steps", nargs="*", default=[], help="Step prefixes to run, such as 01 06. Default: all.")
    parser.add_argument("--analysis-only", action="store_true", help="Only rewrite the analysis files and workbook.")
    args = parser.parse_args()
    asyncio.run(main(args.steps, args.analysis_only))
