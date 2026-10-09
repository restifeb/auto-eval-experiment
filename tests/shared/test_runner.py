"""Tests for running a preset and saving its rows and summary.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

import asyncio
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from auto_eval.agent import AGENT
from auto_eval.agent.rows import TEMPLATE_METRIC_COLUMNS
from auto_eval.shared import runner

RESOURCES = Path(__file__).parents[1] / "resources" / "agent"
JUDGE_MODEL = "openrouter/google/gemini-3.1-flash-lite"


def make_output(passed: bool) -> dict[str, object]:
    """Build an evaluator output whose metrics all pass, or where completeness fails."""
    output: dict[str, object] = {
        "aggregate_success": passed,
        "evaluator_token_usage": {"completeness": {"input_tokens": 2000, "output_tokens": 100}},
    }
    for metric in TEMPLATE_METRIC_COLUMNS:
        success = passed or metric != "completeness"
        output[metric] = {"score": 1.0 if success else 0.0, "success": success, "explanation": f"{metric} checked."}
    return output


def test_run_evaluation_with_agent_export(tmp_path, monkeypatch):
    """
    Condition:
    The agent preset runs on an export with a run that answered and a run that returned no answer.

    Expected:
    `rows.csv` is the export with its evaluation columns filled, and `summary.json` compares Auto RR with
    `Manual_Eval`.
    """

    async def fake_evaluate_suites(suites, experiment_tracker, run_aggregators, dataset_name):
        results = [[{"generation": make_output(bool(case.actual_output))}] for case in suites[0].data]
        return SimpleNamespace(suites={"auto_eval": SimpleNamespace(results=results)}, run_aggregators_result={})

    monkeypatch.setattr(runner, "evaluate_suites", fake_evaluate_suites)
    monkeypatch.setattr(runner, "build_judge", lambda judge_model: None)
    preset = AGENT.presets["default"]
    samples = AGENT.load(RESOURCES / "agent_runs.csv", preset, None)
    output_dir = tmp_path / "agent_runs_default"

    summary = asyncio.run(runner.run_evaluation(AGENT, preset, samples, JUDGE_MODEL, output_dir))

    with (output_dir / runner.ROWS_FILE).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with (RESOURCES / "agent_runs.csv").open(encoding="utf-8", newline="") as file:
        export_columns = next(csv.reader(file))
    assert list(rows[0]) == export_columns
    assert [(row["auto_rr"], row["possible_issues"]) for row in rows] == [("PASS", ""), ("FAIL", "completeness")]
    assert rows[1]["agent_error"].startswith("The agent returned no answer.")
    assert (summary["project"], summary["TP"], summary["TN"], summary["judge_errors"]) == ("agent", 1, 1, 0)
    assert json.loads((output_dir / runner.SUMMARY_FILE).read_text(encoding="utf-8")) == summary
