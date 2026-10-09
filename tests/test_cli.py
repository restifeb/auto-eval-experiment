"""Tests for the `auto-eval` command line.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

from pathlib import Path

import pytest

from auto_eval import cli

AGENT_RUNS = Path(__file__).parent / "resources" / "agent" / "agent_runs.csv"


@pytest.mark.parametrize(
    ("dataset", "preset", "expected"),
    [
        ("data/agent/test_case_agent_evaluation.csv", "default", "test_case_agent_evaluation_default"),
        ("Auto RR Results - datatest-tambahan.csv", "dd_calibrated", "auto_rr_results_datatest_tambahan_dd_calibrated"),
    ],
)
def test_default_run_name_with_dataset(dataset, preset, expected):
    """
    Condition:
    A dataset file name with dashes or spaces, and a preset.

    Expected:
    The run is named after both in lowercase snake case.
    """
    assert cli.default_run_name(Path(dataset), preset) == expected


def test_main_with_list_presets(capsys):
    """
    Condition:
    The agent project's presets are listed.

    Expected:
    Its `default` preset is shown with its description, and nothing runs.
    """
    cli.main(["--project", "agent", "--list-presets"])

    assert capsys.readouterr().out.startswith("default              The five GEval metrics")


def test_main_with_preset_of_other_project():
    """
    Condition:
    A pipeline preset is asked for in the agent project.

    Expected:
    The command stops with a usage error.
    """
    with pytest.raises(SystemExit):
        cli.main(["--project", "agent", "--preset", "dd_calibrated", "--dataset", str(AGENT_RUNS)])


def test_main_with_agent_project(monkeypatch):
    """
    Condition:
    The agent project runs without `--results-dir` or `--run-name`.

    Expected:
    Both numbered runs are evaluated, and the run's folder is `results/agent/<dataset>_<preset>`.
    """
    calls = []

    async def fake_run_evaluation(project, preset, samples, judge_model, output_dir):
        calls.append((project.name, preset.name, len(samples), output_dir))
        return {"project": project.name}

    monkeypatch.setattr(cli, "run_evaluation", fake_run_evaluation)

    cli.main(["--project", "agent", "--dataset", str(AGENT_RUNS)])

    assert calls == [("agent", "default", 2, Path("results/agent/agent_runs_default"))]
