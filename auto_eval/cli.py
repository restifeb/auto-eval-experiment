"""The `auto-eval` command: pick a project, load its CSV, run one preset, and print the summary.

    auto-eval --project pipeline --dataset data/pipeline/test_auto_eval_intern.csv --preset dd_calibrated
    auto-eval --project agent --dataset data/agent/test_case_agent_evaluation.csv
    auto-eval --project agent --list-presets

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

import argparse
import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from auto_eval.agent import AGENT
from auto_eval.pipeline import PIPELINE
from auto_eval.shared.judge import DEFAULT_JUDGE
from auto_eval.shared.runner import ROWS_FILE, run_evaluation

PROJECTS = {project.name: project for project in (PIPELINE, AGENT)}
RESULTS_DIR = Path("results")


def default_run_name(dataset: Path, preset_name: str) -> str:
    """Name a run after its dataset file and preset.

    Args:
        dataset (Path): The dataset CSV.
        preset_name (str): The preset name.

    Returns:
        str: `<dataset stem>_<preset>` in lowercase, with every other character run replaced by an underscore.
    """
    return re.sub(r"[^a-z0-9]+", "_", f"{dataset.stem}_{preset_name}".lower()).strip("_")


def build_parser() -> argparse.ArgumentParser:
    """Build the `auto-eval` argument parser.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser = argparse.ArgumentParser(
        prog="auto-eval",
        description="Run auto eval on a pipeline or agent CSV and compare Auto RR with the manual rating.",
    )
    parser.add_argument(
        "--project", choices=sorted(PROJECTS), default="pipeline", help="The kind of system that produced the CSV."
    )
    parser.add_argument("--dataset", type=Path, help="The CSV the project exported.")
    parser.add_argument("--preset", default="default", help="An evaluator setup of the project. See --list-presets.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--run-name", help="Name of the run's results folder. Defaults to <dataset>_<preset>.")
    parser.add_argument("--results-dir", type=Path, help="Where the run's folder goes. Defaults to results/<project>.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N rows, for a quick check.")
    parser.add_argument("--list-presets", action="store_true", help="Show the project's presets and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the options, run one preset of the chosen project on the dataset, and print the summary.

    Args:
        argv (Sequence[str] | None, optional): The arguments, without the program name. Defaults to None, which
            reads `sys.argv`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    project = PROJECTS[args.project]
    if args.list_presets:
        for name, preset in sorted(project.presets.items()):
            print(f"{name:<20} {preset.description}")
        return
    if args.preset not in project.presets:
        parser.error(f"the {project.name} project has no preset {args.preset!r}; see --list-presets")
    if args.dataset is None:
        parser.error("--dataset is required")

    load_dotenv()
    preset = project.presets[args.preset]
    samples = project.load(args.dataset, preset, args.limit)
    results_dir = args.results_dir or RESULTS_DIR / project.name
    output_dir = results_dir / (args.run_name or default_run_name(args.dataset, preset.name))
    summary = asyncio.run(run_evaluation(project, preset, samples, args.judge_model, output_dir))
    print(json.dumps({key: value for key, value in summary.items() if key != "threshold_sweep"}, indent=2))
    print(f"Rows: {output_dir / ROWS_FILE}")
