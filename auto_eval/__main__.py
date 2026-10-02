"""Command line entry point: `python -m auto_eval --dataset <csv>`.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from auto_eval.dataset import load_test_cases
from auto_eval.judge import DEFAULT_JUDGE
from auto_eval.presets import PRESETS
from auto_eval.runner import run_evaluation


def main() -> None:
    """Parse the options, run one preset on the dataset, and print the summary."""
    parser = argparse.ArgumentParser(description="Run auto eval on a CSV and compare Auto RR with the manual rating.")
    parser.add_argument("--dataset", type=Path, help="CSV in the DD export format.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--run-name", help="Prefix of the output files. Defaults to <dataset>_<preset>.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, help="Evaluate only the first N rows, for a quick check.")
    parser.add_argument("--list-presets", action="store_true", help="Show the presets and exit.")
    args = parser.parse_args()

    if args.list_presets:
        for name, preset in sorted(PRESETS.items()):
            print(f"{name:<20} {preset.description}")
        return
    if args.dataset is None:
        parser.error("--dataset is required")

    load_dotenv()
    preset = PRESETS[args.preset]
    cases = load_test_cases(args.dataset, clean=preset.clean_context, limit=args.limit)
    run_name = args.run_name or re.sub(r"[^a-z0-9]+", "_", f"{args.dataset.stem}_{preset.name}".lower()).strip("_")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(run_evaluation(cases, preset, args.judge_model, args.results_dir / run_name))
    print(json.dumps({key: value for key, value in summary.items() if key != "threshold_sweep"}, indent=2))
    print(f"Rows: {args.results_dir / f'{run_name}_rows.csv'}")


if __name__ == "__main__":
    main()
