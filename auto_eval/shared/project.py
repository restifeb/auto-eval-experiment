"""The kinds of project auto eval supports, and the evaluator presets each one offers.

A project describes one kind of target system: how its exported CSV becomes test cases, and how each verdict becomes
an output row. `pipeline` covers RAG and QnA pipelines, and `agent` covers agents that call tools.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/llmtestcase
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gllm_evals.evaluator.evaluator import BaseEvaluator
from gllm_evals.types import LLMTestCase
from gllm_inference.lm_invoker.lm_invoker import BaseLMInvoker


@dataclass(frozen=True)
class Preset:
    """A named evaluator setup.

    Attributes:
        name (str): The name used on the command line.
        description (str): What the setup does.
        build_evaluator (Callable[[BaseLMInvoker], BaseEvaluator]): Builds the evaluator from the judge.
        clean_context (bool): Whether the loader cleans and chunks the retrieved context. Only the pipeline loader
            reads it. Defaults to False.
        gate_metric (str | None): The metric whose score alone decides Auto RR, for the threshold sweep in the
            summary. Defaults to None.
    """

    name: str
    description: str
    build_evaluator: Callable[[BaseLMInvoker], BaseEvaluator]
    clean_context: bool = False
    gate_metric: str | None = None


@dataclass(frozen=True)
class Sample:
    """One dataset row and the test case built from it.

    Attributes:
        row (dict[str, str]): The row as read from the CSV.
        case (LLMTestCase): The test case the evaluator scores.
    """

    row: dict[str, str]
    case: LLMTestCase


@dataclass(frozen=True)
class Project:
    """One kind of target system: its dataset format, its presets, and its output rows.

    Attributes:
        name (str): The name used on the command line.
        description (str): What the project evaluates.
        presets (dict[str, Preset]): The evaluator setups, keyed by name.
        load (Callable[[Path, Preset, int | None], list[Sample]]): Reads an exported CSV with a preset, keeping only
            the first rows when a limit is given.
        to_record (Callable[[Sample, dict[str, Any], tuple[str, ...]], dict[str, object]]): Builds a sample's output
            row from the dumped evaluator output and the names of the metrics that ran.
    """

    name: str
    description: str
    presets: dict[str, Preset]
    load: Callable[[Path, Preset, int | None], list[Sample]]
    to_record: Callable[[Sample, dict[str, Any], tuple[str, ...]], dict[str, object]]
