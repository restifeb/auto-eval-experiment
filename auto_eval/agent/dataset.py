"""Load an agent export from GL AIP or GLChat into gllm-evals test cases.

Each row is one agent run. `source_tool_<n>` holds the n-th tool step as the export renders it, with the tool name,
its arguments, its output, and its status. `retrieved_context_or_queried_data_<n>` holds the same output as the context
the answer was built from.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/llmtestcase
    [2] https://gdplabs.gitbook.io/sdk/gl-aip-ai-agent-package/guides/agent-evaluations
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gllm_evals.types import LLMTestCase, ToolCall

from auto_eval.shared.project import Preset, Sample

NO_TOOL_CONTEXT = "The agent called no tool, so no context was retrieved."
FINISHED_STATUS = "finished"

_TOOL_NAME = re.compile(r"^\[Tool: (.+)\]$", re.MULTILINE)
_TOOL_ARGS = re.compile(r"^\*\*Args:\*\*\n(.*?)\n\n\*\*Output:\*\*", re.MULTILINE | re.DOTALL)
_TOOL_OUTPUT = re.compile(r"^\*\*Output:\*\*\n(.*?)(?:\n\n\*\*Status:\*\*|\Z)", re.MULTILINE | re.DOTALL)
_TOOL_STATUS = re.compile(r"^\*\*Status:\*\* (\S+)", re.MULTILINE)
_FENCED_OUTPUT = re.compile(r"^\*\*Output:\*\*\s*```[^\n]*\n(.*?)\n```$", re.DOTALL)


@dataclass(frozen=True)
class Columns:
    """Column names of the agent export. The defaults follow the GL AIP and GLChat export.

    Attributes:
        row_id (str): The run number. Rows where it is empty are skipped. Defaults to "Index".
        question (str): The user question. Defaults to "Question".
        expected (str): The expected answer. Defaults to "Expected Answer".
        generated (str): The agent's final answer. Defaults to "Answer".
        manual_rr (str): The manual rating, `Pass` or `Fail`. Defaults to "Manual_Eval".
        issue_category (str): The manual issue category. Defaults to "Issue_Category".
        agent_error (str): An error the export recorded for the run. Defaults to "agent_error".
        tool_step_prefix (str): The prefix of the numbered tool step columns. Defaults to "source_tool_".
        context_prefix (str): The prefix of the numbered tool output columns. Defaults to
            "retrieved_context_or_queried_data_".
    """

    row_id: str = "Index"
    question: str = "Question"
    expected: str = "Expected Answer"
    generated: str = "Answer"
    manual_rr: str = "Manual_Eval"
    issue_category: str = "Issue_Category"
    agent_error: str = "agent_error"
    tool_step_prefix: str = "source_tool_"
    context_prefix: str = "retrieved_context_or_queried_data_"


@dataclass(frozen=True)
class ToolStep:
    """One tool call of an agent run.

    Attributes:
        name (str): The tool name.
        args (dict[str, Any]): The arguments the agent passed. Arguments that are not a JSON object sit under `raw`.
        output (str): What the tool returned.
        status (str): The step status, such as `finished`. Empty when the export shows none.
    """

    name: str
    args: dict[str, Any]
    output: str
    status: str


def _parse_args(text: str) -> dict[str, Any]:
    """Parse the arguments of a tool step, keeping text that is not a JSON object under `raw`."""
    try:
        args = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return {"raw": text}
    return args if isinstance(args, dict) else {"raw": text}


def parse_tool_step(block: str) -> ToolStep:
    """Read one `source_tool_<n>` cell.

    Args:
        block (str): The tool step as the export renders it, with `[Tool: ...]`, `**Args:**`, `**Output:**`, and
            `**Status:**` sections.

    Returns:
        ToolStep: The tool name, arguments, output, and status.
    """
    name = _TOOL_NAME.search(block)
    args = _TOOL_ARGS.search(block)
    output = _TOOL_OUTPUT.search(block)
    status = _TOOL_STATUS.search(block)
    return ToolStep(
        name=name.group(1).strip() if name else "",
        args=_parse_args(args.group(1) if args else ""),
        output=output.group(1).strip() if output else "",
        status=status.group(1) if status else "",
    )


def clean_tool_output(text: str) -> str:
    """Drop the `**Output:**` heading and the code fence the export puts around a tool output.

    Args:
        text (str): A `retrieved_context_or_queried_data_<n>` cell.

    Returns:
        str: The tool output alone, or the stripped cell when it has no such wrapping.
    """
    stripped = text.strip()
    match = _FENCED_OUTPUT.match(stripped)
    return match.group(1) if match else stripped


def numbered_values(row: dict[str, str], prefix: str) -> list[str]:
    """Collect the non-empty cells of the columns `<prefix>1`, `<prefix>2`, and so on, in number order.

    Args:
        row (dict[str, str]): The CSV row.
        prefix (str): The column prefix, such as `source_tool_`.

    Returns:
        list[str]: The cell values.
    """
    numbered = sorted(
        (int(key[len(prefix) :]), key) for key in row if key.startswith(prefix) and key[len(prefix) :].isdigit()
    )
    return [row[key] for _, key in numbered if (row[key] or "").strip()]


def describe_agent_error(row: dict[str, str], columns: Columns, steps: list[ToolStep]) -> str:
    """Describe what went wrong in the agent run itself.

    An error the export recorded is kept as it is. Otherwise an empty answer and every tool step that ended with a
    status other than `finished` are reported.

    Args:
        row (dict[str, str]): The CSV row.
        columns (Columns): The column names.
        steps (list[ToolStep]): The run's tool steps.

    Returns:
        str: The errors, or an empty string when the run had none.
    """
    recorded = (row.get(columns.agent_error) or "").strip()
    if recorded:
        return recorded
    problems = [] if (row.get(columns.generated) or "").strip() else ["The agent returned no answer."]
    problems += [
        f"{step.name} ended with status {step.status}." for step in steps if step.status not in ("", FINISHED_STATUS)
    ]
    return " ".join(problems)


def read_rows(path: Path, columns: Columns) -> list[dict[str, str]]:
    """Read the numbered runs of an agent export.

    Args:
        path (Path): The CSV file.
        columns (Columns): The column names.

    Returns:
        list[dict[str, str]]: The rows whose run number is not empty, in file order.
    """
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [row for row in csv.DictReader(file) if (row.get(columns.row_id) or "").strip()]


def to_case(row: dict[str, str], columns: Columns) -> LLMTestCase:
    """Map one agent run to a test case, with the tool outputs as the retrieved context.

    The context comes from the `retrieved_context_or_queried_data_<n>` cells, or from the tool step outputs when
    those cells are empty. A run without any tool output gets `NO_TOOL_CONTEXT`, because gllm-evals skips
    groundedness on a case without context instead of failing an answer written from memory.

    Args:
        row (dict[str, str]): The CSV row.
        columns (Columns): The column names.

    Returns:
        LLMTestCase: The test case, with every tool call in `tools_called`, the manual rating as `label` when it is
            present, and `agent_error`.
    """
    steps = [parse_tool_step(block) for block in numbered_values(row, columns.tool_step_prefix)]
    context = [clean_tool_output(text) for text in numbered_values(row, columns.context_prefix)]
    context = context or [step.output for step in steps if step.output] or [NO_TOOL_CONTEXT]
    calls = [{"name": step.name, "input_parameters": step.args, "output": step.output} for step in steps]
    row_id = row[columns.row_id].strip()
    manual_rr = (row.get(columns.manual_rr) or "").strip()
    label = {"label": "TRUE" if manual_rr.lower() == "pass" else "FALSE"} if manual_rr else {}
    return LLMTestCase(
        input=row[columns.question],
        expected_output=row[columns.expected],
        actual_output=row[columns.generated],
        retrieved_context=context,
        tools_called=ToolCall.from_dicts(calls) if calls else None,
        row_no=int(row_id) if row_id.isdigit() else row_id,
        manual_rr=manual_rr,
        issue_category=row.get(columns.issue_category) or "",
        agent_error=describe_agent_error(row, columns, steps),
        **label,
    )


def load_samples(path: Path, preset: Preset, limit: int | None = None) -> list[Sample]:
    """Load an agent export into samples.

    Args:
        path (Path): The CSV file.
        preset (Preset): The preset the run uses. The agent loader reads none of its options.
        limit (int | None, optional): Keep only the first runs, for a quick check. Defaults to None.

    Returns:
        list[Sample]: One sample per numbered run.
    """
    del preset
    columns = Columns()
    return [Sample(row, to_case(row, columns)) for row in read_rows(path, columns)[:limit]]
