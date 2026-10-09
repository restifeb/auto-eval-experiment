"""Tests for the agent export loader.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

from pathlib import Path

import pytest

from auto_eval.agent.dataset import NO_TOOL_CONTEXT, Columns, clean_tool_output, load_samples, parse_tool_step, to_case
from auto_eval.agent.presets import PRESETS

RESOURCES = Path(__file__).parents[1] / "resources" / "agent"
DEFAULT = PRESETS["default"]
ARGS = '{"endpoint_name": "contract_by_email", "aggregate": {"kind": "count"}}'
STEP = f"""[Type: TOOL]
[Tool: endpoint_config_tool]
[Agent: 5d2c8e17-93ab-4f61-b0d4-7e1a6c3f9b28]
[Step: call_Hn3Ze6Yu0Pa5Wq8Rs2Tk7Lm4]
[Parent Step: tool_start_004]

**Args:**
{ARGS}

**Output:**
{{"result": {{"kind": "count", "result": 2}}}}

**Status:** finished
**Duration:** 27ms"""


def test_parse_tool_step_with_export_block():
    """
    Condition:
    A tool step as the GLChat export renders it.

    Expected:
    The tool name, the JSON arguments, the output, and the status are read from their sections.
    """
    step = parse_tool_step(STEP)

    assert step.name == "endpoint_config_tool"
    assert step.args == {"endpoint_name": "contract_by_email", "aggregate": {"kind": "count"}}
    assert step.output == '{"result": {"kind": "count", "result": 2}}'
    assert step.status == "finished"


def test_parse_tool_step_with_text_arguments():
    """
    Condition:
    The arguments are not a JSON object.

    Expected:
    They are kept as text under `raw`.
    """
    step = parse_tool_step(STEP.replace(ARGS, "contract_by_email"))

    assert step.args == {"raw": "contract_by_email"}


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ('**Output:**\n```\n{\n  "result": 2\n}\n```', '{\n  "result": 2\n}'),
        ("Request timed out after 30 seconds.", "Request timed out after 30 seconds."),
    ],
)
def test_clean_tool_output_with_wrapping(cell, expected):
    """
    Condition:
    A context cell with the export's heading and code fence, or a plain text cell.

    Expected:
    The heading and the fence are dropped, and plain text is kept as it is.
    """
    assert clean_tool_output(cell) == expected


def test_load_samples_with_export():
    """
    Condition:
    An export with a finished run, a run whose only tool step failed before any answer, and an unnumbered total row.

    Expected:
    The two numbered runs load with their tool calls, unwrapped contexts, labels, and agent errors.
    """
    samples = load_samples(RESOURCES / "agent_runs.csv", DEFAULT)

    finished, failed = (sample.case for sample in samples)
    assert [call.name for call in finished.tools_called] == ["get_runtime_context", "endpoint_config_tool"]
    assert finished.retrieved_context[1].startswith('{\n  "result": {\n    "kind": "count"')
    assert (finished.label, finished.agent_error) == ("TRUE", "")
    assert failed.label == "FALSE"
    assert failed.retrieved_context == ["Request timed out after 30 seconds."]
    assert failed.agent_error == "The agent returned no answer. endpoint_config_tool ended with status failed."


def test_load_samples_with_limit():
    """
    Condition:
    `limit` is 1.

    Expected:
    Only the first run is loaded.
    """
    samples = load_samples(RESOURCES / "agent_runs.csv", DEFAULT, limit=1)

    assert [sample.case.row_no for sample in samples] == [1]


def make_row(**cells: str) -> dict[str, str]:
    """Build the export row of a run that called no tool."""
    row = {
        "Index": "4",
        "Question": "Halo, kamu bisa bantu apa saja?",
        "Expected Answer": "Saya bisa membantu mencari informasi kontrak, nilai kontrak, dan task project.",
        "Answer": "Halo! Saya bisa membantu mencari informasi kontrak dan task project.",
        "Manual_Eval": "",
    }
    return {**row, **cells}


def test_to_case_without_tool_calls():
    """
    Condition:
    A run called no tool and has no manual rating.

    Expected:
    Its context is `NO_TOOL_CONTEXT`, so groundedness still runs, and the case has no tool calls and no label.
    """
    case = to_case(make_row(), Columns())

    assert case.retrieved_context == [NO_TOOL_CONTEXT]
    assert case.tools_called is None
    assert getattr(case, "label", None) is None


def test_to_case_with_recorded_agent_error():
    """
    Condition:
    The export recorded an agent error, and the answer is empty.

    Expected:
    The recorded error is kept as it is.
    """
    case = to_case(make_row(Answer="", agent_error="GLChat returned status 504."), Columns())

    assert case.agent_error == "GLChat returned status 504."
