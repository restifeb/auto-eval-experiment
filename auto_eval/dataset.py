"""Load a labeled CSV into gllm-evals test cases.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/llmtestcase
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from gllm_evals.types import LLMTestCase

_SOURCE_FOOTER = re.compile(r"\n\s*- source:[^\n]*\n\s*- file_id:[^\n]*")
_IMAGE_PLACEHOLDER = re.compile(r"\[image_[0-9a-f]+\]")


@dataclass(frozen=True)
class Columns:
    """Column names of the dataset CSV. The defaults follow the DD team's export.

    Attributes:
        row_id (str): The row number. Rows where it is empty, such as a summary row, are skipped. Defaults to "No.".
        question (str): The user question. Defaults to "question".
        expected (str): The expected response. Defaults to "expected_response".
        generated (str): The response to evaluate. Defaults to "generated_response".
        context (str): The retrieved chunks. Defaults to "retrieved_context".
        manual_rr (str): The manual rating, `Pass` or `Failed`. Defaults to "Relevance Rate".
        issue_category (str): The manual issue category. Defaults to "Issue Category".
    """

    row_id: str = "No."
    question: str = "question"
    expected: str = "expected_response"
    generated: str = "generated_response"
    context: str = "retrieved_context"
    manual_rr: str = "Relevance Rate"
    issue_category: str = "Issue Category"


def clean_context(context: str) -> list[str]:
    """Split a retrieved context into chunks without source metadata or image placeholders.

    DeepEval prints `retrieval_context` as a Python list, so a single string arrives as one line full of
    escaped newlines. One flat string per chunk keeps the judge prompt readable.

    Args:
        context (str): The raw retrieved context.

    Returns:
        list[str]: One whitespace-collapsed string per retrieved chunk.
    """
    chunks = _SOURCE_FOOTER.split(context)
    return [" ".join(_IMAGE_PLACEHOLDER.sub(" ", chunk).split()) for chunk in chunks if chunk.strip()]


def load_test_cases(
    path: Path, columns: Columns | None = None, clean: bool = False, limit: int | None = None
) -> list[LLMTestCase]:
    """Map each CSV row to a test case the way the platform benchmark maps its rows.

    The manual rating is optional. When it is empty, the case has no `label` and the run reports Auto RR without
    comparing it.

    Args:
        path (Path): The CSV file.
        columns (Columns | None, optional): The column names. Defaults to None, which uses `Columns()`.
        clean (bool, optional): Whether to pass the context through `clean_context`. Defaults to False.
        limit (int | None, optional): Keep only the first rows, for a quick check. Defaults to None.

    Returns:
        list[LLMTestCase]: One case per row, with the manual rating as `label` when it is present.
    """
    columns = columns or Columns()
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if (row.get(columns.row_id) or "").strip()]
    return [_to_case(row, columns, clean) for row in rows[:limit]]


def _to_case(row: dict[str, str], columns: Columns, clean: bool) -> LLMTestCase:
    """Build one test case from a CSV row."""
    row_id = row[columns.row_id].strip()
    context = row.get(columns.context) or ""
    manual_rr = (row.get(columns.manual_rr) or "").strip()
    label = {"label": "TRUE" if manual_rr.lower() == "pass" else "FALSE"} if manual_rr else {}
    return LLMTestCase(
        input=row[columns.question],
        expected_output=row[columns.expected],
        actual_output=row[columns.generated],
        retrieved_context=(clean_context(context) if clean else None) or context,
        row_no=int(row_id) if row_id.isdigit() else row_id,
        manual_rr=manual_rr,
        issue_category=row.get(columns.issue_category) or "",
        **label,
    )
