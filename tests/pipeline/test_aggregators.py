"""Tests for the custom metric aggregators.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

import pytest
from gllm_evals.types import MetricResult

from auto_eval.pipeline.aggregators import CompletenessFirstAggregator


def make_result(success: bool) -> MetricResult:
    """Build a GEval metric result that passed or failed."""
    return MetricResult(
        score=1.0 if success else 0.5,
        success=success,
        higher_is_better=True,
        explanation="",
        rubric_score=3 if success else 2,
        threshold=1.0,
        strict_mode=False,
    )


@pytest.mark.parametrize(
    ("completeness", "groundedness", "expected"),
    [(True, False, True), (False, True, False), (True, True, True)],
)
def test_compute_success_with_completeness(completeness, groundedness, expected):
    """
    Condition:
    Completeness and groundedness pass or fail independently.

    Expected:
    The verdict follows completeness only.
    """
    results = {"completeness": make_result(completeness), "groundedness": make_result(groundedness)}

    assert CompletenessFirstAggregator().compute_success(results) is expected


def test_compute_success_without_completeness():
    """
    Condition:
    Completeness did not run, and groundedness failed.

    Expected:
    The default AND gate decides, so the row fails.
    """
    results = {"groundedness": make_result(False), "redundancy": make_result(True)}

    assert CompletenessFirstAggregator().compute_success(results) is False
