"""Tests for the agent evaluator presets.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

from auto_eval.agent.presets import PRESETS
from auto_eval.agent.rows import TEMPLATE_METRIC_COLUMNS


def test_build_evaluator_with_default():
    """
    Condition:
    The agent `default` preset builds its evaluator.

    Expected:
    It runs the five GEval metrics of the agent evaluation template.
    """
    evaluator = PRESETS["default"].build_evaluator(None)

    assert sorted(metric.name for metric in evaluator.metrics) == sorted(TEMPLATE_METRIC_COLUMNS)
