"""Tests for the pipeline evaluator presets.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

import pytest

from auto_eval.pipeline.aggregators import CompletenessFirstAggregator
from auto_eval.pipeline.presets import DD_COMPLETENESS_GUIDANCE, PRESETS


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_build_evaluator_with_gate_metric(name):
    """
    Condition:
    Each preset builds its evaluator.

    Expected:
    The gate metric used for the threshold sweep is one of the evaluator's metrics.
    """
    preset = PRESETS[name]
    metric_names = [metric.name for metric in preset.build_evaluator(None).metrics]

    assert preset.gate_metric is None or preset.gate_metric in metric_names


def test_build_evaluator_with_dd_calibrated():
    """
    Condition:
    The `dd_calibrated` preset builds its evaluator.

    Expected:
    Completeness carries the DD guidance, and completeness alone decides the verdict.
    """
    evaluator = PRESETS["dd_calibrated"].build_evaluator(None)
    completeness = next(metric for metric in evaluator.metrics if metric.name == "completeness")

    assert completeness._additional_info == DD_COMPLETENESS_GUIDANCE
    assert isinstance(evaluator.metrics_aggregator, CompletenessFirstAggregator)


def test_build_evaluator_with_default():
    """
    Condition:
    The `default` preset builds its evaluator.

    Expected:
    It is the GitBook standard evaluator: completeness, groundedness, and redundancy behind the AND gate.
    """
    evaluator = PRESETS["default"].build_evaluator(None)

    assert sorted(metric.name for metric in evaluator.metrics) == ["completeness", "groundedness", "redundancy"]
    assert not isinstance(evaluator.metrics_aggregator, CompletenessFirstAggregator)
