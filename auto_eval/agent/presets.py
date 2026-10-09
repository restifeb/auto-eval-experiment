"""Evaluator presets for agents.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/evaluator/gevalgenerationevaluator
    [2] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/tutorials/agent_qna_evaluations
"""

from gllm_evals.evaluator.evaluator import BaseEvaluator
from gllm_evals.evaluator.geval_generation_evaluator import GEvalGenerationEvaluator
from gllm_evals.metrics.generation import (
    GEvalCompletenessMetric,
    GEvalGroundednessMetric,
    GEvalLanguageConsistencyMetric,
    GEvalRedundancyMetric,
    GEvalRefusalAlignmentMetric,
)
from gllm_inference.lm_invoker.lm_invoker import BaseLMInvoker

from auto_eval.shared.project import Preset


def build_default(judge: BaseLMInvoker) -> BaseEvaluator:
    """Build the generation evaluator with the five GEval metrics of the agent evaluation template.

    Completeness, groundedness, redundancy, language consistency, and refusal alignment must all pass, each at its
    default threshold. Groundedness checks the answer against the tool outputs.

    Args:
        judge (BaseLMInvoker): The judge model.

    Returns:
        BaseEvaluator: The evaluator.
    """
    metrics = [
        GEvalCompletenessMetric(),
        GEvalGroundednessMetric(),
        GEvalRedundancyMetric(),
        GEvalLanguageConsistencyMetric(),
        GEvalRefusalAlignmentMetric(),
    ]
    return GEvalGenerationEvaluator(models=judge, metrics=metrics)


PRESETS = {
    preset.name: preset
    for preset in (
        Preset(
            "default",
            "The five GEval metrics of the agent evaluation template. All of them must pass.",
            build_default,
        ),
    )
}
