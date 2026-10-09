"""Evaluator presets for pipelines: the GitBook default and the setups calibrated against the DD labels.

A preset pairs a name with a function that builds a gllm-evals evaluator from the judge. A custom setup is one more
entry in `PRESETS`.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/evaluator/gevalgenerationevaluator
    [2] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/evaluator/metrics-aggregator
    [3] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/metric/modify-metrics
    [4] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/evaluator/composite-evaluator
"""

from gllm_evals.evaluator.composite_evaluator import CompositeEvaluator
from gllm_evals.evaluator.evaluator import BaseEvaluator
from gllm_evals.evaluator.geval_generation_evaluator import GEvalGenerationEvaluator
from gllm_evals.metrics.generation import GEvalCompletenessMetric, GEvalGroundednessMetric, GEvalRedundancyMetric
from gllm_evals.metrics.generation.deepeval_answer_relevancy import DeepEvalAnswerRelevancyMetric
from gllm_inference.lm_invoker.lm_invoker import BaseLMInvoker

from auto_eval.pipeline.aggregators import CompletenessFirstAggregator
from auto_eval.shared.project import Preset

DD_COMPLETENESS_GUIDANCE = (
    "Scoring standard: a human reviewer passes the Generated Response when it gives the same answer as the Expected "
    "Response, even if the wording differs. Take minimum key facts only from what the Question directly asks. Impacts, "
    "consequences, background, and elaborations that the Expected Response adds beyond that direct answer are not key "
    "facts. When the Question asks for a list or for every item of a group, each requested item is a key fact."
)


def build_default(judge: BaseLMInvoker) -> BaseEvaluator:
    """Build the GitBook's standard generation evaluator: completeness, groundedness, and redundancy must all pass.

    Args:
        judge (BaseLMInvoker): The judge model.

    Returns:
        BaseEvaluator: The evaluator.
    """
    return GEvalGenerationEvaluator(models=judge)


def build_completeness_first(judge: BaseLMInvoker) -> BaseEvaluator:
    """Build the standard evaluator with `CompletenessFirstAggregator`.

    Args:
        judge (BaseLMInvoker): The judge model.

    Returns:
        BaseEvaluator: The evaluator.
    """
    return GEvalGenerationEvaluator(models=judge, metrics_aggregator=CompletenessFirstAggregator())


def build_dd_calibrated(judge: BaseLMInvoker) -> BaseEvaluator:
    """Build the completeness-first evaluator with `DD_COMPLETENESS_GUIDANCE` in the completeness prompt.

    Args:
        judge (BaseLMInvoker): The judge model.

    Returns:
        BaseEvaluator: The evaluator.
    """
    metrics = [
        GEvalCompletenessMetric(additional_info=DD_COMPLETENESS_GUIDANCE),
        GEvalRedundancyMetric(),
        GEvalGroundednessMetric(),
    ]
    return GEvalGenerationEvaluator(models=judge, metrics=metrics, metrics_aggregator=CompletenessFirstAggregator())


def build_answer_relevancy(judge: BaseLMInvoker) -> BaseEvaluator:
    """Build a `CompositeEvaluator` around DeepEval answer relevancy.

    Args:
        judge (BaseLMInvoker): The judge model.

    Returns:
        BaseEvaluator: The evaluator.
    """
    return CompositeEvaluator(metrics=[DeepEvalAnswerRelevancyMetric(models=judge)], name="answer_relevancy")


PRESETS = {
    preset.name: preset
    for preset in (
        Preset(
            "default",
            "GitBook standard GEvalGenerationEvaluator. Completeness, groundedness, and redundancy must all pass.",
            build_default,
        ),
        Preset(
            "completeness_first",
            "The default metrics, but completeness alone decides the verdict, as in DD's first step.",
            build_completeness_first,
            gate_metric="completeness",
        ),
        Preset(
            "dd_calibrated",
            "completeness_first plus DD's comparison standard in the completeness prompt.",
            build_dd_calibrated,
            gate_metric="completeness",
        ),
        Preset(
            "answer_relevancy",
            "DeepEval answer relevancy alone. It reads only the question and the generated response.",
            build_answer_relevancy,
            gate_metric="deepeval_answer_relevancy",
        ),
    )
}
