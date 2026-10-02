"""Custom metric aggregators.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/evaluator/metrics-aggregator
"""

from gllm_evals.aggregation.metrics_aggregator import MetricsAggregator
from gllm_evals.types import MetricResult


class CompletenessFirstAggregator(MetricsAggregator):
    """Aggregator that passes a row on completeness alone, as the DD team's first check does.

    Groundedness and redundancy are still scored and reported, but they do not change `aggregate_success`.
    """

    def compute_success(self, named_results: dict[str, MetricResult]) -> bool:
        """Return the completeness verdict, or the default AND gate when completeness did not run.

        Args:
            named_results (dict[str, MetricResult]): Metric results keyed by metric name.

        Returns:
            bool: Whether the row passes.
        """
        completeness = named_results.get("completeness")
        if completeness is None:
            return super().compute_success(named_results)
        return bool(completeness.success)
