# Auto Eval Experiment

Base code for auto eval, built on `gllm-evals` from the GL SDK and following its [Evaluation tutorials](https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation). It runs an evaluator on a labeled CSV, writes an automatic relevancy rating (Auto RR, `PASS` or `FAIL`) for every row, and compares it with the manual rating from the DD team. The default setup is the GitBook's standard `GEvalGenerationEvaluator`. Other setups are presets, and `experiments/` holds the calibration study that produced them.

## Setup

1. Install Python 3.12, [uv](https://docs.astral.sh/uv/), and the gcloud CLI. `gllm-evals` is a private GL SDK package, and the GitBook's [Getting Started](https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/getting-started) page explains how to get access.
2. Run `gcloud auth login`, then `make install`.
3. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`, which the default judge uses. A `google/...` judge needs `GOOGLE_API_KEY` instead.

The GL SDK and DeepEval versions are pinned to the ones `model-accuracy-benchmarking` uses (`gllm-evals` 0.1.29, `gllm-inference` 0.6.161, `gllm-core` 0.4.49, `deepeval` 3.9.6), so the results match the study and the platform.

## Dataset format

`data/` holds the two DD-labeled batches of the calibration study: `test_auto_eval_intern.csv` (30 rows) and `Auto RR Results - datatest-tambahan.csv` (40 rows). A new dataset follows the same export format. The column names are the defaults of `auto_eval.dataset.Columns` and can be changed there.

| Column | Content |
| --- | --- |
| `No.` | Row number. Rows where it is empty, such as a summary row, are skipped. |
| `question` | The user question. |
| `expected_response` | The expected answer. |
| `generated_response` | The answer to evaluate. |
| `retrieved_context` | The retrieved chunks. |
| `Relevance Rate` | Optional. The manual rating, `Pass` or `Failed`. Without it, the run reports Auto RR only. |
| `Issue Category` | Optional. Copied to the output. |

## Run the default auto eval

```bash
uv run python -m auto_eval --dataset data/test_auto_eval_intern.csv
uv run python -m auto_eval --dataset data/test_auto_eval_intern.csv --limit 3   # first 3 rows only
uv run python -m auto_eval --list-presets
```

Each run writes two files to `results/`:

- `<dataset>_<preset>_rows.csv` has one line per row: the manual and automatic rating, whether they agree, the failed metrics, and every metric's score and explanation.
- `<dataset>_<preset>_summary.json` has the Auto RR pass rate, the tokens, and the cost. When every row has a manual rating, it also has the confusion counts, TPR, TNR, and agreement.

Auto RR is in line with the manual rating when both give the same verdict on a row. With `Pass` as the positive label, TPR is the share of manual `Pass` rows that Auto RR also passes, and TNR is the share of manual `Failed` rows that Auto RR also fails. The Evals Lifecycle page asks for at least 90% on both, because agreement alone can hide a one-sided evaluator.

## Presets

| Preset | What it does |
| --- | --- |
| `default` | The GitBook standard `GEvalGenerationEvaluator`. Completeness, groundedness, and redundancy must all pass. This is what the platform benchmark writes as `autoeval_rr`. |
| `completeness_first` | The same metrics, but completeness alone decides the verdict, as in the first step of the DD team's review. |
| `dd_calibrated` | `completeness_first` plus the DD team's comparison standard in the completeness prompt. It agreed best with the DD labels in the study. |
| `answer_relevancy` | DeepEval answer relevancy alone. It reads only the question and the generated response. |

## Add a custom setup

A preset is a name plus a function that builds a `gllm-evals` evaluator from the judge. To add one, write the function and add an entry to `PRESETS` in `auto_eval/presets.py`. It is then available as `--preset <name>`. This example uses the lenient groundedness from the GitBook's Modify Metrics page:

```python
def build_lenient_groundedness(judge: BaseLMInvoker) -> BaseEvaluator:
    """Build the standard evaluator with groundedness passing at 0.5."""
    metrics = [GEvalCompletenessMetric(), GEvalRedundancyMetric(), GEvalGroundednessMetric(threshold=0.5)]
    return GEvalGenerationEvaluator(models=judge, metrics=metrics)


Preset("lenient_groundedness", "Groundedness passes at 0.5.", build_lenient_groundedness)
```

| To change | Use | GitBook page | Example in this repository |
| --- | --- | --- | --- |
| How metric verdicts combine into one | A `MetricsAggregator` subclass passed as `metrics_aggregator` | Metrics Aggregator | `aggregators.CompletenessFirstAggregator` |
| A metric's prompt or threshold | Metric constructor arguments such as `additional_info`, `rubric`, `fewshot`, or `threshold` | Modify Metrics | `presets.build_dd_calibrated` |
| Metrics outside GEval | `CompositeEvaluator(metrics=[...])` | Composite Evaluator | `presets.build_answer_relevancy` |
| The judge model | `--judge-model <provider>/<model>` | LM Invoker | `judge.build_judge` |
| The CSV column names | `Columns(...)` | LLMTestCase | `dataset.Columns` |

## Calibration study

`experiments/calibration_study.py` reruns the study behind the presets. Steps 01 to 09 use a 30-row calibration batch labeled by the DD team, and steps 11 to 14 repeat four of those setups, unchanged, on a separate 40-row validation batch. Steps 10 and 15 compare the steps without calling the judge and write the GitBook's two validation conditions: the validation batch alone and both batches together. Every step writes to `results/`, which is not committed. To rerun:

```bash
make experiment                                                    # every step, about $1.60 of judge calls
uv run python -m experiments.calibration_study --steps 11 12 13 14  # the validation steps only
uv run python -m experiments.calibration_study --analysis-only      # steps 10 and 15, no judge calls
```

Results with Gemini 3.1 Flash Lite as the judge:

| Preset | Calibration batch (30 rows) | Validation batch (40 rows) |
| --- | --- | --- |
| `default` | TPR 60.7%, TNR 100%, 19 of 30 agree | TPR 71.8%, TNR 100%, 29 of 40 agree |
| `completeness_first` | TPR 82.1%, TNR 100%, 25 of 30 agree | TPR 82.1%, TNR 100%, 33 of 40 agree |
| `dd_calibrated` | TPR 89.3%, TNR 100%, 27 of 30 agree | TPR 89.7%, TNR 100%, 36 of 40 agree |
| `answer_relevancy` | TPR 96.4%, TNR 0%, 27 of 30 agree | TPR 100%, TNR 0%, 39 of 40 agree |

- The default fails many answers the DD team passed. It requires groundedness on every row, while the DD team checks the context only when the answer does not clearly match the expected response.
- Both calibrations carry over to the validation batch, which the setups were not tuned on. `dd_calibrated` reaches 89.7% TPR there, against 89.3% on the calibration batch.
- Answer relevancy alone passed all three DD failures, because those answers stay on topic while being incomplete or wrong.
- Most remaining misses are answers the DD team passed even though they leave out details of the expected response. Writing expected responses as the facts the DD team requires, the criteria-based mode of completeness, should close them.
- The two batches hold only three DD failures, so TNR still needs more negative examples, especially unsupported or off-topic answers.

## Layout

```text
data/                      the calibration and validation batches
auto_eval/
  __main__.py              command line entry point
  aggregators.py           CompletenessFirstAggregator
  dataset.py               CSV loading and context cleaning
  judge.py                 judge model and prices
  presets.py               the evaluator presets
  report.py                Auto RR rows, confusion counts, and the summary
  runner.py                runs one preset and writes its outputs
experiments/
  calibration_study.py     the calibration and validation steps
tests/                     unit tests, with fixtures in tests/resources/
```

## Development

```bash
make test
make lint
```
