# Auto Eval Experiment

Base code for auto eval, built on `gllm-evals` from the GL SDK and following its [Evaluation tutorials](https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation). It runs an evaluator on an exported CSV, writes an automatic relevancy rating (Auto RR, `PASS` or `FAIL`) for every row, and compares it with the manual rating. One repository covers both kinds of project, chosen with `--project`:

| Project | Target system | Dataset | Output rows |
| --- | --- | --- | --- |
| `pipeline` | A RAG or QnA pipeline, such as the Djarum website | The DD team's export | Auto RR next to the DD rating, with every metric's score and explanation |
| `agent` | A tool-calling agent run through GL AIP or GLChat | The GL AIP and GLChat export | The export itself, with its evaluation template filled |

## Setup

1. Install Python 3.12, [uv](https://docs.astral.sh/uv/), and the gcloud CLI. `gllm-evals` is a private GL SDK package, and the GitBook's [Getting Started](https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/evaluation/getting-started) page explains how to get access.
2. Run `gcloud auth login`, then `make install`.
3. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`, which the default judge uses. A `google/...` judge needs `GOOGLE_API_KEY` instead.

The GL SDK and DeepEval versions are pinned to the ones `model-accuracy-benchmarking` uses (`gllm-evals` 0.1.29, `gllm-inference` 0.6.161, `gllm-core` 0.4.49, `deepeval` 3.9.6), so the results match the study and the platform.

## Run

```bash
uv run auto-eval --project pipeline --dataset data/pipeline/test_auto_eval_intern.csv --preset dd_calibrated
uv run auto-eval --project agent --dataset data/agent/test_case_agent_evaluation.csv
uv run auto-eval --project agent --list-presets
uv run auto-eval --project pipeline --dataset data/pipeline/test_auto_eval_intern.csv --limit 3   # first 3 rows only
make run PROJECT=agent                                                                            # the agent export in data/agent/
```

`python -m auto_eval` takes the same options, and `--project` defaults to `pipeline`. Each run writes to its own folder, `results/<project>/<dataset>_<preset>/`:

- `rows.csv` has one line per row, in the project's output format below.
- `summary.json` has the Auto RR pass rate, the number of failed judge calls, the tokens, and the cost. When every row has a manual rating, it also has the confusion counts, TPR, TNR, and agreement.

Auto RR is in line with the manual rating when both give the same verdict on a row. With `Pass` as the positive label, TPR is the share of manual `Pass` rows that Auto RR also passes, and TNR is the share of manual `Failed` rows that Auto RR also fails. The Evals Lifecycle page asks for at least 90% on both, because agreement alone can hide a one-sided evaluator.

## Pipeline project

`data/pipeline/` holds the two DD-labeled batches of the calibration study. A new dataset follows the same export format, and the column names are the defaults of `auto_eval.pipeline.dataset.Columns`.

| Column | Content |
| --- | --- |
| `No.` | Row number. Rows where it is empty, such as a summary row, are skipped. |
| `question` | The user question. |
| `expected_response` | The expected answer. |
| `generated_response` | The answer to evaluate. |
| `retrieved_context` | The retrieved chunks. |
| `Relevance Rate` | Optional. The manual rating, `Pass` or `Failed`. Without it, the run reports Auto RR only. |
| `Issue Category` | Optional. Copied to the output. |

Each output row has the manual and automatic rating, whether they agree, the failed metrics, and every metric's score and explanation.

| Preset | What it does |
| --- | --- |
| `default` | The GitBook standard `GEvalGenerationEvaluator`. Completeness, groundedness, and redundancy must all pass. This is what the platform benchmark writes as `autoeval_rr`. |
| `completeness_first` | The same metrics, but completeness alone decides the verdict, as in the first step of the DD team's review. |
| `dd_calibrated` | `completeness_first` plus the DD team's comparison standard in the completeness prompt. It agreed best with the DD labels in the study. |
| `answer_relevancy` | DeepEval answer relevancy alone. It reads only the question and the generated response. |

## Agent project

Put the GL AIP or GLChat export in `data/agent/`. Exports carry user emails and contract data, and this repository is public, so `data/agent/*.csv` is ignored by git and every export stays local. The column names are the defaults of `auto_eval.agent.dataset.Columns`.

| Column | Content |
| --- | --- |
| `Index` | Run number. Rows where it is empty are skipped. |
| `Question` | The user question. |
| `Expected Answer` | The expected answer. |
| `Answer` | The agent's final answer. |
| `Manual_Eval` | Optional. The manual rating, `Pass` or `Fail`. |
| `source_tool_<n>` | The n-th tool step: the tool name, its arguments, its output, and its status. |
| `retrieved_context_or_queried_data_<n>` | The n-th tool output, which is the context groundedness checks the answer against. |
| Other columns | Copied to the output unchanged, such as `Response_Time`, `conversation_id`, and `Notes`. |

Every tool step becomes a `ToolCall` in `tools_called`, and the tool outputs, without the export's `**Output:**` heading and code fence, become `retrieved_context`. A run that called no tool gets the context "The agent called no tool, so no context was retrieved.", because gllm-evals skips groundedness on a case without context instead of failing an answer written from memory.

The output is the export with its evaluation columns filled:

| Column | Filled with |
| --- | --- |
| `geval_<metric>`, `geval_<metric>_explanation` | The score and the judge's explanation of completeness, groundedness, redundancy, language consistency, and refusal alignment. A metric that did not run is marked `not evaluated`. |
| `auto_rr` | `PASS` when every metric passes. |
| `possible_issues` | The metrics that failed, empty on a pass, as in the platform's generation results. |
| `evaluation_error` | The metrics whose judge call failed, with the error. They are not counted as possible issues. |
| `agent_error` | The error the export recorded, or else an empty answer and every tool step that did not end as `finished`. |

The only preset, `default`, runs the template's five GEval metrics at their default thresholds, and all of them must pass. Two things are not covered yet. A run that needs a chart is judged on its text, because `artifacts` are not read. And tool use is not scored, because `DeepEvalToolCorrectnessMetric` needs the expected tool calls, which the export does not have.

On a local sample of 10 runs of a contract QnA agent, all rated `Pass`, Auto RR with Gemini 3.1 Flash Lite as the judge passed 8: TPR 80%, with TNR not measurable. Both failures are groundedness, and both answers are grounded, since every figure, contract number, and date in them appears in the tool outputs. On run 8 the judge compared the project name with the question instead of the context, and on run 9 it called contract details unsupported although the tool output lists them. Groundedness needs calibrating, and the sample needs runs rated `Fail`, before agent Auto RR is used.

## Add a custom setup

A preset is a name plus a function that builds a `gllm-evals` evaluator from the judge. To add one, write the function and add an entry to `PRESETS` in `auto_eval/pipeline/presets.py` or `auto_eval/agent/presets.py`. It is then available as `--preset <name>` in that project. This example uses the lenient groundedness from the GitBook's Modify Metrics page:

```python
def build_lenient_groundedness(judge: BaseLMInvoker) -> BaseEvaluator:
    """Build the standard evaluator with groundedness passing at 0.5."""
    metrics = [GEvalCompletenessMetric(), GEvalRedundancyMetric(), GEvalGroundednessMetric(threshold=0.5)]
    return GEvalGenerationEvaluator(models=judge, metrics=metrics)


Preset("lenient_groundedness", "Groundedness passes at 0.5.", build_lenient_groundedness)
```

| To change | Use | GitBook page | Example in this repository |
| --- | --- | --- | --- |
| How metric verdicts combine into one | A `MetricsAggregator` subclass passed as `metrics_aggregator` | Metrics Aggregator | `pipeline.aggregators.CompletenessFirstAggregator` |
| A metric's prompt or threshold | Metric constructor arguments such as `additional_info`, `rubric`, `fewshot`, or `threshold` | Modify Metrics | `pipeline.presets.build_dd_calibrated` |
| Metrics outside GEval | `CompositeEvaluator(metrics=[...])` | Composite Evaluator | `pipeline.presets.build_answer_relevancy` |
| The judge model | `--judge-model <provider>/<model>` | LM Invoker | `shared.judge.build_judge` |
| The CSV column names | `Columns(...)` | LLMTestCase | `pipeline.dataset.Columns`, `agent.dataset.Columns` |

A new kind of target system is a new package next to `pipeline/` and `agent/`, with `load_samples` in `dataset.py`, `PRESETS` in `presets.py`, and `to_record` in `rows.py`, put together as a `Project` in its `__init__.py` and added to `PROJECTS` in `cli.py`.

## Pipeline calibration study

`experiments/pipeline_calibration.py` reruns the study behind the pipeline presets. Steps 01 to 09 use a 30-row calibration batch labeled by the DD team, and steps 11 to 14 repeat four of those setups, unchanged, on a separate 40-row validation batch. Steps 10 and 15 compare the steps without calling the judge and write the GitBook's two validation conditions: the validation batch alone and both batches together. Every step writes to its own folder in `results/pipeline_calibration/`, which is not committed. To rerun:

```bash
make experiment                                                          # every step, about $1.60 of judge calls
uv run python -m experiments.pipeline_calibration --steps 11 12 13 14     # the validation steps only
uv run python -m experiments.pipeline_calibration --analysis-only        # steps 10 and 15, no judge calls
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
auto_eval/
  cli.py                   the auto-eval command: --project, --dataset, --preset
  shared/                  what both projects use
    project.py             Preset, Sample, and Project, which every project fills in
    judge.py               judge model and prices
    runner.py              runs one preset and writes the run's rows and summary
    report.py              verdicts, judge errors, confusion counts, and the summary
  pipeline/                RAG and QnA pipelines
    dataset.py             DD export loading and context cleaning
    presets.py             the pipeline presets
    aggregators.py         CompletenessFirstAggregator
    rows.py                Auto RR rows
  agent/                   tool-calling agents
    dataset.py             GL AIP and GLChat export loading: tool steps and tool outputs
    presets.py             the agent presets
    rows.py                the filled evaluation template
experiments/
  pipeline_calibration.py  the pipeline calibration and validation steps
data/                      one folder per project; see data/README.md
results/                   one folder per project and per study, not committed
tests/                     unit tests in the package's layout, with fixtures in tests/resources/
```

## Development

```bash
make test
make lint
make format
```
