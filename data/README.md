# Data

Each file is an export of one target system with manual ratings. The file names follow one snake_case style, and the original name of each file is listed below.

## pipeline/

DD team exports of the Djarum website QnA pipeline, in the format `auto_eval.pipeline.dataset.Columns` reads.

| File | Original name | Rows | Manual rating | Used for |
| --- | --- | --- | --- | --- |
| `test_auto_eval_intern.csv` | `test_auto_eval_intern.csv` | 30 | 28 `Pass`, 2 `Failed` (both `Search Issue`) | Calibration batch, steps 01 to 09 of the pipeline calibration study |
| `auto_rr_results_datatest_tambahan.csv` | `Auto RR Results - datatest-tambahan.csv` | 40 | 39 `Pass`, 1 `Failed` (`Prompt Issue`) | Validation batch, steps 11 to 14 |

## agent/

GL AIP and GLChat exports, in the format `auto_eval.agent.dataset.Columns` reads. An export carries the requesting user's email, internal IDs, and contract data, and this repository is public, so `data/agent/*.csv` is ignored by git and every export stays local.

| File | Original name | Rows | Manual rating | Content |
| --- | --- | --- | --- | --- |
| `test_case_agent_evaluation.csv`, local only | `test-case-agent-evaluation.csv` | 10 | 10 `Pass` | Runs of a contract QnA agent through GLChat. The evaluation columns, from `geval_completeness` to `agent_error`, are the template that `--project agent` fills. |

Every row of the sample is rated `Pass`, so it measures TPR only. Runs rated `Fail` are needed to measure TNR.
