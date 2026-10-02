"""Auto eval base code: run a gllm-evals evaluator on a labeled CSV and compare Auto RR with the manual rating."""

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
