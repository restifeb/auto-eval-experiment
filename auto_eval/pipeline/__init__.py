"""The pipeline project: RAG and QnA pipelines exported in the DD team's CSV format."""

from auto_eval.pipeline.dataset import load_samples
from auto_eval.pipeline.presets import PRESETS
from auto_eval.pipeline.rows import to_record
from auto_eval.shared.project import Project

PIPELINE = Project(
    name="pipeline",
    description="A RAG or QnA pipeline: the question, the retrieved context, and the answer.",
    presets=PRESETS,
    load=load_samples,
    to_record=to_record,
)
