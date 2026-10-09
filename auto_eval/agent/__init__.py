"""The agent project: tool-calling agents exported from GL AIP or GLChat."""

from auto_eval.agent.dataset import load_samples
from auto_eval.agent.presets import PRESETS
from auto_eval.agent.rows import to_record
from auto_eval.shared.project import Project

AGENT = Project(
    name="agent",
    description="A tool-calling agent: the question, every tool call with its output, and the answer.",
    presets=PRESETS,
    load=load_samples,
    to_record=to_record,
)
