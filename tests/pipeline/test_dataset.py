"""Tests for the pipeline dataset loader.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

from pathlib import Path

from auto_eval.pipeline.dataset import clean_context, load_samples
from auto_eval.pipeline.presets import PRESETS, build_default
from auto_eval.shared.project import Preset

RESOURCES = Path(__file__).parents[1] / "resources" / "pipeline"
DEFAULT = PRESETS["default"]


def test_load_samples_skips_rows_without_number():
    """
    Condition:
    The CSV ends with a summary row whose `No.` is empty.

    Expected:
    Only the two numbered rows load, labeled `TRUE` for `Pass` and `FALSE` for `Failed`, each with its raw row.
    """
    samples = load_samples(RESOURCES / "dataset.csv", DEFAULT)

    assert [sample.case.row_no for sample in samples] == [1, 2]
    assert [sample.case.label for sample in samples] == ["TRUE", "FALSE"]
    assert samples[1].case.issue_category == "Prompt Issue"
    assert samples[0].row["No."] == "1"


def test_load_samples_with_limit():
    """
    Condition:
    `limit` is 1.

    Expected:
    Only the first row is loaded.
    """
    samples = load_samples(RESOURCES / "dataset.csv", DEFAULT, limit=1)

    assert [sample.case.row_no for sample in samples] == [1]


def test_load_samples_without_manual_rating():
    """
    Condition:
    The CSV has no `Relevance Rate` column.

    Expected:
    The case has no `label`, so the run reports Auto RR without a comparison.
    """
    case = load_samples(RESOURCES / "dataset_unlabeled.csv", DEFAULT)[0].case

    assert getattr(case, "label", None) is None
    assert case.manual_rr == ""


def test_load_samples_with_clean_context():
    """
    Condition:
    The preset cleans the context.

    Expected:
    The context becomes one string per chunk, without source lines or image placeholders.
    """
    preset = Preset(
        "default_clean_context",
        "The default evaluator on cleaned, chunked contexts.",
        build_default,
        clean_context=True,
    )

    case = load_samples(RESOURCES / "dataset.csv", preset)[0].case

    assert case.retrieved_context == [
        "Sejarah Indonesia Proklamasi kemerdekaan Indonesia dibacakan oleh Soekarno dan Mohammad Hatta pada "
        "17 Agustus 1945 di Jakarta.",
        "Sejarah Indonesia Teks proklamasi diketik oleh Sayuti Melik.",
    ]


def test_clean_context_with_empty_context():
    """
    Condition:
    The context is empty.

    Expected:
    No chunks are returned, so the loader keeps the empty string.
    """
    assert clean_context("") == []
