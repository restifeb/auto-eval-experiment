"""Tests for the dataset loader.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    NONE
"""

from pathlib import Path

from auto_eval.dataset import clean_context, load_test_cases

RESOURCES = Path(__file__).parent / "resources"


def test_load_test_cases_skips_rows_without_number():
    """
    Condition:
    The CSV ends with a summary row whose `No.` is empty.

    Expected:
    Only the two numbered rows become test cases, labeled `TRUE` for `Pass` and `FALSE` for `Failed`.
    """
    cases = load_test_cases(RESOURCES / "dataset.csv")

    assert [case.row_no for case in cases] == [1, 2]
    assert [case.label for case in cases] == ["TRUE", "FALSE"]
    assert cases[1].issue_category == "Prompt Issue"


def test_load_test_cases_with_limit():
    """
    Condition:
    `limit` is 1.

    Expected:
    Only the first row is loaded.
    """
    cases = load_test_cases(RESOURCES / "dataset.csv", limit=1)

    assert [case.row_no for case in cases] == [1]


def test_load_test_cases_without_manual_rating():
    """
    Condition:
    The CSV has no `Relevance Rate` column.

    Expected:
    The case has no `label`, so the run reports Auto RR without a comparison.
    """
    cases = load_test_cases(RESOURCES / "dataset_unlabeled.csv")

    assert getattr(cases[0], "label", None) is None
    assert cases[0].manual_rr == ""


def test_load_test_cases_with_clean_context():
    """
    Condition:
    `clean` is True.

    Expected:
    The context becomes one string per chunk, without source lines or image placeholders.
    """
    cases = load_test_cases(RESOURCES / "dataset.csv", clean=True)

    assert cases[0].retrieved_context == [
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
