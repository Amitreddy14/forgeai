from __future__ import annotations

import pytest

from forgeai.textutil import clean, contains_quote, extract_numbers, normalize

DOC = (
    "2. Daily allowance\n"
    "The daily meal allowance for domestic travel is INR 1,500. For international travel "
    "the allowance is USD 60 per day. Receipts are required for any single expense above "
    "INR 2,000."
)


def test_normalize_is_form_insensitive() -> None:
    assert normalize("The “Daily”  Allowance — INR 1,500!") == "the daily allowance inr 1500"


def test_normalize_joins_pdf_hyphenation() -> None:
    assert normalize("accom-\nmodation") == "accommodation"


def test_clean_keeps_case_and_punctuation() -> None:
    assert clean("  It\u2019s   INR\u00a01,500 ") == "It's INR 1,500"


@pytest.mark.parametrize(
    "quote",
    [
        "the daily meal allowance for domestic travel is INR 1,500",
        "The daily meal allowance for domestic travel is INR 1500.",
        "“the allowance is USD 60 per day”",
        "For international travel ... USD 60 per day",
        "daily meal allowance … above INR 2,000",
    ],
)
def test_real_quotes_are_found(quote: str) -> None:
    assert contains_quote(DOC, quote)


@pytest.mark.parametrize(
    ("quote", "why"),
    [
        ("the daily meal allowance for domestic travel is INR 2,500", "changed number"),
        ("the allowance is USD 80 per day", "changed number"),
        ("Receipts are optional for any single expense", "changed word"),
        ("USD 60 per day ... the daily meal allowance", "fragments out of order"),
        ("allowance", "too short to verify"),
        ("is INR ... per day", "fragments too short"),
        ("", "empty"),
        ("1,5", "partial number"),
    ],
)
def test_fabricated_quotes_are_rejected(quote: str, why: str) -> None:
    assert not contains_quote(DOC, quote), why


def test_word_boundaries_matter() -> None:
    assert not contains_quote("the allowance is USD 600 per day", "the allowance is USD 60")


def test_extract_numbers_canonicalises() -> None:
    assert extract_numbers("INR 1,500.00 or USD 60 and 7.50%, minus -3") == {
        "1500",
        "60",
        "7.5",
        "-3",
    }
