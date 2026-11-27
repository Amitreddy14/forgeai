"""Text normalisation and quote verification: the code that checks the model's claims.

`contains_quote` is deliberately forgiving about *form* (case, whitespace, curly quotes,
punctuation, thousands separators) and strict about *content* (every word, in order).
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT_MAP = str.maketrans(
    {
        0x2018: "'",  # left single quote
        0x2019: "'",  # right single quote
        0x201C: '"',  # left double quote
        0x201D: '"',  # right double quote
        0x2013: "-",  # en dash
        0x2014: "-",  # em dash
        0x2212: "-",  # minus sign
        0x00A0: " ",  # no-break space
        0x00AD: None,  # soft hyphen
    }
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_ELLIPSIS = re.compile(r"\.\.\.|…|\[\.\.\.\]")
_SPACES = re.compile(r"\s+")


def clean(text: str) -> str:
    """Readable cleanup for display and chunking: unicode, quotes, whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_MAP)
    return _SPACES.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Comparison form: lowercase alphanumeric words separated by single spaces.

    `1,500` becomes `1500` so a model writing either matches the document.
    Hyphenated line breaks from PDFs (`exam-\\nple`) are joined first.
    """
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_MAP)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = _THOUSANDS.sub("", text.lower())
    return _NON_ALNUM.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def contains_quote(haystack: str, quote: str, *, min_words: int = 3) -> bool:
    """True if `quote` appears in `haystack` (normalised), fragments in order.

    A quote may elide text with `...`; each fragment must then appear, in order.
    Fragments shorter than `min_words` words are rejected as unverifiable, so a model
    cannot "prove" a claim with a quote like "the".
    """
    fragments = [normalize(f) for f in _ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    if any(len(f.split()) < min_words for f in fragments):
        return False
    target = f" {normalize(haystack)} "
    position = 0
    for fragment in fragments:
        found = target.find(f" {fragment} ", position)
        if found == -1:
            return False
        position = found + len(fragment) + 1
    return True


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numbers(text: str) -> set[str]:
    """Numbers in a text, canonicalised (`1,500.00` -> `1500`, `7.50` -> `7.5`)."""
    cleaned = _THOUSANDS.sub("", unicodedata.normalize("NFKC", text))
    out: set[str] = set()
    for match in _NUMBER.findall(cleaned):
        value = float(match)
        out.add(str(int(value)) if value.is_integer() else f"{value:g}")
    return out
