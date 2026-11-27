from __future__ import annotations

from pathlib import Path

import pytest

from forgeai.ingest import chunk_corpus, load_folder
from forgeai.ingest.chunker import chunk_document
from forgeai.schemas import ChunkingConfig, Document, Page
from forgeai.textutil import contains_quote
from tests.fixtures import build_docs_folder


@pytest.fixture
def corpus(tmp_path: Path):  # type: ignore[no-untyped-def]
    return load_folder(build_docs_folder(tmp_path))


def test_loads_every_supported_type(corpus) -> None:  # type: ignore[no-untyped-def]
    kinds = {d.doc_id: d.kind for d in corpus.documents}
    assert kinds == {
        "berths": "csv",
        "code-of-conduct": "docx",
        "leave-policy": "pdf",
        "notes": "txt",
        "travel-policy": "md",
    }


def test_skips_with_reasons_and_ignores_hidden(corpus) -> None:  # type: ignore[no-untyped-def]
    reasons = {s.filename: s.reason for s in corpus.skipped}
    assert reasons["malware.exe"].startswith("unsupported type")
    assert reasons["empty.txt"] == "no extractable text"
    assert ".hidden.md" not in reasons
    assert "hidden" not in corpus.doc_ids


def test_pdf_keeps_pages_and_text(corpus) -> None:  # type: ignore[no-untyped-def]
    leave = corpus.get("leave-policy")
    assert [p.number for p in leave.pages] == [1, 2]
    assert "24 days of annual leave" in leave.pages[0].text
    assert leave.title == "Leave Policy"


def test_csv_keeps_rows_for_tools(corpus) -> None:  # type: ignore[no-untyped-def]
    berths = corpus.get("berths")
    assert berths.columns == ["berth_id", "port", "max_loa_m", "max_draft_m", "daily_rate_usd"]
    assert berths.rows[2]["berth_id"] == "M4"
    assert berths.pages[0].text.startswith("Row 1: berth_id: B1; port: Kochi")


def test_docx_headings_become_markdown(corpus) -> None:  # type: ignore[no-untyped-def]
    text = corpus.get("code-of-conduct").full_text
    assert "# Code of Conduct" in text
    assert "## Harassment" in text


def test_missing_folder_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_folder(tmp_path / "nope")


def test_duplicate_stems_get_unique_ids(tmp_path: Path) -> None:
    folder = tmp_path / "d"
    (folder / "a").mkdir(parents=True)
    (folder / "b").mkdir()
    (folder / "a" / "policy.md").write_text("Alpha text here.", encoding="utf-8")
    (folder / "b" / "policy.md").write_text("Beta text here.", encoding="utf-8")
    assert load_folder(folder).doc_ids == ["policy", "policy-2"]


# ---------- Chunking ------------------------------------------------------------------


def test_chunks_carry_page_and_heading(corpus) -> None:  # type: ignore[no-untyped-def]
    chunks = chunk_corpus(corpus)
    sick = next(c for c in chunks if "sick leave" in c.text.lower())
    assert sick.doc_id == "leave-policy"
    assert sick.page == 2
    assert sick.heading == "3. Sick leave"
    assert sick.citation == "leave-policy p.2 — 3. Sick leave"


def test_chunks_never_straddle_headings(corpus) -> None:  # type: ignore[no-untyped-def]
    chunks = [c for c in chunk_corpus(corpus) if c.doc_id == "travel-policy"]
    assert [c.heading for c in chunks] == ["1. Booking", "2. Daily allowance", "3. Claims"]


def test_chunk_text_is_verbatim_quotable(corpus) -> None:  # type: ignore[no-untyped-def]
    for chunk in chunk_corpus(corpus):
        doc = corpus.get(chunk.doc_id)
        assert contains_quote(doc.full_text, chunk.text), chunk.chunk_id


def test_csv_rows_group_and_cite_row_numbers(corpus) -> None:  # type: ignore[no-untyped-def]
    berths = corpus.get("berths")
    one = chunk_document(berths, ChunkingConfig(csv_rows_per_chunk=1))
    two = chunk_document(berths, ChunkingConfig(csv_rows_per_chunk=2))
    assert [c.chunk_id for c in one] == ["berths:r1", "berths:r2", "berths:r3"]
    assert len(two) == 2
    assert two[0].text.startswith("Row 1: berth_id: B1")


def test_long_text_respects_size_and_overlap() -> None:
    sentences = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(200))
    doc = Document(
        doc_id="long",
        filename="long.txt",
        kind="txt",
        title="Long",
        pages=[Page(number=1, text=sentences)],
    )
    config = ChunkingConfig(chunk_words=60, overlap_words=10)
    chunks = chunk_document(doc, config)
    assert len(chunks) > 10
    assert all(len(c.text.split()) <= 60 + 10 for c in chunks)
    first_tail = chunks[0].text.split()[-10:]
    assert chunks[1].text.split()[:10] == first_tail


def test_overlap_must_be_smaller_than_chunk() -> None:
    doc = Document(
        doc_id="x", filename="x.txt", kind="txt", title="X", pages=[Page(number=1, text="hi")]
    )
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(doc, ChunkingConfig(chunk_words=50, overlap_words=50))


def test_heading_at_page_foot_stays_with_its_text(tmp_path: Path) -> None:
    """A real PDF put '8. Applying for leave' at the bottom of page 1 and its text on
    page 2; the chunker used to emit a 4-word heading-only chunk."""
    doc = Document(
        doc_id="leave",
        filename="leave.pdf",
        kind="pdf",
        title="Leave",
        pages=[
            Page(
                number=1,
                text="7. Public holidays\nThe company observes 12 paid public holidays each "
                "year, published every December for each office.\n8. Applying for leave",
            ),
            Page(
                number=2,
                text="Annual leave must be applied for through the HR portal at least 10 "
                "working days in advance.\n9. Leave without pay\nUp to 30 days may be requested.",
            ),
        ],
    )
    chunks = chunk_document(doc, ChunkingConfig(chunk_words=120, overlap_words=10))
    applying = next(c for c in chunks if c.heading == "8. Applying for leave")
    assert applying.text.startswith("8. Applying for leave Annual leave must be applied")
    assert applying.page == 1
    assert all(len(c.text.split()) >= 8 for c in chunks)
    assert contains_quote(doc.full_text, applying.text)
