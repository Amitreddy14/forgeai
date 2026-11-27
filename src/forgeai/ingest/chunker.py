"""Split documents into retrievable chunks that remember where they came from.

Prose is packed paragraph by paragraph up to `chunk_words`, with `overlap_words` carried
into the next chunk. Long paragraphs are split on sentence boundaries. Each chunk keeps
its page and the nearest heading above it, so citations read like "leave-policy p.2 —
4. Annual leave". CSV rows are grouped `csv_rows_per_chunk` at a time.
"""

from __future__ import annotations

import re

from forgeai.schemas.corpus import Chunk, ChunkingConfig, Corpus, Document

MIN_CHUNK_WORDS = 8
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_HEADING = re.compile(
    r"^(#{1,6}\s+\S.*"  # markdown heading
    r"|(?:section|clause|part|article|chapter)\s+\d+[\w.]*\b.*"  # "Section 4 ..."
    r"|\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{1,70})$",  # "4.2 Annual leave"
    re.IGNORECASE,
)


def _is_heading(line: str) -> bool:
    return len(line) <= 90 and bool(_HEADING.match(line.strip()))


def _paragraphs(text: str) -> list[tuple[str | None, str]]:
    """Return (heading, paragraph) pairs, tracking the most recent heading."""
    out: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            out.append((heading, " ".join(buffer)))
            buffer.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if _is_heading(line):
            flush()
            heading = line.lstrip("#").strip()
            # Keep the heading line in the text so every chunk stays a verbatim,
            # contiguous span of its document (quotes can then cross the heading).
            out.append((heading, line))
            continue
        buffer.append(line)
    flush()
    return out


def _split_long(paragraph: str, limit: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= limit:
        return [paragraph]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in _SENTENCE.split(paragraph):
        s_words = sentence.split()
        if current and len(current) + len(s_words) > limit:
            pieces.append(" ".join(current))
            current = []
        if len(s_words) > limit:  # one enormous "sentence": hard wrap
            for i in range(0, len(s_words), limit):
                pieces.append(" ".join(s_words[i : i + limit]))
            continue
        current.extend(s_words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _chunk_prose(doc: Document, config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        units: list[tuple[str | None, str]] = []
        for heading, para in _paragraphs(page.text):
            units.extend((heading, piece) for piece in _split_long(para, config.chunk_words))

        current: list[str] = []
        current_heading: str | None = None
        for heading, unit in units:
            unit_words = unit.split()
            starts_new_section = heading != current_heading and bool(current)
            too_big = len(current) + len(unit_words) > config.chunk_words
            if starts_new_section and not too_big and len(current) < MIN_CHUNK_WORDS:
                # A stray title line before the first heading: fold it into the next section
                # rather than emitting a chunk too small to retrieve or quote.
                current_heading = heading
                starts_new_section = False
            if current and (too_big or starts_new_section):
                chunks.append(_make(doc, page.number, len(chunks), current_heading, current))
                carry = current[-config.overlap_words :] if config.overlap_words else []
                # Never carry text across a heading boundary: it would blur citations.
                current = [] if starts_new_section else list(carry)
            if not current:
                current_heading = heading
            current.extend(unit_words)
        if current:
            chunks.append(_make(doc, page.number, len(chunks), current_heading, current))
    return chunks


def _chunk_rows(doc: Document, config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = config.csv_rows_per_chunk
    for start in range(0, len(doc.pages), step):
        group = doc.pages[start : start + step]
        text = "\n".join(p.text for p in group)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:r{group[0].number}",
                doc_id=doc.doc_id,
                page=group[0].number,
                heading=None,
                text=text,
            )
        )
    return chunks


def _make(doc: Document, page: int, index: int, heading: str | None, words: list[str]) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}:p{page}:c{index}",
        doc_id=doc.doc_id,
        page=page,
        heading=heading,
        text=" ".join(words),
    )


def chunk_document(doc: Document, config: ChunkingConfig) -> list[Chunk]:
    if config.overlap_words >= config.chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")
    return _chunk_rows(doc, config) if doc.kind == "csv" else _chunk_prose(doc, config)


def chunk_corpus(corpus: Corpus, config: ChunkingConfig | None = None) -> list[Chunk]:
    config = config or ChunkingConfig()
    return [chunk for doc in corpus.documents for chunk in chunk_document(doc, config)]
