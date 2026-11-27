"""The document corpus: what ForgeAI read from the folder, and how it was chunked."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocKind = Literal["pdf", "docx", "csv", "md", "txt"]


class Page(BaseModel):
    number: int = Field(ge=1)
    text: str


class Document(BaseModel):
    doc_id: str
    filename: str
    kind: DocKind
    title: str
    pages: list[Page]
    # Tabular sources keep their structure for the csv_lookup tool.
    columns: list[str] | None = None
    rows: list[dict[str, str]] | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def words(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)


class SkippedFile(BaseModel):
    filename: str
    reason: str


class Corpus(BaseModel):
    root: str
    documents: list[Document]
    skipped: list[SkippedFile] = Field(default_factory=list)

    def get(self, doc_id: str) -> Document | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)

    @property
    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]


class ChunkingConfig(BaseModel):
    """Chunking parameters. Part of every agent definition, so the optimizer can patch them."""

    chunk_words: int = Field(default=180, ge=40, le=1200)
    overlap_words: int = Field(default=30, ge=0, le=400)
    csv_rows_per_chunk: int = Field(default=1, ge=1, le=50)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    heading: str | None = None
    text: str

    @property
    def citation(self) -> str:
        where = f"{self.doc_id} p.{self.page}"
        return f"{where} — {self.heading}" if self.heading else where
