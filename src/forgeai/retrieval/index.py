"""Hybrid retrieval: BM25 (exact terms, numbers, clause IDs) + dense embeddings (meaning),
fused with reciprocal-rank fusion. Each hit keeps both raw signals so later stages can
explain a retrieval miss and so the agent can refuse when similarity is too low."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from forgeai.retrieval.embed import Embedder
from forgeai.schemas.corpus import Chunk
from forgeai.textutil import normalize

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "can",
        "do",
        "does",
        "how",
        "may",
        "me",
        "my",
        "should",
    ]
)
CANDIDATES = 50


class RetrievalConfig(BaseModel):
    """Retrieval parameters. Part of every agent definition, so the optimizer can patch them."""

    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"
    rrf_k: int = Field(default=60, ge=1, le=200)
    # Hits whose dense cosine similarity is below this are dropped (dense/hybrid modes).
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class Hit(BaseModel):
    chunk: Chunk
    rank: int
    score: float
    bm25: float
    dense: float


def tokenize(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t not in STOPWORDS]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        self.chunks = chunks
        self.embedder = embedder
        self._bm25 = BM25Okapi([tokenize(c.text) or ["_empty_"] for c in chunks])
        self._vectors = embedder.embed_documents([self._embed_text(c) for c in chunks])

    @staticmethod
    def _embed_text(chunk: Chunk) -> str:
        return f"{chunk.heading}\n{chunk.text}" if chunk.heading else chunk.text

    def search(self, query: str, config: RetrievalConfig | None = None) -> list[Hit]:
        config = config or RetrievalConfig()
        n = len(self.chunks)

        terms = tokenize(query)
        bm25 = np.asarray(self._bm25.get_scores(terms), dtype=np.float64) if terms else np.zeros(n)
        dense = self._vectors @ self.embedder.embed_query(query)

        bm25_order = [int(i) for i in np.argsort(-bm25, kind="stable") if bm25[i] > 0]
        dense_order = [int(i) for i in np.argsort(-dense, kind="stable")]

        fused: dict[int, float] = {}
        if config.mode in ("hybrid", "bm25"):
            for rank, i in enumerate(bm25_order[:CANDIDATES]):
                fused[i] = fused.get(i, 0.0) + 1.0 / (config.rrf_k + rank + 1)
        if config.mode in ("hybrid", "dense"):
            for rank, i in enumerate(dense_order[:CANDIDATES]):
                fused[i] = fused.get(i, 0.0) + 1.0 / (config.rrf_k + rank + 1)

        ordered = sorted(fused, key=lambda i: (-fused[i], i))
        hits: list[Hit] = []
        for i in ordered:
            if config.mode != "bm25" and float(dense[i]) < config.min_score:
                continue
            hits.append(
                Hit(
                    chunk=self.chunks[i],
                    rank=len(hits) + 1,
                    score=round(fused[i], 6),
                    bm25=round(float(bm25[i]), 4),
                    dense=round(float(dense[i]), 4),
                )
            )
            if len(hits) == config.top_k:
                break
        return hits
