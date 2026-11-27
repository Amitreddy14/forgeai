from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from forgeai.ingest import chunk_corpus, load_folder
from forgeai.retrieval import HashEmbedder, HybridIndex, RetrievalConfig, make_embedder, tokenize
from tests.fixtures import build_docs_folder


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> HybridIndex:
    corpus = load_folder(build_docs_folder(tmp_path_factory.mktemp("r")))
    return HybridIndex(chunk_corpus(corpus), HashEmbedder())


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("How many days of annual leave do shore staff get?", "leave-policy"),
        ("medical certificate for sick leave", "leave-policy"),
        ("meal allowance for international travel", "travel-policy"),
        ("maximum draft at berth M4", "berths"),
        ("gift declaration threshold", "code-of-conduct"),
    ],
)
def test_hybrid_finds_the_right_document(index: HybridIndex, query: str, expected_doc: str) -> None:
    hits = index.search(query, RetrievalConfig(top_k=3))
    assert hits[0].chunk.doc_id == expected_doc


def test_hits_are_ranked_and_explained(index: HybridIndex) -> None:
    hits = index.search("annual leave carry forward", RetrievalConfig(top_k=4))
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert hits == sorted(hits, key=lambda h: -h.score)
    assert hits[0].bm25 > 0
    assert -1.0 <= hits[0].dense <= 1.0


def test_top_k_is_respected(index: HybridIndex) -> None:
    assert len(index.search("leave", RetrievalConfig(top_k=2))) == 2


def test_bm25_mode_ignores_unmatched_chunks(index: HybridIndex) -> None:
    hits = index.search("zzzz qqqq", RetrievalConfig(mode="bm25"))
    assert hits == []


def test_min_score_drops_weak_dense_matches(index: HybridIndex) -> None:
    loose = index.search("weather forecast for tomorrow", RetrievalConfig(min_score=0.0))
    strict = index.search("weather forecast for tomorrow", RetrievalConfig(min_score=0.9))
    assert loose
    assert strict == []


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("What is the daily allowance for INR 1,500?") == [
        "daily",
        "allowance",
        "inr",
        "1500",
    ]


def test_hash_embedder_is_normalised_and_deterministic() -> None:
    emb = HashEmbedder(dims=64)
    a = emb.embed_documents(["annual leave policy", "annual leave policy"])
    assert np.allclose(a[0], a[1])
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert emb.embed_documents([]).shape == (0, 64)


def test_unknown_embedder_rejected() -> None:
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("word2vec")


def test_empty_index_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        HybridIndex([], HashEmbedder())


def test_fastembed_end_to_end_when_available(tmp_path: Path) -> None:
    """Opt-in (FORGE_TEST_FASTEMBED=1): downloads the ~130 MB bge-small model once."""
    if os.environ.get("FORGE_TEST_FASTEMBED") != "1":
        pytest.skip("set FORGE_TEST_FASTEMBED=1 to run the real-embedding test")
    try:
        embedder = make_embedder("fastembed")
    except Exception as exc:  # offline CI: skip, don't fail
        pytest.skip(f"fastembed model unavailable: {type(exc).__name__}")
    corpus = load_folder(build_docs_folder(tmp_path))
    hits = HybridIndex(chunk_corpus(corpus), embedder).search(
        "time off when a child is born", RetrievalConfig(top_k=3, mode="dense")
    )
    assert any("parental" in h.chunk.text.lower() for h in hits)
