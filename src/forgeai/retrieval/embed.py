"""Text embedders. FastEmbed (bge-small, ONNX on CPU) for real runs; a hashing embedder
for tests and offline use, so CI never downloads a model."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from forgeai.textutil import normalize

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> Vector: ...

    def embed_query(self, text: str) -> Vector: ...


def _l2(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vector = (matrix / norms).astype(np.float32)
    return out


class HashEmbedder:
    """Bag of word unigrams + bigrams hashed into a fixed space. Deterministic, no download.

    Not semantic, but it behaves like a real embedder for tests (similar wording, high
    cosine) and keeps the pipeline runnable with zero model files.
    """

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims
        self.name = f"hash-{dims}"

    def _vector(self, text: str) -> Vector:
        vec = np.zeros(self.dims, dtype=np.float32)
        words = normalize(text).split()
        grams = words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dims
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        return vec

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        stacked: Vector = np.stack([self._vector(t) for t in texts])
        return _l2(stacked)

    def embed_query(self, text: str) -> Vector:
        row: Vector = _l2(self._vector(text)[None, :])[0]
        return row


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 via fastembed. Downloads ~130 MB on first use, then cached.

    Small batches on purpose: fastembed's default batch of 256 passages builds attention
    tensors of several GB, which kills a 512 MB host. Vectors are identical either way.
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", *, batch_size: int = 8, threads: int = 1
    ) -> None:
        from fastembed import TextEmbedding

        self._model: Any = TextEmbedding(model_name=model_name, threads=threads)
        self._batch_size = batch_size
        self.name = model_name

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        stacked: Vector = np.stack(
            list(self._model.passage_embed(list(texts), batch_size=self._batch_size))
        ).astype(np.float32)
        return _l2(stacked)

    def embed_query(self, text: str) -> Vector:
        vectors = list(self._model.query_embed([text]))
        row: Vector = _l2(np.asarray(vectors[0], dtype=np.float32)[None, :])[0]
        return row


def make_embedder(kind: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder {kind!r} (use 'fastembed' or 'hash')")
