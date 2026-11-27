"""What every stage needs: a model client, which model plays which role, an embedder."""

from __future__ import annotations

from dataclasses import dataclass

from forgeai.llm.base import LLMClient
from forgeai.retrieval.embed import Embedder


@dataclass(frozen=True)
class ModelRoles:
    generator: str
    judge: str


@dataclass
class ForgeContext:
    llm: LLMClient
    models: ModelRoles
    embedder: Embedder
