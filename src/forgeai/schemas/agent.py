"""The declarative agent definition: everything needed to run a generated agent, and
everything the optimizer is allowed to patch."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from forgeai.retrieval.index import RetrievalConfig
from forgeai.schemas.corpus import ChunkingConfig
from forgeai.tools.registry import RejectedTool, ToolChoice


class Citation(BaseModel):
    # Optional on purpose: the quote's words are verified against every retrieved passage
    # and the id corrected, so a missing id must not throw away an otherwise good answer.
    chunk_id: str = ""
    quote: str = Field(description="Exact words copied from that passage.")


class AgentAnswer(BaseModel):
    """The output contract every generated agent must follow."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


class RefusalPolicy(BaseModel):
    refuse_topics: list[str]
    message: str
    # Below this best-passage similarity, with no tool evidence, the agent refuses
    # without asking the model. 0 disables. Calibrated from the requirements' topics.
    min_evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class KnowledgeSource(BaseModel):
    docs_dir: str
    fingerprints: dict[str, str] = Field(default_factory=dict, description="file -> sha256")


class Provenance(BaseModel):
    factory_version: str
    generator_model: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    notes: list[str] = Field(default_factory=list)


class AgentDefinition(BaseModel):
    schema_version: Literal["1"] = "1"
    name: str
    version: int = 0
    parent_version: int | None = None
    description: str
    audience: str
    tone: str

    role_prompt: str = Field(description="Who the agent is and what it covers. Patchable.")
    rules: list[str] = Field(default_factory=list, description="Behaviour rules. Patchable.")
    must_answer: list[str]

    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    tools: list[ToolChoice] = Field(default_factory=list)
    rejected_tools: list[RejectedTool] = Field(default_factory=list)
    refusal: RefusalPolicy
    output_contract: Literal["AgentAnswer/v1"] = "AgentAnswer/v1"

    knowledge: KnowledgeSource
    provenance: Provenance

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]
