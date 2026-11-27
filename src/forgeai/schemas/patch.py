"""Typed patches: the only way the optimizer may change an agent. Each one is small,
names the failure category and the cases it targets, and can be applied or reverted."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from forgeai.schemas.evals import FailureCategory


class SetTopK(BaseModel):
    op: Literal["set_top_k"] = "set_top_k"
    value: int = Field(ge=1, le=20)


class SetChunkWords(BaseModel):
    op: Literal["set_chunk_words"] = "set_chunk_words"
    value: int = Field(ge=40, le=1200)


class SetRetrievalMode(BaseModel):
    op: Literal["set_retrieval_mode"] = "set_retrieval_mode"
    value: Literal["hybrid", "bm25", "dense"]


class SetThreshold(BaseModel):
    op: Literal["set_threshold"] = "set_threshold"
    value: float = Field(ge=0.0, le=1.0)


class AddRule(BaseModel):
    op: Literal["add_rule"] = "add_rule"
    text: str = Field(min_length=10, max_length=240)


class AddRefuseTopic(BaseModel):
    op: Literal["add_refuse_topic"] = "add_refuse_topic"
    topic: str = Field(min_length=3, max_length=120)


PatchOp = Annotated[
    SetTopK | SetChunkWords | SetRetrievalMode | SetThreshold | AddRule | AddRefuseTopic,
    Field(discriminator="op"),
]


class Patch(BaseModel):
    op: PatchOp
    category: FailureCategory
    targets: list[str] = Field(description="Dev case ids this patch is meant to fix.")
    why: str
    source: Literal["rule", "critic"] = "rule"

    def describe(self) -> str:
        op = self.op
        if isinstance(op, AddRule):
            return f'add rule: "{op.text}"'
        if isinstance(op, AddRefuseTopic):
            return f'add refuse topic: "{op.topic}"'
        return f"{op.op.removeprefix('set_').replace('_', ' ')} -> {op.value}"


class ChangeLogEntry(BaseModel):
    iteration: int
    patch: Patch
    from_version: int
    to_version: int | None
    accepted: bool
    dev_before: float
    dev_after: float
    fixed: list[str] = Field(default_factory=list)
    broke: list[str] = Field(default_factory=list)
    decision: str
    tokens: int = 0
