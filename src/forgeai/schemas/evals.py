"""Eval set, per-case results, judgements and scorecards."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["answerable", "multi_doc", "ambiguous", "out_of_scope", "tool"]
Split = Literal["dev", "holdout"]
FailureCategory = Literal["retrieval_miss", "prompt_defect", "hallucination", "wrong_refusal"]
FactStatus = Literal["present", "missing", "contradicted"]
CanaryKind = Literal["wrong_number", "confident_omission", "correct_padded", "correct_terse"]


class ExpectedSource(BaseModel):
    doc_id: str
    quote: str = Field(description="Verified verbatim span; chunking-independent anchor.")


class EvalCase(BaseModel):
    id: str
    category: Category
    question: str
    expect_refusal: bool = False
    sources: list[ExpectedSource] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    expected_tool: str | None = None
    rationale: str = ""
    origin: Literal["llm", "template"] = "llm"
    split: Split = "dev"


class RejectedCase(BaseModel):
    category: Category
    question: str
    reason: str


class EvalSet(BaseModel):
    cases: list[EvalCase]
    rejected: list[RejectedCase] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list, description="Prompt templates used.")

    def split(self, name: Split | Literal["all"]) -> list[EvalCase]:
        return self.cases if name == "all" else [c for c in self.cases if c.split == name]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for case in self.cases:
            out[case.category] = out.get(case.category, 0) + 1
        return out


# ---------- Judging -------------------------------------------------------------------


class FactCheck(BaseModel):
    fact: str
    status: FactStatus
    by: Literal["string", "judge"]


class Checks(BaseModel):
    refusal_ok: bool
    grounded: bool
    source_cited: bool | None = None
    source_retrieved: bool | None = None
    unsupported_numbers: list[str] = Field(default_factory=list)
    tool_ok: bool | None = None
    contract_error: str | None = None


class Judgement(BaseModel):
    case_id: str
    passed: bool
    checks: Checks
    facts: list[FactCheck] = Field(default_factory=list)
    failure: FailureCategory | None = None
    reason: str = ""
    judge_used: bool = False

    @property
    def fact_recall(self) -> float:
        if not self.facts:
            return 1.0
        return sum(f.status == "present" for f in self.facts) / len(self.facts)


class CaseResult(BaseModel):
    case: EvalCase
    answer: str
    refused: bool
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    judgement: Judgement


class CanaryResult(BaseModel):
    kind: CanaryKind
    case_id: str
    answer: str
    should_pass: bool
    judge_passed: bool

    @property
    def correct(self) -> bool:
        return self.should_pass == self.judge_passed


class SplitScore(BaseModel):
    cases: int
    passed: int
    pass_rate: float
    by_category: dict[str, float] = Field(default_factory=dict)
    failures: dict[str, int] = Field(default_factory=dict)
    refusal_accuracy: float
    groundedness: float
    fact_recall: float


class Scorecard(BaseModel):
    agent_version: int
    splits: dict[str, SplitScore]
    judge_canary_accuracy: float | None = None
    judge_trusted: bool = True
    mean_latency_s: float = 0.0


class EvalRun(BaseModel):
    agent_version: int
    split: Literal["dev", "holdout", "all"]
    results: list[CaseResult]
    canaries: list[CanaryResult] = Field(default_factory=list)
    scorecard: Scorecard
