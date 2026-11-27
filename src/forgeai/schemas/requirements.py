"""Intake and requirements artifacts: what the requester wants, in a form code can check."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_QUESTIONS = 5

ToolHint = Literal["table_lookup", "arithmetic", "weather", "current_date"]
AnswerSource = Literal["requester", "brief", "default"]


class DocSummary(BaseModel):
    doc_id: str
    title: str
    kind: str
    pages: int
    words: int
    headings: list[str] = Field(default_factory=list)
    columns: list[str] | None = None
    sample_rows: list[str] = Field(default_factory=list)


class CorpusSummary(BaseModel):
    documents: list[DocSummary]
    skipped: list[str] = Field(default_factory=list)


# ---------- Intake --------------------------------------------------------------------


class IntakeQuestion(BaseModel):
    id: str = Field(description="q1, q2, ...")
    question: str = Field(min_length=5)
    why: str = Field(description="What the answer changes in the build.")
    default_answer: str = Field(description="What will be assumed if nobody answers.")


class IntakeQuestions(BaseModel):
    questions: list[IntakeQuestion] = Field(
        default_factory=list, description=f"At most {MAX_QUESTIONS} questions."
    )

    @field_validator("questions")
    @classmethod
    def _cap_and_renumber(cls, value: list[IntakeQuestion]) -> list[IntakeQuestion]:
        # The limit is a hard requirement, so it is enforced in code, not trusted to the model.
        return [
            q.model_copy(update={"id": f"q{i}"}) for i, q in enumerate(value[:MAX_QUESTIONS], 1)
        ]


class ProxyAnswer(BaseModel):
    question_id: str
    answer: str
    from_brief: bool = Field(description="True only if the brief states or clearly implies it.")


class ProxyAnswers(BaseModel):
    answers: list[ProxyAnswer]


class IntakeAnswer(BaseModel):
    question_id: str
    question: str
    answer: str
    source: AnswerSource


class IntakeRecord(BaseModel):
    brief: str
    corpus: CorpusSummary
    questions: list[IntakeQuestion]
    answers: list[IntakeAnswer]
    mode: Literal["interactive", "auto", "file"]

    @property
    def assumptions(self) -> list[IntakeAnswer]:
        return [a for a in self.answers if a.source != "requester"]


# ---------- Requirements --------------------------------------------------------------


class Topic(BaseModel):
    topic: str = Field(min_length=2, description="Short name, e.g. 'annual leave entitlement'.")
    detail: str = Field(default="", description="What exactly must be covered.")
    answered_from: Literal["documents", "tool"] = "documents"
    tool_hints: list[ToolHint] = Field(
        default_factory=list,
        description="Capabilities this topic needs beyond reading the documents.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_single_hint(cls, data: Any) -> Any:
        """Models often write `tool_hint: "weather"`; accept it as a one-item list."""
        if isinstance(data, dict) and "tool_hint" in data:
            data = dict(data)
            hint = data.pop("tool_hint")
            hints = list(data.get("tool_hints") or [])
            if hint and hint not in hints:
                hints.append(hint)
            data["tool_hints"] = hints
        return data


class RefuseTopic(BaseModel):
    topic: str = Field(min_length=2)
    reason: str = ""


class RequirementsDraft(BaseModel):
    """What the model writes. Code adds provenance (assumptions, flags) afterwards."""

    agent_name: str = Field(min_length=2, max_length=60)
    purpose: str
    scope: str
    audience: str
    tone: str
    must_answer: list[Topic] = Field(min_length=1, max_length=15)
    must_refuse: list[RefuseTopic] = Field(min_length=1, max_length=15)


class FlaggedGap(BaseModel):
    topic: str
    reason: str


class Requirements(RequirementsDraft):
    brief: str
    assumptions: list[str] = Field(default_factory=list)
    flagged_gaps: list[FlaggedGap] = Field(default_factory=list)
    approved_by: Literal["requester", "auto"] | None = None

    @property
    def document_topics(self) -> list[Topic]:
        """Topics answered purely from prose; tool topics are checked differently."""
        return [t for t in self.must_answer if t.answered_from == "documents" and not t.tool_hints]

    @property
    def tool_hints(self) -> set[str]:
        return {h for t in self.must_answer for h in t.tool_hints}


# ---------- Groundability -------------------------------------------------------------

SupportStatus = Literal["supported", "partial", "unsupported"]


class TopicJudgement(BaseModel):
    """The model's claim about one topic; code verifies it before believing it."""

    status: SupportStatus
    chunk_id: str | None = None
    quote: str | None = Field(default=None, description="Exact words copied from the passage.")
    note: str = ""


class GroundingCheck(BaseModel):
    topic: str
    kind: Literal["documents", "personal_records", "table"] = "documents"
    claimed: SupportStatus
    verdict: SupportStatus
    chunk_id: str | None = None
    quote: str | None = None
    quote_verified: bool = False
    retrieved: list[str] = Field(default_factory=list)
    note: str = ""


class GroundabilityReport(BaseModel):
    checks: list[GroundingCheck]

    @property
    def unsupported(self) -> list[GroundingCheck]:
        return [c for c in self.checks if c.verdict == "unsupported"]

    @property
    def coverage(self) -> float:
        if not self.checks:
            return 1.0
        return sum(c.verdict != "unsupported" for c in self.checks) / len(self.checks)
