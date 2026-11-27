"""Stage 1 — intake interviewer: read the brief and the folder, ask at most five questions,
write a structured requirements document."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from forgeai.llm import ChatMessage, complete_json, stage
from forgeai.prompts import render
from forgeai.schemas.corpus import Chunk, Corpus
from forgeai.schemas.requirements import (
    CorpusSummary,
    DocSummary,
    IntakeAnswer,
    IntakeQuestion,
    IntakeQuestions,
    IntakeRecord,
    ProxyAnswers,
    RefuseTopic,
    Requirements,
    RequirementsDraft,
    ToolHint,
    Topic,
)
from forgeai.stages.context import ForgeContext
from forgeai.textutil import normalize
from forgeai.tools.registry import brief_asks_for, infer_tool_hints

MAX_HEADINGS = 12
SAMPLE_ROWS = 3


# ---------- Corpus summary (deterministic) --------------------------------------------


def summarize_corpus(corpus: Corpus, chunks: list[Chunk]) -> CorpusSummary:
    docs: list[DocSummary] = []
    for doc in corpus.documents:
        headings: list[str] = []
        for chunk in chunks:
            if chunk.doc_id == doc.doc_id and chunk.heading and chunk.heading not in headings:
                headings.append(chunk.heading)
        docs.append(
            DocSummary(
                doc_id=doc.doc_id,
                title=doc.title,
                kind=doc.kind,
                pages=len(doc.pages),
                words=doc.words,
                headings=headings[:MAX_HEADINGS],
                columns=doc.columns,
                sample_rows=[p.text for p in doc.pages[:SAMPLE_ROWS]] if doc.kind == "csv" else [],
            )
        )
    return CorpusSummary(
        documents=docs, skipped=[f"{s.filename} ({s.reason})" for s in corpus.skipped]
    )


def summary_text(summary: CorpusSummary) -> str:
    lines: list[str] = []
    for d in summary.documents:
        unit = "rows" if d.kind == "csv" else "pages"
        lines.append(f'- {d.doc_id} ({d.kind}, {d.pages} {unit}, {d.words} words): "{d.title}"')
        if d.headings:
            lines.append(f"  sections: {'; '.join(d.headings)}")
        if d.columns:
            lines.append(f"  columns: {', '.join(d.columns)}")
        lines.extend(f"  sample: {row}" for row in d.sample_rows)
    if summary.skipped:
        lines.append(f"- skipped files: {', '.join(summary.skipped)}")
    return "\n".join(lines)


# ---------- Questions -----------------------------------------------------------------


async def ask_questions(
    ctx: ForgeContext, brief: str, summary: CorpusSummary
) -> list[IntakeQuestion]:
    prompt = render("intake_questions", brief=brief, corpus_summary=summary_text(summary))
    with stage("intake"):
        result = await complete_json(
            ctx.llm,
            [ChatMessage(role="user", content=prompt)],
            IntakeQuestions,
            model=ctx.models.generator,
        )
    return result.questions


# ---------- Answerers -----------------------------------------------------------------


class Answerer(Protocol):
    mode: Literal["interactive", "auto", "file"]

    async def answer(self, brief: str, questions: list[IntakeQuestion]) -> list[IntakeAnswer]: ...


AskFn = Callable[[str, str], str]


class InteractiveAnswerer:
    """CLI dialogue. Pressing Enter accepts the shown default, which counts as the
    requester's answer: they saw it and agreed."""

    mode: Literal["interactive", "auto", "file"] = "interactive"

    def __init__(self, ask: AskFn) -> None:
        self._ask = ask

    async def answer(self, brief: str, questions: list[IntakeQuestion]) -> list[IntakeAnswer]:
        out: list[IntakeAnswer] = []
        for q in questions:
            reply = self._ask(q.question, q.default_answer).strip() or q.default_answer
            out.append(
                IntakeAnswer(
                    question_id=q.id, question=q.question, answer=reply, source="requester"
                )
            )
        return out


class FileAnswerer:
    """Answers from a JSON file: {"q1": "...", ...} or {"<question text>": "..."}.
    Unanswered questions fall back to the interviewer's default, marked as such."""

    mode: Literal["interactive", "auto", "file"] = "file"

    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("answers file must be a JSON object")
        self._answers = {str(k): str(v) for k, v in raw.items()}
        self._by_text = {normalize(k): v for k, v in self._answers.items()}

    async def answer(self, brief: str, questions: list[IntakeQuestion]) -> list[IntakeAnswer]:
        out: list[IntakeAnswer] = []
        for q in questions:
            given = self._answers.get(q.id) or self._by_text.get(normalize(q.question))
            out.append(
                IntakeAnswer(
                    question_id=q.id,
                    question=q.question,
                    answer=given or q.default_answer,
                    source="requester" if given else "default",
                )
            )
        return out


class AutoAnswerer:
    """Unattended mode: a requester proxy answers from the brief alone. Nothing it says
    counts as the requester's word; every answer is reported as an assumption."""

    mode: Literal["interactive", "auto", "file"] = "auto"

    def __init__(self, ctx: ForgeContext) -> None:
        self._ctx = ctx

    async def answer(self, brief: str, questions: list[IntakeQuestion]) -> list[IntakeAnswer]:
        if not questions:
            return []
        payload = json.dumps(
            [
                {"question_id": q.id, "question": q.question, "default_answer": q.default_answer}
                for q in questions
            ],
            ensure_ascii=False,
        )
        prompt = render("intake_auto_answer", brief=brief, questions=payload)
        with stage("intake"):
            result = await complete_json(
                self._ctx.llm,
                [ChatMessage(role="user", content=prompt)],
                ProxyAnswers,
                model=self._ctx.models.generator,
            )
        by_id = {a.question_id: a for a in result.answers}
        out: list[IntakeAnswer] = []
        for q in questions:
            proxy = by_id.get(q.id)
            if proxy is None or not proxy.answer.strip():
                out.append(
                    IntakeAnswer(
                        question_id=q.id,
                        question=q.question,
                        answer=q.default_answer,
                        source="default",
                    )
                )
                continue
            out.append(
                IntakeAnswer(
                    question_id=q.id,
                    question=q.question,
                    answer=proxy.answer.strip(),
                    source="brief" if proxy.from_brief else "default",
                )
            )
        return out


# ---------- Requirements --------------------------------------------------------------


def _answers_text(answers: list[IntakeAnswer]) -> str:
    if not answers:
        return "(no questions were needed)"
    return "\n".join(f"- {a.question} -> {a.answer}" for a in answers)


def _dedupe_topics(topics: list[Topic], corpus: Corpus | None = None) -> list[Topic]:
    """Remove duplicates and settle each topic's capabilities in code: the model's own
    hints plus those inferred from the topic's words and the folder's tables."""
    seen: set[str] = set()
    out: list[Topic] = []
    for topic in topics:
        key = normalize(topic.topic)
        if not key or key in seen:
            continue
        seen.add(key)
        hints = list(topic.tool_hints)
        if corpus is not None:
            hints += [
                h
                for h in infer_tool_hints(f"{topic.topic}. {topic.detail}", corpus)
                if h not in hints
            ]
        topic = topic.model_copy(
            update={"tool_hints": hints, "answered_from": "tool" if hints else "documents"}
        )
        out.append(topic)
    return out


def _dedupe_refusals(topics: list[RefuseTopic]) -> list[RefuseTopic]:
    seen: set[str] = set()
    out: list[RefuseTopic] = []
    for topic in topics:
        key = normalize(topic.topic)
        if key and key not in seen:
            seen.add(key)
            out.append(topic)
    return out


# Live capabilities the brief can ask for in plain words, and how a refusal that
# contradicts them tends to be phrased ("Live data access", "Weather forecasts").
_LIVE: dict[ToolHint, tuple[str, re.Pattern[str]]] = {
    "current_date": (
        "Dates counted from today",
        re.compile(r"\b(live|real[- ]?time|today'?s?|current date|dates?)\b", re.I),
    ),
    "weather": (
        "Weather forecast",
        re.compile(r"\b(live|real[- ]?time|weather|forecasts?)\b", re.I),
    ),
}


def _clause_with(text: str, phrase: str) -> str:
    """The comma- or sentence-delimited piece of the brief that contains `phrase`."""
    for piece in re.split(r"[.;:]\s+|,\s+", text):
        if re.search(rf"\b{re.escape(phrase)}\b", piece, re.I):
            return re.sub(r"^(and|or|also)\s+", "", piece.strip(), flags=re.I).rstrip(".")
    return phrase


def keep_brief_capabilities(
    must_answer: list[Topic],
    must_refuse: list[RefuseTopic],
    brief: str,
    corpus: Corpus | None,
) -> tuple[list[Topic], list[RefuseTopic]]:
    """The brief's own words outrank the model's draft.

    If the brief asks for a live capability ("...if the order is placed today") but no
    must-answer topic carries it, add the topic; and drop any must-refuse topic that
    contradicts it (a stand-in answer of "no live data needed" once turned the ask
    into a refusal). Attaching the tool is still decided by the tool contract.
    """
    answer, refuse = list(must_answer), list(must_refuse)
    for capability, (label, conflict) in _LIVE.items():
        phrase = brief_asks_for(capability, brief)
        if phrase is None:
            continue
        dropped = [r.topic for r in refuse if conflict.search(r.topic)]
        refuse = [r for r in refuse if not conflict.search(r.topic)]
        if any(capability in t.tool_hints for t in answer):
            continue
        clause = _clause_with(brief, phrase)
        hints: list[ToolHint] = [capability]
        if corpus is not None:
            hints += [h for h in infer_tool_hints(clause, corpus) if h not in hints]
        note = f'added in code: the brief says "{phrase}"'
        if dropped:
            note += "; replaced the refusal " + ", ".join(f'"{d}"' for d in dropped)
        answer.append(
            Topic(topic=label, detail=f"{clause} ({note})", answered_from="tool", tool_hints=hints)
        )
    return answer, refuse


_SOURCE_LABEL = {"brief": "inferred from the brief", "default": "interviewer default"}


async def write_requirements(
    ctx: ForgeContext,
    brief: str,
    summary: CorpusSummary,
    answers: list[IntakeAnswer],
    corpus: Corpus | None = None,
) -> Requirements:
    prompt = render(
        "intake_requirements",
        brief=brief,
        corpus_summary=summary_text(summary),
        answers=_answers_text(answers),
    )
    with stage("intake"):
        draft = await complete_json(
            ctx.llm,
            [ChatMessage(role="user", content=prompt)],
            RequirementsDraft,
            model=ctx.models.generator,
            max_tokens=3000,
        )
    assumptions = [
        f"{a.question} -> {a.answer} ({_SOURCE_LABEL[a.source]})"
        for a in answers
        if a.source != "requester"
    ]
    must_answer, must_refuse = keep_brief_capabilities(
        _dedupe_topics(draft.must_answer, corpus),
        _dedupe_refusals(draft.must_refuse),
        brief,
        corpus,
    )
    return Requirements(
        **draft.model_dump(exclude={"must_answer", "must_refuse"}),
        must_answer=must_answer,
        must_refuse=must_refuse,
        brief=brief,
        assumptions=assumptions,
    )


async def run_intake(
    ctx: ForgeContext, brief: str, corpus: Corpus, chunks: list[Chunk], answerer: Answerer
) -> tuple[IntakeRecord, Requirements]:
    summary = summarize_corpus(corpus, chunks)
    questions = await ask_questions(ctx, brief, summary)
    answers = await answerer.answer(brief, questions)
    record = IntakeRecord(
        brief=brief, corpus=summary, questions=questions, answers=answers, mode=answerer.mode
    )
    requirements = await write_requirements(ctx, brief, summary, answers, corpus)
    return record, requirements


# ---------- Markdown ------------------------------------------------------------------

_SLUG = re.compile(r"\s+")


def requirements_markdown(req: Requirements) -> str:
    lines = [
        f"# Requirements: {req.agent_name}",
        "",
        f"**Purpose.** {req.purpose}",
        "",
        f"**Scope.** {req.scope}",
        "",
        f"**Audience.** {req.audience}",
        "",
        f"**Tone.** {req.tone}",
        "",
        "## Must answer",
        "",
        "| Topic | Detail | Source |",
        "|---|---|---|",
    ]
    for t in req.must_answer:
        source = "tools: " + ", ".join(t.tool_hints) if t.tool_hints else "documents"
        lines.append(f"| {t.topic} | {_cell(t.detail)} | {source} |")
    lines += ["", "## Must refuse", "", "| Topic | Reason |", "|---|---|"]
    lines += [f"| {t.topic} | {_cell(t.reason)} |" for t in req.must_refuse]
    if req.flagged_gaps:
        lines += [
            "",
            "## Flagged gaps",
            "",
            "The brief asks for these, but the documents do not support them. The agent will",
            "refuse them instead of guessing.",
            "",
        ]
        lines += [f"- **{g.topic}**: {g.reason}" for g in req.flagged_gaps]
    if req.assumptions:
        lines += ["", "## Assumptions", "", "Not confirmed by the requester:", ""]
        lines += [f"- {a}" for a in req.assumptions]
    lines += ["", "## Brief", "", f"> {_SLUG.sub(' ', req.brief).strip()}"]
    if req.approved_by:
        lines += ["", f"_Approved by: {req.approved_by}_"]
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
