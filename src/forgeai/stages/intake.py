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
    Topic,
)
from forgeai.stages.context import ForgeContext
from forgeai.textutil import normalize

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


def _dedupe_topics(topics: list[Topic]) -> list[Topic]:
    seen: set[str] = set()
    out: list[Topic] = []
    for topic in topics:
        key = normalize(topic.topic)
        if key and key not in seen:
            seen.add(key)
            # A tool hint only makes sense on a tool topic, and vice versa.
            if topic.answered_from == "documents" and topic.tool_hint is not None:
                topic = topic.model_copy(update={"tool_hint": None})
            if topic.answered_from == "tool" and topic.tool_hint is None:
                topic = topic.model_copy(update={"answered_from": "documents"})
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


_SOURCE_LABEL = {"brief": "inferred from the brief", "default": "interviewer default"}


async def write_requirements(
    ctx: ForgeContext, brief: str, summary: CorpusSummary, answers: list[IntakeAnswer]
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
    return Requirements(
        **draft.model_dump(exclude={"must_answer", "must_refuse"}),
        must_answer=_dedupe_topics(draft.must_answer),
        must_refuse=_dedupe_refusals(draft.must_refuse),
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
    requirements = await write_requirements(ctx, brief, summary, answers)
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
        source = f"tool: {t.tool_hint}" if t.answered_from == "tool" else "documents"
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
