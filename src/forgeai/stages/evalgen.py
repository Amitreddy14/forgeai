"""Stage 4 — eval generator: build the test set from the documents themselves.

Every generated case is checked before it is kept:
- its quote must be a verbatim span of the source passage;
- every key fact must appear inside that quote (so the expected answer is in the docs);
- the question must not leak its own answer;
- near-duplicate questions are dropped.
Table and tool cases are produced by code templates from real CSV rows, so their
expected answers (including arithmetic) are computed, not guessed. The set is split
dev/holdout, stratified by category; the optimizer only ever sees dev.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from forgeai.errors import ModelOutputError
from forgeai.ingest import chunk_corpus
from forgeai.llm import ChatMessage, complete_json, stage
from forgeai.prompts import render
from forgeai.schemas.corpus import Chunk, ChunkingConfig, Corpus, Document
from forgeai.schemas.evals import Category, EvalCase, EvalSet, ExpectedSource, RejectedCase
from forgeai.schemas.requirements import Requirements
from forgeai.stages.context import ForgeContext
from forgeai.textutil import contains_fact, contains_quote, word_jaccard
from forgeai.tools.calculator import format_number

MIN_CASES = 20
DEV_FRACTION = 0.6
DUPLICATE_JACCARD = 0.7
MIN_PASSAGE_WORDS = 25
PASSAGE_CHARS = 1500


@dataclass(frozen=True)
class EvalTargets:
    answerable: int = 10
    ambiguous: int = 3
    multi_doc: int = 4
    out_of_scope: int = 6
    refill: int = 6  # extra answerable attempts if validation leaves too few cases


class _Answerable(BaseModel):
    question: str = Field(min_length=8)
    key_facts: list[str] = Field(min_length=1, max_length=3)
    quote: str


class _MultiDoc(BaseModel):
    possible: bool = True
    question: str = ""
    fact_a: str = ""
    quote_a: str = ""
    fact_b: str = ""
    quote_b: str = ""


class _Refusal(BaseModel):
    question: str = Field(min_length=8)
    topic: str = ""
    why: str = ""


class _Refusals(BaseModel):
    questions: list[_Refusal]


@dataclass
class _Candidate:
    case: EvalCase
    passage_texts: list[str]


# ---------- Passage selection (deterministic) -----------------------------------------


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def select_passages(corpus: Corpus, chunks: list[Chunk], limit: int) -> list[Chunk]:
    """Round-robin across documents so every document is tested; prefer passages with
    numbers (they make sharp test facts); stable order so runs are reproducible."""
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        doc = corpus.get(chunk.doc_id)
        if doc is None or len(chunk.text.split()) < (6 if doc.kind == "csv" else MIN_PASSAGE_WORDS):
            continue
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    for doc_id, items in by_doc.items():
        cap = 2 if (d := corpus.get(doc_id)) is not None and d.kind == "csv" else len(items)
        items.sort(key=lambda c: (not re.search(r"\d", c.text), _stable_key(c.chunk_id)))
        by_doc[doc_id] = items[:cap]
    picked: list[Chunk] = []
    queues = [by_doc[k] for k in sorted(by_doc)]
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def _pairs(corpus: Corpus, passages: list[Chunk], limit: int) -> list[tuple[Chunk, Chunk]]:
    prose = [c for c in passages if (d := corpus.get(c.doc_id)) is not None and d.kind != "csv"]
    cross = [(a, b) for i, a in enumerate(prose) for b in prose[i + 1 :] if a.doc_id != b.doc_id]
    if not cross:  # single document: pair different sections instead
        cross = [
            (a, b) for i, a in enumerate(prose) for b in prose[i + 1 :] if a.heading != b.heading
        ]
    cross.sort(key=lambda p: _stable_key(p[0].chunk_id + p[1].chunk_id))
    out: list[tuple[Chunk, Chunk]] = []
    used: set[str] = set()
    for a, b in cross:
        if a.chunk_id in used or b.chunk_id in used:
            continue
        out.append((a, b))
        used |= {a.chunk_id, b.chunk_id}
        if len(out) == limit:
            break
    return out


# ---------- Validation ----------------------------------------------------------------


def validate_candidate(
    category: Category,
    question: str,
    facts: list[str],
    quotes: list[tuple[str, str, str]],  # (doc_id, quote, passage_text)
) -> str | None:
    """Return a rejection reason, or None if the case is sound."""
    if len(question.strip()) < 8:
        return "question too short"
    for _doc_id, quote, passage in quotes:
        if not contains_quote(passage, quote):
            return "quote is not a verbatim span of the source passage"
    all_quotes = " ".join(q for _d, q, _p in quotes)
    for fact in facts:
        if not fact.strip():
            return "empty key fact"
        if not contains_fact(all_quotes, fact):
            return f"key fact {fact!r} is not in the quoted source"
    if facts and all(contains_fact(question, f) for f in facts):
        return "question contains its own answer"
    return None


def _dedupe(candidates: list[_Candidate], rejected: list[RejectedCase]) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for cand in candidates:
        twin = next(
            (
                k
                for k in kept
                if word_jaccard(k.case.question, cand.case.question) >= DUPLICATE_JACCARD
            ),
            None,
        )
        if twin is not None:
            rejected.append(
                RejectedCase(
                    category=cand.case.category,
                    question=cand.case.question,
                    reason="near-duplicate of another question",
                )
            )
            continue
        kept.append(cand)
    return kept


# ---------- LLM case writers ----------------------------------------------------------


async def _single_passage_case(
    ctx: ForgeContext,
    req: Requirements,
    corpus: Corpus,
    chunk: Chunk,
    category: Category,
    rejected: list[RejectedCase],
) -> _Candidate | None:
    doc = corpus.get(chunk.doc_id)
    template = "evals_answerable" if category == "answerable" else "evals_ambiguous"
    prompt = render(
        template,
        audience=req.audience,
        chunk_id=chunk.chunk_id,
        doc_title=doc.title if doc else chunk.doc_id,
        passage=chunk.text[:PASSAGE_CHARS],
    )
    try:
        out = await complete_json(
            ctx.llm,
            [ChatMessage(role="user", content=prompt)],
            _Answerable,
            model=ctx.models.generator,
            temperature=0.0,
        )
    except ModelOutputError as exc:
        rejected.append(RejectedCase(category=category, question="", reason=f"model output: {exc}"))
        return None
    facts = [f.strip() for f in out.key_facts if f.strip()]
    reason = validate_candidate(
        category, out.question, facts, [(chunk.doc_id, out.quote, chunk.text)]
    )
    if reason:
        rejected.append(RejectedCase(category=category, question=out.question, reason=reason))
        return None
    case = EvalCase(
        id="",
        category=category,
        question=out.question.strip(),
        sources=[ExpectedSource(doc_id=chunk.doc_id, quote=out.quote.strip())],
        key_facts=facts,
        rationale=f"generated from {chunk.citation}",
    )
    return _Candidate(case, [chunk.text])


async def _multi_doc_case(
    ctx: ForgeContext,
    req: Requirements,
    corpus: Corpus,
    a: Chunk,
    b: Chunk,
    rejected: list[RejectedCase],
) -> _Candidate | None:
    title = {d.doc_id: d.title for d in corpus.documents}
    prompt = render(
        "evals_multi_doc",
        audience=req.audience,
        chunk_a=a.chunk_id,
        title_a=title.get(a.doc_id, a.doc_id),
        passage_a=a.text[:PASSAGE_CHARS],
        chunk_b=b.chunk_id,
        title_b=title.get(b.doc_id, b.doc_id),
        passage_b=b.text[:PASSAGE_CHARS],
    )
    try:
        out = await complete_json(
            ctx.llm,
            [ChatMessage(role="user", content=prompt)],
            _MultiDoc,
            model=ctx.models.generator,
        )
    except ModelOutputError as exc:
        rejected.append(
            RejectedCase(category="multi_doc", question="", reason=f"model output: {exc}")
        )
        return None
    if not out.possible or not out.question:
        rejected.append(
            RejectedCase(
                category="multi_doc",
                question=out.question,
                reason="model found no sensible pairing",
            )
        )
        return None
    facts = [out.fact_a.strip(), out.fact_b.strip()]
    reason = validate_candidate(
        "multi_doc",
        out.question,
        facts,
        [(a.doc_id, out.quote_a, a.text), (b.doc_id, out.quote_b, b.text)],
    )
    # Each fact must come from its own passage, or the case does not really need both.
    if reason is None and not contains_fact(out.quote_a, out.fact_a):
        reason = "fact A is not in quote A"
    if reason is None and not contains_fact(out.quote_b, out.fact_b):
        reason = "fact B is not in quote B"
    if reason:
        rejected.append(RejectedCase(category="multi_doc", question=out.question, reason=reason))
        return None
    case = EvalCase(
        id="",
        category="multi_doc",
        question=out.question.strip(),
        sources=[
            ExpectedSource(doc_id=a.doc_id, quote=out.quote_a.strip()),
            ExpectedSource(doc_id=b.doc_id, quote=out.quote_b.strip()),
        ],
        key_facts=facts,
        rationale=f"needs {a.citation} and {b.citation}",
    )
    return _Candidate(case, [a.text, b.text])


async def _refusal_cases(
    ctx: ForgeContext,
    req: Requirements,
    chunks: list[Chunk],
    count: int,
    rejected: list[RejectedCase],
) -> list[_Candidate]:
    sections = sorted({c.heading for c in chunks if c.heading})[:20]
    refuse = [f"- {t.topic}: {t.reason}" for t in req.must_refuse]
    prompt = render(
        "evals_out_of_scope",
        purpose=req.purpose,
        audience=req.audience,
        refuse_topics="\n".join(refuse),
        sections=", ".join(sections) or "(none)",
        count=count,
    )
    try:
        out = await complete_json(
            ctx.llm,
            [ChatMessage(role="user", content=prompt)],
            _Refusals,
            model=ctx.models.generator,
            max_tokens=1500,
        )
    except ModelOutputError as exc:
        rejected.append(
            RejectedCase(category="out_of_scope", question="", reason=f"model output: {exc}")
        )
        return []
    candidates: list[_Candidate] = []
    for item in out.questions[: count + 2]:
        case = EvalCase(
            id="",
            category="out_of_scope",
            question=item.question.strip(),
            expect_refusal=True,
            rationale=f"{item.topic}: {item.why}".strip(": "),
        )
        candidates.append(_Candidate(case, []))
    return candidates


# ---------- Template cases (computed by code) -----------------------------------------

_MONEY = re.compile(r"(rate|price|cost|fee|charge|tariff|dues|usd|inr|eur)", re.IGNORECASE)
_PLACE = re.compile(r"^(port|city|location|place|town)$", re.IGNORECASE)


def _human(column: str) -> str:
    return column.replace("_", " ").strip()


def _number(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def template_cases(req: Requirements, corpus: Corpus) -> list[_Candidate]:
    hints = req.tool_hints
    tables = [d for d in corpus.documents if d.kind == "csv" and d.rows]
    out: list[_Candidate] = []
    if tables and "table_lookup" in hints:
        out.extend(_table_cases(tables[0], with_math="arithmetic" in hints))
    if "weather" in hints:
        place = _a_place(tables)
        if place:
            case = EvalCase(
                id="",
                category="tool",
                question=f"What is the weather forecast for {place} for the next two days?",
                expected_tool="weather",
                origin="template",
                rationale="live data: must call the weather tool, not answer from memory",
            )
            out.append(_Candidate(case, []))
    if "current_date" in hints:
        case = EvalCase(
            id="",
            category="tool",
            question="What is today's date?",
            expected_tool="current_date",
            origin="template",
            rationale="live data: must call the current_date tool",
        )
        out.append(_Candidate(case, []))
    return out


def _a_place(tables: list[Document]) -> str | None:
    for table in tables:
        for column in table.columns or []:
            if _PLACE.match(column) and table.rows:
                return table.rows[0].get(column) or None
    return None


def _table_cases(table: Document, with_math: bool) -> list[_Candidate]:
    rows = table.rows or []
    columns = table.columns or []
    if len(rows) < 2 or len(columns) < 2:
        return []
    key = columns[0]
    numeric = [c for c in columns[1:] if all(_number(r.get(c, "")) is not None for r in rows)]
    if not numeric:
        return []
    index = len(rows) // 2
    row = rows[index]
    page_text = table.pages[index].text
    column = numeric[0]
    cases = [
        _Candidate(
            EvalCase(
                id="",
                category="tool",
                question=f"What is the {_human(column)} for {_human(key)} {row[key]}?",
                sources=[ExpectedSource(doc_id=table.doc_id, quote=page_text)],
                key_facts=[row[column]],
                expected_tool="csv_lookup",
                origin="template",
                rationale=f"table lookup in {table.doc_id} row {index + 1}",
            ),
            [page_text],
        )
    ]
    money = next((c for c in numeric if _MONEY.search(c)), None)
    if with_math and money:
        value = _number(row[money]) or 0.0
        units = 3
        total = format_number(value * units)
        per_day = "day" in money.lower() or "daily" in money.lower()
        unit_word = "days" if per_day else "units"
        cases.append(
            _Candidate(
                EvalCase(
                    id="",
                    category="tool",
                    question=(
                        f"What is the total {_human(money)} for {units} {unit_word} at "
                        f"{_human(key)} {row[key]}?"
                    ),
                    sources=[ExpectedSource(doc_id=table.doc_id, quote=page_text)],
                    key_facts=[total],
                    expected_tool="calculator",
                    origin="template",
                    rationale=f"{units} x {row[money]} = {total} (computed by code)",
                ),
                [page_text],
            )
        )
    return cases


# ---------- Split + assembly ----------------------------------------------------------


def assign_splits(cases: list[EvalCase]) -> list[EvalCase]:
    """Stratified, deterministic dev/holdout split."""
    by_category: dict[str, list[EvalCase]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)
    out: list[EvalCase] = []
    for _category, items in sorted(by_category.items()):
        items.sort(key=lambda c: _stable_key(c.question))
        n_dev = max(1, math.ceil(len(items) * DEV_FRACTION)) if len(items) > 1 else 1
        out += [
            c.model_copy(update={"split": "dev" if i < n_dev else "holdout"})
            for i, c in enumerate(items)
        ]
    return out


_ORDER: dict[str, int] = {
    "answerable": 0,
    "ambiguous": 1,
    "multi_doc": 2,
    "tool": 3,
    "out_of_scope": 4,
}


async def generate_evals(
    ctx: ForgeContext, req: Requirements, corpus: Corpus, targets: EvalTargets | None = None
) -> EvalSet:
    targets = targets or EvalTargets()
    chunks = chunk_corpus(corpus, ChunkingConfig())  # reference chunking, agent-independent
    rejected: list[RejectedCase] = []

    total_single = targets.answerable + targets.ambiguous + targets.refill
    pool = select_passages(corpus, chunks, total_single)
    answer_pool = pool[: targets.answerable]
    ambiguous_pool = pool[targets.answerable : targets.answerable + targets.ambiguous]
    refill_pool = pool[targets.answerable + targets.ambiguous :]
    if len(ambiguous_pool) < targets.ambiguous:  # small corpus: reuse passages
        ambiguous_pool = answer_pool[: targets.ambiguous]

    with stage("evalgen"):
        jobs = [
            *(
                _single_passage_case(ctx, req, corpus, c, "answerable", rejected)
                for c in answer_pool
            ),
            *(
                _single_passage_case(ctx, req, corpus, c, "ambiguous", rejected)
                for c in ambiguous_pool
            ),
            *(
                _multi_doc_case(ctx, req, corpus, a, b, rejected)
                for a, b in _pairs(corpus, pool, targets.multi_doc)
            ),
        ]
        singles = [c for c in await asyncio.gather(*jobs) if c is not None]
        refusals = await _refusal_cases(ctx, req, chunks, targets.out_of_scope, rejected)
        candidates = _dedupe([*singles, *template_cases(req, corpus), *refusals], rejected)

        if len(candidates) < MIN_CASES and refill_pool:
            extra = await asyncio.gather(
                *(
                    _single_passage_case(ctx, req, corpus, c, "answerable", rejected)
                    for c in refill_pool
                )
            )
            candidates = _dedupe([*candidates, *(c for c in extra if c is not None)], rejected)

    ordered = sorted(candidates, key=lambda c: _ORDER[c.case.category])
    cases = [c.case.model_copy(update={"id": f"c{i:02d}"}) for i, c in enumerate(ordered, 1)]
    return EvalSet(
        cases=sorted(assign_splits(cases), key=lambda c: c.id),
        rejected=rejected,
        prompts=["evals_answerable", "evals_ambiguous", "evals_multi_doc", "evals_out_of_scope"],
    )


def evals_markdown(evals: EvalSet) -> str:
    counts = evals.counts()
    lines = [
        "# Generated eval set",
        "",
        f"{len(evals.cases)} cases kept, {len(evals.rejected)} rejected by validation. "
        f"Dev {len(evals.split('dev'))} / holdout {len(evals.split('holdout'))}.",
        "",
        "Categories: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "| ID | Split | Category | Question | Expected |",
        "|---|---|---|---|---|",
    ]
    for c in evals.cases:
        if c.expect_refusal:
            expected = "refuse"
        elif c.expected_tool and not c.key_facts:
            expected = f"call {c.expected_tool}"
        else:
            expected = "; ".join(c.key_facts) + (
                f" (via {c.expected_tool})" if c.expected_tool else ""
            )
        lines.append(
            f"| {c.id} | {c.split} | {c.category} | {c.question.replace('|', '/')} | {expected.replace('|', '/')} |"
        )
    if evals.rejected:
        lines += [
            "",
            "## Rejected candidates",
            "",
            "| Category | Question | Reason |",
            "|---|---|---|",
        ]
        lines += [
            f"| {r.category} | {r.question.replace('|', '/') or '-'} | {r.reason.replace('|', '/')} |"
            for r in evals.rejected
        ]
    return "\n".join(lines) + "\n"
