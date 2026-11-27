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
from forgeai.retrieval import Embedder, HybridIndex, RetrievalConfig
from forgeai.schemas.corpus import Chunk, ChunkingConfig, Corpus, Document
from forgeai.schemas.evals import Category, EvalCase, EvalSet, ExpectedSource, RejectedCase
from forgeai.schemas.requirements import Requirements
from forgeai.stages.context import ForgeContext
from forgeai.stages.guard import PERSONAL_RECORDS
from forgeai.textutil import contains_fact, contains_quote, normalize, word_jaccard
from forgeai.tools.calculator import format_number

MIN_CASES = 20
MAX_FACT_WORDS = 10
_CONTEXT_DEPENDENT = re.compile(
    r"\b(this|that|the above|the following|the given|the attached)\s+"
    r"(policy|document|passage|text|section|file|excerpt)\b",
    re.IGNORECASE,
)
_GENERIC = frozenset(
    (
        "the a an and or of for to in on at by with is are be can may must what which who how "
        "many much does do per any all staff employee employees policy inr usd eur"
    ).split()
)
DEV_FRACTION = 0.6
DUPLICATE_JACCARD = 0.7
MIN_PASSAGE_WORDS = 25
PASSAGE_CHARS = 1500


@dataclass(frozen=True)
class EvalTargets:
    answerable: int = 12
    ambiguous: int = 3
    multi_doc: int = 4
    out_of_scope: int = 7
    refill: int = 8  # extra answerable attempts if validation leaves too few cases


class _Answerable(BaseModel):
    question: str = Field(min_length=8)
    key_facts: list[str] = Field(min_length=1, max_length=3)
    quote: str


class _Refusal(BaseModel):
    question: str = Field(min_length=8)
    topic: str = ""
    why: str = ""


class _Refusals(BaseModel):
    questions: list[_Refusal]


class _Answerability(BaseModel):
    answered: bool = False
    chunk_id: str | None = None
    quote: str | None = None


async def answered_by_documents(
    ctx: ForgeContext, index: HybridIndex, question: str
) -> tuple[bool, str]:
    """Is this would-be refusal actually answered by the documents? Believed only with a
    quote that code finds in a retrieved passage."""
    hits = index.search(question, RetrievalConfig(top_k=4))
    if not hits:
        return False, ""
    passages = "\n\n".join(f"[{h.chunk.chunk_id}]\n{h.chunk.text[:PASSAGE_CHARS]}" for h in hits)
    prompt = render("evals_answerability", question=question, passages=passages)
    try:
        out = await complete_json(
            ctx.llm,
            [ChatMessage(role="user", content=prompt)],
            _Answerability,
            model=ctx.models.generator,
        )
    except ModelOutputError:
        return False, ""
    if not out.answered or not out.quote:
        return False, ""
    for hit in hits:
        if contains_quote(hit.chunk.text, out.quote):
            return True, f'{hit.chunk.chunk_id}: "{out.quote}"'
    return False, ""  # claimed an answer it could not quote: keep the case


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


# ---------- Validation ----------------------------------------------------------------


def _content_words(text: str) -> set[str]:
    return {w for w in normalize(text).split() if w not in _GENERIC and not w.isdigit()}


def clean_facts(question: str, facts: list[str]) -> list[str]:
    """Keep facts that are short and answer what was asked.

    Small models pad key facts with whole sentences or with rules the question never
    asked about; such facts would fail a correct answer. The first fact is the direct
    answer; later facts must share at least two content words with the question.
    """
    out: list[str] = []
    q_words = _content_words(question)
    for i, raw in enumerate(facts):
        fact = raw.strip().rstrip(".;").strip()
        if not fact or len(fact.split()) > MAX_FACT_WORDS:
            continue
        if i > 0 and len(_content_words(fact) & q_words) < 2:
            continue
        out.append(fact)
    return out


def validate_candidate(
    category: Category,
    question: str,
    facts: list[str],
    quotes: list[tuple[str, str, str]],  # (doc_id, quote, passage_text)
) -> str | None:
    """Return a rejection reason, or None if the case is sound."""
    if len(question.strip()) < 8:
        return "question too short"
    if _CONTEXT_DEPENDENT.search(question):
        return "question depends on unseen context (e.g. 'this policy')"
    if category != "out_of_scope" and not facts:
        return "no short, relevant key fact"
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
    facts = clean_facts(out.question, out.key_facts)
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


async def _refusal_cases(
    ctx: ForgeContext,
    req: Requirements,
    chunks: list[Chunk],
    count: int,
    rejected: list[RejectedCase],
    similarity_floor: float | None = None,
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
    items = [i for i in out.questions[: count + 2] if not _CONTEXT_DEPENDENT.search(i.question)]
    index = HybridIndex(chunks, ctx.embedder)
    checks = await asyncio.gather(*(answered_by_documents(ctx, index, i.question) for i in items))
    candidates: list[_Candidate] = []
    for item, (answered, evidence) in zip(items, checks, strict=True):
        named = in_scope_topic(item.question, req)
        if not answered and named is not None:
            answered, evidence = True, f'names the must-answer topic "{named}"'
        closeness = best_similarity(index, item.question)
        if not answered and similarity_floor is not None and closeness >= similarity_floor:
            # Backstop for a lenient model: the question sits as close to the documents
            # as a typical answerable question does, so it is probably answerable.
            answered = True
            evidence = (
                f"similarity {closeness:.2f} to the documents is at or above the "
                f"answerable-question floor {similarity_floor:.2f}"
            )
        if answered:
            # The model wrote a question the documents DO answer: not out of scope.
            rejected.append(
                RejectedCase(
                    category="out_of_scope",
                    question=item.question,
                    reason=f"answerable from the documents ({evidence})",
                )
            )
            continue
        case = EvalCase(
            id="",
            category="out_of_scope",
            question=item.question.strip(),
            expect_refusal=True,
            rationale=f"{item.topic}: {item.why}".strip(": "),
        )
        candidates.append(_Candidate(case, []))
    return candidates


_TOPIC_FILLER = frozenset(
    "policy policies process procedure procedures rules rule information general guidelines "
    "details detail questions question".split()
)
# Signs that a question about an in-scope topic still asks for something to refuse:
# personal data, the future, or an action instead of an explanation.
_REFUSAL_CUE = re.compile(
    r"\b(my|me|mine|i|colleague'?s?|next|future|upcoming|plans?|planned|recent|changes?|changed|"
    r"will|book|submit|approve|status)\b",
    re.IGNORECASE,
)


def _stem_word(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def in_scope_topic(question: str, req: Requirements) -> str | None:
    """The must-answer topic a would-be refusal plainly asks about, if any.

    "What are the HR policies for remote work?" names the must-answer topic "Remote work
    policy" and asks for nothing personal, future or actionable: it is in scope, whatever
    the generator labelled it. Questions with a refusal cue are left alone.
    """
    if _REFUSAL_CUE.search(question) or PERSONAL_RECORDS.search(question):
        return None
    q_words = {_stem_word(w) for w in normalize(question).split()}
    for topic in req.must_answer:
        words = {
            _stem_word(w)
            for w in normalize(topic.topic).split()
            if w not in _GENERIC and w not in _TOPIC_FILLER
        }
        if words and words <= q_words:
            return topic.topic
    return None


def best_similarity(index: HybridIndex, question: str) -> float:
    hits = index.search(question, RetrievalConfig(top_k=1, mode="dense"))
    return hits[0].dense if hits else 0.0


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _join_questions(first: str, second: str) -> str:
    second = second.strip()
    return f"{first.strip().rstrip('?. ')}? Also, {second[:1].lower()}{second[1:]}"


def compose_multi_doc(
    singles: list[_Candidate], limit: int, embedder: Embedder
) -> tuple[list[_Candidate], list[_Candidate]]:
    """Build two-document cases from verified single-document cases.

    Asking a 7B model to invent a question that needs two passages produced forced
    pairings ("compare the monsoon draft limit with port dues"). Joining two verified
    questions from different documents gives a case whose expected facts are already
    proven, and which can only be answered fully by retrieving both documents.
    Returns (two-document cases, singles not used as components).
    """
    if limit <= 0 or len(singles) < 2:
        return [], singles
    vectors = embedder.embed_documents([c.case.question for c in singles])
    pairs = [
        (float(vectors[i] @ vectors[j]), i, j)
        for i in range(len(singles))
        for j in range(i + 1, len(singles))
        if singles[i].case.sources[0].doc_id != singles[j].case.sources[0].doc_id
    ]
    pairs.sort(key=lambda p: (-round(p[0], 4), p[1], p[2]))
    used: set[int] = set()
    composed: list[_Candidate] = []
    for _score, i, j in pairs:
        if i in used or j in used:
            continue
        a, b = singles[i].case, singles[j].case
        composed.append(
            _Candidate(
                EvalCase(
                    id="",
                    category="multi_doc",
                    question=_join_questions(a.question, b.question),
                    sources=[*a.sources, *b.sources],
                    key_facts=[a.key_facts[0], b.key_facts[0]],
                    rationale=f"joins two verified questions from {a.sources[0].doc_id} "
                    f"and {b.sources[0].doc_id}",
                ),
                [*singles[i].passage_texts, *singles[j].passage_texts],
            )
        )
        used |= {i, j}
        if len(composed) == limit:
            break
    return composed, [c for k, c in enumerate(singles) if k not in used]


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
        # At least one case on each side whenever a category has two or more.
        n_dev = (
            min(len(items) - 1, max(1, math.ceil(len(items) * DEV_FRACTION)))
            if len(items) > 1
            else 1
        )
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
    answer_pool = [*pool[: targets.answerable], *pool[targets.answerable + targets.ambiguous :]]
    ambiguous_pool = pool[targets.answerable : targets.answerable + targets.ambiguous]
    if len(ambiguous_pool) < targets.ambiguous:  # small corpus: reuse passages
        ambiguous_pool = answer_pool[: targets.ambiguous]
    index = HybridIndex(chunks, ctx.embedder)

    with stage("evalgen"):
        answerable = [
            c
            for c in await asyncio.gather(
                *(
                    _single_passage_case(ctx, req, corpus, c, "answerable", rejected)
                    for c in answer_pool
                )
            )
            if c is not None
        ]
        ambiguous = [
            c
            for c in await asyncio.gather(
                *(
                    _single_passage_case(ctx, req, corpus, c, "ambiguous", rejected)
                    for c in ambiguous_pool
                )
            )
            if c is not None
        ]
        answerable = _dedupe(answerable, rejected)
        # Surplus answerable cases become components of two-document cases first.
        surplus = answerable[targets.answerable :]
        kept = answerable[: targets.answerable]
        multi, _unused = compose_multi_doc(surplus, targets.multi_doc, ctx.embedder)
        if len(multi) < targets.multi_doc:
            # Small corpus: also join kept cases, which stay in the set on their own too.
            more, _ = compose_multi_doc(kept, targets.multi_doc - len(multi), ctx.embedder)
            multi += more

        scores = [best_similarity(index, c.case.question) for c in [*answerable, *ambiguous]]
        floor = _percentile(scores, 0.25) if len(scores) >= 4 else None
        refusals = await _refusal_cases(
            ctx, req, chunks, targets.out_of_scope, rejected, similarity_floor=floor
        )
        candidates = _dedupe(
            [*kept, *ambiguous, *multi, *template_cases(req, corpus), *refusals], rejected
        )

    ordered = sorted(candidates, key=lambda c: _ORDER[c.case.category])
    cases = [c.case.model_copy(update={"id": f"c{i:02d}"}) for i, c in enumerate(ordered, 1)]
    return EvalSet(
        cases=sorted(assign_splits(cases), key=lambda c: c.id),
        rejected=rejected,
        prompts=[
            "evals_answerable",
            "evals_ambiguous",
            "evals_out_of_scope",
            "evals_answerability",
        ],
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
