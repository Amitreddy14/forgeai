"""Stage 2 — groundability guard: every documents topic in the requirements must be backed
by a passage, proven with an exact quote that code finds in that passage. Topics that fail
are flagged and moved to must-refuse, so the generated agent refuses instead of guessing.
"""

from __future__ import annotations

import asyncio
import re

from forgeai.errors import GroundingError
from forgeai.llm import ChatMessage, complete_json, stage
from forgeai.prompts import render
from forgeai.retrieval import Hit, HybridIndex, RetrievalConfig
from forgeai.schemas.corpus import Corpus
from forgeai.schemas.requirements import (
    FlaggedGap,
    GroundabilityReport,
    GroundingCheck,
    RefuseTopic,
    Requirements,
    Topic,
    TopicJudgement,
)
from forgeai.stages.context import ForgeContext
from forgeai.textutil import contains_quote, content_stems

GUARD_TOP_K = 5
PASSAGE_CHARS = 1200
GAP_REASON = "The documents do not cover this (flagged by the groundability guard)."


def _passages(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{h.chunk.chunk_id}] ({h.chunk.citation})\n{h.chunk.text[:PASSAGE_CHARS]}" for h in hits
    )


def verify_judgement(topic: Topic, judgement: TopicJudgement, hits: list[Hit]) -> GroundingCheck:
    """Believe a 'supported' or 'partial' claim only if its quote is really in a retrieved
    passage. The chunk id may be wrong (small models mix them up); the words may not."""
    retrieved = [h.chunk.chunk_id for h in hits]
    base = GroundingCheck(
        topic=topic.topic,
        claimed=judgement.status,
        verdict="unsupported",
        retrieved=retrieved,
        note=judgement.note,
    )
    if judgement.status == "unsupported":
        return base
    if not judgement.quote:
        return base.model_copy(update={"note": "claimed support without a quote"})

    ordered = sorted(hits, key=lambda h: h.chunk.chunk_id != judgement.chunk_id)
    for hit in ordered:
        if contains_quote(hit.chunk.text, judgement.quote):
            return base.model_copy(
                update={
                    "verdict": judgement.status,
                    "chunk_id": hit.chunk.chunk_id,
                    "quote": judgement.quote,
                    "quote_verified": True,
                }
            )
    return base.model_copy(
        update={
            "quote": judgement.quote,
            "note": "claimed support, but the quote is not in any retrieved passage",
        }
    )


_GENERIC_TOPIC_WORDS = frozenset(
    "policy policies process procedure procedures rule rules information general guideline "
    "guidelines detail details".split()
)
_SENTENCE = re.compile(r"[^.!?]+[.!?]")


def section_support(topic: Topic, hits: list[Hit]) -> GroundingCheck | None:
    """Second look at a topic the model called unsupported: a retrieved passage whose
    section heading names the topic ('Splitting orders') does support it. Needs at least
    two content words on the matching side, so one generic word never decides it. The
    quote is the section's first sentence, so it is verbatim by construction."""
    wanted = content_stems(topic.topic, extra_stopwords=_GENERIC_TOPIC_WORDS)
    for hit in hits:
        heading = hit.chunk.heading or ""
        have = content_stems(heading, extra_stopwords=_GENERIC_TOPIC_WORDS)
        matched = (len(wanted) >= 2 and wanted <= have) or (len(have) >= 2 and have <= wanted)
        if not matched:
            continue
        body = hit.chunk.text.lstrip("# ")
        if body.startswith(heading):
            body = body[len(heading) :]
        first = _SENTENCE.search(body)
        quote = first.group(0).strip() if first else ""
        if quote and contains_quote(hit.chunk.text, quote):
            return GroundingCheck(
                topic=topic.topic,
                claimed="unsupported",
                verdict="supported",
                chunk_id=hit.chunk.chunk_id,
                quote=quote,
                quote_verified=True,
                retrieved=[h.chunk.chunk_id for h in hits],
                note=f'the guard model said unsupported, but section "{heading}" covers it',
            )
    return None


async def check_topic(ctx: ForgeContext, index: HybridIndex, topic: Topic) -> GroundingCheck:
    query = f"{topic.topic}. {topic.detail}".strip()
    hits = index.search(query, RetrievalConfig(top_k=GUARD_TOP_K))
    if not hits:
        return GroundingCheck(
            topic=topic.topic,
            claimed="unsupported",
            verdict="unsupported",
            note="no passages retrieved",
        )
    prompt = render(
        "guard_topic", topic=topic.topic, detail=topic.detail or "-", passages=_passages(hits)
    )
    judgement = await complete_json(
        ctx.llm,
        [ChatMessage(role="user", content=prompt)],
        TopicJudgement,
        model=ctx.models.generator,
    )
    check = verify_judgement(topic, judgement, hits)
    # Only an honest "unsupported" gets the second look. A claimed quote that is not in
    # the passages is a fabrication, and the topic stays flagged.
    if judgement.status == "unsupported":
        rescued = section_support(topic, hits)
        if rescued is not None:
            return rescued
    return check


def _tool_checks(req: Requirements, corpus: Corpus) -> list[GroundingCheck]:
    """Tool topics are not checked against prose, but a table lookup needs a table."""
    has_table = any(d.kind == "csv" for d in corpus.documents)
    checks: list[GroundingCheck] = []
    for topic in req.must_answer:
        if "table_lookup" not in topic.tool_hints or has_table:
            continue
        checks.append(
            GroundingCheck(
                topic=topic.topic,
                kind="table",
                claimed="supported",
                verdict="unsupported",
                note="needs a table lookup, but the folder contains no CSV table",
            )
        )
    return checks


# Asks that need records about individual people. General policy text never answers
# them, however closely related it sounds ("24 days a year" is not "6 days left").
PERSONAL_RECORDS = re.compile(
    r"\b(days?|leave|balance)\s+(left|remaining)\b|\bremaining\s+(leave|days|balance)\b"
    r"|\b(leave|account|salary|pay)\s+balance\b|\beach employee'?s?\b|\bindividual employee'?s?\b"
    r"|\bpersonal (record|records|data|history)\b|\b(employee|staff|crew) records?\b"
    r"|\bhow many .{0,40}\b(left|remaining)\b|\bremaining\b.{0,20}\bbalance\b",
    re.IGNORECASE,
)


def _personal_record_checks(req: Requirements, corpus: Corpus) -> list[GroundingCheck]:
    has_people_table = any(
        d.kind == "csv" and any(re.search(r"employee|staff|name", c, re.I) for c in d.columns or [])
        for d in corpus.documents
    )
    if has_people_table:
        return []
    return [
        GroundingCheck(
            topic=t.topic,
            kind="personal_records",
            claimed="supported",
            verdict="unsupported",
            note=(t.detail or t.topic)
            + " needs records about individual people; the documents contain only general rules",
        )
        for t in req.must_answer
        if PERSONAL_RECORDS.search(f"{t.topic}. {t.detail}")
    ]


async def check_groundability(
    ctx: ForgeContext, req: Requirements, corpus: Corpus, index: HybridIndex
) -> GroundabilityReport:
    personal = _personal_record_checks(req, corpus)
    personal_topics = {c.topic for c in personal}
    prose = [t for t in req.document_topics if t.topic not in personal_topics]
    with stage("guard"):
        checks = await asyncio.gather(*(check_topic(ctx, index, t) for t in prose))
    return GroundabilityReport(checks=[*checks, *personal, *_tool_checks(req, corpus)])


def apply_guard(req: Requirements, report: GroundabilityReport) -> Requirements:
    """Move unsupported topics from must-answer to must-refuse and record them as gaps.

    A topic flagged for needing personal records is split instead: the general topic
    stays answerable (its detail is dropped), and the personal ask becomes a refusal.
    """
    failed = {c.topic: c for c in report.unsupported}
    if not failed:
        return req
    kept: list[Topic] = []
    refusals = list(req.must_refuse)
    known = {r.topic for r in refusals}
    gaps: list[FlaggedGap] = list(req.flagged_gaps)
    for topic in req.must_answer:
        check = failed.get(topic.topic)
        if check is None:
            kept.append(topic)
            continue
        if check.kind == "personal_records":
            ask = topic.detail.strip().rstrip(".") or topic.topic
            gaps.append(FlaggedGap(topic=ask, reason=f"{GAP_REASON} Detail: {check.note}"))
            if ask not in known:
                refusals.append(RefuseTopic(topic=ask, reason="needs individual people's records"))
            # Keep the general topic ("annual leave entitlement") only if its own name is
            # not itself the personal ask ("remaining leave balance").
            if topic.detail and not PERSONAL_RECORDS.search(topic.topic):
                kept.append(topic.model_copy(update={"detail": ""}))
            continue
        reason = GAP_REASON if not check.note else f"{GAP_REASON} Detail: {check.note}"
        gaps.append(FlaggedGap(topic=topic.topic, reason=reason))
        if topic.topic not in known:
            refusals.append(RefuseTopic(topic=topic.topic, reason=GAP_REASON))
    if not kept:
        raise GroundingError(
            "the documents support none of the topics the brief asks for; "
            "no agent was generated (see groundability.json)"
        )
    return req.model_copy(
        update={"must_answer": kept, "must_refuse": refusals, "flagged_gaps": gaps}
    )


def groundability_markdown(report: GroundabilityReport) -> str:
    lines = [
        "# Groundability check",
        "",
        f"Coverage: {report.coverage:.0%} of checked topics are supported by the documents.",
        "",
        "| Topic | Model claimed | Verdict | Evidence |",
        "|---|---|---|---|",
    ]
    for c in report.checks:
        verified = c.quote_verified and c.quote
        evidence = f'`{c.chunk_id}`: "{c.quote}"' if verified else (c.note or "-")
        lines.append(
            f"| {c.topic} | {c.claimed} | **{c.verdict}** | {evidence.replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"
