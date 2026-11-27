"""Stage 2 — groundability guard: every documents topic in the requirements must be backed
by a passage, proven with an exact quote that code finds in that passage. Topics that fail
are flagged and moved to must-refuse, so the generated agent refuses instead of guessing.
"""

from __future__ import annotations

import asyncio

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
from forgeai.textutil import contains_quote

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
    return verify_judgement(topic, judgement, hits)


def _tool_checks(req: Requirements, corpus: Corpus) -> list[GroundingCheck]:
    """Tool topics are not checked against prose, but a table lookup needs a table."""
    has_table = any(d.kind == "csv" for d in corpus.documents)
    checks: list[GroundingCheck] = []
    for topic in req.must_answer:
        if topic.answered_from != "tool" or topic.tool_hint != "table_lookup" or has_table:
            continue
        checks.append(
            GroundingCheck(
                topic=topic.topic,
                claimed="supported",
                verdict="unsupported",
                note="needs a table lookup, but the folder contains no CSV table",
            )
        )
    return checks


async def check_groundability(
    ctx: ForgeContext, req: Requirements, corpus: Corpus, index: HybridIndex
) -> GroundabilityReport:
    with stage("guard"):
        checks = await asyncio.gather(*(check_topic(ctx, index, t) for t in req.document_topics))
    return GroundabilityReport(checks=[*checks, *_tool_checks(req, corpus)])


def apply_guard(req: Requirements, report: GroundabilityReport) -> Requirements:
    """Move unsupported topics from must-answer to must-refuse and record them as gaps."""
    failed = {c.topic: c for c in report.unsupported}
    if not failed:
        return req
    kept = [t for t in req.must_answer if t.topic not in failed]
    if not kept:
        raise GroundingError(
            "the documents support none of the topics the brief asks for; "
            "no agent was generated (see groundability.json)"
        )
    refusals = list(req.must_refuse)
    known = {r.topic for r in refusals}
    gaps: list[FlaggedGap] = list(req.flagged_gaps)
    for topic, check in failed.items():
        reason = GAP_REASON if not check.note else f"{GAP_REASON} Detail: {check.note}"
        gaps.append(FlaggedGap(topic=topic, reason=reason))
        if topic not in known:
            refusals.append(RefuseTopic(topic=topic, reason=GAP_REASON))
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
