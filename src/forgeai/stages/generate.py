"""Stage 3 — agent generator: requirements + documents -> a declarative AgentDefinition.

The model writes what models are good at (the role prompt, domain rules, the refusal
line) and proposes tools. Code sets what a 7B model cannot reason about reliably:
chunk size from the documents' real section lengths, top-k from corpus shape, and a
refusal threshold calibrated on the requirements' own topics. Tool proposals pass
through the evidence contracts in `tools.registry`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from forgeai import __version__
from forgeai.ingest import chunk_corpus
from forgeai.llm import ChatMessage, complete_json, stage
from forgeai.prompts import render
from forgeai.retrieval import HybridIndex, RetrievalConfig
from forgeai.runtime.loader import fingerprint_folder
from forgeai.schemas.agent import AgentDefinition, KnowledgeSource, Provenance, RefusalPolicy
from forgeai.schemas.corpus import ChunkingConfig, Corpus
from forgeai.schemas.requirements import IntakeRecord, Requirements
from forgeai.stages.context import ForgeContext
from forgeai.stages.intake import summarize_corpus, summary_text
from forgeai.tools import REGISTRY, ToolProposal, ToolSelection, select_tools, tool_card

MIN_CHUNK, MAX_CHUNK = 120, 400


class AgentDraft(BaseModel):
    role_prompt: str = Field(min_length=40)
    rules: list[str] = Field(default_factory=list)
    refusal_message: str = Field(min_length=10)
    tools: list[ToolProposal] = Field(default_factory=list)


@dataclass
class Calibration:
    threshold: float
    answer_scores: list[float]
    refuse_scores: list[float]
    note: str


# ---------- Deterministic parameter choices -------------------------------------------


def choose_chunking(corpus: Corpus) -> tuple[ChunkingConfig, str]:
    """Size chunks to the documents' typical section, so one chunk ~ one section."""
    sections = chunk_corpus(corpus, ChunkingConfig(chunk_words=1200, overlap_words=0))
    prose_ids = {d.doc_id for d in corpus.documents if d.kind != "csv"}
    prose = [len(c.text.split()) for c in sections if c.doc_id in prose_ids]
    if not prose:
        return ChunkingConfig(), "no prose documents; default chunking"
    median = statistics.median(prose)
    words = int(min(MAX_CHUNK, max(MIN_CHUNK, round(median * 1.2 / 20) * 20)))
    config = ChunkingConfig(chunk_words=words, overlap_words=max(10, words // 6))
    note = f"median section {median:.0f} words -> chunk {words}, overlap {config.overlap_words}"
    return config, note


def choose_retrieval(corpus: Corpus) -> tuple[RetrievalConfig, str]:
    docs = len(corpus.documents)
    has_csv = any(d.kind == "csv" for d in corpus.documents)
    top_k = min(8, 4 + (1 if docs > 2 else 0) + (1 if has_csv else 0))
    note = f"{docs} documents{' incl. CSV' if has_csv else ''} -> top_k {top_k}"
    return RetrievalConfig(top_k=top_k, mode="hybrid"), note


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def calibrate_threshold(req: Requirements, index: HybridIndex) -> Calibration:
    """Find a similarity floor that separates must-answer topics from must-refuse topics.

    If the two groups overlap, the threshold is disabled: refusal is left to the model
    rather than refusing real questions on a guess.
    """
    config = RetrievalConfig(top_k=1, mode="dense")

    def best(query: str) -> float:
        hits = index.search(query, config)
        return hits[0].dense if hits else 0.0

    answer = [best(f"{t.topic}. {t.detail}") for t in req.document_topics]
    refuse = [best(f"{t.topic}. {t.reason}") for t in req.must_refuse]
    if not answer or not refuse:
        return Calibration(0.0, answer, refuse, "not enough topics to calibrate; threshold off")
    low_answer = _percentile(answer, 0.2)
    high_refuse = _percentile(refuse, 0.8)
    if low_answer <= high_refuse + 0.05:
        return Calibration(
            0.0,
            answer,
            refuse,
            f"topics overlap (answer p20 {low_answer:.2f} vs refuse p80 {high_refuse:.2f}); "
            "threshold off, refusal left to the model",
        )
    # Sit closer to the refuse side: a false refusal is worse than a model-handled one.
    threshold = round(high_refuse + 0.3 * (low_answer - high_refuse), 3)
    return Calibration(
        threshold,
        answer,
        refuse,
        f"answer p20 {low_answer:.2f}, refuse p80 {high_refuse:.2f} -> threshold {threshold:.2f}",
    )


# ---------- Generation ----------------------------------------------------------------


def _requirements_text(req: Requirements) -> str:
    lines = [
        f"Name: {req.agent_name}",
        f"Purpose: {req.purpose}",
        f"Scope: {req.scope}",
        f"Audience: {req.audience}",
        f"Tone: {req.tone}",
        "Must answer:",
        *[
            f"- {t.topic}: {t.detail}"
            + (f" [needs tool capabilities: {', '.join(t.tool_hints)}]" if t.tool_hints else "")
            for t in req.must_answer
        ],
        "Must refuse:",
        *[f"- {t.topic}: {t.reason}" for t in req.must_refuse],
    ]
    return "\n".join(lines)


@dataclass
class GeneratedAgent:
    definition: AgentDefinition
    draft: AgentDraft
    selection: ToolSelection
    calibration: Calibration


async def generate_agent(
    ctx: ForgeContext,
    req: Requirements,
    record: IntakeRecord,
    corpus: Corpus,
    docs_dir: Path,
) -> GeneratedAgent:
    chunking, chunk_note = choose_chunking(corpus)
    retrieval, retrieval_note = choose_retrieval(corpus)

    prompt = render(
        "generate_agent",
        requirements=_requirements_text(req),
        corpus_summary=summary_text(summarize_corpus(corpus, chunk_corpus(corpus, chunking))),
        tools="\n".join(tool_card(t) for t in REGISTRY.values()),
    )
    with stage("generate"):
        draft = await complete_json(
            ctx.llm,
            [ChatMessage(role="user", content=prompt)],
            AgentDraft,
            model=ctx.models.generator,
            max_tokens=1500,
        )

    requester_answers = [a.answer for a in record.answers if a.source == "requester"]
    selection = select_tools(draft.tools, req, corpus, requester_answers)

    index = HybridIndex(chunk_corpus(corpus, chunking), ctx.embedder)
    calibration = calibrate_threshold(req, index)

    rules = [r.strip() for r in draft.rules if r.strip()][:6]
    definition = AgentDefinition(
        name=req.agent_name,
        description=req.purpose,
        audience=req.audience,
        tone=req.tone,
        role_prompt=draft.role_prompt.strip(),
        rules=rules,
        must_answer=[t.topic for t in req.must_answer],
        chunking=chunking,
        retrieval=retrieval,
        tools=selection.attached,
        rejected_tools=selection.rejected,
        refusal=RefusalPolicy(
            refuse_topics=[t.topic for t in req.must_refuse],
            message=draft.refusal_message.strip(),
            min_evidence_score=calibration.threshold,
        ),
        knowledge=KnowledgeSource(
            docs_dir=docs_dir.as_posix(), fingerprints=fingerprint_folder(docs_dir)
        ),
        provenance=Provenance(
            factory_version=__version__,
            generator_model=ctx.models.generator,
            notes=[chunk_note, retrieval_note, calibration.note],
        ),
    )
    return GeneratedAgent(definition, draft, selection, calibration)
