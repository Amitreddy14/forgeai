"""Runs the stages in order and writes every artifact to the run folder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from forgeai.errors import GroundingError
from forgeai.ingest import chunk_corpus, load_folder
from forgeai.retrieval import HybridIndex
from forgeai.runtime.agent import system_prompt
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.requirements import GroundabilityReport, IntakeRecord, Requirements
from forgeai.stages.context import ForgeContext
from forgeai.stages.generate import GeneratedAgent, generate_agent
from forgeai.stages.guard import apply_guard, check_groundability, groundability_markdown
from forgeai.stages.intake import Answerer, requirements_markdown, run_intake
from forgeai.store import RunStore

ApproveFn = Callable[[Requirements], Requirements]


@dataclass
class RequirementsOutcome:
    record: IntakeRecord
    requirements: Requirements
    groundability: GroundabilityReport


async def build_requirements(
    ctx: ForgeContext,
    store: RunStore,
    brief: str,
    docs_dir: Path,
    answerer: Answerer,
    *,
    approve: ApproveFn | None = None,
) -> RequirementsOutcome:
    """Intake -> groundability guard -> (optional human checkpoint) -> requirements."""
    brief = brief.strip()
    if len(brief) < 20:
        raise ValueError("the brief is too short: describe the assistant in a sentence or more")
    corpus = load_folder(docs_dir)
    if not corpus.documents:
        raise GroundingError(f"no readable documents in {docs_dir}")
    chunks = chunk_corpus(corpus)

    store.write_text("brief.md", brief)
    record, draft = await run_intake(ctx, brief, corpus, chunks, answerer)
    store.write_json("intake.json", record)
    store.write_json("requirements.draft.json", draft)

    index = HybridIndex(chunks, ctx.embedder)
    report = await check_groundability(ctx, draft, corpus, index)
    store.write_json("groundability.json", report)
    store.write_text("groundability.md", groundability_markdown(report))

    requirements = apply_guard(draft, report)
    if approve is not None:
        requirements = approve(requirements).model_copy(update={"approved_by": "requester"})
    else:
        requirements = requirements.model_copy(update={"approved_by": "auto"})
    store.write_json("requirements.json", requirements)
    store.write_text("requirements.md", requirements_markdown(requirements))
    return RequirementsOutcome(record=record, requirements=requirements, groundability=report)


def agent_markdown(gen: GeneratedAgent) -> str:
    d = gen.definition
    lines = [
        f"# Agent definition: {d.name} (v{d.version})",
        "",
        "## Tools",
        "",
    ]
    if d.tools:
        lines += [f"- **{t.name}**: {t.justification} _(evidence: {t.evidence})_" for t in d.tools]
    else:
        lines.append("- none attached: no requirement needs a tool")
    if d.rejected_tools:
        lines += ["", "Rejected proposals:", ""]
        lines += [f"- ~~{t.name}~~: {t.reason}" for t in d.rejected_tools]
    if gen.selection.added:
        lines += [
            "",
            f"Added by contract (generator omitted them): {', '.join(gen.selection.added)}",
        ]
    lines += [
        "",
        "## Retrieval",
        "",
        f"- chunking: {d.chunking.chunk_words} words, overlap {d.chunking.overlap_words}",
        f"- retrieval: {d.retrieval.mode}, top_k {d.retrieval.top_k}",
        f"- refusal threshold: {d.refusal.min_evidence_score:.2f}",
        *[f"- {note}" for note in d.provenance.notes],
        "",
        "## Compiled system prompt",
        "",
        "```text",
        system_prompt(d),
        "```",
    ]
    return "\n".join(lines) + "\n"


async def build_agent(
    ctx: ForgeContext,
    store: RunStore,
    outcome: RequirementsOutcome,
    docs_dir: Path,
) -> AgentDefinition:
    """Requirements -> agent.v0.json (+ a readable agent.v0.md)."""
    corpus = load_folder(docs_dir)
    generated = await generate_agent(ctx, outcome.requirements, outcome.record, corpus, docs_dir)
    store.write_json("agent.v0.json", generated.definition)
    store.write_json("tool_selection.json", generated.selection)
    store.write_text("agent.v0.md", agent_markdown(generated))
    return generated.definition


def load_requirements_outcome(store: RunStore) -> RequirementsOutcome:
    return RequirementsOutcome(
        record=store.read_json("intake.json", IntakeRecord),
        requirements=store.read_json("requirements.json", Requirements),
        groundability=store.read_json("groundability.json", GroundabilityReport),
    )
