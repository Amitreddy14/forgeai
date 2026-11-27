"""Runs the stages in order and writes every artifact to the run folder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from forgeai.errors import GroundingError
from forgeai.ingest import chunk_corpus, load_folder
from forgeai.retrieval import HybridIndex
from forgeai.schemas.requirements import GroundabilityReport, IntakeRecord, Requirements
from forgeai.stages.context import ForgeContext
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
