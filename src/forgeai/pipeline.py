"""Runs the stages in order and writes every artifact to the run folder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from forgeai.errors import GroundingError
from forgeai.ingest import chunk_corpus, load_folder
from forgeai.llm import Budget, stage
from forgeai.retrieval import HybridIndex
from forgeai.runtime.agent import AgentRuntime, system_prompt
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import CanaryResult, EvalRun, EvalSet
from forgeai.schemas.patch import ChangeLogEntry
from forgeai.schemas.requirements import GroundabilityReport, IntakeRecord, Requirements
from forgeai.stages.context import ForgeContext
from forgeai.stages.evalgen import MIN_CASES, evals_markdown, generate_evals
from forgeai.stages.evaluate import SplitName, evaluate, scorecard_markdown
from forgeai.stages.generate import GeneratedAgent, generate_agent
from forgeai.stages.guard import apply_guard, check_groundability, groundability_markdown
from forgeai.stages.intake import Answerer, requirements_markdown, run_intake
from forgeai.stages.judge import run_canaries
from forgeai.stages.optimize import apply_patch, optimize
from forgeai.stages.report import (
    FinalScorecard,
    build_final_scorecard,
    changelog_markdown,
    report_markdown,
)
from forgeai.stages.tournament import TournamentResult, prompt_tournament
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


async def build_evals(
    ctx: ForgeContext,
    store: RunStore,
    requirements: Requirements,
    docs_dir: Path,
    *,
    min_cases: int = MIN_CASES,
) -> EvalSet:
    """Generate, validate and split the eval set; write evals.json + evals.md."""
    corpus = load_folder(docs_dir)
    evals = await generate_evals(ctx, requirements, corpus)
    store.write_json("evals.json", evals)
    store.write_text("evals.md", evals_markdown(evals))
    if len(evals.cases) < min_cases:
        raise GroundingError(
            f"only {len(evals.cases)} eval cases survived validation (need {min_cases}); "
            "see evals.md for the rejected candidates"
        )
    return evals


async def evaluate_agent(
    ctx: ForgeContext,
    store: RunStore,
    definition: AgentDefinition,
    evals: EvalSet,
    docs_dir: Path,
    split: SplitName,
    *,
    canaries: list[CanaryResult] | None = None,
    tag: str | None = None,
) -> EvalRun:
    corpus = load_folder(docs_dir)
    runtime = AgentRuntime(definition, corpus, ctx.embedder, ctx.llm, ctx.models.generator)
    run = await evaluate(ctx, runtime, evals, split, canaries=canaries)
    tag = tag or f"v{definition.version}-{split}"
    store.write_json(f"evals/{tag}.json", run)
    store.write_text(f"evals/{tag}.md", scorecard_markdown(run))
    return run


# ---------- The whole factory ---------------------------------------------------------


@dataclass
class FactoryResult:
    requirements: Requirements
    final: AgentDefinition
    scorecard: FinalScorecard
    report_path: Path


def _save_agent(store: RunStore, defn: AgentDefinition) -> None:
    store.write_json(f"agent.v{defn.version}.json", defn)
    store.write_text(
        f"agent.v{defn.version}.prompt.md", "```text\n" + system_prompt(defn) + "\n```"
    )


async def run_factory(
    ctx: ForgeContext,
    store: RunStore,
    brief: str,
    docs_dir: Path,
    answerer: Answerer,
    budget: Budget,
    *,
    approve: ApproveFn | None = None,
    max_iterations: int = 3,
    tournament: bool = False,
    resume: bool = False,
    min_cases: int = MIN_CASES,
    progress: Callable[[str], None] = lambda _m: None,
) -> FactoryResult:
    """Brief + documents -> requirements -> agent -> evals -> optimize -> report.
    With `resume`, stages whose artifacts already exist are loaded instead of re-run."""
    if resume and store.exists("requirements.json"):
        progress("requirements: reusing requirements.json")
        outcome = load_requirements_outcome(store)
    else:
        progress("intake + groundability guard")
        outcome = await build_requirements(ctx, store, brief, docs_dir, answerer, approve=approve)
    req = outcome.requirements

    if resume and store.exists("agent.v0.json"):
        progress("agent: reusing agent.v0.json")
        v0 = store.read_json("agent.v0.json", AgentDefinition)
    else:
        progress("generating agent v0")
        v0 = await build_agent(ctx, store, outcome, docs_dir)

    if resume and store.exists("evals.json"):
        progress("evals: reusing evals.json")
        evals = store.read_json("evals.json", EvalSet)
    else:
        progress("generating eval set")
        evals = await build_evals(ctx, store, req, docs_dir, min_cases=min_cases)

    progress("judge canaries")
    canaries = await run_canaries(ctx, evals.split("dev"))
    store.write_json("canaries.json", canaries)

    counter = {"n": 0}

    async def evaluate_dev(defn: AgentDefinition) -> EvalRun:
        counter["n"] += 1
        progress(f"evaluating candidate {counter['n']} on dev")
        return await evaluate_agent(
            ctx,
            store,
            defn,
            evals,
            docs_dir,
            "dev",
            canaries=canaries,
            tag=f"dev-run{counter['n']:02d}-v{defn.version}",
        )

    v0_run = await evaluate_dev(v0)
    tournament_result: TournamentResult | None = None
    if tournament:
        progress("prompt tournament")
        v0, v0_run, tournament_result = await prompt_tournament(ctx, v0, v0_run, evaluate_dev)
        store.write_json("tournament.json", tournament_result)
        _save_agent(store, v0)

    progress("critic-optimizer loop")
    with stage("optimize"):
        result = await optimize(
            ctx,
            v0,
            v0_run,
            evaluate_dev,
            max_iterations=max_iterations,
            tokens_used=lambda: budget.used_tokens,
            tokens_left=lambda: budget.remaining_tokens,
        )
    for entry in result.changelog:
        if entry.accepted:
            _save_agent(store, apply_patch_chain(v0, result.changelog, entry.to_version))
    store.write_json("changelog.json", result.changelog)
    store.write_text("changelog.md", changelog_markdown(result.changelog))

    progress("holdout evaluation (before and after)")
    holdout_v0 = await evaluate_agent(
        ctx, store, v0, evals, docs_dir, "holdout", canaries=canaries, tag="holdout-v0"
    )
    final = result.best
    holdout_final = holdout_v0
    if final.version != v0.version:
        holdout_final = await evaluate_agent(
            ctx,
            store,
            final,
            evals,
            docs_dir,
            "holdout",
            canaries=canaries,
            tag=f"holdout-v{final.version}",
        )

    _save_agent(store, final)
    store.write_json("agent.final.json", final)
    card = build_final_scorecard(
        final, result.history, holdout_v0, holdout_final, result.stop_reason, budget.report()
    )
    store.write_json("scorecard.json", card)
    store.write_json("budget.json", budget.report())
    report = report_markdown(
        card,
        final,
        req,
        outcome.groundability,
        evals,
        result.changelog,
        result.best_run,
        holdout_final,
        tournament_result,
    )
    path = store.write_text("report.md", report)
    return FactoryResult(requirements=req, final=final, scorecard=card, report_path=path)


def apply_patch_chain(
    v0: AgentDefinition, changelog: list[ChangeLogEntry], up_to: int | None
) -> AgentDefinition:
    """Rebuild an accepted version from v0 by replaying the accepted patches in order."""
    defn = v0
    for entry in changelog:
        if entry.accepted:
            defn = apply_patch(defn, entry.patch)
            if defn.version == up_to:
                break
    return defn
