"""Stage 7 — honest report. Every number here is computed from per-case results on disk;
no model writes any part of it."""

from __future__ import annotations

import math
from collections import defaultdict

from pydantic import BaseModel, Field

from forgeai.llm.budget import BudgetReport
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import CaseResult, EvalRun, EvalSet
from forgeai.schemas.patch import ChangeLogEntry
from forgeai.schemas.requirements import GroundabilityReport, Requirements
from forgeai.stages.tournament import TournamentResult


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval: honest error bars for small eval sets."""
    if n == 0:
        return 0.0, 0.0
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class IterationScore(BaseModel):
    version: int
    dev_pass_rate: float
    dev_passed: int
    dev_cases: int
    failures: dict[str, int]


class FinalScorecard(BaseModel):
    agent: str
    final_version: int
    iterations: list[IterationScore]
    holdout_before: float | None
    holdout_after: float | None
    holdout_passed: int
    holdout_cases: int
    holdout_ci95: tuple[float, float]
    judge_canary_accuracy: float | None
    judge_trusted: bool
    stop_reason: str
    tokens_used: int
    by_stage_tokens: dict[str, int] = Field(default_factory=dict)


def build_final_scorecard(
    final: AgentDefinition,
    dev_history: list[EvalRun],
    holdout_v0: EvalRun | None,
    holdout_final: EvalRun | None,
    stop_reason: str,
    budget: BudgetReport,
) -> FinalScorecard:
    iterations = [
        IterationScore(
            version=run.agent_version,
            dev_pass_rate=run.scorecard.splits["dev"].pass_rate,
            dev_passed=run.scorecard.splits["dev"].passed,
            dev_cases=run.scorecard.splits["dev"].cases,
            failures=run.scorecard.splits["dev"].failures,
        )
        for run in dev_history
        if "dev" in run.scorecard.splits
    ]
    hf = holdout_final.scorecard.splits.get("holdout") if holdout_final else None
    hb = holdout_v0.scorecard.splits.get("holdout") if holdout_v0 else None
    last = dev_history[-1].scorecard
    return FinalScorecard(
        agent=final.name,
        final_version=final.version,
        iterations=iterations,
        holdout_before=hb.pass_rate if hb else None,
        holdout_after=hf.pass_rate if hf else None,
        holdout_passed=hf.passed if hf else 0,
        holdout_cases=hf.cases if hf else 0,
        holdout_ci95=wilson(hf.passed, hf.cases) if hf else (0.0, 0.0),
        judge_canary_accuracy=last.judge_canary_accuracy,
        judge_trusted=last.judge_trusted,
        stop_reason=stop_reason,
        tokens_used=budget.used_tokens,
        by_stage_tokens={k: v.total_tokens for k, v in budget.by_stage.items()},
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _open_failures(results: list[CaseResult]) -> dict[str, list[CaseResult]]:
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        if not r.judgement.passed:
            grouped[r.judgement.failure or "unclassified"].append(r)
    return dict(grouped)


WHY = {
    "retrieval_miss": "the passage with the answer was not retrieved, so the agent never saw it",
    "prompt_defect": "the right passage was retrieved, but the answer missed or garbled the facts",
    "hallucination": "the answer stated something not supported by the passages",
    "wrong_refusal": "the agent refused an answerable question, or answered one it should refuse",
}


def report_markdown(
    card: FinalScorecard,
    final: AgentDefinition,
    requirements: Requirements,
    groundability: GroundabilityReport,
    evals: EvalSet,
    changelog: list[ChangeLogEntry],
    final_dev: EvalRun,
    holdout_final: EvalRun | None,
    tournament: TournamentResult | None = None,
) -> str:
    lo, hi = card.holdout_ci95
    lines = [
        f"# ForgeAI report: {card.agent}",
        "",
        "## Headline",
        "",
        f"**Holdout: {card.holdout_passed}/{card.holdout_cases} passed ({_pct(card.holdout_after)})**, "
        f"95% interval {lo:.0%}-{hi:.0%}. Before optimization: {_pct(card.holdout_before)}.",
        "",
        "The holdout cases were never shown to the optimizer; it tuned the agent on the dev "
        "cases only. With this few cases the interval is wide: treat the number as a range.",
        "",
        "## Score per iteration (dev)",
        "",
        "| Agent version | Dev passed | Dev pass rate | Failures |",
        "|---|---|---|---|",
    ]
    for it in card.iterations:
        failures = ", ".join(f"{k} {v}" for k, v in it.failures.items()) or "none"
        lines.append(
            f"| v{it.version} | {it.dev_passed}/{it.dev_cases} | {it.dev_pass_rate:.0%} | {failures} |"
        )
    lines += ["", f"Stopped because: {card.stop_reason}.", ""]

    lines += ["## Change log", ""]
    if changelog:
        lines += [
            "| Iter | Category | Change | Targets | Dev before -> after | Decision |",
            "|---|---|---|---|---|---|",
        ]
        for e in changelog:
            lines.append(
                f"| {e.iteration} | {e.patch.category} | {e.patch.describe().replace('|', '/')} | "
                f"{', '.join(e.patch.targets)} | {e.dev_before:.0%} -> {e.dev_after:.0%} | {e.decision} |"
            )
        lines += ["", "Why each change was tried:", ""]
        lines += [f"- iter {e.iteration}: {e.patch.why}" for e in changelog]
    else:
        lines.append("No changes were tried.")
    lines.append("")

    if tournament:
        lines += [
            "## Prompt tournament (dev)",
            "",
            "| Prompt | Dev passed | Rate |",
            "|---|---|---|",
        ]
        for c in tournament.contenders:
            mark = " (winner)" if c.label == tournament.winner else ""
            lines.append(f"| {c.label}{mark} | {c.passed}/{c.cases} | {c.dev_pass_rate:.0%} |")
        lines.append("")

    lines += ["## What still fails, and why", ""]
    open_results = [*final_dev.results, *(holdout_final.results if holdout_final else [])]
    grouped = _open_failures(open_results)
    if not grouped:
        lines.append("No eval case fails. Read this with the eval set's limits below in mind.")
    for category, items in sorted(grouped.items()):
        lines += [f"### {category} ({len(items)})", "", f"_{WHY.get(category, '')}._", ""]
        for r in items:
            lines.append(
                f"- **{r.case.id}** ({r.case.split}, {r.case.category}): {r.case.question}"
            )
            lines.append(f"  - judge: {r.judgement.reason or 'failed'}")
            if r.answer:
                lines.append(f"  - agent said: {r.answer[:220].replace(chr(10), ' ')}")
        lines.append("")

    lines += ["## How far to trust these numbers", ""]
    if card.judge_canary_accuracy is not None:
        verdict = (
            "trusted"
            if card.judge_trusted
            else "NOT trusted, so only the deterministic checks decided the verdicts"
        )
        lines.append(f"- Judge canary accuracy {card.judge_canary_accuracy:.0%}: {verdict}.")
    lines.append(
        f"- Eval set: {len(evals.cases)} generated cases kept, {len(evals.rejected)} rejected by "
        "validation (see evals.md). Every answerable case's facts were verified inside a quote "
        "from the documents."
    )
    lines.append(
        f"- Groundability: {groundability.coverage:.0%} of requested topics are supported by the documents."
    )
    if requirements.flagged_gaps:
        lines.append(
            "- Flagged gaps (the agent refuses these instead of guessing): "
            + "; ".join(g.topic for g in requirements.flagged_gaps)
            + "."
        )
    if requirements.assumptions:
        lines.append(
            f"- {len(requirements.assumptions)} requirement(s) rest on assumptions, not the "
            "requester's word (see requirements.md)."
        )
    lines.append(
        "- The eval set is generated by the same small model family that builds the agent. Its "
        "blind spots are shared: a topic the generator never asks about is never tested."
    )
    lines.append("")

    lines += ["## The agent", ""]
    lines.append(
        f"- Tools: {', '.join(final.tool_names) or 'none'}"
        + (
            f" (rejected: {', '.join(t.name for t in final.rejected_tools)})"
            if final.rejected_tools
            else ""
        )
    )
    lines.append(
        f"- Retrieval: {final.retrieval.mode}, top_k {final.retrieval.top_k}; chunks of "
        f"{final.chunking.chunk_words} words; refusal threshold {final.refusal.min_evidence_score:.2f}"
    )
    lines.append(f"- Rules: {len(final.rules)}; refuse topics: {len(final.refusal.refuse_topics)}")
    lines.append(f"- Definition: `agent.v{final.version}.json`")
    lines += ["", "## Cost", "", f"- Tokens used: {card.tokens_used:,}"]
    lines += [f"  - {k}: {v:,}" for k, v in sorted(card.by_stage_tokens.items())]
    return "\n".join(lines) + "\n"


def changelog_markdown(changelog: list[ChangeLogEntry]) -> str:
    lines = ["# Change log", ""]
    if not changelog:
        return "# Change log\n\nNo changes were tried.\n"
    for e in changelog:
        lines += [
            f"## Iteration {e.iteration}: {e.patch.describe()}",
            "",
            f"- category: {e.patch.category} (proposed by {e.patch.source})",
            f"- targets: {', '.join(e.patch.targets)}",
            f"- why: {e.patch.why}",
            f"- dev pass rate: {e.dev_before:.0%} -> {e.dev_after:.0%}",
            f"- fixed: {', '.join(e.fixed) or 'none'}; broke: {', '.join(e.broke) or 'none'}",
            f"- decision: {e.decision}"
            + (f" (now agent v{e.to_version})" if e.to_version is not None else ""),
            f"- tokens: {e.tokens:,}",
            "",
        ]
    return "\n".join(lines)
