"""Run an agent over eval cases and judge it. All agent calls (generator model) run
first, then all judge calls (judge model), so a local GPU swaps models once, not per case."""

from __future__ import annotations

import asyncio
import statistics
from typing import Literal

from forgeai.llm import stage
from forgeai.runtime.agent import AgentRuntime, AgentTurn
from forgeai.schemas.evals import (
    CanaryResult,
    CaseResult,
    EvalCase,
    EvalRun,
    EvalSet,
    Judgement,
    Scorecard,
    SplitScore,
)
from forgeai.stages.context import ForgeContext
from forgeai.stages.judge import TRUST_THRESHOLD, canary_accuracy, judge_case, run_canaries

SplitName = Literal["dev", "holdout", "all"]


async def answer_cases(runtime: AgentRuntime, cases: list[EvalCase]) -> list[AgentTurn]:
    with stage("agent"):
        return list(await asyncio.gather(*(runtime.answer(c.question) for c in cases)))


async def judge_turns(
    ctx: ForgeContext,
    cases: list[EvalCase],
    turns: list[AgentTurn],
    chunk_text: dict[str, str],
    trust_judge: bool,
) -> list[Judgement]:
    with stage("judge"):
        return list(
            await asyncio.gather(
                *(
                    judge_case(ctx, c, t, chunk_text, trust_judge=trust_judge)
                    for c, t in zip(cases, turns, strict=True)
                )
            )
        )


def score_split(results: list[CaseResult]) -> SplitScore:
    n = len(results)
    passed = sum(r.judgement.passed for r in results)
    by_cat: dict[str, list[bool]] = {}
    failures: dict[str, int] = {}
    for r in results:
        by_cat.setdefault(r.case.category, []).append(r.judgement.passed)
        if r.judgement.failure:
            failures[r.judgement.failure] = failures.get(r.judgement.failure, 0) + 1
    answerable = [r for r in results if not r.case.expect_refusal]
    return SplitScore(
        cases=n,
        passed=passed,
        pass_rate=round(passed / n, 4) if n else 0.0,
        by_category={k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
        failures=dict(sorted(failures.items())),
        refusal_accuracy=round(sum(r.judgement.checks.refusal_ok for r in results) / n, 4)
        if n
        else 0.0,
        groundedness=round(
            sum(r.judgement.checks.grounded for r in answerable) / len(answerable), 4
        )
        if answerable
        else 1.0,
        fact_recall=round(statistics.mean(r.judgement.fact_recall for r in answerable), 4)
        if answerable
        else 1.0,
    )


def build_scorecard(
    version: int, results: list[CaseResult], canaries: list[CanaryResult]
) -> Scorecard:
    splits: dict[str, SplitScore] = {}
    for name in ("dev", "holdout"):
        subset = [r for r in results if r.case.split == name]
        if subset:
            splits[name] = score_split(subset)
    accuracy = canary_accuracy(canaries)
    return Scorecard(
        agent_version=version,
        splits=splits,
        judge_canary_accuracy=accuracy,
        judge_trusted=accuracy is None or accuracy >= TRUST_THRESHOLD,
        mean_latency_s=round(statistics.mean(r.latency_s for r in results), 2) if results else 0.0,
    )


async def evaluate(
    ctx: ForgeContext,
    runtime: AgentRuntime,
    evals: EvalSet,
    split: SplitName = "dev",
    *,
    canaries: list[CanaryResult] | None = None,
) -> EvalRun:
    cases = evals.split(split)
    turns = await answer_cases(runtime, cases)
    if canaries is None:
        canaries = await run_canaries(ctx, evals.split("dev"))
    accuracy = canary_accuracy(canaries)
    trusted = accuracy is None or accuracy >= TRUST_THRESHOLD
    chunk_text = {c.chunk_id: c.text for c in runtime.index.chunks}
    judgements = await judge_turns(ctx, cases, turns, chunk_text, trusted)
    results = [
        CaseResult(
            case=case,
            answer=turn.answer.answer,
            refused=turn.answer.refused,
            citations=[c.chunk_id for c in turn.valid_citations],
            tool_calls=[f"{t.name}:{'ok' if t.ok else 'failed'}" for t in turn.tool_calls],
            latency_s=turn.latency_s,
            judgement=judgement,
        )
        for case, turn, judgement in zip(cases, turns, judgements, strict=True)
    ]
    return EvalRun(
        agent_version=runtime.definition.version,
        split=split,
        results=results,
        canaries=canaries,
        scorecard=build_scorecard(runtime.definition.version, results, canaries),
    )


def scorecard_markdown(run: EvalRun) -> str:
    sc = run.scorecard
    lines = [f"# Scorecard: agent v{sc.agent_version} ({run.split})", ""]
    for name, s in sc.splits.items():
        lines += [
            f"## {name}: {s.passed}/{s.cases} passed ({s.pass_rate:.0%})",
            "",
            f"- refusal accuracy {s.refusal_accuracy:.0%}, groundedness {s.groundedness:.0%}, "
            f"fact recall {s.fact_recall:.0%}",
            "- by category: " + ", ".join(f"{k} {v:.0%}" for k, v in s.by_category.items()),
            "- failures: " + (", ".join(f"{k} {v}" for k, v in s.failures.items()) or "none"),
            "",
        ]
    if sc.judge_canary_accuracy is not None:
        trust = "trusted" if sc.judge_trusted else "NOT trusted: only deterministic checks count"
        lines += [f"Judge canary accuracy {sc.judge_canary_accuracy:.0%} ({trust}).", ""]
    lines += [
        "| Case | Split | Category | Result | Failure | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in run.results:
        j = r.judgement
        lines.append(
            f"| {r.case.id} | {r.case.split} | {r.case.category} | "
            f"{'pass' if j.passed else '**fail**'} | {j.failure or ''} | "
            f"{j.reason.replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"
