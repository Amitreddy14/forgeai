"""Stretch: prompt tournament. The generator writes alternative role prompts; the dev set
picks the winner. Ties keep the original (no change without evidence)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from forgeai.errors import ModelOutputError
from forgeai.llm import ChatMessage, complete_json, stage
from forgeai.prompts import render
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import EvalRun
from forgeai.stages.context import ForgeContext


class _Variants(BaseModel):
    prompts: list[str] = Field(default_factory=list)


class Contender(BaseModel):
    label: str
    role_prompt: str
    dev_pass_rate: float
    passed: int
    cases: int


class TournamentResult(BaseModel):
    contenders: list[Contender]
    winner: str


async def prompt_tournament(
    ctx: ForgeContext,
    v0: AgentDefinition,
    v0_run: EvalRun,
    evaluate_dev: Callable[[AgentDefinition], Awaitable[EvalRun]],
    count: int = 2,
) -> tuple[AgentDefinition, EvalRun, TournamentResult]:
    def entry(label: str, defn: AgentDefinition, run: EvalRun) -> Contender:
        dev = run.scorecard.splits["dev"]
        return Contender(
            label=label,
            role_prompt=defn.role_prompt,
            dev_pass_rate=dev.pass_rate,
            passed=dev.passed,
            cases=dev.cases,
        )

    contenders = [entry("original", v0, v0_run)]
    best, best_run = v0, v0_run
    prompt = render("generate_variants", role_prompt=v0.role_prompt, count=count)
    try:
        with stage("tournament"):
            variants = await complete_json(
                ctx.llm,
                [ChatMessage(role="user", content=prompt)],
                _Variants,
                model=ctx.models.generator,
                max_tokens=1500,
            )
    except ModelOutputError:
        return best, best_run, TournamentResult(contenders=contenders, winner="original")
    labels = ["precision", "helpfulness", "caution"]
    for i, text in enumerate(v for v in variants.prompts[:count] if len(v.strip()) >= 40):
        candidate = v0.model_copy(update={"role_prompt": text.strip()})
        run = await evaluate_dev(candidate)
        label = labels[i] if i < len(labels) else f"variant {i + 1}"
        contenders.append(entry(label, candidate, run))
        if run.scorecard.splits["dev"].passed > best_run.scorecard.splits["dev"].passed:
            best, best_run = candidate, run
    winner = next(c.label for c in contenders if c.role_prompt == best.role_prompt)
    return best, best_run, TournamentResult(contenders=contenders, winner=winner)
