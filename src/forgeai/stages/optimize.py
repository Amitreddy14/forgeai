"""Stage 6 — critic-optimizer: evaluate on dev -> classify failures -> propose one targeted
patch -> re-evaluate on dev -> keep it only if the dev score rises and no passing case
breaks, otherwise revert. At most three iterations. Every decision is logged.

The critic is mostly code. Retrieval misses and threshold refusals have mechanical fixes
(more passages, bigger chunks, a lower threshold) that code can compute from the failing
cases. Only prompt-level problems go to the model, and a model-written rule is rejected
if it smuggles in an expected answer (overfitting to the test set).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from forgeai.errors import BudgetExceeded, ModelOutputError
from forgeai.llm import ChatMessage, complete_json, stage
from forgeai.prompts import render
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import CaseResult, EvalRun, FailureCategory
from forgeai.schemas.patch import (
    AddRefuseTopic,
    AddRule,
    ChangeLogEntry,
    Patch,
    SetChunkWords,
    SetRetrievalMode,
    SetThreshold,
    SetTopK,
)
from forgeai.stages.context import ForgeContext
from forgeai.textutil import contains_fact, extract_numbers, normalize

log = logging.getLogger(__name__)

PRIORITY: list[FailureCategory] = [
    "retrieval_miss",
    "wrong_refusal",
    "hallucination",
    "prompt_defect",
]
CATEGORY_HELP: dict[str, str] = {
    "retrieval_miss": "the passage with the answer was not retrieved",
    "prompt_defect": "the right passage was retrieved but the answer missed or garbled the facts",
    "hallucination": "the answer stated things, often numbers, that are not in the passages",
    "wrong_refusal": "the assistant refused an answerable question or answered one it must refuse",
}
DEFAULT_RULES: dict[str, str] = {
    "hallucination": (
        "State only facts and numbers that appear in the passages or tool results; if a "
        "detail is not there, say it is not covered instead of estimating it."
    ),
    "prompt_defect": (
        "Answer the exact question first with the specific figure or rule from the "
        "passage, using the document's own wording, then add brief context."
    ),
    "wrong_refusal": (
        "If the passages contain the answer, even under different wording, answer it; "
        "refuse only when the topic is listed for refusal or truly absent."
    ),
}
MAX_TOP_K = 10
MAX_CHUNK_WORDS = 600


class _CriticRule(BaseModel):
    rule: str = Field(min_length=10)
    rationale: str = ""


# ---------- Patches -------------------------------------------------------------------


def apply_patch(defn: AgentDefinition, patch: Patch) -> AgentDefinition:
    """Return a new definition version with the patch applied (the input is untouched)."""
    op = patch.op
    update: dict[str, object] = {"version": defn.version + 1, "parent_version": defn.version}
    if isinstance(op, SetTopK):
        update["retrieval"] = defn.retrieval.model_copy(update={"top_k": op.value})
    elif isinstance(op, SetRetrievalMode):
        update["retrieval"] = defn.retrieval.model_copy(update={"mode": op.value})
    elif isinstance(op, SetChunkWords):
        overlap = min(defn.chunking.overlap_words, op.value // 4)
        update["chunking"] = defn.chunking.model_copy(
            update={"chunk_words": op.value, "overlap_words": overlap}
        )
    elif isinstance(op, SetThreshold):
        update["refusal"] = defn.refusal.model_copy(update={"min_evidence_score": op.value})
    elif isinstance(op, AddRule):
        update["rules"] = [*defn.rules, op.text]
    elif isinstance(op, AddRefuseTopic):
        update["refusal"] = defn.refusal.model_copy(
            update={"refuse_topics": [*defn.refusal.refuse_topics, op.topic]}
        )
    return defn.model_copy(update=update)


def leaks_answers(text: str, failures: list[CaseResult]) -> str | None:
    """A rule that contains an expected answer would pass the test by memorising it."""
    numbers = extract_numbers(text) - {str(n) for n in range(10)}
    for r in failures:
        for fact in r.case.key_facts:
            if len(normalize(fact).split()) >= 2 and contains_fact(text, fact):
                return f"contains the expected fact {fact!r} of {r.case.id}"
            if numbers & (extract_numbers(fact) - {str(n) for n in range(10)}):
                return f"contains a number from the expected answer of {r.case.id}"
    return None


# ---------- Critic --------------------------------------------------------------------


def _ids(failures: list[CaseResult]) -> list[str]:
    return [r.case.id for r in failures]


def mechanical_patches(
    category: FailureCategory,
    failures: list[CaseResult],
    defn: AgentDefinition,
    run: EvalRun,
) -> list[Patch]:
    """Fixes code can compute from the evidence, in the order they should be tried."""
    out: list[Patch] = []
    ids = _ids(failures)
    if category == "retrieval_miss":
        if defn.retrieval.top_k < MAX_TOP_K:
            value = min(MAX_TOP_K, defn.retrieval.top_k + 3)
            out.append(
                Patch(
                    op=SetTopK(value=value),
                    category=category,
                    targets=ids,
                    why=f"expected source not among the top {defn.retrieval.top_k} passages",
                )
            )
        longest = max((len(s.quote.split()) for r in failures for s in r.case.sources), default=0)
        if defn.chunking.chunk_words < MAX_CHUNK_WORDS:
            value = min(MAX_CHUNK_WORDS, max(defn.chunking.chunk_words + 80, longest * 2))
            out.append(
                Patch(
                    op=SetChunkWords(value=value),
                    category=category,
                    targets=ids,
                    why="answers may be split across chunks; larger chunks keep them together",
                )
            )
        if defn.retrieval.mode != "hybrid":
            out.append(
                Patch(
                    op=SetRetrievalMode(value="hybrid"),
                    category=category,
                    targets=ids,
                    why="hybrid retrieval combines exact terms with meaning",
                )
            )
    elif category == "wrong_refusal":
        over = [r for r in failures if not r.case.expect_refusal and r.refused_by == "threshold"]
        under = [r for r in failures if r.case.expect_refusal and not r.refused]
        if over and defn.refusal.min_evidence_score > 0:
            lowest = min(r.best_dense for r in over)
            lowered = round(max(0.0, lowest - 0.03), 3)
            out.append(
                Patch(
                    op=SetThreshold(value=lowered),
                    category=category,
                    targets=_ids(over),
                    why=f"threshold {defn.refusal.min_evidence_score:.2f} refused answerable "
                    f"questions scoring down to {lowest:.2f}",
                )
            )
        if under:
            answerable = [
                r.best_dense
                for r in run.results
                if not r.case.expect_refusal and r.judgement.passed
            ]
            highest_bad = max(r.best_dense for r in under)
            if answerable and highest_bad < min(answerable) - 0.02:
                raised = round(highest_bad + 0.01, 3)
                out.append(
                    Patch(
                        op=SetThreshold(value=raised),
                        category=category,
                        targets=_ids(under),
                        why=f"must-refuse questions score up to {highest_bad:.2f}, every "
                        f"passing answerable question scores above {min(answerable):.2f}",
                    )
                )
    return out


async def critic_rule_patch(
    ctx: ForgeContext, category: FailureCategory, failures: list[CaseResult], defn: AgentDefinition
) -> Patch | None:
    """Ask the model for one general rule; reject it if it leaks expected answers."""
    cases = "\n".join(
        f"- Q: {r.case.question}\n  expected: {'refuse' if r.case.expect_refusal else '; '.join(r.case.key_facts)}"
        f"\n  got: {r.answer[:300] or '(no answer)'}\n  judge: {r.judgement.reason[:200]}"
        for r in failures[:6]
    )
    prompt = render(
        "critic_rule",
        category=category,
        category_help=CATEGORY_HELP[category],
        cases=cases,
        rules="\n".join(f"- {r}" for r in defn.rules) or "(none)",
    )
    try:
        with stage("critic"):
            out = await complete_json(
                ctx.llm,
                [ChatMessage(role="user", content=prompt)],
                _CriticRule,
                model=ctx.models.generator,
            )
        text = out.rule.strip()
        why = out.rationale or CATEGORY_HELP[category]
        source = "critic"
    except ModelOutputError:
        text, why, source = DEFAULT_RULES.get(category, ""), "fallback rule", "rule"
    if not text or len(text) > 240:
        return None
    leak = leaks_answers(text, failures)
    if leak:
        log.warning("critic rule rejected (%s): %s", leak, text)
        text, why, source = DEFAULT_RULES.get(category, ""), f"critic rule rejected: {leak}", "rule"
        if not text:
            return None
    if normalize(text) in {normalize(r) for r in defn.rules}:
        return None
    return Patch(
        op=AddRule(text=text),
        category=category,
        targets=_ids(failures),
        why=why,
        source="critic" if source == "critic" else "rule",
    )


def failure_categories(run: EvalRun) -> list[tuple[FailureCategory, list[CaseResult]]]:
    """Failing categories, most failures first (ties broken by PRIORITY)."""
    failures = [r for r in run.results if not r.judgement.passed and r.judgement.failure]
    counts = Counter(r.judgement.failure for r in failures)
    ordered = sorted(
        (c for c in PRIORITY if counts.get(c)), key=lambda c: (-counts[c], PRIORITY.index(c))
    )
    return [(c, [r for r in failures if r.judgement.failure == c]) for c in ordered]


async def candidate_patches(
    ctx: ForgeContext, run: EvalRun, defn: AgentDefinition
) -> AsyncIterator[Patch]:
    """Yield fixes lazily, category by category, mechanical fixes before model-written
    rules, so the critic model is only asked when a cheaper fix did not work."""
    for category, targeted in failure_categories(run):
        for patch in mechanical_patches(category, targeted, defn, run):
            yield patch
        if category != "retrieval_miss":
            rule = await critic_rule_patch(ctx, category, targeted, defn)
            if rule is not None:
                yield rule


# ---------- Loop ----------------------------------------------------------------------

Evaluator = Callable[[AgentDefinition], Awaitable[EvalRun]]


@dataclass
class OptimizeResult:
    best: AgentDefinition
    best_run: EvalRun
    history: list[EvalRun] = field(default_factory=list)  # accepted dev runs, v0 first
    changelog: list[ChangeLogEntry] = field(default_factory=list)
    stop_reason: str = ""


def _passed(run: EvalRun) -> set[str]:
    return {r.case.id for r in run.results if r.judgement.passed}


def _rate(run: EvalRun) -> float:
    return run.scorecard.splits["dev"].pass_rate if "dev" in run.scorecard.splits else 0.0


async def optimize(
    ctx: ForgeContext,
    v0: AgentDefinition,
    v0_run: EvalRun,
    evaluate_dev: Evaluator,
    *,
    max_iterations: int = 3,
    max_candidates: int = 3,
    tokens_used: Callable[[], int] = lambda: 0,
    tokens_left: Callable[[], int] = lambda: 1 << 62,
) -> OptimizeResult:
    best, best_run = v0, v0_run
    result = OptimizeResult(best=best, best_run=best_run, history=[v0_run])
    eval_cost = 0
    for iteration in range(1, max_iterations + 1):
        if not failure_categories(best_run):
            result.stop_reason = "every dev case passes"
            break
        accepted = False
        tried = 0
        async for patch in candidate_patches(ctx, best_run, best):
            if tried == max_candidates:
                break
            tried += 1
            if eval_cost and tokens_left() < eval_cost * 1.2:
                result.stop_reason = "token budget would not cover another evaluation"
                return _finish(result, best, best_run)
            candidate = apply_patch(best, patch)
            before_tokens = tokens_used()
            try:
                run = await evaluate_dev(candidate)
            except BudgetExceeded as exc:
                result.stop_reason = f"budget exceeded: {exc}"
                return _finish(result, best, best_run)
            eval_cost = max(eval_cost, tokens_used() - before_tokens)
            before, after = _passed(best_run), _passed(run)
            fixed, broke = sorted(after - before), sorted(before - after)
            keep = len(after) > len(before) and not broke
            decision = (
                f"kept: +{len(fixed)} fixed, none broken"
                if keep
                else f"reverted: {len(fixed)} fixed, {len(broke)} broken"
                + (" (no net gain)" if len(after) <= len(before) else "")
            )
            result.changelog.append(
                ChangeLogEntry(
                    iteration=iteration,
                    patch=patch,
                    from_version=best.version,
                    to_version=candidate.version if keep else None,
                    accepted=keep,
                    dev_before=_rate(best_run),
                    dev_after=_rate(run),
                    fixed=fixed,
                    broke=broke,
                    decision=decision,
                    tokens=tokens_used() - before_tokens,
                )
            )
            if keep:
                best, best_run = candidate, run
                result.history.append(run)
                accepted = True
                break
        if tried == 0:
            result.stop_reason = "no applicable fix for the remaining failures"
            break
        if not accepted:
            result.stop_reason = (
                f"iteration {iteration} found no patch that improved dev without regressions"
            )
            break
    else:
        result.stop_reason = f"reached the iteration limit ({max_iterations})"
    return _finish(result, best, best_run)


def _finish(result: OptimizeResult, best: AgentDefinition, run: EvalRun) -> OptimizeResult:
    result.best, result.best_run = best, run
    return result
