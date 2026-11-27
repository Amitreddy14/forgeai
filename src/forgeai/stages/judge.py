"""Stage 5 — layered judge.

Layer 1, deterministic (decides everything it can):
  refusal flag vs expectation, citation quotes verified, expected source retrieved and
  cited, key facts found by string/number match, numbers in the answer that appear in
  no evidence the agent saw (a hallucination signal), required tool actually called.
Layer 2, LLM judge (only for facts layer 1 could not confirm):
  a model from a *different family* than the generator (self-preference), grading one
  answer at a time against facts, not a reference essay (no position bias, no reward
  for length), told explicitly that length earns nothing (verbosity bias). It must quote
  the answer as evidence; code checks that quote and its numbers (leniency).
Layer 3, canaries: planted answers with known verdicts (a wrong number, a long confident
  answer that omits the fact, a padded correct answer, a terse correct answer). If the
  judge gets them wrong, its verdicts are not trusted and only layer 1 counts.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel

from forgeai.errors import ModelOutputError
from forgeai.llm import ChatMessage, complete_json, stage
from forgeai.prompts import render
from forgeai.runtime.agent import AgentTurn
from forgeai.schemas.evals import (
    CanaryKind,
    CanaryResult,
    Checks,
    EvalCase,
    FactCheck,
    FactStatus,
    FailureCategory,
    Judgement,
)
from forgeai.stages.context import ForgeContext
from forgeai.textutil import contains_fact, contains_quote, extract_numbers, normalize

TRUST_THRESHOLD = 0.75
ANCHOR_WORDS = 10
PADDING = (
    "This is covered by the relevant company policy, which applies to all eligible staff. "
    "Employees are encouraged to read the full policy document and to speak with their "
    "manager or the HR team if anything is unclear. Policies may be reviewed from time to "
    "time, and the latest approved version always applies. "
)


class _JudgedFact(BaseModel):
    fact: str = ""
    status: FactStatus
    evidence: str = ""


class _JudgeOut(BaseModel):
    facts: list[_JudgedFact]


# ---------- Layer 1 -------------------------------------------------------------------


def _doc_of(chunk_id: str) -> str:
    return chunk_id.split(":", 1)[0]


def _anchor(quote: str) -> str:
    words = quote.split()
    return " ".join(words[:ANCHOR_WORDS]) if len(words) >= 3 else quote


def source_retrieved(case: EvalCase, turn: AgentTurn, chunk_text: dict[str, str]) -> bool | None:
    if not case.sources:
        return None
    texts = [chunk_text.get(r.chunk_id, "") for r in turn.retrieved]
    return all(any(contains_quote(t, _anchor(s.quote)) for t in texts) for s in case.sources)


def source_cited(case: EvalCase, turn: AgentTurn) -> bool | None:
    if not case.sources:
        return None
    cited = {_doc_of(c.chunk_id) for c in turn.valid_citations}
    tool_used = any(t.ok for t in turn.tool_calls)
    return all(s.doc_id in cited for s in case.sources) or (tool_used and bool(case.expected_tool))


def unsupported_numbers(turn: AgentTurn, chunk_text: dict[str, str]) -> list[str]:
    """Numbers stated in the answer that appear nowhere in what the agent was given."""
    if turn.answer.refused:
        return []
    evidence = " ".join(
        [
            turn.question,
            *(chunk_text.get(r.chunk_id, "") for r in turn.retrieved),
            *(t.output for t in turn.tool_calls),
        ]
    )
    known = extract_numbers(evidence)
    stated = extract_numbers(turn.answer.answer)
    return sorted(n for n in stated - known if not (n.isdigit() and int(n) < 10))


def string_fact_check(fact: str, answer: str) -> FactStatus | None:
    """`present` if clearly stated; None if only a model can tell."""
    if contains_fact(answer, fact):
        return "present"
    fact_numbers = extract_numbers(fact)
    if fact_numbers and fact_numbers <= extract_numbers(answer):
        words = [w for w in normalize(fact).split() if not w.isdigit()]
        answer_norm = f" {normalize(answer)} "
        if not words or sum(f" {w} " in answer_norm for w in words) / len(words) >= 0.5:
            return "present"
    return None


def layer_one(
    case: EvalCase, turn: AgentTurn, chunk_text: dict[str, str]
) -> tuple[Checks, list[FactCheck], list[str]]:
    """Returns checks, facts confirmed by string match, and facts still undecided."""
    checks = Checks(
        refusal_ok=turn.answer.refused == case.expect_refusal,
        grounded=turn.grounded,
        source_retrieved=source_retrieved(case, turn, chunk_text),
        source_cited=source_cited(case, turn),
        unsupported_numbers=unsupported_numbers(turn, chunk_text),
        tool_ok=(
            any(t.name == case.expected_tool and t.ok for t in turn.tool_calls)
            if case.expected_tool
            else None
        ),
        contract_error=turn.error,
    )
    confirmed: list[FactCheck] = []
    undecided: list[str] = []
    if not case.expect_refusal and not turn.answer.refused and turn.error is None:
        for fact in case.key_facts:
            status = string_fact_check(fact, turn.answer.answer)
            if status:
                confirmed.append(FactCheck(fact=fact, status=status, by="string"))
            else:
                undecided.append(fact)
    return checks, confirmed, undecided


# ---------- Layer 2 -------------------------------------------------------------------


def check_judge_evidence(fact: str, status: FactStatus, evidence: str, answer: str) -> FactStatus:
    """The judge must quote the answer. Code checks the quote, so a lenient judge cannot
    call a fact present when the answer never states it, or wave through a wrong number.
    """
    if status == "missing":
        return "missing"
    if not evidence.strip() or not contains_fact(answer, evidence):
        return "missing"  # evidence invented: the answer does not contain those words
    fact_numbers = extract_numbers(fact)
    if status == "present" and fact_numbers and not fact_numbers <= extract_numbers(evidence):
        # The quoted words carry different numbers than the fact: that is a contradiction.
        return "contradicted" if extract_numbers(evidence) else "missing"
    return status


async def llm_fact_judge(
    ctx: ForgeContext, question: str, facts: list[str], answer: str
) -> list[FactCheck]:
    if not facts:
        return []
    prompt = render(
        "judge_facts",
        question=question,
        facts="\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1)),
        answer=answer or "(empty answer)",
    )
    try:
        out = await complete_json(
            ctx.llm,
            [ChatMessage(role="user", content=prompt)],
            _JudgeOut,
            model=ctx.models.judge,
        )
    except ModelOutputError:
        return [FactCheck(fact=f, status="missing", by="judge") for f in facts]
    checks: list[FactCheck] = []
    for i, fact in enumerate(facts):
        if i < len(out.facts):  # align by position; dropped facts count as missing
            judged = out.facts[i]
            status = check_judge_evidence(fact, judged.status, judged.evidence, answer)
        else:
            status = "missing"
        checks.append(FactCheck(fact=fact, status=status, by="judge"))
    return checks


# ---------- Verdict -------------------------------------------------------------------


def classify(case: EvalCase, checks: Checks, facts: list[FactCheck]) -> FailureCategory:
    if not checks.refusal_ok:
        return "wrong_refusal"
    if checks.contract_error:
        return "prompt_defect"
    missing = any(f.status != "present" for f in facts)
    if checks.source_retrieved is False and (missing or not facts):
        return "retrieval_miss"
    if (
        any(f.status == "contradicted" for f in facts)
        or checks.unsupported_numbers
        or not checks.grounded
    ):
        return "hallucination"
    return "prompt_defect"


def verdict(case: EvalCase, checks: Checks, facts: list[FactCheck], judge_used: bool) -> Judgement:
    reasons: list[str] = []
    if case.expect_refusal:
        passed = checks.refusal_ok
        if not passed:
            reasons.append("answered a question it must refuse")
    else:
        if not checks.refusal_ok:
            reasons.append(
                "refused an answerable question" if not checks.contract_error else "no valid answer"
            )
        if checks.contract_error:
            reasons.append(f"output contract error: {checks.contract_error[:120]}")
        bad_facts = [f.fact for f in facts if f.status != "present"]
        if bad_facts:
            reasons.append("facts not stated correctly: " + "; ".join(bad_facts))
        if checks.unsupported_numbers:
            reasons.append("numbers not in any evidence: " + ", ".join(checks.unsupported_numbers))
        if not checks.grounded:
            reasons.append("citations not verified")
        if checks.tool_ok is False:
            reasons.append(f"did not successfully call {case.expected_tool}")
        passed = not reasons
    failure = None if passed else classify(case, checks, facts)
    return Judgement(
        case_id=case.id,
        passed=passed,
        checks=checks,
        facts=facts,
        failure=failure,
        reason="; ".join(reasons),
        judge_used=judge_used,
    )


async def judge_case(
    ctx: ForgeContext,
    case: EvalCase,
    turn: AgentTurn,
    chunk_text: dict[str, str],
    *,
    trust_judge: bool = True,
) -> Judgement:
    checks, confirmed, undecided = layer_one(case, turn, chunk_text)
    judged: list[FactCheck] = []
    if undecided:
        judged = await llm_fact_judge(ctx, case.question, undecided, turn.answer.answer)
        if not trust_judge:  # canaries failed: the judge may not mark anything present
            judged = [
                f if f.status != "present" else f.model_copy(update={"status": "missing"})
                for f in judged
            ]
    by_fact = {f.fact: f for f in [*confirmed, *judged]}
    facts = [by_fact[f] for f in case.key_facts if f in by_fact]
    return verdict(case, checks, facts, judge_used=bool(undecided))


# ---------- Layer 3: canaries ---------------------------------------------------------

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _corrupt(fact: str) -> str | None:
    match = _NUM.search(fact)
    if not match:
        return None
    value = float(match.group().replace(",", ""))
    wrong = value * 2 + 7
    text = str(int(wrong)) if wrong.is_integer() else f"{wrong:g}"
    return fact[: match.start()] + text + fact[match.end() :]


def canary_answers(case: EvalCase) -> list[tuple[CanaryKind, str, bool]]:
    """(kind, planted answer, should the judge pass it?)"""
    facts = case.key_facts
    out: list[tuple[CanaryKind, str, bool]] = [
        ("correct_terse", "; ".join(facts) + ".", True),
        ("correct_padded", PADDING + "In short: " + "; ".join(facts) + ". " + PADDING * 2, True),
        ("confident_omission", "Yes, absolutely. " + PADDING * 3, False),
    ]
    corrupted = [_corrupt(f) for f in facts]
    if any(corrupted):
        wrong = [c or f for c, f in zip(corrupted, facts, strict=True)]
        out.append(("wrong_number", "According to the policy: " + "; ".join(wrong) + ".", False))
    return out


async def run_canaries(
    ctx: ForgeContext, cases: list[EvalCase], limit: int = 2
) -> list[CanaryResult]:
    """Plant answers with known verdicts and see whether the judge gets them right."""
    pool = [c for c in cases if c.category == "answerable" and c.key_facts][:limit]
    jobs: list[tuple[EvalCase, CanaryKind, str, bool]] = [
        (case, kind, answer, should)
        for case in pool
        for kind, answer, should in canary_answers(case)
    ]

    async def one(case: EvalCase, answer: str) -> bool:
        facts = await llm_fact_judge(ctx, case.question, case.key_facts, answer)
        return all(f.status == "present" for f in facts)

    with stage("judge"):
        verdicts = await asyncio.gather(*(one(case, answer) for case, _k, answer, _s in jobs))
    return [
        CanaryResult(
            kind=kind, case_id=case.id, answer=answer, should_pass=should, judge_passed=got
        )
        for (case, kind, answer, should), got in zip(jobs, verdicts, strict=True)
    ]


def canary_accuracy(canaries: list[CanaryResult]) -> float | None:
    if not canaries:
        return None
    return sum(c.correct for c in canaries) / len(canaries)
