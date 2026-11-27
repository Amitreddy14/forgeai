from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeai.llm import Budget, BudgetedLLM, FakeLLM
from forgeai.pipeline import run_factory
from forgeai.retrieval import HashEmbedder
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import CaseResult, Checks, EvalCase, EvalRun, Judgement
from forgeai.schemas.patch import (
    AddRefuseTopic,
    AddRule,
    Patch,
    SetChunkWords,
    SetThreshold,
    SetTopK,
)
from forgeai.stages.context import ForgeContext, ModelRoles
from forgeai.stages.evaluate import build_scorecard
from forgeai.stages.intake import AutoAnswerer
from forgeai.stages.optimize import (
    apply_patch,
    critic_rule_patch,
    leaks_answers,
    mechanical_patches,
    optimize,
)
from forgeai.stages.report import wilson
from forgeai.store import RunStore
from tests import sim
from tests.fixtures import build_docs_folder

BRIEF = "I need an assistant that answers HR policy questions for shore staff."


def definition() -> AgentDefinition:
    return AgentDefinition.model_validate(
        {
            "name": "HR Desk",
            "description": "d",
            "audience": "staff",
            "tone": "plain",
            "role_prompt": "You are the HR desk for shore staff and answer from the policies.",
            "rules": ["Name the section."],
            "must_answer": ["leave"],
            "chunking": {"chunk_words": 160, "overlap_words": 30},
            "retrieval": {"top_k": 4},
            "refusal": {
                "refuse_topics": ["legal advice"],
                "message": "Please contact HR.",
                "min_evidence_score": 0.65,
            },
            "knowledge": {"docs_dir": "docs"},
            "provenance": {"factory_version": "t", "generator_model": "gen"},
        }
    )


def case(cid: str, *, refuse: bool = False, facts: list[str] | None = None) -> EvalCase:
    return EvalCase(
        id=cid,
        category="out_of_scope" if refuse else "answerable",
        question=f"Question {cid}?",
        expect_refusal=refuse,
        key_facts=facts if facts is not None else ([] if refuse else ["24 days"]),
    )


def result(c: EvalCase, passed: bool, failure: str | None = None, **kw: object) -> CaseResult:
    return CaseResult(
        case=c,
        answer=str(kw.get("answer", "")),
        refused=bool(kw.get("refused", False)),
        refused_by=kw.get("refused_by"),  # type: ignore[arg-type]
        best_dense=float(kw.get("best_dense", 0.8)),  # type: ignore[arg-type]
        judgement=Judgement(
            case_id=c.id,
            passed=passed,
            checks=Checks(refusal_ok=True, grounded=True),
            failure=failure,
            reason="" if passed else "failed",
        ),  # type: ignore[arg-type]
    )


def run_of(results: list[CaseResult], version: int = 0) -> EvalRun:
    return EvalRun(
        agent_version=version,
        split="dev",
        results=results,
        scorecard=build_scorecard(version, results, []),
    )


# ---------- Patches -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "check"),
    [
        (SetTopK(value=7), lambda d: d.retrieval.top_k == 7),
        (
            SetChunkWords(value=300),
            lambda d: d.chunking.chunk_words == 300 and d.chunking.overlap_words == 30,
        ),
        (SetChunkWords(value=80), lambda d: d.chunking.overlap_words == 20),
        (SetThreshold(value=0.4), lambda d: d.refusal.min_evidence_score == 0.4),
        (
            AddRule(text="Quote the clause number you used."),
            lambda d: d.rules[-1].startswith("Quote"),
        ),
        (AddRefuseTopic(topic="tax advice"), lambda d: d.refusal.refuse_topics[-1] == "tax advice"),
    ],
)
def test_apply_patch_makes_a_new_version(op, check) -> None:  # type: ignore[no-untyped-def]
    v0 = definition()
    v1 = apply_patch(v0, Patch(op=op, category="prompt_defect", targets=["c1"], why="w"))
    assert check(v1)
    assert (v1.version, v1.parent_version) == (1, 0)
    assert v0.version == 0
    assert v0.rules == ["Name the section."]  # original untouched


def test_patch_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="less than or equal to 20"):
        SetTopK(value=50)
    with pytest.raises(ValueError, match="at least 10 characters"):
        AddRule(text="short")


def test_rules_that_leak_answers_are_caught() -> None:
    failures = [result(case("c1", facts=["24 days of annual leave"]), False, "prompt_defect")]
    assert leaks_answers("Say staff get 24 days of annual leave.", failures)
    assert leaks_answers("Mention that the figure is 24.", failures)
    assert leaks_answers("Quote the exact figure and the section it comes from.", failures) is None


async def test_leaky_critic_rule_is_replaced_by_safe_default() -> None:
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "rule": "Always answer that annual leave is 24 days of annual leave.",
                    "rationale": "fixes c1",
                }
            )
        ]
    )
    ctx = ForgeContext(llm=llm, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())
    failures = [result(case("c1", facts=["24 days of annual leave"]), False, "prompt_defect")]
    patch = await critic_rule_patch(ctx, "prompt_defect", failures, definition())
    assert patch is not None
    assert isinstance(patch.op, AddRule)
    assert "24" not in patch.op.text
    assert patch.why.startswith("critic rule rejected")


def test_mechanical_fixes() -> None:
    d = definition()
    miss = [result(case("c1"), False, "retrieval_miss")]
    patches = mechanical_patches("retrieval_miss", miss, d, run_of(miss))
    assert [p.op.op for p in patches] == ["set_top_k", "set_chunk_words"]
    assert patches[0].op.value == 7  # type: ignore[union-attr]

    over = [
        result(
            case("c2"),
            False,
            "wrong_refusal",
            refused=True,
            refused_by="threshold",
            best_dense=0.61,
        )
    ]
    patches = mechanical_patches("wrong_refusal", over, d, run_of(over))
    assert patches[0].op == SetThreshold(value=0.58)


# ---------- Loop ----------------------------------------------------------------------


async def test_loop_keeps_improvements_and_reverts_regressions() -> None:
    """Scripted world: top_k 7 fixes c1; a chunk-size change would break c3."""
    cases = [case("c1"), case("c2"), case("c3"), case("c4", refuse=True)]
    llm = FakeLLM(
        responder=lambda _m: json.dumps(
            {"rule": "Answer the exact question first.", "rationale": "r"}
        )
    )
    ctx = ForgeContext(llm=llm, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())

    async def evaluate_dev(d: AgentDefinition) -> EvalRun:
        c1 = d.retrieval.top_k >= 7
        c3 = d.chunking.chunk_words == 160
        c2 = any("exact question" in r for r in d.rules)
        return run_of(
            [
                result(cases[0], c1, None if c1 else "retrieval_miss"),
                result(cases[1], c2, None if c2 else "prompt_defect", answer="something"),
                result(cases[2], c3, None if c3 else "retrieval_miss"),
                result(cases[3], True),
            ],
            d.version,
        )

    v0 = definition()
    out = await optimize(ctx, v0, await evaluate_dev(v0), evaluate_dev, max_iterations=3)
    decisions = [(e.iteration, e.patch.op.op, e.accepted) for e in out.changelog]
    assert decisions == [(1, "set_top_k", True), (2, "add_rule", True)]
    assert out.best.version == 2
    assert out.best_run.scorecard.splits["dev"].pass_rate == 1.0
    assert out.stop_reason == "every dev case passes"


async def test_loop_stops_when_nothing_helps() -> None:
    cases = [case("c1")]
    llm = FakeLLM(
        responder=lambda _m: json.dumps({"rule": "Be more careful with facts.", "rationale": "r"})
    )
    ctx = ForgeContext(llm=llm, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())

    async def stuck(d: AgentDefinition) -> EvalRun:
        return run_of([result(cases[0], False, "prompt_defect", answer="x")], d.version)

    v0 = definition()
    out = await optimize(ctx, v0, await stuck(v0), stuck)
    assert out.best.version == 0
    assert all(not e.accepted for e in out.changelog)
    assert "found no patch" in out.stop_reason


async def test_loop_respects_budget() -> None:
    cases = [case("c1")]
    ctx = ForgeContext(
        llm=FakeLLM(["{}"]), models=ModelRoles("gen", "judge"), embedder=HashEmbedder()
    )
    spent = {"t": 0}

    async def costly(d: AgentDefinition) -> EvalRun:
        spent["t"] += 1000
        return run_of([result(cases[0], False, "retrieval_miss")], d.version)

    v0 = definition()
    out = await optimize(
        ctx,
        v0,
        await costly(v0),
        costly,
        tokens_used=lambda: spent["t"],
        tokens_left=lambda: 1500 - spent["t"],
    )
    assert out.stop_reason == "token budget would not cover another evaluation"
    assert len(out.changelog) == 1


def test_wilson_interval() -> None:
    lo, hi = wilson(7, 9)
    assert 0.4 < lo < 0.5
    assert 0.9 < hi < 0.97
    assert wilson(0, 0) == (0.0, 0.0)


# ---------- Whole factory -------------------------------------------------------------


async def test_run_factory_end_to_end(tmp_path: Path) -> None:
    docs = build_docs_folder(tmp_path)
    budget = Budget(max_tokens=2_000_000)
    llm = BudgetedLLM(FakeLLM(responder=sim.respond), budget)
    ctx = ForgeContext(llm=llm, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())
    store = RunStore(tmp_path / "runs", "factory")
    messages: list[str] = []
    result = await run_factory(
        ctx,
        store,
        BRIEF,
        docs,
        AutoAnswerer(ctx),
        budget,
        min_cases=5,
        tournament=True,
        progress=messages.append,
    )

    files = set(store.list_artifacts())
    for name in (
        "requirements.md",
        "groundability.md",
        "evals.json",
        "evals.md",
        "canaries.json",
        "agent.v0.json",
        "agent.final.json",
        "changelog.md",
        "scorecard.json",
        "report.md",
        "evals/holdout-v0.json",
        "budget.json",
        "tournament.json",
    ):
        assert name in files, name
    report = store.read_text("report.md")
    for heading in (
        "## Headline",
        "## Score per iteration (dev)",
        "## Change log",
        "## What still fails, and why",
        "## How far to trust these numbers",
        "## Prompt tournament (dev)",
    ):
        assert heading in report
    assert result.scorecard.holdout_cases > 0
    assert result.scorecard.iterations[0].version == 0
    assert any("critic-optimizer" in m for m in messages)

    # Resume reuses requirements, agent and evals instead of regenerating them.
    calls_before = len(llm._inner.calls)  # type: ignore[attr-defined]
    await run_factory(
        ctx,
        store,
        BRIEF,
        docs,
        AutoAnswerer(ctx),
        budget,
        min_cases=5,
        resume=True,
        max_iterations=0,
    )
    tasks = [m.messages[-1].content.split("\n", 1)[0] for m in llm._inner.calls[calls_before:]]  # type: ignore[attr-defined]
    assert not any(
        t.startswith(("TASK: intake", "TASK: generate.agent", "TASK: evals")) for t in tasks
    )


async def test_refuse_topic_patch_generalises_and_rejects_copies() -> None:
    from forgeai.stages.optimize import critic_refuse_topic_patch

    answered = [result(case("c9", refuse=True), False, "wrong_refusal", refused=False, answer="x")]
    answered[0].case.question = "Are there any changes to the payment terms for port services?"
    good = FakeLLM([json.dumps({"topic": "future or planned changes to rules", "rationale": "r"})])
    ctx = ForgeContext(llm=good, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())
    patch = await critic_refuse_topic_patch(ctx, answered, definition())
    assert patch is not None
    assert patch.op == AddRefuseTopic(topic="future or planned changes to rules")

    copy = FakeLLM([json.dumps({"topic": "changes to the payment terms for port services"})])
    ctx = ForgeContext(llm=copy, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())
    assert await critic_refuse_topic_patch(ctx, answered, definition()) is None
