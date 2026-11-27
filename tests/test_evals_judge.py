from __future__ import annotations

from pathlib import Path

import pytest

from forgeai.errors import GroundingError
from forgeai.ingest import load_folder
from forgeai.llm import FakeLLM
from forgeai.pipeline import build_agent, build_evals, build_requirements, evaluate_agent
from forgeai.retrieval import HashEmbedder
from forgeai.runtime.agent import AgentTurn, RetrievedRef, ToolCallRecord
from forgeai.schemas.agent import AgentAnswer, Citation
from forgeai.schemas.evals import Checks, EvalCase, EvalSet, ExpectedSource, FactCheck
from forgeai.stages.context import ForgeContext, ModelRoles
from forgeai.stages.evalgen import (
    EvalTargets,
    assign_splits,
    generate_evals,
    template_cases,
    validate_candidate,
)
from forgeai.stages.intake import AutoAnswerer
from forgeai.stages.judge import (
    canary_answers,
    classify,
    layer_one,
    run_canaries,
    string_fact_check,
    unsupported_numbers,
)
from forgeai.store import RunStore
from tests import sim
from tests.fixtures import build_docs_folder

BRIEF = "I need an assistant that answers HR policy questions for shore staff."


@pytest.fixture
def ctx() -> ForgeContext:
    llm = FakeLLM(responder=sim.respond)
    return ForgeContext(llm=llm, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())


async def _setup(ctx: ForgeContext, tmp_path: Path):  # type: ignore[no-untyped-def]
    docs = build_docs_folder(tmp_path)
    store = RunStore(tmp_path / "runs", "eval-test")
    outcome = await build_requirements(ctx, store, BRIEF, docs, AutoAnswerer(ctx))
    definition = await build_agent(ctx, store, outcome, docs)
    return docs, store, outcome.requirements, definition


# ---------- Eval generation -----------------------------------------------------------


async def test_generated_cases_are_all_verified(ctx: ForgeContext, tmp_path: Path) -> None:
    docs, _store, req, _ = await _setup(ctx, tmp_path)
    corpus = load_folder(docs)
    evals = await generate_evals(ctx, req, corpus)

    counts = evals.counts()
    assert counts["answerable"] >= 5
    assert counts["out_of_scope"] == 6
    assert counts.get("multi_doc", 0) >= 1
    assert counts.get("tool", 0) == 1  # one table lookup template (no arithmetic hint)

    # The planted lie (a sick-leave fact not in the passage) was caught and logged.
    reasons = [r.reason for r in evals.rejected]
    assert any("not in the quoted source" in r for r in reasons)

    for case in evals.cases:
        assert case.id.startswith("c")
        for source in case.sources:
            assert corpus.get(source.doc_id) is not None
        if not case.expect_refusal and case.origin == "llm":
            joined = " ".join(s.quote for s in case.sources)
            assert all(
                f.lower().replace(",", "") in joined.lower().replace(",", "")
                for f in case.key_facts
            )

    # Stratified split: every category with 2+ cases appears in both halves.
    for category, n in counts.items():
        if n >= 2:
            splits = {c.split for c in evals.cases if c.category == category}
            assert splits == {"dev", "holdout"}, category


async def test_too_few_cases_stops_the_run(ctx: ForgeContext, tmp_path: Path) -> None:
    docs, store, req, _ = await _setup(ctx, tmp_path)
    import forgeai.pipeline as pipeline

    original = pipeline.generate_evals

    async def tiny(*args, **kwargs):  # type: ignore[no-untyped-def]
        return await original(
            *args,
            targets=EvalTargets(answerable=1, ambiguous=0, multi_doc=0, out_of_scope=1, refill=0),
        )

    pipeline.generate_evals = tiny  # type: ignore[assignment]
    try:
        with pytest.raises(GroundingError, match="survived validation"):
            await build_evals(ctx, store, req, docs)
    finally:
        pipeline.generate_evals = original  # type: ignore[assignment]
    assert store.exists("evals.md")


@pytest.mark.parametrize(
    ("question", "facts", "quote", "reason"),
    [
        (
            "How many leave days?",
            ["24 days"],
            "Staff earn 25 days of leave.",
            "not in the quoted source",
        ),
        (
            "Do staff earn 24 days?",
            ["24 days"],
            "Staff earn 24 days of leave.",
            "contains its own answer",
        ),
        (
            "How many leave days?",
            ["24 days"],
            "Staff earn 24 days of leave per year here.",
            "verbatim span",
        ),
        ("Hm?", ["24 days"], "Staff earn 24 days of leave.", "too short"),
    ],
)
def test_validate_candidate_rejections(
    question: str, facts: list[str], quote: str, reason: str
) -> None:
    passage = "Staff earn 24 days of leave. Staff earn 25 days of leave."
    if reason == "verbatim span":
        passage = "Staff earn 24 days of leave."
    got = validate_candidate("answerable", question, facts, [("doc", quote, passage)])
    assert got is not None
    assert reason in got


def test_template_cases_compute_arithmetic(tmp_path: Path) -> None:
    from forgeai.schemas.requirements import Requirements, Topic

    req = Requirements(
        agent_name="Port desk",
        purpose="p",
        scope="s",
        audience="a",
        tone="t",
        must_answer=[
            Topic(topic="berths", answered_from="tool", tool_hints=["table_lookup"]),
            Topic(topic="dues", answered_from="tool", tool_hints=["arithmetic"]),
            Topic(topic="weather", answered_from="tool", tool_hints=["weather"]),
        ],
        must_refuse=[{"topic": "legal"}],  # type: ignore[list-item]
        brief="b",
    )
    corpus = load_folder(build_docs_folder(tmp_path))
    cases = [c.case for c in template_cases(req, corpus)]
    lookup, calc, weather = cases
    assert lookup.expected_tool == "csv_lookup"
    assert lookup.key_facts == ["180"]  # berth B2 (middle row), first numeric column max_loa_m
    assert calc.key_facts == ["7800"]  # 3 x 2600, computed by code
    assert "3 days" in calc.question
    assert weather.expected_tool == "weather"
    assert "Kochi" in weather.question


def test_split_is_deterministic_and_stratified() -> None:
    cases = [
        EvalCase(id=f"c{i}", category="answerable", question=f"Question number {i}?")
        for i in range(10)
    ]
    a, b = assign_splits(cases), assign_splits(list(reversed(cases)))
    assert sorted((c.question, c.split) for c in a) == sorted((c.question, c.split) for c in b)
    assert sum(c.split == "dev" for c in a) == 6


# ---------- Judge: deterministic layer -------------------------------------------------


def _turn(
    answer: str,
    *,
    refused: bool = False,
    citations: list[Citation] | None = None,
    retrieved: list[str] | None = None,
    tools: list[ToolCallRecord] | None = None,
) -> AgentTurn:
    return AgentTurn(
        question="How many days of annual leave?",
        answer=AgentAnswer(answer=answer, refused=refused, citations=citations or []),
        retrieved=[RetrievedRef(chunk_id=c, score=1, bm25=1, dense=0.9) for c in retrieved or []],
        valid_citations=citations or [],
        tool_calls=tools or [],
        grounded=True,
    )


CHUNKS = {
    "leave:p1:c1": "Shore staff earn 24 days of annual leave per calendar year. Up to 5 unused days carry forward."
}
CASE = EvalCase(
    id="c01",
    category="answerable",
    question="How many days of annual leave?",
    sources=[
        ExpectedSource(
            doc_id="leave", quote="Shore staff earn 24 days of annual leave per calendar year."
        )
    ],
    key_facts=["24 days"],
)


def test_layer_one_passes_a_correct_cited_answer() -> None:
    turn = _turn(
        "You get 24 days per year.",
        citations=[Citation(chunk_id="leave:p1:c1", quote="x")],
        retrieved=["leave:p1:c1"],
    )
    checks, confirmed, undecided = layer_one(CASE, turn, CHUNKS)
    assert checks.refusal_ok
    assert checks.source_retrieved
    assert checks.source_cited
    assert [f.status for f in confirmed] == ["present"]
    assert undecided == []


def test_unsupported_number_is_flagged_as_hallucination() -> None:
    turn = _turn("You get 24 days, plus 12 bonus days after 3 years.", retrieved=["leave:p1:c1"])
    assert unsupported_numbers(turn, CHUNKS) == ["12"]  # 3 is below 10 and ignored


def test_classify_priorities() -> None:
    base = Checks(refusal_ok=True, grounded=True, source_retrieved=True)
    present = [FactCheck(fact="24 days", status="present", by="string")]
    missing = [FactCheck(fact="24 days", status="missing", by="judge")]
    wrong = [FactCheck(fact="24 days", status="contradicted", by="judge")]
    assert classify(CASE, base.model_copy(update={"refusal_ok": False}), present) == "wrong_refusal"
    assert (
        classify(CASE, base.model_copy(update={"source_retrieved": False}), missing)
        == "retrieval_miss"
    )
    assert classify(CASE, base, wrong) == "hallucination"
    assert (
        classify(CASE, base.model_copy(update={"unsupported_numbers": ["12"]}), present)
        == "hallucination"
    )
    assert classify(CASE, base, missing) == "prompt_defect"
    assert (
        classify(CASE, base.model_copy(update={"contract_error": "bad json"}), missing)
        == "prompt_defect"
    )


@pytest.mark.parametrize(
    ("fact", "answer", "expected"),
    [
        ("24 days", "You get 24 days.", "present"),
        ("INR 1,500", "The allowance is INR 1500 per day.", "present"),
        (
            "INR 1,500",
            "The allowance is 1,500 rupees.",
            None,
        ),  # number matches, words don't: ask judge
        ("24 days", "You get twenty-four days.", None),
        ("26 weeks", "Mothers get 26 weeks of paid leave.", "present"),
    ],
)
def test_string_fact_check(fact: str, answer: str, expected: str | None) -> None:
    assert string_fact_check(fact, answer) == expected


def test_tool_expectation_checked() -> None:
    case = CASE.model_copy(update={"expected_tool": "csv_lookup", "category": "tool"})
    ok = _turn(
        "24 days", tools=[ToolCallRecord(name="csv_lookup", args={}, ok=True, output="24 days")]
    )
    bad = _turn("24 days")
    assert layer_one(case, ok, CHUNKS)[0].tool_ok
    assert layer_one(case, bad, CHUNKS)[0].tool_ok is False


# ---------- Canaries ------------------------------------------------------------------


def test_canary_answers_cover_the_bias_modes() -> None:
    kinds = {k: should for k, _a, should in canary_answers(CASE)}
    assert kinds == {
        "correct_terse": True,
        "correct_padded": True,
        "confident_omission": False,
        "wrong_number": False,
    }
    wrong = next(a for k, a, _ in canary_answers(CASE) if k == "wrong_number")
    assert "55 days" in wrong  # 24 * 2 + 7


async def test_good_judge_passes_canaries(ctx: ForgeContext) -> None:
    results = await run_canaries(ctx, [CASE])
    assert len(results) == 4
    assert all(r.correct for r in results)


async def test_lenient_judge_is_corrected_by_evidence_check() -> None:
    """The real llama judge on hr-mini passed a confident omission and a wrong number.
    Making it quote the answer, and checking that quote in code, catches both."""
    import json

    def lenient(messages):  # type: ignore[no-untyped-def]
        answer = (
            messages[-1].content.split("ANSWER TO GRADE:", 1)[1].split("For each", 1)[0].strip()
        )
        # Always "present"; quotes the first words of the answer as evidence.
        return json.dumps({"facts": [{"fact": "x", "status": "present", "evidence": answer[:40]}]})

    ctx = ForgeContext(
        llm=FakeLLM(responder=lenient), models=ModelRoles("gen", "judge"), embedder=HashEmbedder()
    )
    results = {r.kind: r for r in await run_canaries(ctx, [CASE])}
    assert not results["confident_omission"].judge_passed
    assert not results["wrong_number"].judge_passed


@pytest.mark.parametrize(
    ("fact", "status", "evidence", "answer", "expected"),
    [
        ("24 days", "present", "24 days a year", "You get 24 days a year.", "present"),
        ("24 days", "present", "24 days a year", "You get a generous allowance.", "missing"),
        (
            "6 hours",
            "present",
            "under 19 hours",
            "Economy for flights under 19 hours.",
            "contradicted",
        ),
        ("24 days", "present", "", "You get 24 days.", "missing"),
        ("Economy class", "present", "economy class", "Book economy class.", "present"),
        ("24 days", "contradicted", "30 days", "You get 30 days.", "contradicted"),
    ],
)
def test_judge_evidence_check(
    fact: str, status: str, evidence: str, answer: str, expected: str
) -> None:
    from forgeai.stages.judge import check_judge_evidence

    assert check_judge_evidence(fact, status, evidence, answer) == expected  # type: ignore[arg-type]


# ---------- End to end ----------------------------------------------------------------


async def test_evaluate_end_to_end(ctx: ForgeContext, tmp_path: Path) -> None:
    docs, store, req, definition = await _setup(ctx, tmp_path)
    evals = await generate_evals(ctx, req, load_folder(docs))
    store.write_json("evals.json", evals)
    run = await evaluate_agent(ctx, store, definition, evals, docs, "all")

    assert {r.case.split for r in run.results} == {"dev", "holdout"}
    assert set(run.scorecard.splits) == {"dev", "holdout"}
    assert run.scorecard.judge_trusted
    dev = run.scorecard.splits["dev"]
    assert 0 < dev.passed <= dev.cases
    # Every failure is classified into one of the four categories.
    for r in run.results:
        if not r.judgement.passed:
            assert r.judgement.failure in {
                "retrieval_miss",
                "prompt_defect",
                "hallucination",
                "wrong_refusal",
            }
            assert r.judgement.reason
    assert store.exists("evals/v0-all.json")
    md = store.read_text("evals/v0-all.md")
    assert "Judge canary accuracy" in md
    # Judge calls used the judge model; agent calls used the generator.
    models = {c.model for c in ctx.llm.calls}  # type: ignore[attr-defined]
    assert models == {"gen", "judge"}


def test_evalset_helpers() -> None:
    evals = EvalSet(cases=[CASE, CASE.model_copy(update={"id": "c02", "split": "holdout"})])
    assert len(evals.split("dev")) == 1
    assert len(evals.split("all")) == 2
    assert evals.counts() == {"answerable": 2}


# ---------- Phase 4.1: fixes found on the first real-model run ---------------------------


@pytest.mark.parametrize(
    ("question", "facts", "kept"),
    [
        # Direct answer kept; an unrelated rule the question never asked about is dropped.
        (
            "What is the daily meal allowance for domestic travel?",
            ["INR 1,500", "Receipts are required for any single expense above INR 2,000"],
            ["INR 1,500"],
        ),
        # A second fact the question does ask about is kept.
        (
            "How many unused annual leave days can be carried forward, and how many are earned?",
            ["24 days", "5 unused days may be carried forward"],
            ["24 days", "5 unused days may be carried forward"],
        ),
        # Whole sentences are not key facts.
        (
            "How do I apply for leave?",
            [
                "Apply through the HR portal at least 10 working days before planned leave in all cases"
            ],
            [],
        ),
        ("How many days?", ["24 days."], ["24 days"]),
    ],
)
def test_clean_facts(question: str, facts: list[str], kept: list[str]) -> None:
    from forgeai.stages.evalgen import clean_facts

    assert clean_facts(question, facts) == kept


def test_context_dependent_question_rejected() -> None:
    reason = validate_candidate(
        "answerable",
        "What does this policy explain?",
        ["leave"],
        [("d", "This policy explains leave for staff.", "This policy explains leave for staff.")],
    )
    assert reason is not None
    assert "unseen context" in reason


async def test_in_scope_question_cannot_pose_as_out_of_scope(tmp_path: Path) -> None:
    """On hr-mini the generator wrote 'What are the leave entitlements for a new employee?'
    as out-of-scope. The answerability check now rejects such questions."""
    import json

    from forgeai.ingest import chunk_corpus
    from forgeai.schemas.corpus import ChunkingConfig
    from forgeai.stages.evalgen import _refusal_cases
    from tests import scripted

    def handler(messages):  # type: ignore[no-untyped-def]
        task = scripted._task(messages)
        if task == "evals.out_of_scope":
            return json.dumps(
                {
                    "questions": [
                        {
                            "question": "How many days of annual leave do shore staff get?",
                            "topic": "x",
                        },
                        {"question": "What is the CEO's home address?", "topic": "privacy"},
                    ]
                }
            )
        if task == "evals.answerability":
            if "annual leave" in messages[-1].content.split("PASSAGES:")[0]:
                return json.dumps(
                    {
                        "answered": True,
                        "chunk_id": "?",
                        "quote": "Shore staff earn 24 days of annual leave per calendar year",
                    }
                )
            return json.dumps({"answered": False})
        raise AssertionError(task)

    ctx = ForgeContext(
        llm=FakeLLM(responder=handler), models=ModelRoles("gen", "judge"), embedder=HashEmbedder()
    )
    docs = build_docs_folder(tmp_path)
    chunks = chunk_corpus(load_folder(docs), ChunkingConfig())
    from forgeai.schemas.requirements import Requirements

    req = Requirements.model_validate(
        {**__import__("tests.scripted", fromlist=["x"]).REQUIREMENTS, "brief": BRIEF}
    )
    rejected: list = []  # type: ignore[type-arg]
    kept = await _refusal_cases(ctx, req, chunks, 2, rejected)
    assert [c.case.question for c in kept] == ["What is the CEO's home address?"]
    assert "answerable from the documents" in rejected[0].reason


def test_composed_multi_doc_joins_verified_cases_from_two_documents() -> None:
    from forgeai.stages.evalgen import _Candidate, compose_multi_doc

    def single(doc: str, q: str, fact: str) -> _Candidate:
        return _Candidate(
            EvalCase(
                id="",
                category="answerable",
                question=q,
                key_facts=[fact],
                sources=[ExpectedSource(doc_id=doc, quote=f"{fact} is the rule here.")],
            ),
            [f"{fact} is the rule here."],
        )

    singles = [
        single("leave", "How many days of sick leave do staff get?", "12 days"),
        single("leave", "How many days of annual leave do staff get?", "24 days"),
        single("travel", "What is the daily allowance for domestic travel?", "INR 1,800"),
    ]
    multi, unused = compose_multi_doc(singles, 2, HashEmbedder())
    assert len(multi) == 1  # only one cross-document pair is possible without reuse
    case = multi[0].case
    assert {s.doc_id for s in case.sources} == {"leave", "travel"}
    assert "? Also, " in case.question
    assert len(case.key_facts) == 2
    assert len(unused) == 1


def test_wrong_label_out_of_scope_caught_by_similarity_backstop(tmp_path: Path) -> None:
    """Real HR run: qwen said 'How do we handle sick leave for our shore staff?' was not
    answered by the documents. Its similarity to the documents gives it away."""
    import asyncio
    import json

    from forgeai.ingest import chunk_corpus
    from forgeai.schemas.corpus import ChunkingConfig
    from forgeai.schemas.requirements import Requirements
    from forgeai.stages.evalgen import _refusal_cases
    from tests import scripted

    def handler(messages):  # type: ignore[no-untyped-def]
        task = scripted._task(messages)
        if task == "evals.out_of_scope":
            return json.dumps(
                {
                    "questions": [
                        {
                            "question": "How do employees get paid sick days and a medical certificate?"
                        },
                        {"question": "What is the CEO's home address?"},
                    ]
                }
            )
        return json.dumps({"answered": False})  # a lenient model: says nothing is answered

    ctx = ForgeContext(
        llm=FakeLLM(responder=handler), models=ModelRoles("gen", "judge"), embedder=HashEmbedder()
    )
    chunks = chunk_corpus(load_folder(build_docs_folder(tmp_path)), ChunkingConfig())
    # No "Sick leave" must-answer topic here, so only the similarity backstop can catch it.
    topics = [t for t in scripted.REQUIREMENTS["must_answer"] if "sick" not in str(t).lower()]
    req = Requirements.model_validate(
        {**scripted.REQUIREMENTS, "must_answer": topics, "brief": BRIEF}
    )
    rejected: list = []  # type: ignore[type-arg]
    kept = asyncio.run(_refusal_cases(ctx, req, chunks, 2, rejected, similarity_floor=0.3))
    assert [c.case.question for c in kept] == ["What is the CEO's home address?"]
    assert "answerable-question floor" in rejected[0].reason


@pytest.mark.parametrize(
    ("question", "topic"),
    [
        # From the second real HR run: labelled out-of-scope, but plainly in scope.
        ("What are the HR policies for remote work?", "Remote work policy"),
        ("How do we handle gifts and hospitality for our clients?", "Gifts and hospitality policy"),
        # Real refusals that mention an in-scope topic: personal, future, or an action.
        ("I want to know the status of my leave balance this year.", None),
        ("What are the plans for travel advances in the next quarter?", None),
        ("Could you tell me about the recent changes in remote work rules?", None),
        ("How many days has John taken off this year?", None),
    ],
)
def test_in_scope_topic(question: str, topic: str | None) -> None:
    from forgeai.schemas.requirements import Requirements, Topic
    from forgeai.stages.evalgen import in_scope_topic

    req = Requirements(
        agent_name="HR Desk",
        purpose="p",
        scope="s",
        audience="a",
        tone="t",
        must_answer=[
            Topic(topic="Remote work policy"),
            Topic(topic="Gifts and hospitality policy"),
            Topic(topic="Annual leave entitlement"),
            Topic(topic="Travel advances"),
        ],
        must_refuse=[{"topic": "Legal advice"}],  # type: ignore[list-item]
        brief=BRIEF,
    )
    assert in_scope_topic(question, req) == topic


def test_personal_topic_name_is_dropped_not_kept() -> None:
    """Second real HR run: the model made a separate topic 'Remaining leave balance'."""
    from forgeai.schemas.corpus import Corpus, Document, Page
    from forgeai.schemas.requirements import GroundabilityReport, Requirements, Topic
    from forgeai.stages.guard import _personal_record_checks, apply_guard

    req = Requirements(
        agent_name="HR Desk",
        purpose="p",
        scope="s",
        audience="a",
        tone="t",
        must_answer=[
            Topic(topic="Annual leave entitlement", detail="days earned per year"),
            Topic(topic="Remaining leave balance", detail="days an employee has left this year"),
        ],
        must_refuse=[{"topic": "Legal advice"}],  # type: ignore[list-item]
        brief=BRIEF,
    )
    corpus = Corpus(
        root=".",
        documents=[
            Document(
                doc_id="l",
                filename="l.pdf",
                kind="pdf",
                title="L",
                pages=[Page(number=1, text="x")],
            )
        ],
    )
    out = apply_guard(req, GroundabilityReport(checks=_personal_record_checks(req, corpus)))
    assert [t.topic for t in out.must_answer] == ["Annual leave entitlement"]
    assert "days an employee has left this year" in [r.topic for r in out.must_refuse]


def test_refusal_naming_a_document_section_is_answerable(tmp_path: Path) -> None:
    from forgeai.ingest import chunk_corpus
    from forgeai.schemas.corpus import ChunkingConfig
    from forgeai.schemas.requirements import Requirements
    from forgeai.stages.evalgen import names_document_section
    from tests import scripted

    chunks = chunk_corpus(load_folder(build_docs_folder(tmp_path)), ChunkingConfig())
    req = Requirements.model_validate({**scripted.REQUIREMENTS, "brief": BRIEF})
    assert names_document_section("How is parental leave handled?", chunks, req) is not None
    # Names the section, but asks about a group the documents never mention.
    assert names_document_section("Do seafarers get the same parental leave?", chunks, req) is None
    # A refusal cue keeps it a refusal.
    assert names_document_section("What is my parental leave balance?", chunks, req) is None
