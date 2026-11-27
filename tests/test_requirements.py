from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeai.errors import GroundingError
from forgeai.ingest import chunk_corpus, load_folder
from forgeai.llm import FakeLLM
from forgeai.pipeline import build_requirements
from forgeai.prompts import render
from forgeai.retrieval import HashEmbedder
from forgeai.schemas.requirements import (
    GroundabilityReport,
    GroundingCheck,
    IntakeQuestion,
    IntakeQuestions,
    Requirements,
    Topic,
    TopicJudgement,
)
from forgeai.stages.context import ForgeContext, ModelRoles
from forgeai.stages.guard import apply_guard, verify_judgement
from forgeai.stages.intake import (
    AutoAnswerer,
    FileAnswerer,
    InteractiveAnswerer,
    requirements_markdown,
    summarize_corpus,
    summary_text,
)
from forgeai.store import RunStore
from tests import scripted
from tests.fixtures import build_docs_folder

BRIEF = "I need an assistant that answers HR policy questions for shore staff."


def make_ctx(guard: dict[str, dict[str, str]] | None = None) -> tuple[ForgeContext, FakeLLM]:
    table = scripted.GUARD if guard is None else guard
    llm = FakeLLM(responder=lambda m: scripted.responder(m, table))
    ctx = ForgeContext(llm=llm, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())
    return ctx, llm


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    return build_docs_folder(tmp_path)


# ---------- End to end (scripted model) -----------------------------------------------


async def test_requirements_pipeline_end_to_end(tmp_path: Path, docs: Path) -> None:
    ctx, llm = make_ctx()
    store = RunStore(tmp_path / "runs", "hr-test")
    outcome = await build_requirements(ctx, store, BRIEF, docs, AutoAnswerer(ctx))

    # At most five questions, renumbered, even though the model asked seven.
    assert [q.id for q in outcome.record.questions] == ["q1", "q2", "q3", "q4", "q5"]

    sources = [a.source for a in outcome.record.answers]
    assert sources == ["brief", "default", "default", "default", "default"]

    req = outcome.requirements
    topics = [t.topic for t in req.must_answer]
    # Duplicate removed; topics the documents cannot support moved out.
    assert topics == ["Annual leave", "Berth lookup"]
    gaps = {g.topic for g in req.flagged_gaps}
    assert gaps == {"Sick leave", "Salary bands", "Odd hint"}
    assert {"Sick leave", "Salary bands"} <= {r.topic for r in req.must_refuse}
    assert len(req.assumptions) == 5
    assert req.approved_by == "auto"

    checks = {c.topic: c for c in outcome.groundability.checks}
    assert checks["Annual leave"].quote_verified
    assert checks["Annual leave"].chunk_id != "wrong-id"  # corrected to the real chunk
    assert checks["Sick leave"].claimed == "supported"
    assert checks["Sick leave"].verdict == "unsupported"  # fabricated quote caught
    assert "quote is not in any retrieved passage" in checks["Sick leave"].note
    # A tool topic without a tool hint is treated as a documents topic and checked.
    assert checks["Odd hint"].verdict == "unsupported"

    assert set(store.list_artifacts()) >= {
        "brief.md",
        "intake.json",
        "requirements.draft.json",
        "groundability.json",
        "groundability.md",
        "requirements.json",
        "requirements.md",
    }
    assert {c.model for c in llm.calls} == {"gen"}


async def test_human_checkpoint_can_edit(tmp_path: Path, docs: Path) -> None:
    ctx, _ = make_ctx()
    store = RunStore(tmp_path / "runs", "hr-approve")

    def approve(req: Requirements) -> Requirements:
        return req.model_copy(update={"tone": "Formal"})

    outcome = await build_requirements(ctx, store, BRIEF, docs, AutoAnswerer(ctx), approve=approve)
    assert outcome.requirements.tone == "Formal"
    assert outcome.requirements.approved_by == "requester"
    assert "_Approved by: requester_" in store.read_text("requirements.md")


async def test_factory_refuses_to_build_when_nothing_is_grounded(
    tmp_path: Path, docs: Path
) -> None:
    # Prose only (no CSV), and the guard model finds nothing: every topic fails.
    folder = tmp_path / "prose"
    folder.mkdir()
    (folder / "canteen.md").write_text("# Canteen\n\nLunch is served from 12 to 2.", "utf-8")
    ctx, _ = make_ctx(guard={})
    store = RunStore(tmp_path / "runs", "hr-empty")
    with pytest.raises(GroundingError, match="support none"):
        await build_requirements(ctx, store, BRIEF, folder, AutoAnswerer(ctx))
    assert store.exists("groundability.json")
    assert not store.exists("requirements.json")


async def test_table_lookup_needs_a_table(tmp_path: Path) -> None:
    folder = tmp_path / "prose"
    folder.mkdir()
    (folder / "leave.md").write_text(
        "# Leave\n\nShore staff earn 24 days of annual leave per calendar year.",
        encoding="utf-8",
    )
    ctx, _ = make_ctx()
    store = RunStore(tmp_path / "runs", "no-table")
    outcome = await build_requirements(ctx, store, BRIEF, folder, AutoAnswerer(ctx))
    assert "Berth lookup" in {g.topic for g in outcome.requirements.flagged_gaps}


async def test_rejects_empty_brief_and_empty_folder(tmp_path: Path, docs: Path) -> None:
    ctx, _ = make_ctx()
    store = RunStore(tmp_path / "runs", "bad-input")
    with pytest.raises(ValueError, match="too short"):
        await build_requirements(ctx, store, "help", docs, AutoAnswerer(ctx))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(GroundingError, match="no readable documents"):
        await build_requirements(ctx, store, BRIEF, empty, AutoAnswerer(ctx))


# ---------- Units ---------------------------------------------------------------------


def test_question_cap_is_enforced_in_code() -> None:
    qs = IntakeQuestions.model_validate(scripted.QUESTIONS)
    assert len(qs.questions) == 5


def _questions() -> list[IntakeQuestion]:
    return [
        IntakeQuestion(id="q1", question="Who uses it?", why="w", default_answer="All staff"),
        IntakeQuestion(id="q2", question="Which tone?", why="w", default_answer="Neutral"),
    ]


async def test_interactive_enter_accepts_default() -> None:
    replies = iter(["Shore staff", ""])
    answerer = InteractiveAnswerer(lambda _q, _d: next(replies))
    answers = await answerer.answer(BRIEF, _questions())
    assert [a.answer for a in answers] == ["Shore staff", "Neutral"]
    assert {a.source for a in answers} == {"requester"}


async def test_file_answers_by_id_or_text(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"who uses it?": "Engineers"}), encoding="utf-8")
    answers = await FileAnswerer(path).answer(BRIEF, _questions())
    assert answers[0].answer == "Engineers"
    assert answers[0].source == "requester"
    assert answers[1].answer == "Neutral"
    assert answers[1].source == "default"


def test_file_answers_must_be_an_object(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        FileAnswerer(path)


def test_corpus_summary_lists_sections_and_columns(docs: Path) -> None:
    corpus = load_folder(docs)
    text = summary_text(summarize_corpus(corpus, chunk_corpus(corpus)))
    assert "3. Sick leave" in text
    assert "columns: berth_id, port" in text
    assert "malware.exe (unsupported type .exe)" in text


def test_verify_rejects_quote_without_words_from_passage(docs: Path) -> None:
    from forgeai.retrieval import HybridIndex

    corpus = load_folder(docs)
    hits = HybridIndex(chunk_corpus(corpus), HashEmbedder()).search("annual leave")
    topic = Topic(topic="Annual leave")
    no_quote = verify_judgement(topic, TopicJudgement(status="partial"), hits)
    assert no_quote.verdict == "unsupported"
    partial = verify_judgement(
        topic,
        TopicJudgement(status="partial", quote="Up to 5 unused days may be carried forward"),
        hits,
    )
    assert partial.verdict == "partial"
    assert partial.quote_verified


def test_apply_guard_is_a_noop_when_all_supported() -> None:
    req = Requirements(
        agent_name="HR Desk",
        purpose="p",
        scope="s",
        audience="a",
        tone="t",
        must_answer=[Topic(topic="Leave")],
        must_refuse=[{"topic": "Legal", "reason": "r"}],  # type: ignore[list-item]
        brief=BRIEF,
    )
    report = GroundabilityReport(
        checks=[GroundingCheck(topic="Leave", claimed="supported", verdict="supported")]
    )
    assert apply_guard(req, report) is req
    assert report.coverage == 1.0


def test_markdown_has_every_section(tmp_path: Path) -> None:
    req = Requirements.model_validate(
        {
            **scripted.REQUIREMENTS,
            "must_answer": scripted.REQUIREMENTS["must_answer"][:1],
            "brief": BRIEF,
            "assumptions": ["Who? -> All staff (interviewer default)"],
            "flagged_gaps": [{"topic": "Salary bands", "reason": "not in docs"}],
        }
    )
    md = requirements_markdown(req)
    for heading in ("## Must answer", "## Must refuse", "## Flagged gaps", "## Assumptions"):
        assert heading in md


def test_prompt_render_fails_on_missing_variable() -> None:
    with pytest.raises(KeyError):
        render("intake_questions", brief="x")
    assert render("intake_questions", brief="B", corpus_summary="S").startswith("TASK:")


# ---------- Phase 6.1: personal-records asks --------------------------------------------


def test_personal_records_ask_is_split_off_and_refused() -> None:
    """Real HR run: the model merged 'how many leave days are left' into 'annual leave
    entitlement', and a real entitlement quote let it pass the guard."""
    from forgeai.schemas.corpus import Corpus, Document, Page
    from forgeai.stages.guard import _personal_record_checks

    req = Requirements(
        agent_name="HR Desk",
        purpose="p",
        scope="s",
        audience="a",
        tone="t",
        must_answer=[
            Topic(
                topic="annual leave entitlement",
                detail="How many leave days an employee has left this year.",
            ),
            Topic(topic="remote work policy", detail="days per week allowed"),
        ],
        must_refuse=[{"topic": "Legal advice", "reason": "r"}],  # type: ignore[list-item]
        brief=BRIEF,
    )
    corpus = Corpus(
        root=".",
        documents=[
            Document(
                doc_id="leave",
                filename="l.pdf",
                kind="pdf",
                title="L",
                pages=[Page(number=1, text="x")],
            )
        ],
    )
    checks = _personal_record_checks(req, corpus)
    assert [c.topic for c in checks] == ["annual leave entitlement"]
    out = apply_guard(req, GroundabilityReport(checks=checks))
    assert [(t.topic, t.detail) for t in out.must_answer] == [
        ("annual leave entitlement", ""),  # the general topic stays answerable
        ("remote work policy", "days per week allowed"),
    ]
    assert out.must_refuse[-1].topic == "How many leave days an employee has left this year"
    assert out.flagged_gaps[0].topic == "How many leave days an employee has left this year"


@pytest.mark.parametrize(
    ("text", "personal"),
    [
        ("Remaining leave balance: days an employee has left this year", True),
        ("Leave balance for each employee", True),
        ("How many sick days do I have remaining", True),
        ("Annual leave entitlement: days earned per year", False),
        ("Carry forward of unused days into the next year", False),
    ],
)
def test_personal_records_pattern(text: str, personal: bool) -> None:
    from forgeai.stages.guard import PERSONAL_RECORDS

    assert bool(PERSONAL_RECORDS.search(text)) is personal
