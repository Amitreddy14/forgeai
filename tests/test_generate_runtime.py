from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forgeai.ingest import chunk_corpus, load_folder
from forgeai.llm import ChatMessage, FakeLLM
from forgeai.pipeline import agent_markdown, build_agent, build_requirements
from forgeai.retrieval import HashEmbedder, HybridIndex, RetrievalConfig
from forgeai.runtime import AgentRuntime, load_definition, load_runtime, system_prompt
from forgeai.runtime.agent import verify_citations
from forgeai.runtime.loader import changed_documents
from forgeai.schemas.agent import AgentDefinition, Citation
from forgeai.schemas.corpus import ChunkingConfig
from forgeai.stages.context import ForgeContext, ModelRoles
from forgeai.stages.generate import calibrate_threshold, choose_chunking, choose_retrieval
from forgeai.stages.intake import AutoAnswerer
from forgeai.store import RunStore
from tests import scripted
from tests.fixtures import build_docs_folder

BRIEF = "I need an assistant that answers HR policy questions for shore staff."


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    return build_docs_folder(tmp_path)


async def _generated(tmp_path: Path, docs: Path) -> tuple[AgentDefinition, RunStore]:
    llm = FakeLLM(responder=scripted.generator_responder)
    ctx = ForgeContext(llm=llm, models=ModelRoles("gen", "judge"), embedder=HashEmbedder())
    store = RunStore(tmp_path / "runs", "gen-test")
    outcome = await build_requirements(ctx, store, BRIEF, docs, AutoAnswerer(ctx))
    definition = await build_agent(ctx, store, outcome, docs)
    return definition, store


# ---------- Generator -----------------------------------------------------------------


async def test_generator_writes_a_complete_definition(tmp_path: Path, docs: Path) -> None:
    definition, store = await _generated(tmp_path, docs)
    # The brief mentions no weather: the proposed weather tool is rejected.
    assert "weather" not in definition.tool_names
    assert {r.name for r in definition.rejected_tools} == {"weather"}
    # "Berth lookup" (table_lookup) survived the guard because the folder has a CSV, and
    # a table lookup's evidence is the CSV itself, so csv_lookup is added by contract.
    assert definition.tool_names == ["csv_lookup"]
    assert definition.refusal.refuse_topics[:2] == ["Legal advice", "Other employees' records"]
    assert "Sick leave" in definition.refusal.refuse_topics  # moved there by the guard
    assert definition.knowledge.fingerprints  # documents fingerprinted
    assert len(definition.provenance.notes) == 3
    assert store.exists("agent.v0.json")
    md = store.read_text("agent.v0.md")
    assert "~~weather~~" in md
    assert "## Compiled system prompt" in md
    reloaded = load_definition(store.path("agent.v0.json"))
    assert reloaded == definition


def test_chunking_follows_section_length(docs: Path) -> None:
    config, note = choose_chunking(load_folder(docs))
    assert 120 <= config.chunk_words <= 400
    assert config.overlap_words < config.chunk_words
    assert "median section" in note


def test_retrieval_grows_with_corpus(docs: Path) -> None:
    config, _ = choose_retrieval(load_folder(docs))
    assert config.top_k == 6  # 5 documents incl. a CSV


def test_threshold_disabled_when_topics_overlap(docs: Path) -> None:
    from forgeai.schemas.requirements import Requirements

    corpus = load_folder(docs)
    index = HybridIndex(chunk_corpus(corpus), HashEmbedder())
    req = Requirements.model_validate(
        {
            **scripted.REQUIREMENTS,
            "brief": BRIEF,
            "must_refuse": [{"topic": "Annual leave", "reason": "same as an answer topic"}],
        }
    )
    calibration = calibrate_threshold(req, index)
    assert calibration.threshold == 0.0
    assert "overlap" in calibration.note


# ---------- Runtime -------------------------------------------------------------------


def runtime_for(
    definition: AgentDefinition, docs: Path, handler: Any
) -> tuple[AgentRuntime, FakeLLM]:
    llm = FakeLLM(responder=handler)
    corpus = load_folder(docs)
    return AgentRuntime(definition, corpus, HashEmbedder(), llm, "gen"), llm


def answer_json(**kwargs: Any) -> str:
    return json.dumps({"answer": "", "citations": [], "refused": False, **kwargs})


async def test_runtime_answers_with_verified_citation(tmp_path: Path, docs: Path) -> None:
    definition, _ = await _generated(tmp_path, docs)
    definition = definition.model_copy(update={"tools": []})

    def handler(messages: list[ChatMessage]) -> str:
        assert "TASK: agent.answer" in messages[-1].content
        assert "Shore Staff HR Desk" in messages[0].content
        return answer_json(
            answer="Shore staff get 24 days of annual leave a year.",
            citations=[
                {
                    "chunk_id": "made-up",
                    "quote": "Shore staff earn 24 days of annual leave per calendar year",
                }
            ],
        )

    runtime, llm = runtime_for(definition, docs, handler)
    turn = await runtime.answer("How many days of annual leave do I get?")
    assert turn.grounded
    assert turn.valid_citations[0].chunk_id.startswith("leave-policy")
    assert not turn.repaired
    assert len(llm.calls) == 1  # no tools attached -> no planning call


async def test_runtime_repairs_a_fabricated_citation_once(tmp_path: Path, docs: Path) -> None:
    definition, _ = await _generated(tmp_path, docs)
    definition = definition.model_copy(update={"tools": []})
    replies = iter(
        [
            answer_json(
                answer="30 days.",
                citations=[
                    {"chunk_id": "x", "quote": "Employees receive 30 days of paid sick leave"}
                ],
            ),
            answer_json(
                answer="30 days.",
                citations=[
                    {"chunk_id": "x", "quote": "Employees receive 30 days of paid sick leave"}
                ],
            ),
        ]
    )
    runtime, llm = runtime_for(definition, docs, lambda _m: next(replies))
    turn = await runtime.answer("How much sick leave?")
    assert turn.repaired
    # Still fabricated after the repair: the answer is withdrawn, never shown as fact.
    assert turn.answer.refused
    assert turn.refused_by == "grounding"
    assert "30 days" not in turn.answer.answer
    assert len(turn.invalid_citations) == 1
    assert "not in the cited passage" in llm.calls[1].messages[-1].content


async def test_runtime_plans_and_uses_tools(tmp_path: Path, docs: Path) -> None:
    definition, _ = await _generated(tmp_path, docs)
    plans = iter(
        [
            json.dumps(
                {
                    "calls": [
                        {
                            "name": "csv_lookup",
                            "args": {"table": "berths", "filters": {"berth_id": "M4"}},
                        },
                        {"name": "weather", "args": {"location": "Kochi"}},
                    ]
                }
            ),
            json.dumps({"calls": []}),
        ]
    )

    def handler(messages: list[ChatMessage]) -> str:
        task = scripted._task(messages)
        if task == "agent.plan":
            return next(plans)
        assert "[csv_lookup] 1 matching row(s)" in messages[-1].content
        return answer_json(answer="Berth M4 has a maximum draft of 14.0 m (csv_lookup).")

    runtime, _ = runtime_for(definition, docs, handler)
    turn = await runtime.answer("What is the max draft at berth M4?")
    names = [(c.name, c.ok) for c in turn.tool_calls]
    assert names == [("csv_lookup", True), ("weather", False)]  # weather not attached
    assert "not available to this agent" in turn.tool_calls[1].output
    assert turn.grounded  # tool evidence counts


def test_quote_copied_from_a_tool_result_is_tool_evidence(tmp_path: Path, docs: Path) -> None:
    chunks = chunk_corpus(load_folder(docs), ChunkingConfig())
    hits = HybridIndex(chunks, HashEmbedder()).search("annual leave", RetrievalConfig(top_k=3))
    output = "Today is Saturday, 28 November 2026, 09:00 (Asia/Kolkata)."
    from_tool = Citation(chunk_id="current_date", quote="Saturday, 28 November 2026")
    made_up = Citation(chunk_id="current_date", quote="Sunday, 29 November 2026")
    valid, invalid = verify_citations([from_tool, made_up], hits, [output])
    assert valid == []
    assert invalid == [made_up]  # a quote found nowhere is still a broken citation


async def test_refusal_after_tool_outage_says_so(tmp_path: Path, docs: Path) -> None:
    definition, _ = await _generated(tmp_path, docs)
    refusal = definition.refusal.model_copy(update={"min_evidence_score": 0.0})
    definition = definition.model_copy(update={"refusal": refusal})
    plans = iter(
        [
            json.dumps({"calls": [{"name": "weather", "args": {"location": "Kochi"}}]}),
            '{"calls": []}',
        ]
    )

    def handler(messages: list[ChatMessage]) -> str:
        if scripted._task(messages) == "agent.plan":
            return next(plans)
        return answer_json(
            answer=definition.refusal.message, refused=True, refusal_reason="tool failed"
        )

    runtime, _ = runtime_for(definition, docs, handler)
    turn = await runtime.answer("What's the weather in Kochi?")
    assert turn.answer.refused
    assert all(not c.ok for c in turn.tool_calls)
    assert "weather service did not respond" in turn.answer.answer
    assert turn.answer.answer != definition.refusal.message


async def test_runtime_threshold_refuses_without_calling_model(tmp_path: Path, docs: Path) -> None:
    definition, _ = await _generated(tmp_path, docs)
    refusal = definition.refusal.model_copy(update={"min_evidence_score": 0.99})
    definition = definition.model_copy(update={"tools": [], "refusal": refusal})
    runtime, llm = runtime_for(definition, docs, lambda _m: pytest.fail("model called"))
    turn = await runtime.answer("What is the capital of France?")
    assert turn.answer.refused
    assert turn.refused_by == "threshold"
    assert llm.calls == []


async def test_runtime_survives_garbage_output(tmp_path: Path, docs: Path) -> None:
    definition, _ = await _generated(tmp_path, docs)
    definition = definition.model_copy(update={"tools": []})
    runtime, _ = runtime_for(definition, docs, lambda _m: "I am not JSON")
    turn = await runtime.answer("Annual leave?")
    assert turn.error is not None
    assert not turn.grounded


def test_verify_citations_splits_real_and_fake(docs: Path) -> None:
    corpus = load_folder(docs)
    hits = HybridIndex(chunk_corpus(corpus), HashEmbedder()).search(
        "annual leave", RetrievalConfig(top_k=3)
    )
    valid, invalid = verify_citations(
        [
            Citation(chunk_id="?", quote="Up to 5 unused days may be carried forward"),
            Citation(chunk_id="?", quote="Up to 50 unused days may be carried forward"),
        ],
        hits,
    )
    assert len(valid) == 1
    assert len(invalid) == 1


async def test_system_prompt_includes_refusals_and_rules(tmp_path: Path, docs: Path) -> None:
    definition, _ = await _generated(tmp_path, docs)
    prompt = system_prompt(definition)
    assert "- Name the policy section you used." in prompt
    assert "- Legal advice" in prompt
    assert "hr@example.com" in prompt


async def test_loader_detects_changed_documents(tmp_path: Path, docs: Path) -> None:
    definition, _ = await _generated(tmp_path, docs)
    assert changed_documents(definition, docs) == []
    (docs / "notes.txt").write_text("Office hours changed to 10 to 7.", encoding="utf-8")
    assert changed_documents(definition, docs) == ["notes.txt"]
    runtime = load_runtime(definition, FakeLLM(["{}"]), "gen", HashEmbedder(), docs_dir=docs)
    assert runtime.definition.name == definition.name
    with pytest.raises(FileNotFoundError):
        load_runtime(definition, FakeLLM(["{}"]), "gen", HashEmbedder(), docs_dir=tmp_path / "nope")


async def test_agent_markdown_lists_tools_and_prompt(tmp_path: Path, docs: Path) -> None:
    _, store = await _generated(tmp_path, docs)
    md = store.read_text("agent.v0.md")
    assert "**csv_lookup**" in md
    assert "Added by contract (generator omitted them): csv_lookup" in md
    assert callable(agent_markdown)


async def test_system_prompt_says_tool_results_are_evidence(tmp_path: Path, docs: Path) -> None:
    """Second real port run: the weather tool returned a forecast and the agent still
    refused, because nothing said tool output counts when the documents are silent."""
    definition, _ = await _generated(tmp_path, docs)
    prompt = system_prompt(definition)
    assert "Tool results are live, trusted evidence" in prompt
    assert "TOOLS: csv_lookup" in prompt
    no_tools = system_prompt(definition.model_copy(update={"tools": []}))
    assert "TOOLS:" not in no_tools


async def test_answer_marked_refused_is_repaired(tmp_path: Path, docs: Path) -> None:
    """Real port run: 'total berth hire for 3 days' came back as the right number with
    refused=true. The contradiction now triggers one repair."""
    definition, _ = await _generated(tmp_path, docs)
    definition = definition.model_copy(update={"tools": []})
    quote = "Shore staff earn 24 days of annual leave per calendar year"
    replies = iter(
        [
            answer_json(
                answer="You get 24 days of annual leave each year.",
                refused=True,
                citations=[{"chunk_id": "?", "quote": quote}],
            ),
            answer_json(
                answer="You get 24 days of annual leave each year.",
                refused=False,
                citations=[{"chunk_id": "?", "quote": quote}],
            ),
        ]
    )
    runtime, llm = runtime_for(definition, docs, lambda _m: next(replies))
    turn = await runtime.answer("How many days of annual leave?")
    assert turn.repaired
    assert not turn.answer.refused
    assert turn.grounded
    assert "set refused to true but also gave an answer" in llm.calls[1].messages[-1].content


def test_plain_refusal_is_not_a_contract_problem() -> None:
    from forgeai.runtime.agent import contract_problems
    from forgeai.schemas.agent import AgentAnswer

    msg = "I can't help with that here; please contact HR."
    ok = AgentAnswer(answer=msg, refused=True, refusal_reason="out of scope")
    assert contract_problems(ok, msg, [], [], False) == ""
    short = AgentAnswer(answer="Sorry, I cannot answer that.", refused=True)
    assert contract_problems(short, msg, [], [], False) == ""


async def test_citation_without_chunk_id_is_still_verified(tmp_path: Path, docs: Path) -> None:
    """Real port run: one citation lacked chunk_id and the whole answer failed validation."""
    definition, _ = await _generated(tmp_path, docs)
    definition = definition.model_copy(update={"tools": []})
    reply = json.dumps(
        {
            "answer": "24 days a year.",
            "refused": False,
            "citations": [{"quote": "Shore staff earn 24 days of annual leave per calendar year"}],
        }
    )
    runtime, _ = runtime_for(definition, docs, lambda _m: reply)
    turn = await runtime.answer("How many days of annual leave?")
    assert turn.error is None
    assert turn.grounded
    assert turn.valid_citations[0].chunk_id.startswith("leave-policy")


async def test_unsupported_non_answer_becomes_a_refusal(tmp_path: Path, docs: Path) -> None:
    """Real port run: 'the recent changes are not mentioned in the passages', unrefused,
    uncited. With no evidence at all it must not ship as an answer."""
    definition, _ = await _generated(tmp_path, docs)
    definition = definition.model_copy(update={"tools": []})
    no_threshold = definition.refusal.model_copy(update={"min_evidence_score": 0.0})
    definition = definition.model_copy(update={"refusal": no_threshold})
    reply = answer_json(answer="The recent changes are not mentioned in the provided passages.")
    runtime, _ = runtime_for(definition, docs, lambda _m: reply)
    turn = await runtime.answer("What changed in the rules recently?")
    assert turn.answer.refused
    assert turn.refused_by == "grounding"
    assert turn.answer.answer == definition.refusal.message
