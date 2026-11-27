"""The generic agent runtime: loads any AgentDefinition and answers questions with a
plan -> act -> answer -> check loop. The same code runs evals, `forge chat` and the web
chat, so what is tested is exactly what ships."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from forgeai.errors import ModelOutputError
from forgeai.ingest import chunk_corpus
from forgeai.llm import ChatMessage, LLMClient, complete_json
from forgeai.prompts import render
from forgeai.retrieval import Embedder, Hit, HybridIndex
from forgeai.schemas.agent import AgentAnswer, AgentDefinition, Citation
from forgeai.schemas.corpus import Corpus
from forgeai.textutil import contains_quote
from forgeai.tools import REGISTRY, ToolEnv, call_tool, tool_card

MAX_STEPS = 3
MAX_CALLS_PER_STEP = 3
PASSAGE_CHARS = 1400
PLAN_PREVIEW_CHARS = 160


class PlannedCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentPlan(BaseModel):
    calls: list[PlannedCall] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]
    ok: bool
    output: str


class RetrievedRef(BaseModel):
    chunk_id: str
    score: float
    bm25: float
    dense: float


class AgentTurn(BaseModel):
    question: str
    answer: AgentAnswer
    retrieved: list[RetrievedRef] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    valid_citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False
    refused_by: Literal["model", "threshold"] | None = None
    repaired: bool = False
    error: str | None = None
    latency_s: float = 0.0

    @property
    def best_dense(self) -> float:
        return max((r.dense for r in self.retrieved), default=0.0)


def system_prompt(defn: AgentDefinition) -> str:
    rules = "".join(f"- {r}\n" for r in defn.rules)
    refuse = "\n".join(f"- {t}" for t in defn.refusal.refuse_topics) or "- (none listed)"
    tools = [REGISTRY[t.name] for t in defn.tools if t.name in REGISTRY]
    tools_section = (
        "\nTools were called for you before this answer; their results are in TOOL RESULTS.\n"
        if tools
        else ""
    )
    return render(
        "agent_system",
        role=defn.role_prompt.strip(),
        audience=defn.audience,
        tone=defn.tone,
        rules=rules,
        refuse_topics=refuse,
        refusal_message=defn.refusal.message.replace('"', "'"),
        tools_section=tools_section,
    )


def _passages(hits: list[Hit], limit: int | None = None) -> str:
    if not hits:
        return "(no passages found)"
    blocks = []
    for h in hits:
        text = h.chunk.text if limit is None else h.chunk.text[:limit]
        blocks.append(f"[{h.chunk.chunk_id}] ({h.chunk.citation})\n{text}")
    return "\n\n".join(blocks)


def verify_citations(
    citations: list[Citation], hits: list[Hit]
) -> tuple[list[Citation], list[Citation]]:
    """Split citations into those whose quote really is in the cited (or any retrieved)
    passage, and those that are not. A wrong chunk id with a real quote is corrected."""
    valid: list[Citation] = []
    invalid: list[Citation] = []
    for citation in citations:
        ordered = sorted(hits, key=lambda h: h.chunk.chunk_id != citation.chunk_id)
        match = next((h for h in ordered if contains_quote(h.chunk.text, citation.quote)), None)
        if match is None:
            invalid.append(citation)
        else:
            valid.append(Citation(chunk_id=match.chunk.chunk_id, quote=citation.quote))
    return valid, invalid


class AgentRuntime:
    def __init__(
        self,
        definition: AgentDefinition,
        corpus: Corpus,
        embedder: Embedder,
        llm: LLMClient,
        model: str,
        tool_env: ToolEnv | None = None,
    ) -> None:
        self.definition = definition
        self.llm = llm
        self.model = model
        chunks = chunk_corpus(corpus, definition.chunking)
        self.index = HybridIndex(chunks, embedder)
        self.env = tool_env or ToolEnv()
        if not self.env.tables:
            self.env.tables = {d.doc_id: d.rows or [] for d in corpus.documents if d.kind == "csv"}
        self.tools = [REGISTRY[t.name] for t in definition.tools if t.name in REGISTRY]
        self._system = system_prompt(definition)

    @property
    def system(self) -> str:
        return self._system

    async def _plan_and_act(self, question: str, hits: list[Hit]) -> list[ToolCallRecord]:
        if not self.tools:
            return []
        allowed = {t.name: t for t in self.tools}
        cards = "\n".join(tool_card(t) for t in self.tools)
        records: list[ToolCallRecord] = []
        for _ in range(MAX_STEPS):
            history = (
                "\n".join(f"- {r.name}({json.dumps(r.args)}) -> {r.output[:600]}" for r in records)
                or "(none)"
            )
            prompt = render(
                "agent_plan",
                tools=cards,
                question=question,
                passages=_passages(hits, PLAN_PREVIEW_CHARS),
                history=history,
            )
            plan = await complete_json(
                self.llm, [ChatMessage(role="user", content=prompt)], AgentPlan, model=self.model
            )
            calls = plan.calls[:MAX_CALLS_PER_STEP]
            done_before = {(r.name, json.dumps(r.args, sort_keys=True)) for r in records if r.ok}
            fresh = [
                c for c in calls if (c.name, json.dumps(c.args, sort_keys=True)) not in done_before
            ]
            if not fresh:
                break
            for call in fresh:
                tool = allowed.get(call.name)
                if tool is None:
                    records.append(
                        ToolCallRecord(
                            name=call.name,
                            args=call.args,
                            ok=False,
                            output=f"ERROR: tool {call.name!r} is not available to this agent",
                        )
                    )
                    continue
                result = await call_tool(tool, call.args, self.env)
                records.append(
                    ToolCallRecord(
                        name=call.name, args=call.args, ok=result.ok, output=result.output
                    )
                )
        return records

    async def answer(self, question: str) -> AgentTurn:
        started = time.perf_counter()
        defn = self.definition
        hits = self.index.search(question, defn.retrieval)
        retrieved = [
            RetrievedRef(chunk_id=h.chunk.chunk_id, score=h.score, bm25=h.bm25, dense=h.dense)
            for h in hits
        ]
        turn = AgentTurn(question=question, answer=AgentAnswer(answer=""), retrieved=retrieved)
        try:
            turn.tool_calls = await self._plan_and_act(question, hits)
            tool_evidence = any(r.ok for r in turn.tool_calls)

            threshold = defn.refusal.min_evidence_score
            if threshold > 0 and turn.best_dense < threshold and not tool_evidence:
                turn.answer = AgentAnswer(
                    answer=defn.refusal.message,
                    refused=True,
                    refusal_reason=(
                        f"best passage similarity {turn.best_dense:.2f} is below "
                        f"the evidence threshold {threshold:.2f}"
                    ),
                )
                turn.refused_by = "threshold"
                return turn

            tool_text = "\n".join(f"[{r.name}] {r.output}" for r in turn.tool_calls) or "(none)"
            messages = [
                ChatMessage(role="system", content=self._system),
                ChatMessage(
                    role="user",
                    content=render(
                        "agent_answer",
                        question=question,
                        passages=_passages(hits, PASSAGE_CHARS),
                        tool_results=tool_text,
                    ),
                ),
            ]
            answer = await complete_json(
                self.llm, messages, AgentAnswer, model=self.model, max_tokens=1200
            )
            valid, invalid = verify_citations(answer.citations, hits)

            needs_repair = not answer.refused and (
                bool(invalid) or (not valid and not tool_evidence)
            )
            if needs_repair:
                problems = (
                    f"{len(invalid)} citation(s) quote text that is not in the cited passage"
                    if invalid
                    else "no citations were given"
                )
                messages += [
                    ChatMessage(role="assistant", content=answer.model_dump_json()),
                    ChatMessage(role="user", content=render("agent_repair", problems=problems)),
                ]
                answer = await complete_json(
                    self.llm, messages, AgentAnswer, model=self.model, max_tokens=1200
                )
                valid, invalid = verify_citations(answer.citations, hits)
                turn.repaired = True

            turn.answer = answer
            turn.valid_citations = valid
            turn.invalid_citations = invalid
            turn.refused_by = "model" if answer.refused else None
            turn.grounded = answer.refused or (not invalid and (bool(valid) or tool_evidence))
        except ModelOutputError as exc:
            turn.error = str(exc)
            turn.answer = AgentAnswer(answer="", refused=False)
        finally:
            turn.latency_s = round(time.perf_counter() - started, 3)
        return turn
