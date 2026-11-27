from __future__ import annotations

import pytest

from forgeai.llm import ChatMessage, FakeLLM, LLMClient


def msgs(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


async def test_scripted_replies_in_order() -> None:
    llm = FakeLLM(["first", "second"])
    assert (await llm.complete(msgs("a"), model="m")).text == "first"
    assert (await llm.complete(msgs("b"), model="m")).text == "second"
    assert [c.messages[0].content for c in llm.calls] == ["a", "b"]


async def test_responder_routes_on_prompt() -> None:
    llm = FakeLLM(responder=lambda m: "judge" if "grade" in m[-1].content else "gen")
    assert (await llm.complete(msgs("please grade this"), model="m")).text == "judge"
    assert (await llm.complete(msgs("write a prompt"), model="m")).text == "gen"


async def test_records_schema_and_usage() -> None:
    llm = FakeLLM(['{"ok": true}'])
    schema = {"type": "object"}
    response = await llm.complete(msgs("x" * 40), model="m", json_schema=schema)
    assert llm.calls[0].json_schema == schema
    assert response.usage.prompt_tokens == 10
    assert response.usage.total_tokens > response.usage.prompt_tokens


async def test_running_out_of_replies_is_loud() -> None:
    llm = FakeLLM(["only one"])
    await llm.complete(msgs("a"), model="m")
    with pytest.raises(AssertionError):
        await llm.complete(msgs("b"), model="m")


def test_satisfies_protocol() -> None:
    assert isinstance(FakeLLM(["x"]), LLMClient)


def test_needs_replies_or_responder() -> None:
    with pytest.raises(ValueError, match="replies or a responder"):
        FakeLLM()
