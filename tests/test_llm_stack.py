from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field

from forgeai.errors import BudgetExceeded, ModelOutputError, ModelUnavailable
from forgeai.llm import (
    Budget,
    BudgetedLLM,
    CachedLLM,
    ChatMessage,
    FakeLLM,
    OpenAICompatClient,
    ResponseCache,
    complete_json,
    extract_json_object,
    stage,
)


def user(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


def completion(text: str, prompt: int = 10, completion_tokens: int = 5) -> dict[str, Any]:
    return {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion_tokens},
    }


# ---------- OpenAI-compatible client -------------------------------------------------


async def test_client_sends_openai_payload_and_parses_usage() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer k"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion('{"a": 1}', 12, 3))

    client = OpenAICompatClient("http://x/v1", api_key="k", transport=httpx.MockTransport(handler))
    async with client:
        out = await client.complete(user("hi"), model="m", json_schema={"type": "object"})
    assert out.text == '{"a": 1}'
    assert out.usage.total_tokens == 15
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["temperature"] == 0.0


async def test_client_retries_rate_limit_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json=completion("ok"))

    client = OpenAICompatClient(
        "http://x/v1", transport=httpx.MockTransport(handler), backoff_base_s=0
    )
    async with client:
        assert (await client.complete(user("hi"), model="m")).text == "ok"
    assert attempts["n"] == 3


async def test_client_gives_up_with_typed_error() -> None:
    client = OpenAICompatClient(
        "http://x/v1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(503)),
        max_retries=2,
        backoff_base_s=0,
    )
    async with client:
        with pytest.raises(ModelUnavailable, match="after retries"):
            await client.complete(user("hi"), model="m")


async def test_client_does_not_retry_auth_errors() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    client = OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))
    async with client:
        with pytest.raises(ModelUnavailable, match="401"):
            await client.complete(user("hi"), model="m")
    assert calls["n"] == 1


async def test_client_rejects_malformed_completion() -> None:
    client = OpenAICompatClient(
        "http://x/v1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"oops": True})),
    )
    async with client:
        with pytest.raises(ModelOutputError, match="unexpected completion shape"):
            await client.complete(user("hi"), model="m")


# ---------- Structured output ---------------------------------------------------------


class Verdict(BaseModel):
    grounded: bool
    score: int = Field(ge=0, le=5)


@pytest.mark.parametrize(
    "text",
    [
        '{"grounded": true, "score": 4}',
        '```json\n{"grounded": true, "score": 4}\n```',
        'Sure! Here is the JSON: {"grounded": true, "score": 4} Hope it helps.',
        'Thinking {not json} ... {"grounded": true, "score": 4}',
    ],
)
def test_extract_json_tolerates_wrapping(text: str) -> None:
    assert extract_json_object(text) == {"grounded": True, "score": 4}


def test_extract_json_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object("I cannot answer that.")


async def test_complete_json_happy_path_puts_schema_in_system_prompt() -> None:
    llm = FakeLLM(['{"grounded": false, "score": 1}'])
    out = await complete_json(llm, user("grade"), Verdict, model="m")
    assert out == Verdict(grounded=False, score=1)
    system = llm.calls[0].messages[0]
    assert system.role == "system"
    assert '"grounded"' in system.content


async def test_complete_json_repairs_once_with_the_error() -> None:
    llm = FakeLLM(['{"grounded": true, "score": 9}', '{"grounded": true, "score": 5}'])
    out = await complete_json(llm, user("grade"), Verdict, model="m")
    assert out.score == 5
    repair_msg = llm.calls[1].messages[-1].content
    assert "score" in repair_msg
    assert "less than or equal to 5" in repair_msg


async def test_complete_json_raises_after_repairs_exhausted() -> None:
    llm = FakeLLM(["nope", "still nope"])
    with pytest.raises(ModelOutputError, match="Verdict"):
        await complete_json(llm, user("grade"), Verdict, model="m", repair_attempts=1)


# ---------- Cache ---------------------------------------------------------------------


async def test_cache_returns_identical_answer_without_calling_model(tmp_path: Path) -> None:
    inner = FakeLLM(["first answer"])
    cached = CachedLLM(inner, ResponseCache(tmp_path / "c.sqlite"))
    a = await cached.complete(user("q"), model="m")
    b = await cached.complete(user("q"), model="m")
    assert a.text == b.text == "first answer"
    assert b.cached
    assert not a.cached
    assert len(inner.calls) == 1
    assert (cached.hits, cached.misses) == (1, 1)


async def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    await CachedLLM(FakeLLM(["saved"]), ResponseCache(path)).complete(user("q"), model="m")
    again = CachedLLM(FakeLLM(["should not be used"]), ResponseCache(path))
    assert (await again.complete(user("q"), model="m")).text == "saved"


async def test_cache_skips_nonzero_temperature(tmp_path: Path) -> None:
    inner = FakeLLM(["one", "two"])
    cached = CachedLLM(inner, ResponseCache(tmp_path / "c.sqlite"))
    await cached.complete(user("q"), model="m", temperature=0.7)
    second = await cached.complete(user("q"), model="m", temperature=0.7)
    assert second.text == "two"


async def test_cache_key_depends_on_model(tmp_path: Path) -> None:
    cached = CachedLLM(FakeLLM(["gen", "judge"]), ResponseCache(tmp_path / "c.sqlite"))
    assert (await cached.complete(user("q"), model="gen-model")).text == "gen"
    assert (await cached.complete(user("q"), model="judge-model")).text == "judge"


# ---------- Budget --------------------------------------------------------------------


async def test_budget_attributes_usage_to_stages() -> None:
    budget = Budget(max_tokens=10_000)
    llm = BudgetedLLM(FakeLLM(responder=lambda _m: "x" * 40), budget)
    with stage("intake"):
        await llm.complete(user("a" * 40), model="m")
    with stage("judge"):
        await llm.complete(user("b" * 40), model="m")
        await llm.complete(user("c" * 40), model="m")
    report = budget.report()
    assert report.by_stage["intake"].calls == 1
    assert report.by_stage["judge"].calls == 2
    assert report.used_tokens == 60


async def test_budget_blocks_calls_once_spent() -> None:
    budget = Budget(max_tokens=25)
    llm = BudgetedLLM(FakeLLM(responder=lambda _m: "x" * 40), budget)
    await llm.complete(user("a" * 40), model="m")  # 20 tokens: still under
    await llm.complete(user("a" * 40), model="m")  # 40 tokens: now over
    with pytest.raises(BudgetExceeded, match="token budget"):
        await llm.complete(user("a" * 40), model="m")


async def test_cache_hits_are_free(tmp_path: Path) -> None:
    budget = Budget(max_tokens=10_000)
    llm = BudgetedLLM(CachedLLM(FakeLLM(["answer"]), ResponseCache(tmp_path / "c.sqlite")), budget)
    await llm.complete(user("q" * 40), model="m")
    spent = budget.used_tokens
    await llm.complete(user("q" * 40), model="m")
    assert budget.used_tokens == spent
    assert budget.report().by_stage["unattributed"].cached_calls == 1


async def test_build_llm_wires_cache_and_budget(tmp_path: Path) -> None:
    from forgeai.config import Settings
    from forgeai.llm.factory import build_llm

    settings = Settings(cache_path=tmp_path / "c.sqlite", max_tokens_per_run=5_000, llm_api_key="k")
    stack = build_llm(settings)
    try:
        assert stack.budget.max_tokens == 5_000
        assert stack.cache is not None
        assert isinstance(stack.llm, BudgetedLLM)
    finally:
        await stack.aclose()


async def test_client_hands_back_groq_json_validation_failure() -> None:
    body = {
        "error": {
            "message": "Failed to generate JSON.",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": '{"answer": "half',
        }
    }
    client = OpenAICompatClient(
        "http://x/v1", transport=httpx.MockTransport(lambda _r: httpx.Response(400, json=body))
    )
    async with client:
        out = await client.complete(user("hi"), model="m", json_schema={"type": "object"})
    assert out.text == '{"answer": "half'


async def test_client_still_raises_on_other_400s() -> None:
    client = OpenAICompatClient(
        "http://x/v1",
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(400, json={"error": {"code": "model_not_found"}})
        ),
    )
    async with client:
        with pytest.raises(ModelUnavailable, match="400"):
            await client.complete(user("hi"), model="m")
