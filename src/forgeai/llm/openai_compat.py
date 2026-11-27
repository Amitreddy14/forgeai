"""Async client for any OpenAI-compatible chat endpoint (Ollama `/v1`, Groq, vLLM, ...)."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

import httpx

from forgeai.errors import ModelOutputError, ModelUnavailable
from forgeai.llm.base import ChatMessage, LLMResponse, Usage

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatClient:
    """Thin, dependable wrapper over POST {base_url}/chat/completions.

    - Bounded concurrency (one GPU serialises Ollama anyway; Groq rate-limits).
    - Retries on network errors and retryable status codes, honouring Retry-After.
    - JSON mode via `response_format: json_object`. The schema itself travels in the prompt
      (see `structured.py`): json_object is supported by both Ollama and Groq for every
      model, while json_schema support varies. Validation happens on our side regardless.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_concurrency: int = 2,
        max_retries: int = 4,
        backoff_base_s: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        async with self._semaphore:
            data = await self._post_with_retries(payload)
        return _parse_completion(data, model)

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error = "no attempt made"
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        body: dict[str, Any] = response.json()
                    except json.JSONDecodeError as exc:
                        raise ModelOutputError("endpoint returned non-JSON body") from exc
                    return body
                failed = _failed_json_generation(response)
                if failed is not None:
                    # Groq's JSON mode rejects malformed output with a 400 and hands back
                    # the text. Pass it on so the structured layer's repair loop sees it.
                    log.info("endpoint rejected JSON output; passing it to repair")
                    return {"choices": [{"message": {"content": failed}}]}
                if response.status_code not in RETRYABLE_STATUS:
                    raise ModelUnavailable(
                        f"model endpoint answered {response.status_code}: {response.text[:300]}"
                    )
                last_error = f"HTTP {response.status_code}"
                retry_after = _retry_after_seconds(response)
                if retry_after is not None and attempt < self._max_retries:
                    log.warning("rate limited; waiting %.1fs", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
            if attempt < self._max_retries:
                delay = self._backoff_base_s * (2**attempt) * (1 + random.random() / 4)  # noqa: S311
                log.warning(
                    "LLM call failed (%s); retry %d in %.1fs", last_error, attempt + 1, delay
                )
                await asyncio.sleep(delay)
        raise ModelUnavailable(f"model endpoint failed after retries: {last_error}")


def _failed_json_generation(response: httpx.Response) -> str | None:
    """The rejected text from a JSON-mode validation 400 (Groq), or None."""
    if response.status_code != 400:
        return None
    try:
        error = response.json().get("error") or {}
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(error, dict) or error.get("code") != "json_validate_failed":
        return None
    return str(error.get("failed_generation") or "")


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return min(float(value), 60.0)
    except ValueError:
        return None


def _parse_completion(data: dict[str, Any], requested_model: str) -> LLMResponse:
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelOutputError(f"unexpected completion shape: {str(data)[:300]}") from exc
    usage_raw = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
    )
    return LLMResponse(text=text, model=str(data.get("model", requested_model)), usage=usage)
