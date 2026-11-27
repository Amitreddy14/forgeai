"""A scripted LLM for tests. It lets a test make the model lie on purpose."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from forgeai.llm.base import ChatMessage, LLMResponse, Usage

Responder = Callable[[list[ChatMessage]], str]


class RecordedCall(BaseModel):
    messages: list[ChatMessage]
    model: str
    temperature: float
    max_tokens: int
    json_schema: dict[str, Any] | None


class FakeLLM:
    """Returns scripted replies in order, or computes them with a responder function.

    Use a list of strings for simple tests, or a responder to route on prompt content
    when stages run concurrently and order is not guaranteed.
    """

    def __init__(
        self,
        replies: Iterable[str] | None = None,
        *,
        responder: Responder | None = None,
    ) -> None:
        if replies is None and responder is None:
            raise ValueError("FakeLLM needs replies or a responder")
        self._replies: deque[str] = deque(replies or [])
        self._responder = responder
        self.calls: list[RecordedCall] = []

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            RecordedCall(
                messages=list(messages),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )
        )
        if self._replies:
            text = self._replies.popleft()
        elif self._responder is not None:
            text = self._responder(messages)
        else:
            raise AssertionError("FakeLLM ran out of scripted replies")
        usage = Usage(
            prompt_tokens=sum(_rough_tokens(m.content) for m in messages),
            completion_tokens=_rough_tokens(text),
        )
        return LLMResponse(text=text, model=model, usage=usage)


def _rough_tokens(text: str) -> int:
    """About 4 characters per token; good enough for budget tests."""
    return max(1, len(text) // 4)
