"""The contract every LLM backend implements. Stages depend on this protocol, never on a vendor."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class Usage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    cached: bool = False


@runtime_checkable
class LLMClient(Protocol):
    """One chat-completion call. `json_schema`, when given, asks for JSON matching it."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...
