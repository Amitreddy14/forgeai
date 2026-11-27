"""Token and call budget for a run. Every model call is attributed to a stage, so the report
can say where the budget went and the optimizer can stop when an iteration stops paying."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, Field

from forgeai.errors import BudgetExceeded
from forgeai.llm.base import ChatMessage, LLMClient, LLMResponse

_current_stage: ContextVar[str] = ContextVar("forge_stage", default="unattributed")


class StageUsage(BaseModel):
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetReport(BaseModel):
    max_tokens: int
    max_calls: int
    used_tokens: int
    used_calls: int
    by_stage: dict[str, StageUsage] = Field(default_factory=dict)


class Budget:
    """Counts only tokens actually spent: cache hits are free."""

    def __init__(self, max_tokens: int, max_calls: int = 5_000) -> None:
        self.max_tokens = max_tokens
        self.max_calls = max_calls
        self.by_stage: dict[str, StageUsage] = {}

    @property
    def used_tokens(self) -> int:
        return sum(s.total_tokens for s in self.by_stage.values())

    @property
    def used_calls(self) -> int:
        return sum(s.calls - s.cached_calls for s in self.by_stage.values())

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def check(self) -> None:
        if self.used_tokens >= self.max_tokens:
            raise BudgetExceeded(f"token budget spent ({self.used_tokens}/{self.max_tokens})")
        if self.used_calls >= self.max_calls:
            raise BudgetExceeded(f"call budget spent ({self.used_calls}/{self.max_calls})")

    def record(self, response: LLMResponse, stage: str | None = None) -> None:
        name = stage or _current_stage.get()
        usage = self.by_stage.setdefault(name, StageUsage())
        usage.calls += 1
        if response.cached:
            usage.cached_calls += 1
            return
        usage.prompt_tokens += response.usage.prompt_tokens
        usage.completion_tokens += response.usage.completion_tokens

    def report(self) -> BudgetReport:
        return BudgetReport(
            max_tokens=self.max_tokens,
            max_calls=self.max_calls,
            used_tokens=self.used_tokens,
            used_calls=self.used_calls,
            by_stage={k: v.model_copy() for k, v in self.by_stage.items()},
        )


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Attribute every model call inside this block to `name`."""
    token = _current_stage.set(name)
    try:
        yield
    finally:
        _current_stage.reset(token)


class BudgetedLLM:
    """Refuses to call the model once the budget is spent; records usage per stage."""

    def __init__(self, inner: LLMClient, budget: Budget) -> None:
        self._inner = inner
        self.budget = budget

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.budget.check()
        response = await self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )
        self.budget.record(response)
        return response
