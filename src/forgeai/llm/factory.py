"""Build the production client stack: HTTP client -> cache -> budget."""

from __future__ import annotations

from dataclasses import dataclass

from forgeai.config import Settings
from forgeai.llm.budget import Budget, BudgetedLLM
from forgeai.llm.cache import CachedLLM, ResponseCache
from forgeai.llm.openai_compat import OpenAICompatClient


@dataclass
class LLMStack:
    llm: BudgetedLLM
    http: OpenAICompatClient
    cache: ResponseCache
    budget: Budget

    async def aclose(self) -> None:
        await self.http.aclose()
        self.cache.close()


def build_llm(settings: Settings, budget: Budget | None = None) -> LLMStack:
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    http = OpenAICompatClient(
        settings.llm_base_url,
        api_key=api_key,
        timeout_s=settings.llm_timeout_s,
        max_concurrency=settings.max_concurrency,
    )
    cache = ResponseCache(settings.cache_path)
    budget = budget or Budget(max_tokens=settings.max_tokens_per_run)
    llm = BudgetedLLM(CachedLLM(http, cache), budget)
    return LLMStack(llm=llm, http=http, cache=cache, budget=budget)
