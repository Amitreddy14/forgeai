"""Response cache: identical requests return identical answers, so runs are reproducible
and re-running a finished stage costs nothing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from forgeai.llm.base import ChatMessage, LLMClient, LLMResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def request_key(
    messages: list[ChatMessage],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    json_schema: dict[str, Any] | None,
) -> str:
    blob = json.dumps(
        {
            "messages": [m.model_dump() for m in messages],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_schema": json_schema,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """A tiny SQLite key-value store. Parameterised queries only."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = asyncio.Lock()

    def get(self, key: str) -> LLMResponse | None:
        row = self._conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return LLMResponse.model_validate_json(row[0])

    def put(self, key: str, response: LLMResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, model, response) VALUES (?, ?, ?)",
            (key, response.model, response.model_dump_json()),
        )
        self._conn.commit()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._conn.close()


class CachedLLM:
    """Wraps any client. Only deterministic calls (temperature 0) are cached."""

    def __init__(self, inner: LLMClient, cache: ResponseCache) -> None:
        self._inner = inner
        self._cache = cache
        self.hits = 0
        self.misses = 0

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        cacheable = temperature == 0.0
        key = request_key(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )
        if cacheable:
            hit = self._cache.get(key)
            if hit is not None:
                self.hits += 1
                return hit.model_copy(update={"cached": True})
        self.misses += 1
        response = await self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )
        if cacheable:
            self._cache.put(key, response)
        return response
