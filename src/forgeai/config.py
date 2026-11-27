"""Runtime settings, read from FORGE_* environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables in one place. Secrets are SecretStr so they never print by accident."""

    model_config = SettingsConfigDict(
        env_prefix="FORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "prod", "test"] = "dev"

    # Any OpenAI-compatible endpoint: Ollama (default) or Groq.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr | None = None
    generator_model: str = "qwen2.5:7b"
    judge_model: str = "llama3.1:8b"
    llm_timeout_s: float = Field(default=120.0, gt=0)
    max_concurrency: int = Field(default=2, ge=1, le=16)

    # Budget and loop limits.
    max_tokens_per_run: int = Field(default=400_000, ge=1_000)
    max_iterations: int = Field(default=3, ge=0, le=3)

    # Retrieval: "fastembed" (bge-small, downloads once) or "hash" (offline, for tests).
    embedder: Literal["fastembed", "hash"] = "fastembed"

    # Storage.
    runs_dir: Path = Path("runs")
    cache_path: Path = Path(".forge_cache/llm.sqlite")

    def redacted(self) -> dict[str, object]:
        """Settings safe to show a user: the API key is reduced to set / not set."""
        data = self.model_dump()
        data["llm_api_key"] = "set" if self.llm_api_key else "not set"
        return {
            key: (str(value) if isinstance(value, Path) else value) for key, value in data.items()
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
