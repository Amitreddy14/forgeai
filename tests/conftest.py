from __future__ import annotations

from collections.abc import Iterator

import pytest

from forgeai.config import get_settings


@pytest.fixture(autouse=True)
def isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """Every test gets clean settings: no real .env, no real API key, temp storage."""
    for key in ("FORGE_LLM_API_KEY", "FORGE_LLM_BASE_URL", "FORGE_ENV"):
        monkeypatch.delenv(key, raising=False)
    root = tmp_path_factory.mktemp("forge")
    monkeypatch.setenv("FORGE_ENV", "test")
    monkeypatch.setenv("FORGE_RUNS_DIR", str(root / "runs"))
    monkeypatch.setenv("FORGE_CACHE_PATH", str(root / "cache.sqlite"))
    monkeypatch.chdir(root)  # so a developer's .env is never picked up
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
