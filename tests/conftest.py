from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from forgeai.config import get_settings


@pytest.fixture(autouse=True)
def isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """Every test gets clean settings: no real .env, no real API key, temp storage."""
    # Clear every FORGE_* variable from the developer's shell: tests must not depend on
    # which models (or keys) happen to be configured on the machine running them.
    for key in [k for k in os.environ if k.upper().startswith("FORGE_")]:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path_factory.mktemp("forge")
    monkeypatch.setenv("FORGE_ENV", "test")
    monkeypatch.setenv("FORGE_EMBEDDER", "hash")
    monkeypatch.setenv("FORGE_RUNS_DIR", str(root / "runs"))
    monkeypatch.setenv("FORGE_CACHE_PATH", str(root / "cache.sqlite"))
    monkeypatch.chdir(root)  # so a developer's .env is never picked up
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
