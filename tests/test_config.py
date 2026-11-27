from __future__ import annotations

import pytest
from pydantic import ValidationError

from forgeai.config import Settings, get_settings


def test_defaults_point_at_local_ollama() -> None:
    settings = get_settings()
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.generator_model != settings.judge_model, "judge must differ from generator"
    assert settings.env == "test"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_GENERATOR_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("FORGE_MAX_ITERATIONS", "2")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.generator_model == "llama-3.1-8b-instant"
    assert settings.max_iterations == 2


def test_iteration_cap_is_three() -> None:
    with pytest.raises(ValidationError):
        Settings(max_iterations=4)


def test_redacted_never_leaks_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_LLM_API_KEY", "gsk_super_secret")
    get_settings.cache_clear()
    shown = get_settings().redacted()
    assert shown["llm_api_key"] == "set"
    assert "gsk_super_secret" not in repr(shown)
    assert "gsk_super_secret" not in repr(get_settings())
