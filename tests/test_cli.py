from __future__ import annotations

from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from forgeai import __version__, cli
from forgeai.config import Settings

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_offline_shows_settings_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_LLM_API_KEY", "gsk_super_secret")
    cli.get_settings.cache_clear()
    result = runner.invoke(cli.app, ["doctor", "--offline"])
    assert result.exit_code == 0
    assert "generator_model" in result.stdout
    assert "gsk_super_secret" not in result.stdout


class FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> Any:
        return self._body


def _patch_get(monkeypatch: pytest.MonkeyPatch, outcome: FakeResponse | Exception) -> None:
    def fake_get(*_args: Any, **_kwargs: Any) -> FakeResponse:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(cli.httpx, "get", fake_get)


def test_endpoint_check_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"data": [{"id": "qwen2.5:7b"}, {"id": "llama3.1:8b"}]}
    _patch_get(monkeypatch, FakeResponse(200, body))
    ok, detail = cli.check_model_endpoint(Settings())
    assert ok, detail


def test_endpoint_check_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, FakeResponse(200, {"data": [{"id": "qwen2.5:7b"}]}))
    ok, detail = cli.check_model_endpoint(Settings())
    assert not ok
    assert "llama3.1:8b" in detail


def test_endpoint_check_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx.ConnectError("refused"))
    ok, detail = cli.check_model_endpoint(Settings())
    assert not ok
    assert "could not reach" in detail


def test_doctor_exits_nonzero_when_endpoint_down(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx.ConnectError("refused"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
