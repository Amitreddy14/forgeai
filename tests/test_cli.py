from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from forgeai import __version__, cli
from forgeai.config import Settings
from tests.fixtures import build_docs_folder

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


def test_ingest_lists_documents_and_skips(tmp_path: Path) -> None:
    folder = build_docs_folder(tmp_path)
    result = runner.invoke(cli.app, ["ingest", str(folder)])
    assert result.exit_code == 0, result.stdout
    assert "leave-policy" in result.stdout
    assert "malware.exe" in result.stdout


def test_ingest_empty_folder_fails(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", str(tmp_path)])
    assert result.exit_code == 1


def test_search_prints_ranked_citations(tmp_path: Path) -> None:
    folder = build_docs_folder(tmp_path)
    result = runner.invoke(cli.app, ["search", str(folder), "sick leave certificate", "-k", "2"])
    assert result.exit_code == 0, result.stdout
    assert "1. leave-policy p.2" in result.stdout


def test_search_rejects_bad_mode(tmp_path: Path) -> None:
    folder = build_docs_folder(tmp_path)
    result = runner.invoke(cli.app, ["search", str(folder), "x", "--mode", "magic"])
    assert result.exit_code != 0


class _FakeStack:
    def __init__(self, replies: list[str]) -> None:
        from forgeai.llm import Budget, BudgetedLLM, FakeLLM

        self.budget = Budget(max_tokens=10_000)
        self.llm = BudgetedLLM(FakeLLM(replies), self.budget)

    async def aclose(self) -> None:
        return None


def test_smoke_reports_each_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import forgeai.llm.factory as factory

    reply = '{"capital": "Thiruvananthapuram", "confidence": 0.9}'
    monkeypatch.setattr(factory, "build_llm", lambda _s: _FakeStack([reply, reply]))
    result = runner.invoke(cli.app, ["smoke"])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("OK") == 2
    assert "llama3.1:8b" in result.stdout


def test_smoke_fails_cleanly_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import forgeai.llm.factory as factory

    monkeypatch.setattr(factory, "build_llm", lambda _s: _FakeStack(["nope", "still nope"]))
    result = runner.invoke(cli.app, ["smoke"])
    assert result.exit_code == 1
    assert "ModelOutputError" in result.stdout
