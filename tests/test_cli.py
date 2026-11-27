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


def test_intake_unattended_writes_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json

    from forgeai.llm import Budget, BudgetedLLM, FakeLLM
    from tests import scripted

    class Stack:
        def __init__(self) -> None:
            self.budget = Budget(max_tokens=100_000)
            self.llm = BudgetedLLM(FakeLLM(responder=scripted.responder), self.budget)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(cli, "build_llm", lambda _s: Stack())
    folder = build_docs_folder(tmp_path)
    brief = tmp_path / "hr-brief.md"
    brief.write_text("I need an assistant that answers HR policy questions.", encoding="utf-8")
    result = runner.invoke(
        cli.app,
        [
            "intake",
            "--brief",
            str(brief),
            "--docs",
            str(folder),
            "--answers",
            "auto",
            "--run-id",
            "cli-intake",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Flagged" in result.stdout
    run = cli.get_settings().runs_dir / "cli-intake"
    budget = _json.loads((run / "budget.json").read_text())
    assert budget["by_stage"]["intake"]["calls"] == 3
    assert budget["by_stage"]["guard"]["calls"] == 4
    assert (run / "requirements.md").exists()


def test_intake_rejects_bad_answers_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = build_docs_folder(tmp_path)
    brief = tmp_path / "b.md"
    brief.write_text("I need an assistant that answers HR policy questions.", encoding="utf-8")
    result = runner.invoke(
        cli.app, ["intake", "--brief", str(brief), "--docs", str(folder), "--answers", "nope.json"]
    )
    assert result.exit_code != 0


def _scripted_stack(handler: Any) -> Any:
    from forgeai.llm import Budget, BudgetedLLM, FakeLLM

    class Stack:
        def __init__(self) -> None:
            self.budget = Budget(max_tokens=100_000)
            self.llm = BudgetedLLM(FakeLLM(responder=handler), self.budget)

        async def aclose(self) -> None:
            return None

    return Stack()


def test_generate_then_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    from tests import scripted

    def handler(messages: list[Any]) -> str:
        task = scripted._task(messages)
        if task == "agent.plan":
            return '{"calls": []}'
        if task == "agent.answer":
            return _json.dumps(
                {
                    "answer": "24 days a year.",
                    "citations": [
                        {
                            "chunk_id": "?",
                            "quote": "Shore staff earn 24 days of annual leave per calendar year",
                        }
                    ],
                    "refused": False,
                }
            )
        return scripted.generator_responder(messages)

    monkeypatch.setattr(cli, "build_llm", lambda _s: _scripted_stack(handler))
    folder = build_docs_folder(tmp_path)
    brief = tmp_path / "hr.md"
    brief.write_text("I need an assistant that answers HR policy questions.", encoding="utf-8")
    common = ["--docs", str(folder)]
    result = runner.invoke(
        cli.app,
        ["intake", "--brief", str(brief), *common, "--answers", "auto", "--run-id", "cli-gen"],
    )
    assert result.exit_code == 0, result.stdout
    result = runner.invoke(cli.app, ["generate", "--run-id", "cli-gen", *common])
    assert result.exit_code == 0, result.stdout
    assert "Rejected weather" in result.stdout

    definition = cli.get_settings().runs_dir / "cli-gen" / "agent.v0.json"
    result = runner.invoke(cli.app, ["chat", str(definition), *common, "-q", "Annual leave?"])
    assert result.exit_code == 0, result.stdout
    assert "24 days a year." in result.stdout
    assert "verified" in result.stdout


def test_generate_without_requirements_fails(tmp_path: Path) -> None:
    folder = build_docs_folder(tmp_path)
    result = runner.invoke(cli.app, ["generate", "--run-id", "no-such-run", "--docs", str(folder)])
    assert result.exit_code == 1


def test_evalgen_and_evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests import sim

    monkeypatch.setattr(cli, "build_llm", lambda _s: _scripted_stack(sim.respond))
    folder = build_docs_folder(tmp_path)
    brief = tmp_path / "hr.md"
    brief.write_text("I need an assistant that answers HR policy questions.", encoding="utf-8")
    common = ["--run-id", "cli-eval", "--docs", str(folder)]
    assert (
        runner.invoke(
            cli.app, ["intake", "--brief", str(brief), "--answers", "auto", *common]
        ).exit_code
        == 0
    )
    assert runner.invoke(cli.app, ["generate", *common]).exit_code == 0

    too_strict = runner.invoke(cli.app, ["evalgen", *common])
    assert too_strict.exit_code == 1
    assert "survived validation" in too_strict.stdout

    result = runner.invoke(cli.app, ["evalgen", *common, "--min-cases", "5"])
    assert result.exit_code == 0, result.stdout
    assert "eval cases" in result.stdout

    result = runner.invoke(cli.app, ["evaluate", *common, "--split", "all"])
    assert result.exit_code == 0, result.stdout
    assert "dev:" in result.stdout
    assert "judge canary accuracy 100%" in result.stdout

    bad = runner.invoke(cli.app, ["evaluate", *common, "--split", "train"])
    assert bad.exit_code != 0


def test_build_runs_the_whole_factory_unattended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import sim

    monkeypatch.setattr(cli, "build_llm", lambda _s: _scripted_stack(sim.respond))
    folder = build_docs_folder(tmp_path)
    brief = tmp_path / "hr.md"
    brief.write_text("I need an assistant that answers HR policy questions.", encoding="utf-8")
    result = runner.invoke(
        cli.app,
        [
            "build",
            "--brief",
            str(brief),
            "--docs",
            str(folder),
            "--run-id",
            "cli-build",
            "--min-cases",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Built Shore Staff HR Desk" in result.stdout
    assert "holdout:" in result.stdout
    assert (cli.get_settings().runs_dir / "cli-build" / "report.md").exists()


def test_build_reports_where_artifacts_are_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import sim

    monkeypatch.setattr(cli, "build_llm", lambda _s: _scripted_stack(sim.respond))
    folder = build_docs_folder(tmp_path)
    brief = tmp_path / "hr.md"
    brief.write_text("I need an assistant that answers HR policy questions.", encoding="utf-8")
    result = runner.invoke(
        cli.app, ["build", "--brief", str(brief), "--docs", str(folder), "--run-id", "cli-fail"]
    )
    assert result.exit_code == 1  # default 20 cases cannot come from the tiny fixture
    assert "Artifacts so far are in" in result.stdout
