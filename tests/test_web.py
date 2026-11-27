from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from forgeai import __version__
from forgeai.config import Settings
from forgeai.llm import Budget, BudgetedLLM, FakeLLM
from forgeai.pipeline import run_factory
from forgeai.retrieval import HashEmbedder
from forgeai.stages.context import ForgeContext, ModelRoles
from forgeai.stages.intake import AutoAnswerer
from forgeai.store import RunStore
from forgeai.web import app as web
from forgeai.web.app import RateLimiter, create_app
from tests import sim
from tests.fixtures import build_docs_folder


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One complete scripted build, shared by the page tests."""
    root = tmp_path_factory.mktemp("web")
    docs = build_docs_folder(root)
    budget = Budget(max_tokens=2_000_000)
    ctx = ForgeContext(
        llm=BudgetedLLM(FakeLLM(responder=sim.respond), budget),
        models=ModelRoles("gen", "judge"),
        embedder=HashEmbedder(),
    )
    store = RunStore(root / "runs", "hr-demo")
    asyncio.run(
        run_factory(
            ctx,
            store,
            "I need an assistant that answers HR policy questions.",
            docs,
            AutoAnswerer(ctx),
            budget,
            min_cases=5,
        )
    )
    partial = RunStore(root / "runs", "half-done")
    partial.write_text("brief.md", "x")
    partial.write_json("progress.json", [{"t": 0, "message": "intake + groundability guard"}])
    return root / "runs"


class _Stack:
    def __init__(self) -> None:
        self.budget = Budget(max_tokens=10**9)
        self.llm = BudgetedLLM(FakeLLM(responder=sim.respond), self.budget)

    async def aclose(self) -> None:
        return None


@pytest.fixture
def client(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(web, "build_llm", lambda *_a, **_k: _Stack())
    return TestClient(create_app(Settings(runs_dir=runs_dir, embedder="hash", env="prod")))


def test_healthz_reports_version(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_security_headers_present(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert "unsafe-inline" not in response.headers["Content-Security-Policy"]


def test_api_docs_hidden_in_production(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_builds_register_lists_runs(client: TestClient) -> None:
    html = client.get("/").text
    assert "Shore Staff HR Desk" in html
    assert "half-done" in html
    assert "Building" in html


def test_certificate_page_shows_evidence(client: TestClient) -> None:
    html = client.get("/runs/hr-demo").text
    for text in (
        "Certificate of survey",
        "holdout pass rate",
        "Conditions",
        "Particulars",
        "Survey log",
        "Chat with this agent",
        "Full report",
    ):
        assert text in html, text
    assert "<svg" in html


def test_in_progress_page_polls(client: TestClient) -> None:
    html = client.get("/runs/half-done").text
    assert 'hx-get="/runs/half-done/progress"' in html
    assert "intake + groundability guard" in html


def test_artifacts_are_served_safely(client: TestClient) -> None:
    ok = client.get("/runs/hr-demo/files/report.md")
    assert ok.status_code == 200
    assert ok.text.startswith("# ForgeAI report")
    assert json.loads(client.get("/runs/hr-demo/files/changelog.json").text) is not None
    assert client.get("/runs/hr-demo/files/../../../etc/passwd").status_code == 404
    assert client.get("/runs/hr-demo/files/docs/leave.pdf").status_code == 404
    assert client.get("/runs/NOT_VALID/").status_code == 404
    assert client.get("/runs/missing-run").status_code == 404


def test_chat_round_trip_shows_verified_quote(client: TestClient) -> None:
    assert "Your question" in client.get("/runs/hr-demo/chat").text
    response = client.post(
        "/runs/hr-demo/chat",
        data={"question": "How many days of annual leave do shore staff earn?"},
    )
    assert response.status_code == 200
    assert "Quote verified" in response.text or "Refused" in response.text


def test_chat_validates_and_rate_limits(client: TestClient) -> None:
    assert client.post("/runs/hr-demo/chat", data={"question": "x"}).status_code == 400
    assert (
        client.post("/runs/missing-run/chat", data={"question": "hello there"}).status_code == 404
    )


def test_rate_limiter_window() -> None:
    limiter = RateLimiter(limit=2, window_s=60)
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_build_upload_checks(client: TestClient) -> None:
    brief = "I need an assistant for HR questions about leave."
    no_files = client.post("/builds", data={"brief": brief})
    assert no_files.status_code in (400, 422)
    short = client.post(
        "/builds", data={"brief": "hi"}, files=[("files", ("a.md", b"# A\n\ntext"))]
    )
    assert short.status_code == 400
    bad_type = client.post("/builds", data={"brief": brief}, files=[("files", ("evil.exe", b"MZ"))])
    assert bad_type.status_code == 400
    assert "allowed types" in bad_type.json()["detail"]


def test_build_starts_in_background(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, Any] = {}

    async def fake_build(settings: Settings, store: RunStore, brief: str, docs: Path) -> None:
        started.update(run=store.run_id, files=sorted(p.name for p in docs.iterdir()))  # noqa: ASYNC240

    monkeypatch.setattr(web, "_run_build", fake_build)
    response = client.post(
        "/builds",
        data={"brief": "I need an assistant for HR questions about leave rules."},
        files=[("files", ("leave policy (v2).md", b"# Leave\n\nStaff get 24 days."))],
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/runs/i-need-an-assistant")
    assert started["files"] == ["leave-policy-v2-.md"]


def test_builds_can_be_switched_off(runs_dir: Path) -> None:
    client = TestClient(
        create_app(Settings(runs_dir=runs_dir, embedder="hash", allow_builds=False))
    )
    assert "switched off" in client.get("/new").text
    response = client.post("/builds", data={"brief": "x" * 30}, files=[("files", ("a.md", b"a"))])
    assert response.status_code == 403
