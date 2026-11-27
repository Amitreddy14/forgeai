from __future__ import annotations

from fastapi.testclient import TestClient

from forgeai import __version__
from forgeai.config import Settings
from forgeai.web.app import create_app


def test_healthz_reports_version() -> None:
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_security_headers_present() -> None:
    response = TestClient(create_app()).get("/healthz")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_api_docs_hidden_in_production() -> None:
    client = TestClient(create_app(Settings(env="prod")))
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
