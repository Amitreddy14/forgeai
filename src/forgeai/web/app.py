"""ForgeAI web app: browse builds as survey certificates, chat with any built agent, and
start a new build. Server-rendered (Jinja) with HTMX for the live parts."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from forgeai import __version__
from forgeai.config import Settings, get_settings
from forgeai.errors import ForgeError
from forgeai.ingest.loaders import ALLOWED
from forgeai.llm import Budget
from forgeai.llm.factory import build_llm
from forgeai.retrieval import Embedder, make_embedder
from forgeai.runtime import AgentRuntime, AgentTurn, load_runtime
from forgeai.stages.context import ForgeContext, ModelRoles
from forgeai.stages.intake import AutoAnswerer
from forgeai.store import InvalidName, RunStore, new_run_id, validate_run_id
from forgeai.web.data import list_run_views, load_run, render_markdown

log = logging.getLogger(__name__)
HERE = Path(__file__).parent

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; frame-ancestors 'self'"
    ),
}
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class RateLimiter:
    """Small in-memory sliding window per client: enough to stop a free tier being drained."""

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit, self.window_s = limit, window_s
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self.hits[key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


def _client(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    is_prod = settings.env == "prod"
    app = FastAPI(
        title="ForgeAI",
        version=__version__,
        debug=False,
        docs_url=None if is_prod else "/docs",
        redoc_url=None,
        openapi_url=None if is_prod else "/openapi.json",
    )
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters["pct"] = _pct
    templates.env.filters["markdown"] = render_markdown
    templates.env.globals["version"] = __version__
    templates.env.globals["ly"] = lambda v: round(20 + (1 - float(v)) * 220, 1)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    runs_dir = settings.runs_dir
    state: dict[str, Any] = {"embedder": None, "stack": None, "runtimes": {}, "build": None}
    chat_limit = RateLimiter(limit=20, window_s=60)
    build_limit = RateLimiter(limit=2, window_s=3600)

    def embedder() -> Embedder:
        if state["embedder"] is None:
            state["embedder"] = make_embedder(settings.embedder)
        return state["embedder"]  # type: ignore[no-any-return]

    def llm_stack() -> Any:
        if state["stack"] is None:
            state["stack"] = build_llm(settings, Budget(max_tokens=10**12))
        return state["stack"]

    @app.middleware("http")
    async def add_security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    def run_or_404(run_id: str) -> Any:
        try:
            validate_run_id(run_id)
        except InvalidName as exc:
            raise HTTPException(status_code=404, detail="No such build") from exc
        if not (runs_dir / run_id).is_dir():
            raise HTTPException(status_code=404, detail="No such build")
        return load_run(runs_dir, run_id)

    # ---------- Pages --------------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/", response_class=HTMLResponse)
    async def builds(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"runs": list_run_views(runs_dir), "settings": settings, "nav": "builds"},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(request: Request, run_id: str) -> HTMLResponse:
        view = run_or_404(run_id)
        return templates.TemplateResponse(request, "run.html", {"run": view, "settings": settings})

    @app.get("/runs/{run_id}/progress", response_class=HTMLResponse)
    async def run_progress(request: Request, run_id: str) -> Response:
        view = run_or_404(run_id)
        response = templates.TemplateResponse(request, "_progress.html", {"run": view})
        if view.complete:
            response.headers["HX-Redirect"] = f"/runs/{run_id}"
        return response

    @app.get("/runs/{run_id}/files/{name:path}", response_class=PlainTextResponse)
    async def run_file(run_id: str, name: str) -> PlainTextResponse:
        view = run_or_404(run_id)
        try:
            path = view.store.path(name)
        except InvalidName as exc:
            raise HTTPException(status_code=404, detail="No such file") from exc
        if not path.is_file() or path.suffix not in {".json", ".md"}:
            raise HTTPException(status_code=404, detail="No such file")
        media = "application/json" if path.suffix == ".json" else "text/markdown"
        return PlainTextResponse(
            path.read_text(encoding="utf-8"), media_type=f"{media}; charset=utf-8"
        )

    # ---------- Chat ---------------------------------------------------------------

    def runtime_for(run_id: str) -> AgentRuntime:
        cached = state["runtimes"].get(run_id)
        if cached is not None:
            return cached  # type: ignore[no-any-return]
        view = run_or_404(run_id)
        if view.agent is None:
            raise HTTPException(status_code=409, detail="This build has no agent yet.")
        runtime = load_runtime(view.agent, llm_stack().llm, settings.generator_model, embedder())
        state["runtimes"][run_id] = runtime
        return runtime

    @app.get("/runs/{run_id}/chat", response_class=HTMLResponse)
    async def chat_page(request: Request, run_id: str) -> HTMLResponse:
        view = run_or_404(run_id)
        if view.agent is None:
            raise HTTPException(status_code=409, detail="This build has no agent yet.")
        return templates.TemplateResponse(request, "chat.html", {"run": view, "settings": settings})

    @app.post("/runs/{run_id}/chat", response_class=HTMLResponse)
    async def chat_turn(request: Request, run_id: str, question: str = Form(...)) -> HTMLResponse:
        question = question.strip()
        if not 2 <= len(question) <= 500:
            raise HTTPException(status_code=400, detail="Ask a question of 2 to 500 characters.")
        if not chat_limit.allow(_client(request)):
            raise HTTPException(
                status_code=429, detail="Too many questions. Wait a minute and try again."
            )
        try:
            runtime = runtime_for(run_id)
            turn: AgentTurn = await runtime.answer(question)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503, detail="The agent's documents are missing on this server."
            ) from exc
        except ForgeError as exc:
            raise HTTPException(
                status_code=503, detail="The model server did not answer. Try again shortly."
            ) from exc
        return templates.TemplateResponse(request, "_turn.html", {"turn": turn})

    # ---------- New build ----------------------------------------------------------

    @app.get("/new", response_class=HTMLResponse)
    async def new_build(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "new.html",
            {
                "settings": settings,
                "busy": state["build"] is not None and not state["build"].done(),
                "allowed": ", ".join(sorted(ALLOWED)),
            },
        )

    @app.post("/builds")
    async def start_build(
        request: Request,
        brief: str = Form(...),
        files: list[UploadFile] | None = None,
    ) -> Response:
        if not settings.allow_builds:
            raise HTTPException(status_code=403, detail="Building is switched off on this server.")
        if state["build"] is not None and not state["build"].done():
            raise HTTPException(
                status_code=409, detail="Another build is running. Try again when it finishes."
            )
        brief = brief.strip()
        if not 20 <= len(brief) <= 2000:
            raise HTTPException(status_code=400, detail="Write a brief of 20 to 2000 characters.")
        uploads = [f for f in files or [] if f.filename]
        if not uploads:
            raise HTTPException(status_code=400, detail="Add at least one document.")
        if len(uploads) > settings.max_upload_files:
            raise HTTPException(
                status_code=413, detail=f"Upload at most {settings.max_upload_files} files."
            )

        # Count only builds that would really start, not rejected forms.
        if not build_limit.allow(_client(request)):
            raise HTTPException(
                status_code=429, detail="Build limit reached for now. Try again later."
            )
        run_id = new_run_id(brief.split(".")[0][:40] or "build")
        store = RunStore(runs_dir, run_id)
        docs_dir = store.root / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        limit = settings.max_upload_mb * 1024 * 1024
        try:
            for upload in uploads:
                name = (
                    SAFE_FILENAME.sub("-", Path(upload.filename or "file").name).strip("-.")
                    or "file"
                )
                if Path(name).suffix.lower() not in ALLOWED:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{name}: allowed types are {', '.join(sorted(ALLOWED))}.",
                    )
                data = await upload.read(limit + 1)
                if len(data) > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{name} is larger than {settings.max_upload_mb} MB.",
                    )
                (docs_dir / name).write_bytes(data)
        except HTTPException:
            shutil.rmtree(store.root, ignore_errors=True)
            raise

        state["build"] = asyncio.create_task(_run_build(settings, store, brief, docs_dir))
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    return app


async def _run_build(settings: Settings, store: RunStore, brief: str, docs_dir: Path) -> None:
    """Run the factory in the background; progress and errors go to the run folder."""
    from forgeai.pipeline import run_factory

    events: list[dict[str, Any]] = []
    started = time.monotonic()

    def progress(message: str) -> None:
        events.append({"t": round(time.monotonic() - started), "message": message})
        store.write_json("progress.json", events)

    stack = build_llm(settings)
    ctx = ForgeContext(
        llm=stack.llm,
        models=ModelRoles(generator=settings.generator_model, judge=settings.judge_model),
        embedder=make_embedder(settings.embedder),
    )
    try:
        await run_factory(
            ctx,
            store,
            brief,
            docs_dir,
            AutoAnswerer(ctx),
            stack.budget,
            max_iterations=settings.max_iterations,
            min_cases=10,
            progress=progress,
        )
    except Exception as exc:  # a failed build must leave a readable explanation behind
        log.exception("build %s failed", store.run_id)
        store.write_json("stage_error.json", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        store.write_json("budget.json", stack.budget.report())
        await stack.aclose()


app = create_app()
