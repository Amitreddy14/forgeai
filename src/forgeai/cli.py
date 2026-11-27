"""The `forge` command. Phase 0 ships `version` and `doctor`; later phases add build/chat/report."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import click
import httpx
import typer
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from forgeai import __version__
from forgeai.config import Settings, get_settings
from forgeai.errors import ForgeError
from forgeai.ingest import chunk_corpus, load_folder
from forgeai.llm.factory import build_llm
from forgeai.pipeline import RequirementsOutcome, build_requirements
from forgeai.retrieval import HybridIndex, RetrievalConfig, make_embedder
from forgeai.schemas import ChunkingConfig
from forgeai.schemas.requirements import Requirements
from forgeai.stages.context import ForgeContext, ModelRoles
from forgeai.stages.intake import (
    Answerer,
    AutoAnswerer,
    FileAnswerer,
    InteractiveAnswerer,
    requirements_markdown,
)
from forgeai.store import RunStore, new_run_id, validate_run_id

app = typer.Typer(
    name="forge",
    help="ForgeAI: build, test and improve grounded agents from a one-paragraph brief.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the ForgeAI version."""
    console.print(f"forgeai {__version__}")


def check_model_endpoint(settings: Settings, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Ask the OpenAI-compatible endpoint for its model list. Returns (ok, detail)."""
    headers = {}
    if settings.llm_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value()}"
    url = settings.llm_base_url.rstrip("/") + "/models"
    try:
        response = httpx.get(url, headers=headers, timeout=timeout_s)
    except httpx.HTTPError as exc:
        return False, f"could not reach {url} ({type(exc).__name__})"
    if response.status_code != 200:
        return False, f"{url} answered {response.status_code}"
    try:
        ids = {item["id"] for item in response.json().get("data", [])}
    except (ValueError, KeyError, TypeError):
        return False, f"{url} did not return an OpenAI-style model list"
    missing = [m for m in (settings.generator_model, settings.judge_model) if m not in ids]
    if missing:
        return False, f"endpoint reachable, but these models are missing: {', '.join(missing)}"
    return True, f"endpoint reachable, both models available ({len(ids)} listed)"


@app.command()
def doctor(
    offline: bool = typer.Option(False, "--offline", help="Skip the model endpoint check."),
) -> None:
    """Show the active settings and check that the model endpoint answers."""
    settings = get_settings()
    table = Table(title="ForgeAI settings", show_header=True)
    table.add_column("Setting")
    table.add_column("Value")
    for key, value in settings.redacted().items():
        table.add_row(key, str(value))
    console.print(table)

    if offline:
        console.print("Model check skipped (--offline).")
        return
    ok, detail = check_model_endpoint(settings)
    if ok:
        console.print(f"[green]OK[/green]  {detail}")
    else:
        console.print(f"[red]FAIL[/red]  {detail}")
        raise typer.Exit(code=1)


class _SmokeAnswer(BaseModel):
    capital: str
    confidence: float = Field(ge=0, le=1)


async def _smoke(settings: Settings) -> list[tuple[str, str, float, int]]:
    from forgeai.llm import ChatMessage, complete_json
    from forgeai.llm.factory import build_llm

    stack = build_llm(settings)
    results: list[tuple[str, str, float, int]] = []
    prompt = [ChatMessage(role="user", content="What is the capital of Kerala, India?")]
    try:
        for model in (settings.generator_model, settings.judge_model):
            started = time.perf_counter()
            before = stack.budget.used_tokens
            answer = await complete_json(stack.llm, prompt, _SmokeAnswer, model=model)
            results.append(
                (
                    model,
                    answer.capital,
                    time.perf_counter() - started,
                    stack.budget.used_tokens - before,
                )
            )
    finally:
        await stack.aclose()
    return results


@app.command()
def smoke() -> None:
    """Make one structured-JSON call to each configured model and time it."""
    settings = get_settings()
    try:
        results = asyncio.run(_smoke(settings))
    except ForgeError as exc:
        console.print(f"[red]FAIL[/red]  {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc
    for model, capital, seconds, tokens in results:
        console.print(f"[green]OK[/green]  {model}: {capital!r} in {seconds:.1f}s, {tokens} tokens")


DocsArg = typer.Argument(..., exists=True, file_okay=False, help="Folder of documents.")


@app.command()
def ingest(
    docs: Path = DocsArg,
    chunk_words: int = typer.Option(180, help="Words per chunk."),
    overlap_words: int = typer.Option(30, help="Words carried into the next chunk."),
) -> None:
    """Read a documents folder and show what ForgeAI extracted and how it chunks."""
    corpus = load_folder(docs)
    chunks = chunk_corpus(
        corpus, ChunkingConfig(chunk_words=chunk_words, overlap_words=overlap_words)
    )
    table = Table(title=f"Corpus: {docs}")
    for column in ("Document", "Type", "Title", "Pages/rows", "Words", "Chunks"):
        table.add_column(column)
    for doc in corpus.documents:
        n_chunks = sum(1 for c in chunks if c.doc_id == doc.doc_id)
        table.add_row(
            doc.doc_id,
            doc.kind,
            doc.title,
            str(len(doc.pages)),
            str(doc.words),
            str(n_chunks),
        )
    console.print(table)
    for skip in corpus.skipped:
        console.print(f"[yellow]Skipped[/yellow] {skip.filename}: {skip.reason}")
    if not corpus.documents:
        console.print("[red]No readable documents found.[/red]")
        raise typer.Exit(code=1)


@app.command()
def search(
    docs: Path = DocsArg,
    query: str = typer.Argument(..., help="What to look for."),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, max=20),
    mode: str = typer.Option("hybrid", help="hybrid, bm25 or dense."),
) -> None:
    """Search a documents folder with the same retriever the generated agents use."""
    if mode not in ("hybrid", "bm25", "dense"):
        raise typer.BadParameter("mode must be hybrid, bm25 or dense")
    corpus = load_folder(docs)
    chunks = chunk_corpus(corpus)
    if not chunks:
        console.print("[red]No readable documents found.[/red]")
        raise typer.Exit(code=1)
    index = HybridIndex(chunks, make_embedder(get_settings().embedder))
    config = RetrievalConfig.model_validate({"top_k": top_k, "mode": mode})
    for hit in index.search(query, config):
        console.print(
            f"[bold]{hit.rank}. {hit.chunk.citation}[/bold]  "
            f"rrf={hit.score:.4f} bm25={hit.bm25:.2f} dense={hit.dense:.2f}"
        )
        console.print(f"   {hit.chunk.text[:240]}{'…' if len(hit.chunk.text) > 240 else ''}")


def _approve_interactively(req: Requirements) -> Requirements:
    """Human checkpoint: approve the requirements, or edit them as JSON and re-check."""
    while True:
        console.print(Markdown(requirements_markdown(req)))
        if typer.confirm("Approve these requirements?", default=True):
            return req
        edited = click.edit(req.model_dump_json(indent=2), extension=".json")
        if not edited:
            console.print("No changes made.")
            continue
        try:
            req = Requirements.model_validate_json(edited)
        except ValidationError as exc:
            console.print(f"[red]Those edits are not valid requirements:[/red]\n{exc}")


def _make_answerer(answers: str, ctx: ForgeContext) -> Answerer:
    if answers == "auto":
        return AutoAnswerer(ctx)
    if answers == "interactive":
        return InteractiveAnswerer(lambda q, default: typer.prompt(q, default=default))
    path = Path(answers)
    if not path.is_file():
        raise typer.BadParameter("--answers must be auto, interactive, or a JSON file path")
    return FileAnswerer(path)


async def _intake(
    brief_text: str, docs: Path, answers: str, approve: bool, run_id: str
) -> RequirementsOutcome:
    settings = get_settings()
    stack = build_llm(settings)
    ctx = ForgeContext(
        llm=stack.llm,
        models=ModelRoles(generator=settings.generator_model, judge=settings.judge_model),
        embedder=make_embedder(settings.embedder),
    )
    store = RunStore(settings.runs_dir, run_id)
    try:
        return await build_requirements(
            ctx,
            store,
            brief_text,
            docs,
            _make_answerer(answers, ctx),
            approve=_approve_interactively if approve else None,
        )
    finally:
        store.write_json("budget.json", stack.budget.report())
        await stack.aclose()


@app.command()
def intake(
    brief: Path = typer.Option(..., exists=True, dir_okay=False, help="Brief (text/markdown)."),
    docs: Path = typer.Option(..., exists=True, file_okay=False, help="Documents folder."),
    answers: str = typer.Option(
        "interactive", help="interactive, auto (unattended), or a JSON file of answers."
    ),
    approve: bool = typer.Option(False, help="Review and approve requirements before saving."),
    run_id: str | None = typer.Option(None, help="Run folder name (default: generated)."),
) -> None:
    """Interview the requester, check the brief against the documents, write requirements."""
    brief_text = brief.read_text(encoding="utf-8")
    rid = validate_run_id(run_id) if run_id else new_run_id(brief.stem)
    try:
        outcome = asyncio.run(_intake(brief_text, docs, answers, approve, rid))
    except (ForgeError, ValueError) as exc:
        console.print(f"[red]Stopped:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    req = outcome.requirements
    console.print(f"[green]Requirements written[/green] to {get_settings().runs_dir / rid}")
    console.print(
        f"  {req.agent_name}: {len(req.must_answer)} must-answer, "
        f"{len(req.must_refuse)} must-refuse, {len(req.flagged_gaps)} flagged gaps, "
        f"{len(req.assumptions)} assumptions"
    )
    for gap in req.flagged_gaps:
        console.print(f"  [yellow]Flagged[/yellow] {gap.topic}")


if __name__ == "__main__":  # pragma: no cover
    app()
