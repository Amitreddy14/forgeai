"""The `forge` command. Phase 0 ships `version` and `doctor`; later phases add build/chat/report."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import click
import httpx
import typer
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from forgeai import __version__
from forgeai.config import Settings, get_settings
from forgeai.errors import ForgeError
from forgeai.ingest import chunk_corpus, load_folder
from forgeai.llm.factory import build_llm
from forgeai.pipeline import (
    RequirementsOutcome,
    build_agent,
    build_evals,
    build_requirements,
    evaluate_agent,
    load_requirements_outcome,
)
from forgeai.retrieval import HybridIndex, RetrievalConfig, make_embedder
from forgeai.runtime import AgentTurn, load_definition, load_runtime
from forgeai.schemas import ChunkingConfig
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import EvalRun, EvalSet
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
    ctx, stack = _context(settings)
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


def _context(settings: Settings) -> tuple[ForgeContext, Any]:
    stack = build_llm(settings)
    ctx = ForgeContext(
        llm=stack.llm,
        models=ModelRoles(generator=settings.generator_model, judge=settings.judge_model),
        embedder=make_embedder(settings.embedder),
    )
    return ctx, stack


async def _generate(run_id: str, docs: Path) -> AgentDefinition:
    settings = get_settings()
    ctx, stack = _context(settings)
    store = RunStore(settings.runs_dir, run_id)
    try:
        outcome = load_requirements_outcome(store)
        return await build_agent(ctx, store, outcome, docs)
    finally:
        store.write_json("budget.generate.json", stack.budget.report())
        await stack.aclose()


@app.command()
def generate(
    run_id: str = typer.Option(..., help="Run folder that already has requirements.json."),
    docs: Path = typer.Option(..., exists=True, file_okay=False, help="Documents folder."),
) -> None:
    """Generate the agent definition (agent.v0.json) from a run's requirements."""
    rid = validate_run_id(run_id)
    try:
        definition = asyncio.run(_generate(rid, docs))
    except (ForgeError, FileNotFoundError, ValidationError) as exc:
        console.print(f"[red]Stopped:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    tools = ", ".join(definition.tool_names) or "none"
    console.print(f"[green]Agent generated[/green]: {definition.name} (tools: {tools})")
    for rejected in definition.rejected_tools:
        console.print(f"  [yellow]Rejected[/yellow] {rejected.name}: {rejected.reason}")


def print_turn(turn: AgentTurn) -> None:
    if turn.error:
        console.print(f"[red]Agent error:[/red] {turn.error}")
        return
    label = "[yellow]Refused[/yellow]" if turn.answer.refused else "[bold]Answer[/bold]"
    console.print(f"{label}: {escape(turn.answer.answer)}")
    if turn.answer.refused and turn.answer.refusal_reason:
        console.print(f"  reason: {escape(turn.answer.refusal_reason)}")
    for c in turn.valid_citations:
        console.print(f'  [green]verified[/green] {escape(f"[{c.chunk_id}]")} "{escape(c.quote)}"')
    for c in turn.invalid_citations:
        console.print(f'  [red]not in source[/red] {escape(f"[{c.chunk_id}]")} "{escape(c.quote)}"')
    for call in turn.tool_calls:
        status = "[green]ok[/green]" if call.ok else "[red]failed[/red]"
        console.print(f"  tool {call.name} {status}: {escape(call.output[:200])}")
    console.print(
        f"  [dim]{turn.latency_s:.1f}s, best passage similarity {turn.best_dense:.2f}[/dim]"
    )


async def _chat(definition_path: Path, docs: Path | None, question: str | None) -> None:
    settings = get_settings()
    stack = build_llm(settings)
    try:
        runtime = load_runtime(
            load_definition(definition_path),
            stack.llm,
            settings.generator_model,
            make_embedder(settings.embedder),
            docs_dir=docs,
        )
        console.print(f"[bold]{runtime.definition.name}[/bold] is ready. Empty line to quit.")
        while True:
            q = question or typer.prompt("You", default="", show_default=False)
            if not q.strip():
                break
            print_turn(await runtime.answer(q.strip()))
            if question:
                break
    finally:
        await stack.aclose()


@app.command()
def chat(
    definition: Path = typer.Argument(..., exists=True, dir_okay=False, help="agent.vN.json"),
    docs: Path | None = typer.Option(None, file_okay=False, help="Override the documents folder."),
    question: str | None = typer.Option(None, "--question", "-q", help="Ask once and exit."),
) -> None:
    """Chat with any generated agent definition."""
    try:
        asyncio.run(_chat(definition, docs, question))
    except (ForgeError, FileNotFoundError, ValidationError) as exc:
        console.print(f"[red]Stopped:[/red] {exc}")
        raise typer.Exit(code=1) from exc


async def _evalgen(run_id: str, docs: Path, min_cases: int) -> EvalSet:
    settings = get_settings()
    ctx, stack = _context(settings)
    store = RunStore(settings.runs_dir, run_id)
    try:
        requirements = store.read_json("requirements.json", Requirements)
        return await build_evals(ctx, store, requirements, docs, min_cases=min_cases)
    finally:
        store.write_json("budget.evalgen.json", stack.budget.report())
        await stack.aclose()


@app.command()
def evalgen(
    run_id: str = typer.Option(..., help="Run folder with requirements.json."),
    docs: Path = typer.Option(..., exists=True, file_okay=False, help="Documents folder."),
    min_cases: int = typer.Option(20, min=1, help="Stop if fewer cases survive validation."),
) -> None:
    """Generate and validate the eval set from the documents (evals.json, evals.md)."""
    rid = validate_run_id(run_id)
    try:
        evals = asyncio.run(_evalgen(rid, docs, min_cases))
    except (ForgeError, FileNotFoundError, ValidationError) as exc:
        console.print(f"[red]Stopped:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    counts = ", ".join(f"{k} {v}" for k, v in sorted(evals.counts().items()))
    console.print(
        f"[green]{len(evals.cases)} eval cases[/green] ({counts}); "
        f"{len(evals.rejected)} candidates rejected by validation; "
        f"dev {len(evals.split('dev'))} / holdout {len(evals.split('holdout'))}"
    )


async def _evaluate(run_id: str, docs: Path, agent: Path | None, split: str) -> EvalRun:
    settings = get_settings()
    ctx, stack = _context(settings)
    store = RunStore(settings.runs_dir, run_id)
    try:
        definition = load_definition(agent or store.path("agent.v0.json"))
        evals = store.read_json("evals.json", EvalSet)
        return await evaluate_agent(ctx, store, definition, evals, docs, split)  # type: ignore[arg-type]
    finally:
        store.write_json("budget.evaluate.json", stack.budget.report())
        await stack.aclose()


@app.command()
def evaluate(
    run_id: str = typer.Option(..., help="Run folder with evals.json."),
    docs: Path = typer.Option(..., exists=True, file_okay=False, help="Documents folder."),
    agent: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Default agent.v0.json."
    ),
    split: str = typer.Option("dev", help="dev, holdout or all."),
) -> None:
    """Run the agent over the eval set and judge every answer."""
    if split not in ("dev", "holdout", "all"):
        raise typer.BadParameter("split must be dev, holdout or all")
    rid = validate_run_id(run_id)
    try:
        run = asyncio.run(_evaluate(rid, docs, agent, split))
    except (ForgeError, FileNotFoundError, ValidationError) as exc:
        console.print(f"[red]Stopped:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    for name, score in run.scorecard.splits.items():
        console.print(
            f"[bold]{name}[/bold]: {score.passed}/{score.cases} passed ({score.pass_rate:.0%}); "
            f"failures: {score.failures or 'none'}"
        )
    if run.scorecard.judge_canary_accuracy is not None:
        console.print(f"judge canary accuracy {run.scorecard.judge_canary_accuracy:.0%}")


if __name__ == "__main__":  # pragma: no cover
    app()
