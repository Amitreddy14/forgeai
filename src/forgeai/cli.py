"""The `forge` command. Phase 0 ships `version` and `doctor`; later phases add build/chat/report."""

from __future__ import annotations

import httpx
import typer
from rich.console import Console
from rich.table import Table

from forgeai import __version__
from forgeai.config import Settings, get_settings

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


if __name__ == "__main__":  # pragma: no cover
    app()
