"""Read a run folder into what the pages need. Everything is computed from the committed
artifacts; the web layer never re-scores anything."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from pydantic import BaseModel, ValidationError

from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import CaseResult, EvalRun, EvalSet
from forgeai.schemas.patch import ChangeLogEntry
from forgeai.schemas.requirements import GroundabilityReport, IntakeRecord, Requirements
from forgeai.stages.report import FinalScorecard
from forgeai.store import RunStore, list_runs

_md = MarkdownIt("commonmark", {"html": False}).enable("table")

# Stages in order, with the artifact that proves each one finished.
STAGES: list[tuple[str, str, str]] = [
    ("Intake", "intake.json", "Questions asked and answered"),
    ("Groundability", "groundability.json", "Brief checked against documents"),
    ("Requirements", "requirements.json", "Scope, audience, must answer and refuse"),
    ("Agent", "agent.v0.json", "Prompt, retrieval, tools"),
    ("Eval set", "evals.json", "Test cases written and verified"),
    ("Judge check", "canaries.json", "Planted answers graded"),
    ("Optimizer", "changelog.json", "Patches tried, kept or reverted"),
    ("Report", "report.md", "Holdout score and open failures"),
]


def render_markdown(text: str) -> str:
    html: str = _md.render(text)
    return html


@dataclass
class StageState:
    name: str
    hint: str
    done: bool
    active: bool = False


@dataclass
class RunView:
    run_id: str
    store: RunStore
    requirements: Requirements | None = None
    intake: IntakeRecord | None = None
    groundability: GroundabilityReport | None = None
    agent: AgentDefinition | None = None
    evals: EvalSet | None = None
    scorecard: FinalScorecard | None = None
    changelog: list[ChangeLogEntry] = field(default_factory=list)
    final_dev: EvalRun | None = None
    holdout: EvalRun | None = None
    progress: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def name(self) -> str:
        if self.agent:
            return self.agent.name
        if self.requirements:
            return self.requirements.agent_name
        return self.run_id

    @property
    def complete(self) -> bool:
        return self.scorecard is not None

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def stages(self) -> list[StageState]:
        states = [StageState(n, h, self.store.exists(f)) for n, f, h in STAGES]
        if not self.complete and not self.failed:
            for s in states:
                if not s.done:
                    s.active = True
                    break
        return states

    @property
    def open_failures(self) -> list[CaseResult]:
        runs = [r for r in (self.final_dev, self.holdout) if r is not None]
        return [res for run in runs for res in run.results if not res.judgement.passed]

    def artifacts(self) -> list[str]:
        return self.store.list_artifacts()

    def report_html(self) -> str:
        return (
            render_markdown(self.store.read_text("report.md"))
            if self.store.exists("report.md")
            else ""
        )

    def requirements_html(self) -> str:
        if not self.store.exists("requirements.md"):
            return ""
        return render_markdown(self.store.read_text("requirements.md"))


def _read(store: RunStore, name: str, model: type[BaseModel]) -> Any:
    if not store.exists(name):
        return None
    try:
        return store.read_json(name, model)
    except (ValidationError, ValueError):
        return None


def load_run(runs_dir: Path, run_id: str) -> RunView:
    store = RunStore(runs_dir, run_id)
    view = RunView(run_id=run_id, store=store)
    view.requirements = _read(store, "requirements.json", Requirements)
    view.intake = _read(store, "intake.json", IntakeRecord)
    view.groundability = _read(store, "groundability.json", GroundabilityReport)
    view.agent = _read(store, "agent.final.json", AgentDefinition) or _read(
        store, "agent.v0.json", AgentDefinition
    )
    view.evals = _read(store, "evals.json", EvalSet)
    view.scorecard = _read(store, "scorecard.json", FinalScorecard)
    if store.exists("changelog.json"):
        raw = json.loads(store.read_text("changelog.json"))
        view.changelog = [ChangeLogEntry.model_validate(e) for e in raw]
    if view.scorecard:
        version = view.scorecard.final_version
        view.final_dev = _read(store, "evals/final-dev.json", EvalRun)
        view.holdout = _read(store, f"evals/holdout-v{version}.json", EvalRun) or _read(
            store, "evals/holdout-v0.json", EvalRun
        )
    if store.exists("progress.json"):
        view.progress = json.loads(store.read_text("progress.json"))
    if store.exists("stage_error.json"):
        view.error = json.loads(store.read_text("stage_error.json")).get("error", "unknown error")
    return view


def list_run_views(runs_dir: Path) -> list[RunView]:
    views = [load_run(runs_dir, run_id) for run_id in list_runs(runs_dir)]
    # Finished runs first, then by name.
    return sorted(views, key=lambda v: (not v.complete, v.name.lower()))
