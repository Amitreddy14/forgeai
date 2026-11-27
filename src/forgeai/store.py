"""RunStore: every artifact of a run lives in runs/<run_id>/ as JSON or Markdown.

Run IDs and artifact names are validated so nothing (including a web request) can read
or write outside the runs folder.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
ARTIFACT_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class InvalidName(ValueError):
    pass


def validate_run_id(run_id: str) -> str:
    if not RUN_ID.match(run_id):
        raise InvalidName(f"invalid run id {run_id!r}: use 3-64 lowercase letters, digits, '-'")
    return run_id


def _validate_artifact(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or not all(ARTIFACT_PART.match(p) for p in path.parts):
        raise InvalidName(f"invalid artifact name {name!r}")
    return path


def new_run_id(slug: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")[:40] or "run"
    return validate_run_id(f"{base}-{stamp}")


def _plain(value: Any) -> Any:
    """JSON-ready copy: models (also nested in lists and dicts) become plain data.
    Anything else that JSON cannot hold is a bug, so it raises instead of str()-ing."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"cannot store {type(value).__name__} as JSON")


class RunStore:
    def __init__(self, runs_dir: Path, run_id: str) -> None:
        self.run_id = validate_run_id(run_id)
        self.root = (runs_dir / self.run_id).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        target = (self.root / _validate_artifact(name)).resolve()
        if not target.is_relative_to(self.root):
            raise InvalidName(f"artifact {name!r} escapes the run folder")
        return target

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def write_json(self, name: str, value: BaseModel | dict[str, Any] | list[Any]) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, BaseModel):
            text = value.model_dump_json(indent=2)
        else:
            text = json.dumps(_plain(value), indent=2, ensure_ascii=False)
        target.write_text(text + "\n", encoding="utf-8")
        return target

    def read_json(self, name: str, model_type: type[T]) -> T:
        return model_type.model_validate_json(self.path(name).read_text(encoding="utf-8"))

    def write_text(self, name: str, text: str) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        return target

    def read_text(self, name: str) -> str:
        return self.path(name).read_text(encoding="utf-8")

    def list_artifacts(self) -> list[str]:
        return sorted(
            p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()
        )


def list_runs(runs_dir: Path) -> list[str]:
    if not runs_dir.is_dir():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if p.is_dir() and RUN_ID.match(p.name))
