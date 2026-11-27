from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from forgeai.store import InvalidName, RunStore, list_runs, new_run_id, validate_run_id


class Thing(BaseModel):
    name: str
    score: float


def test_round_trips_models_and_text(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "hr-demo")
    store.write_json("agent.v0.json", Thing(name="hr", score=0.78))
    store.write_text("report.md", "# Report")
    store.write_json("iterations/iter_1/results.json", [{"case": "a1", "pass": True}])
    assert store.read_json("agent.v0.json", Thing).score == 0.78
    assert store.read_text("report.md") == "# Report\n"
    assert store.list_artifacts() == [
        "agent.v0.json",
        "iterations/iter_1/results.json",
        "report.md",
    ]
    assert list_runs(tmp_path) == ["hr-demo"]


@pytest.mark.parametrize("bad", ["../escape", "UPPER", "a", "has space", "x/y", "-lead", "a" * 65])
def test_rejects_bad_run_ids(bad: str) -> None:
    with pytest.raises(InvalidName):
        validate_run_id(bad)


@pytest.mark.parametrize(
    "bad", ["../../etc/passwd", "/abs.json", "a/../../b.json", ".hidden", "x/.git/config", ""]
)
def test_rejects_path_traversal(tmp_path: Path, bad: str) -> None:
    store = RunStore(tmp_path, "safe-run")
    with pytest.raises(InvalidName):
        store.write_text(bad, "nope")


def test_new_run_id_is_valid_and_slugged() -> None:
    run_id = new_run_id("HR Policy Desk!!")
    assert run_id.startswith("hr-policy-desk-")
    assert validate_run_id(run_id) == run_id


def test_list_runs_on_missing_dir(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "missing") == []
