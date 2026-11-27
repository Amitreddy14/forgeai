"""Load any agent definition from disk and get a runnable agent."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from forgeai.ingest import load_folder
from forgeai.ingest.loaders import ALLOWED
from forgeai.llm import LLMClient
from forgeai.retrieval import Embedder
from forgeai.runtime.agent import AgentRuntime
from forgeai.schemas.agent import AgentDefinition
from forgeai.tools import ToolEnv

log = logging.getLogger(__name__)


def fingerprint_folder(folder: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder)
        if not path.is_file() or path.suffix.lower() not in ALLOWED:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        out[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def load_definition(path: Path) -> AgentDefinition:
    return AgentDefinition.model_validate_json(path.read_text(encoding="utf-8"))


def changed_documents(defn: AgentDefinition, docs_dir: Path) -> list[str]:
    """Files added, removed or edited since the agent was generated."""
    now = fingerprint_folder(docs_dir)
    before = defn.knowledge.fingerprints
    return sorted({f for f in before.keys() | now.keys() if before.get(f) != now.get(f)})


def load_runtime(
    definition: AgentDefinition,
    llm: LLMClient,
    model: str,
    embedder: Embedder,
    *,
    docs_dir: Path | None = None,
    tool_env: ToolEnv | None = None,
) -> AgentRuntime:
    folder = docs_dir or Path(definition.knowledge.docs_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"documents folder {folder} not found; pass --docs to point at it")
    changed = changed_documents(definition, folder)
    if changed:
        log.warning("documents changed since generation: %s", ", ".join(changed))
    corpus = load_folder(folder)
    return AgentRuntime(definition, corpus, embedder, llm, model, tool_env)
