"""Tool contract. Every tool validates its own arguments with Pydantic, never trusts the
model, and returns a ToolResult instead of raising, so the agent loop can show the model
what went wrong and let it try again."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from forgeai.schemas.requirements import ToolHint

A = TypeVar("A", bound=BaseModel)


def expect_args(args: BaseModel, model: type[A]) -> A:
    if not isinstance(args, model):
        raise TypeError(f"expected {model.__name__}, got {type(args).__name__}")
    return args


class ToolResult(BaseModel):
    ok: bool
    output: str
    data: Any = None

    @classmethod
    def fail(cls, message: str) -> ToolResult:
        return cls(ok=False, output=f"ERROR: {message}")


@dataclass
class ToolEnv:
    """What tools may touch: the documents' tables, an HTTP client, and a clock."""

    tables: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    http: httpx.AsyncClient | None = None
    now: Callable[[], datetime] = datetime.now
    timezone: str = "Asia/Kolkata"


class Tool(Protocol):
    name: ClassVar[str]
    capability: ClassVar[ToolHint]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    async def run(self, args: BaseModel, env: ToolEnv) -> ToolResult: ...


async def call_tool(tool: Tool, raw_args: dict[str, Any], env: ToolEnv) -> ToolResult:
    try:
        args = tool.args_model.model_validate(raw_args)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}" for e in exc.errors()
        )
        return ToolResult.fail(f"invalid arguments for {tool.name}: {problems}")
    try:
        return await tool.run(args, env)
    except Exception as exc:  # a tool must never crash the agent loop
        return ToolResult.fail(f"{tool.name} failed: {type(exc).__name__}: {exc}")


def tool_card(tool: Tool) -> str:
    """How a tool is described to the model: name, purpose, argument schema."""
    schema = tool.args_model.model_json_schema()
    props = schema.get("properties", {})
    compact = {k: {kk: vv for kk, vv in v.items() if kk != "title"} for k, v in props.items()}
    return f"- {tool.name}: {tool.description}\n  arguments: {json.dumps(compact)}"
