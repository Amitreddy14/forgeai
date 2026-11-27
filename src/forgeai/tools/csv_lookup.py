"""Look up rows in a CSV table from the documents folder, with simple typed filters."""

from __future__ import annotations

import json
import re
from typing import ClassVar

from pydantic import BaseModel, Field

from forgeai.schemas.requirements import ToolHint
from forgeai.tools.base import ToolEnv, ToolResult, expect_args

_OP = re.compile(r"^\s*(>=|<=|!=|>|<|=)?\s*(.+?)\s*$")
MAX_ROWS = 20


class CsvLookupArgs(BaseModel):
    table: str = Field(description="Table id (the CSV document id).")
    filters: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "column -> condition. Text matches case-insensitively (exact, or contains with "
            "'~value'). Numbers accept >=, <=, >, <, =, != (e.g. '>=250')."
        ),
    )
    columns: list[str] | None = Field(default=None, description="Columns to return (all if empty).")
    limit: int = Field(default=10, ge=1, le=MAX_ROWS)


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def matches(cell: str, condition: str) -> bool:
    condition = condition.strip()
    if condition.startswith("~"):
        return condition[1:].strip().lower() in cell.lower()
    m = _OP.match(condition)
    op, value = (m.group(1) or "=", m.group(2)) if m else ("=", condition)
    left, right = _to_float(cell), _to_float(value)
    if left is not None and right is not None:
        return {
            "=": left == right,
            "!=": left != right,
            ">": left > right,
            "<": left < right,
            ">=": left >= right,
            "<=": left <= right,
        }[op]
    if op == "=":
        return cell.strip().lower() == value.strip().lower()
    if op == "!=":
        return cell.strip().lower() != value.strip().lower()
    return False  # ordering comparison on text: no match rather than a guess


class CsvLookup:
    name: ClassVar[str] = "csv_lookup"
    capability: ClassVar[ToolHint] = "table_lookup"
    description: ClassVar[str] = (
        "Find rows in a CSV table from the documents. Returns matching rows as JSON "
        "with their row numbers."
    )
    args_model: ClassVar[type[BaseModel]] = CsvLookupArgs

    async def run(self, args: BaseModel, env: ToolEnv) -> ToolResult:
        args = expect_args(args, CsvLookupArgs)
        rows = env.tables.get(args.table)
        if rows is None:
            available = ", ".join(sorted(env.tables)) or "none"
            return ToolResult.fail(f"unknown table {args.table!r}; available: {available}")
        columns = list(rows[0].keys()) if rows else []
        lower = {c.lower(): c for c in columns}
        unknown = [c for c in [*args.filters, *(args.columns or [])] if c.lower() not in lower]
        if unknown:
            return ToolResult.fail(
                f"unknown column(s) {unknown}; columns are: {', '.join(columns)}"
            )
        filters = {lower[k.lower()]: v for k, v in args.filters.items()}
        wanted = [lower[c.lower()] for c in args.columns] if args.columns else columns

        found: list[dict[str, str | int]] = []
        total = 0
        for number, row in enumerate(rows, start=1):
            if all(matches(row.get(col, ""), cond) for col, cond in filters.items()):
                total += 1
                if len(found) < args.limit:
                    found.append({"row": number, **{c: row.get(c, "") for c in wanted}})
        summary = f"{total} matching row(s) in {args.table}"
        if total > len(found):
            summary += f" (showing first {len(found)})"
        return ToolResult(
            ok=True,
            output=f"{summary}:\n{json.dumps(found, ensure_ascii=False)}",
            data={"rows": found, "total": total},
        )
