"""Today's date and time, so the agent never guesses it from training data."""

from __future__ import annotations

from typing import ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from forgeai.schemas.requirements import ToolHint
from forgeai.tools.base import ToolEnv, ToolResult, expect_args


class CurrentDateArgs(BaseModel):
    timezone: str | None = Field(default=None, description="IANA zone, e.g. 'Asia/Kolkata'.")


class CurrentDate:
    name: ClassVar[str] = "current_date"
    capability: ClassVar[ToolHint] = "current_date"
    description: ClassVar[str] = "Today's date, weekday and time in a time zone."
    args_model: ClassVar[type[BaseModel]] = CurrentDateArgs

    async def run(self, args: BaseModel, env: ToolEnv) -> ToolResult:
        args = expect_args(args, CurrentDateArgs)
        zone_name = args.timezone or env.timezone
        try:
            zone = ZoneInfo(zone_name)
        except (ZoneInfoNotFoundError, ValueError):
            return ToolResult.fail(f"unknown time zone {zone_name!r}")
        now = env.now().astimezone(zone)
        return ToolResult(
            ok=True,
            output=f"{now:%A, %d %B %Y, %H:%M} ({zone_name}); ISO date {now:%Y-%m-%d}",
            data={"date": now.date().isoformat(), "weekday": now.strftime("%A")},
        )
