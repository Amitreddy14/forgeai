"""Typed model output: ask for JSON, extract it, validate it with Pydantic, repair once.

Small open models wrap JSON in prose or code fences, drop required fields, or invent
extra ones. This module turns any of that into either a validated object or a typed
`ModelOutputError` — never a half-parsed dict.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from forgeai.errors import ModelOutputError
from forgeai.llm.base import ChatMessage, LLMClient

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_INSTRUCTIONS = (
    "Reply with one JSON object only: no prose, no code fences, no comments. "
    "It must validate against this JSON Schema:\n{schema}"
)
_REPAIR_PROMPT = (
    "Your previous reply could not be used: {error}\n"
    "Reply again with one corrected JSON object only, valid against the same schema."
)


def extract_json_object(text: str) -> Any:
    """Return the first JSON object in `text`, tolerating fences and surrounding prose."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in the reply")


def schema_for(model_type: type[BaseModel]) -> dict[str, Any]:
    return model_type.model_json_schema()


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
            for err in exc.errors()[:6]
        ]
        return "validation failed: " + "; ".join(parts)
    return str(exc)


async def complete_json(
    llm: LLMClient,
    messages: list[ChatMessage],
    output_type: type[T],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    repair_attempts: int = 1,
) -> T:
    """Call the model and return a validated `output_type`, or raise ModelOutputError."""
    schema = schema_for(output_type)
    instructions = _JSON_INSTRUCTIONS.format(schema=json.dumps(schema, separators=(",", ":")))
    convo = list(messages)
    if convo and convo[0].role == "system":
        convo[0] = ChatMessage(role="system", content=f"{convo[0].content}\n\n{instructions}")
    else:
        convo.insert(0, ChatMessage(role="system", content=instructions))

    last_error = "no attempt made"
    for attempt in range(repair_attempts + 1):
        response = await llm.complete(
            convo,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=schema,
        )
        try:
            return output_type.model_validate(extract_json_object(response.text))
        except (ValueError, ValidationError) as exc:
            last_error = _short_error(exc)
            log.info("structured output attempt %d failed: %s", attempt + 1, last_error)
            convo = [
                *convo,
                ChatMessage(role="assistant", content=response.text[:4000]),
                ChatMessage(role="user", content=_REPAIR_PROMPT.format(error=last_error)),
            ]
    raise ModelOutputError(
        f"{output_type.__name__}: model output invalid after "
        f"{repair_attempts + 1} attempts ({last_error})"
    )
