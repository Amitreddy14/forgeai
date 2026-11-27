"""The tool registry, and the contracts that decide whether a brief really needs a tool.

The model proposes tools. Code then checks each proposal against two pieces of
evidence it cannot fake:
  1. the requirements contain a topic with that tool's capability, and
  2. the *human's own words* (the brief and the requester's answers) mention it, or
     for table lookup, the folder actually contains a CSV table.
An HR agent with a weather tool would need the requester to have asked about weather.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from forgeai.schemas.corpus import Corpus
from forgeai.schemas.requirements import Requirements, ToolHint
from forgeai.tools.base import Tool
from forgeai.tools.calculator import Calculator
from forgeai.tools.csv_lookup import CsvLookup
from forgeai.tools.current_date import CurrentDate
from forgeai.tools.weather import Weather

REGISTRY: dict[str, Tool] = {
    t.name: t for t in (CsvLookup(), Calculator(), Weather(), CurrentDate())
}
BY_CAPABILITY: dict[ToolHint, str] = {t.capability: t.name for t in REGISTRY.values()}

# Words a requester would use when they need the capability. Matched on whole words.
EVIDENCE: dict[ToolHint, tuple[str, ...]] = {
    "weather": (
        "weather",
        "forecast",
        "wind",
        "winds",
        "rain",
        "storm",
        "swell",
        "waves",
        "sea state",
        "monsoon",
        "cyclone",
        "visibility",
        "gusts",
    ),
    "current_date": (
        "today",
        "todays",
        "today's",
        "deadline",
        "deadlines",
        "days until",
        "days left",
        "expiry",
        "expires",
        "due date",
        "this week",
        "this month",
        "how long until",
    ),
    "arithmetic": (
        "calculate",
        "calculation",
        "compute",
        "estimate",
        "estimates",
        "total",
        "totals",
        "cost",
        "costs",
        "sum",
        "how much",
        "convert",
        "charges",
        "dues",
        "fee",
        "fees",
        "price",
        "prices",
        "rate",
        "rates",
        "amount",
    ),
    "table_lookup": (),  # evidence is the CSV itself
}


class ToolProposal(BaseModel):
    name: str
    justification: str = ""


class ToolChoice(BaseModel):
    name: str
    capability: ToolHint
    justification: str
    evidence: str


class RejectedTool(BaseModel):
    name: str
    reason: str


class ToolSelection(BaseModel):
    attached: list[ToolChoice]
    rejected: list[RejectedTool]
    added: list[str]  # required by the contract but not proposed by the model


def human_words(req: Requirements, requester_answers: list[str]) -> str:
    return " ".join([req.brief, *requester_answers]).lower()


def _evidence_for(capability: ToolHint, text: str, corpus: Corpus) -> str | None:
    if capability == "table_lookup":
        tables = [d.doc_id for d in corpus.documents if d.kind == "csv"]
        return f"CSV table(s) in the folder: {', '.join(tables)}" if tables else None
    for phrase in EVIDENCE[capability]:
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return f'requester wrote "{phrase}"'
    return None


def select_tools(
    proposals: list[ToolProposal],
    req: Requirements,
    corpus: Corpus,
    requester_answers: list[str],
) -> ToolSelection:
    text = human_words(req, requester_answers)
    needed: dict[ToolHint, str] = {}
    for hint in req.tool_hints:
        capability: ToolHint = hint  # type: ignore[assignment]
        evidence = _evidence_for(capability, text, corpus)
        if evidence is not None:
            needed[capability] = evidence

    attached: list[ToolChoice] = []
    rejected: list[RejectedTool] = []
    seen: set[str] = set()
    for proposal in proposals:
        name = proposal.name.strip()
        if name in seen:
            continue
        seen.add(name)
        tool = REGISTRY.get(name)
        if tool is None:
            rejected.append(RejectedTool(name=name, reason="not in the tool registry"))
            continue
        if tool.capability not in req.tool_hints:
            rejected.append(
                RejectedTool(name=name, reason="no requirement topic needs this capability")
            )
            continue
        if tool.capability not in needed:
            rejected.append(
                RejectedTool(
                    name=name,
                    reason="the requester's own words never ask for it"
                    if tool.capability != "table_lookup"
                    else "the folder contains no CSV table",
                )
            )
            continue
        attached.append(
            ToolChoice(
                name=name,
                capability=tool.capability,
                justification=proposal.justification or "proposed by generator",
                evidence=needed[tool.capability],
            )
        )

    added: list[str] = []
    for capability, evidence in needed.items():
        name = BY_CAPABILITY[capability]
        if name not in {a.name for a in attached}:
            attached.append(
                ToolChoice(
                    name=name,
                    capability=capability,
                    justification="required by a requirement topic; the generator omitted it",
                    evidence=evidence,
                )
            )
            added.append(name)
    attached.sort(key=lambda c: c.name)
    return ToolSelection(attached=attached, rejected=rejected, added=added)


# ---------- Inferring capabilities from requirement topics (deterministic) -------------

_TOPIC_WORDS: dict[ToolHint, tuple[str, ...]] = {
    "weather": EVIDENCE["weather"],
    "current_date": EVIDENCE["current_date"],
    "arithmetic": (
        "estimate",
        "estimates",
        "estimating",
        "estimation",
        "calculate",
        "calculation",
        "compute",
        "total",
        "totals",
        "cost",
        "costs",
        "charges",
        "fees",
        "dues",
        "how much",
    ),
}
_TABLE_NOISE = frozenset({"id", "m", "t", "usd", "inr", "eur", "max", "min", "type", "no", "of"})


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in re.split(r"[^a-z0-9]+", text.lower()) if w} - _TABLE_NOISE


def _needs_table(topic_text: str, corpus: Corpus) -> bool:
    """A topic needs a table if it names the table itself (`berths`) or shares at least
    two words with its columns (`length`, `draft`). One generic shared word such as
    `port` is not enough."""
    words = _words(topic_text)
    for doc in corpus.documents:
        if doc.kind != "csv":
            continue
        if words & _words(doc.doc_id):
            return True
        columns: set[str] = set()
        for column in doc.columns or []:
            columns |= _words(column)
        if len(words & columns) >= 2:
            return True
    return False


def infer_tool_hints(text: str, corpus: Corpus) -> list[ToolHint]:
    """Capabilities a requirement topic needs, read from its own words and the folder.

    The 7B model often marks 'check the weather forecast' as a documents topic. Code
    reads the topic text instead of trusting that label. (Attaching a tool still needs
    the requester's own words: see `select_tools`.)
    """
    lowered = text.lower()
    hints: list[ToolHint] = []
    for capability, words in _TOPIC_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in words):
            hints.append(capability)
    if _needs_table(text, corpus):
        hints.append("table_lookup")
    return hints
