"""A scripted 'model' for pipeline tests: routes on the TASK line each prompt starts with,
and can be told to lie (fake quotes, too many questions) on purpose."""

from __future__ import annotations

import json
import re

from forgeai.llm import ChatMessage

QUESTIONS = {
    "questions": [
        {
            "id": f"x{i}",
            "question": f"Question number {i}?",
            "why": "changes scope",
            "default_answer": f"default {i}",
        }
        for i in range(1, 8)  # 7: the factory must cap this at 5
    ]
}

REQUIREMENTS = {
    "agent_name": "Shore Staff HR Desk",
    "purpose": "Answer HR policy questions for shore staff.",
    "scope": "Leave and travel policies only.",
    "audience": "Shore-based employees",
    "tone": "Clear and polite",
    "must_answer": [
        {"topic": "Annual leave", "detail": "days per year, carry forward"},
        {"topic": "annual leave", "detail": "duplicate that must be removed"},
        {"topic": "Sick leave", "detail": "certificate rules"},
        {"topic": "Salary bands", "detail": "pay scales by grade"},
        {"topic": "Berth lookup", "answered_from": "tool", "tool_hint": "table_lookup"},
        {"topic": "Odd hint", "answered_from": "tool"},
    ],
    "must_refuse": [
        {"topic": "Legal advice", "reason": "not qualified"},
        {"topic": "Other employees' records", "reason": "confidential"},
    ],
}

# What the guard "model" claims per topic. Sick leave gets a fabricated quote.
GUARD = {
    "Annual leave": {
        "status": "supported",
        "chunk_id": "wrong-id",
        "quote": "Shore staff earn 24 days of annual leave per calendar year",
    },
    "Sick leave": {
        "status": "supported",
        "chunk_id": "leave-policy:p2:c0",
        "quote": "Employees receive 30 days of paid sick leave per year",
    },
    "Salary bands": {"status": "unsupported", "note": "no pay data"},
    "Odd hint": {"status": "unsupported", "note": "nothing"},
}


def _task(messages: list[ChatMessage]) -> str:
    match = re.search(r"TASK: ([\w.]+)", messages[-1].content)
    return match.group(1) if match else ""


def responder(messages: list[ChatMessage], guard: dict[str, dict[str, str]] = GUARD) -> str:
    task = _task(messages)
    prompt = messages[-1].content
    if task == "intake.questions":
        return json.dumps(QUESTIONS)
    if task == "intake.auto_answer":
        return json.dumps(
            {
                "answers": [
                    {"question_id": "q1", "answer": "Shore staff only", "from_brief": True},
                    {"question_id": "q2", "answer": "default 2", "from_brief": False},
                    # q3..q5 missing on purpose: the factory must fall back to defaults
                ]
            }
        )
    if task == "intake.requirements":
        return "Here you go:\n```json\n" + json.dumps(REQUIREMENTS) + "\n```"
    if task == "guard.topic":
        topic = re.search(r"TOPIC: (.+)", prompt)
        name = topic.group(1).strip() if topic else ""
        return json.dumps(guard.get(name, {"status": "unsupported"}))
    raise AssertionError(f"unscripted task {task!r}")
