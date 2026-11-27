"""A crude simulated model for eval/judge tests. It reads the prompt like a model would
(passages, facts, questions) and answers by copying sentences, so the factory's
verification code has real text to check. It also lies in a few known places."""

from __future__ import annotations

import json
import re

from forgeai.llm import ChatMessage
from forgeai.textutil import normalize
from tests.scripted import _task, generator_responder

_SENT = re.compile(r"(?<=[.!?])\s+")
_FACT = re.compile(r"(?:INR |USD )?\d[\d,.]*\s+\w+(?:\s+\w+)?")
STOP = {
    "the",
    "a",
    "of",
    "for",
    "is",
    "what",
    "how",
    "many",
    "much",
    "to",
    "in",
    "and",
    "do",
    "i",
    "get",
}


def _section(prompt: str, start: str, end: str | None = None) -> str:
    body = prompt.split(start, 1)[1]
    return body.split(end, 1)[0].strip() if end and end in body else body.strip()


def _numeric_sentence(passage: str) -> str | None:
    for sentence in _SENT.split(passage.replace("\n", " ")):
        if _FACT.search(sentence):
            return sentence.strip()
    return None


def _question_from(sentence: str, fact: str) -> str:
    return sentence.replace(fact, "how much").rstrip(".") + "?"


def write_case(prompt: str, casual: bool) -> str:
    passage = _section(prompt, "PASSAGE [", "\n\nRules:").split("\n", 1)[1]
    sentence = _numeric_sentence(passage)
    if sentence is None:
        sentence = _SENT.split(passage.replace("\n", " "))[-1].strip()
        fact = " ".join(sentence.split()[:3])
    else:
        fact = _FACT.search(sentence).group().strip()  # type: ignore[union-attr]
    if "sick leave" in sentence:  # planted lie: a fact the passage does not contain
        fact = "30 days of paid sick leave"
    question = _question_from(sentence, fact)
    if casual:
        question = "hey quick one - " + question.lower()
    return json.dumps({"question": question, "key_facts": [fact], "quote": sentence})


def write_pair(prompt: str) -> str:
    a = _section(prompt, "PASSAGE A [", "\n\nPASSAGE B").split("\n", 1)[1]
    b = _section(prompt, "PASSAGE B [", "\n\nWrite ONE").split("\n", 1)[1]
    sa, sb = _numeric_sentence(a), _numeric_sentence(b)
    if not sa or not sb:
        return json.dumps({"possible": False})
    fa, fb = _FACT.search(sa).group().strip(), _FACT.search(sb).group().strip()  # type: ignore[union-attr]
    question = f"Compare these two: {_question_from(sa, fa)} And {_question_from(sb, fb)}"
    return json.dumps(
        {
            "possible": True,
            "question": question,
            "fact_a": fa,
            "quote_a": sa,
            "fact_b": fb,
            "quote_b": sb,
        }
    )


REFUSALS = [
    "Can you give me legal advice on my employment contract?",
    "What is my colleague Ravi's home address?",
    "How much does the CEO earn per year?",
    "What will next year's leave policy be?",
    "Do seafarers on board get the same parental leave?",
    "Can you book my flight to Dubai for me?",
]


def answer_question(prompt: str) -> str:
    question = _section(prompt, "QUESTION:", "PASSAGES:")
    passages = _section(prompt, "PASSAGES:", "TOOL RESULTS:")
    q_words = {w for w in normalize(question).split() if w not in STOP}
    best: tuple[int, str, str] = (0, "", "")
    for block in passages.split("\n\n["):
        header, _, text = block.lstrip("[").partition("\n")
        chunk_id = header.split("]", 1)[0]
        for sentence in _SENT.split(text.replace("\n", " ")):
            overlap = len(q_words & set(normalize(sentence).split()))
            if overlap > best[0]:
                best = (overlap, chunk_id, sentence.strip())
    if best[0] < 3:
        return json.dumps(
            {
                "answer": "I can't help with that.",
                "refused": True,
                "refusal_reason": "not in the documents",
                "citations": [],
            }
        )
    return json.dumps(
        {
            "answer": best[2],
            "refused": False,
            "citations": [{"chunk_id": best[1], "quote": best[2]}],
        }
    )


def judge(prompt: str) -> str:
    facts = re.findall(
        r"^\d+\. (.+)$", _section(prompt, "REQUIRED FACTS:", "ANSWER TO GRADE:"), re.M
    )
    answer = normalize(_section(prompt, "ANSWER TO GRADE:", "For each required fact"))
    return json.dumps(
        {
            "facts": [
                {"fact": f, "status": "present", "evidence": f}
                if normalize(f) in answer
                else {"fact": f, "status": "missing", "evidence": ""}
                for f in facts
            ]
        }
    )


def respond(messages: list[ChatMessage]) -> str:
    task = _task(messages)
    prompt = messages[-1].content
    if task == "evals.answerable":
        return write_case(prompt, casual=False)
    if task == "evals.ambiguous":
        return write_case(prompt, casual=True)
    if task == "evals.multi_doc":
        return write_pair(prompt)
    if task == "evals.out_of_scope":
        return json.dumps(
            {"questions": [{"question": q, "topic": "x", "why": "y"} for q in REFUSALS]}
        )
    if task == "agent.answer":
        return answer_question(prompt)
    if task == "agent.plan":
        return json.dumps({"calls": []})
    if task == "judge.facts":
        return judge(prompt)
    if task == "generate.variants":
        return json.dumps(
            {
                "prompts": [
                    "You are a precise HR assistant for shore staff. Quote exact figures and name the "
                    "policy section they come from. Answer only from the policy documents.",
                    "You are a friendly HR helper for shore staff. Answer the need in plain words "
                    "first, then give the exact policy details from the documents.",
                ]
            }
        )
    if task == "critic.rule":
        return json.dumps(
            {
                "rule": "Answer the exact question first, using the document's own wording.",
                "rationale": "answers drifted from the asked fact",
            }
        )
    return generator_responder(messages)
