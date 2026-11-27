# ForgeAI — Product Requirements

> The agent that builds agents, tests them like a QA engineer, and ships them with evidence.

## 1. Problem

OceanAI's platform lets admins configure agents by hand: a system prompt, knowledge
bases, tools. Doing that well is a GenAI engineer's job — write the prompt, pick the
retrieval settings, write test questions, read the failures, fix the right thing,
re-test. The next step is an admin who writes **one paragraph** and drops in a folder
of documents.

ForgeAI automates that engineer's loop. The hard part is not generating an agent; it is
**trusting** one. Every stage runs on a small, imperfect open model, so every stage can
lie: the requirements can promise topics the documents never cover, the eval set can
contain questions whose "expected answer" is not in any document, the judge can reward
long answers, and the optimizer can overfit the tests it sees. ForgeAI's design goal is
to catch each of those lies with a check that does not depend on the model being honest.

## 2. Users

| User | What they do | What they need |
|---|---|---|
| Requester (platform admin) | Writes a brief, answers ≤ 5 questions, optionally approves the requirements | A working agent and a plain report of what it can and cannot do |
| Reviewer (OceanAI engineer) | Runs ForgeAI unattended on an unseen brief + folder | One command, no babysitting, honest numbers, every artifact on disk |
| End user of the produced agent | Chats with the generated agent | Grounded answers with citations, clean refusals outside scope |

## 3. Core requirements (all mandatory — from the assignment)

| # | Requirement | ForgeAI feature |
|---|---|---|
| R1 | Intake interviewer: read brief, inspect folder, ≤ 5 clarifying questions, structured requirements (scope, audience, tone, must-answer, must-refuse) | `intake` stage; CLI dialogue, web form, or `--answers auto` (a requester proxy answers from the brief and every auto answer is marked *assumed*) |
| R2 | Groundability guard: flag brief asks the documents cannot support | `guard` stage: each must-answer topic needs a retrieved passage **and** a verbatim quote that code verifies exists in the document; unsupported topics are flagged and moved to must-refuse, never silently kept |
| R3 | Agent generator: system prompt, chunking + retrieval params, tool selection (only what the brief needs), output contract | `generate` stage → `AgentDefinition` (versioned JSON). Each tool must cite the requirement that needs it; a deterministic tool-need check rejects unjustified tools |
| R4 | Eval generator: ≥ 20 generated cases — answerable (source doc + key facts), out-of-scope refusals, edge cases (ambiguous, two-document) | `evalgen` stage; every answerable case carries a verbatim source quote, and its key facts must appear in that quote or the case is discarded. Generation prompts committed |
| R5 | Judge: grounded? correct vs expected facts? refused when it should? defend vs position / self-preference / verbosity bias | Layered judge: deterministic checks first (citations exist, cited doc matches, facts/numbers present, refusal flag), then an LLM judge from a **different model family** with a length-blind rubric and order shuffling, then **canary cases** (planted bad answers) to measure the judge itself |
| R6 | Critic–optimizer: classify failures (retrieval miss / prompt defect / hallucination / wrong refusal) → targeted fix → re-run; ≤ 3 iterations; log everything | Failures classified by deterministic signals first; fixes are typed **patches** on the definition; a patch is kept only if the dev score rises without breaking passing cases, else reverted. Changelog records what, why, which cases, score before/after |
| R7 | Honest reporting: definition, per-iteration scorecard, what still fails and why | `report.md` + `scorecard.json`; headline score is on the **holdout** split the optimizer never saw; lists assumptions, flagged gaps, judge reliability, open failures |
| R8 | Runnable product: generic chat runner that loads any definition | `forge chat runs/<run>/agent.json` (CLI) and the web chat page |
| R9 | Two full demonstrations, all artifacts committed | `runs/hr-shore-staff/` and `runs/port-call-ops/` |

## 4. Stretch goals (after R1–R9 work)

In priority order:
1. **Human checkpoint** — requester approves or edits requirements before generation (`--approve`).
2. **Budget enforcement** — per-run token/latency budget; early stop when an iteration gains less than a threshold.
3. **Prompt tournament** — 2–3 candidate system prompts, eval set picks the winner, comparison in the report.
4. **UI and hosting** — web UI for factory + chat runner, hosted on Hugging Face Spaces.
5. **Regression mode** — documents change → re-run the old eval set against the regenerated agent, report what broke.
6. **Factory vs human baseline** — hand-written HR agent scored on the same evals.

## 5. Demo runs (deliberately different)

| | Run A: `hr-shore-staff` | Run B: `port-call-ops` |
|---|---|---|
| Brief | Answer HR policy questions for shore staff | Help a port agent plan a vessel call: berth fit, dues estimate, weather window |
| Documents | 3 HR policy PDFs (leave, travel & expenses, code of conduct) | Berth & tariff CSV + 2 port-notice PDFs |
| Expected tools | **None** (proves the factory does not over-attach) | `csv_lookup`, `calculator`, `weather`; `current_date` only if the brief needs it |
| Stresses | Refusals (legal advice, salary of others), two-doc answers | Numeric correctness, tool use, out-of-scope ports |

Documents are LLM-synthesized; synthesis prompts are committed under `data/synthesis/`.

## 6. Non-functional requirements

- **Unattended:** `forge build --brief … --docs … --answers auto` finishes with no input.
- **Model-agnostic:** any OpenAI-compatible endpoint (Ollama locally, Groq hosted). Generator and judge models configured separately.
- **Reproducible:** LLM responses cached by request hash; temperature 0 for checks; every artifact written to `runs/<run_id>/`.
- **Budgeted:** hard caps on LLM calls and tokens per run; the run stops cleanly and reports why.
- **Tested:** CI runs lint, type check and tests against a fake LLM — no network, no API key.
- **Secure:** no secrets in code or artifacts; calculator never uses `eval`; HTTP tool restricted to an allowlisted host; uploaded files type- and size-checked.

## 7. Out of scope

Multi-tenant auth, a database server, fine-tuning, document formats beyond PDF / DOCX / CSV / Markdown / TXT, non-English documents.

## 8. Success criteria

1. On an unseen brief + folder, a single command produces every artifact with no crash.
2. The HR agent has zero tools; the port-call agent has exactly the tools its brief needs.
3. ≥ 20 eval cases per run, each answerable case with a verified source quote.
4. The report's holdout score and open failures match the per-case results on disk.
5. CI green on `main`; hosted app reachable.
