# ForgeAI — Memory (running log)

Newest first. Decisions, bugs, changes, and anything the next session must know.

## 2026-11-27 — Phase 5

- One canary set per run (dev cases), reused by every evaluation so judge trust is stable.
- The optimizer sees dev only; holdout runs twice at the end (v0 and final) for the
  before/after headline. Wilson 95% interval shown because holdout is small (~8-10 cases).
- Candidate order per iteration: categories by failure count; within a category,
  mechanical patches first, then one critic rule. Up to 3 candidates per iteration.
- Accept rule: dev passed count strictly up AND no previously passing case breaks.
- Critic rules are rejected if they contain an expected fact or a number (≥10) from a
  failing case's expected answer; a safe default rule is used instead.
- Every evaluation is saved as evals/dev-runNN-vK.json (also reverted candidates).

## 2026-11-27 — Phase 4.1 (fixes from the first real eval run)

First real `forge evaluate` on hr-mini: dev 46%, holdout 40%, judge canaries 62% (judge
distrusted). Reading the cases showed most failures were the factory's own defects:
- Out-of-scope generator wrote in-scope questions (leave entitlements). Fix: every
  out-of-scope candidate goes through the groundability check; if the documents answer
  it (verified quote), it is rejected as "answerable from the documents".
- Key facts were whole sentences or rules the question never asked ("receipts above INR
  2,000" on a meal-allowance question). Fix: facts ≤ 10 words; the first fact is the
  answer, later facts must share ≥ 2 content words with the question; prompt example.
- Questions like "What does this policy explain?" depend on unseen context. Fix: rejected.
- Multi-doc pairs were random (leave + flights). Fix: pair by embedding similarity.
- llama3.1 judge passed a confident omission and a wrong number (6 h -> 19 h). Fix: the
  judge must quote the answer as evidence; code checks the quote exists in the answer
  and carries the fact's numbers, else downgrades to missing/contradicted.
This is the report's headline story: the canaries caught the judge, the guard caught
the eval generator.

## 2026-11-27 — Phase 4

- Real run of Phase 3 on hr-mini: generator proposed csv_lookup + calculator for an HR
  agent; both rejected by contract (no tools attached). Chat answers verified; salary and
  weather questions refused by the calibrated threshold (0.65) without a model call.
  Hotel question scored 0.70, close to the threshold: a likely wrong-refusal risk.
- Rich markup swallowed `[chunk-id]` in chat output; fixed with `rich.markup.escape`.
- Expected sources are anchored by quote text, not chunk id, because the optimizer may
  re-chunk; "source retrieved" = some retrieved chunk contains the quote's first 10 words.
- Unsupported-number check ignores integers below 10 (list numbering, small counts).
- Canary judge test: a judge that always says "present" scores 50% and is distrusted.
- Evals use a fixed reference chunking (default config), independent of the agent.
- Ruff: formatter wraps at 100; lint tolerates 120 for lines it cannot split.
- Small corpora may yield < 20 cases: `forge evalgen --min-cases N` for examples.

## 2026-11-27 — Phase 3

- Real-model check (hr-mini, qwen2.5:7b): intake + guard ran end to end and flagged the
  planted "salary band" gap. Warm qwen on the 4060: ~85 tok/s, 100% GPU.
- `OLLAMA_CONTEXT_LENGTH` env var was not picked up (ollama ps showed 4096), so named
  variants with `PARAMETER num_ctx 8192` are used: `qwen2.5-8k`, `llama3.1-8k`.
- Tool contract = capability in requirements AND evidence in the human's own words
  (brief + requester answers, not proxy answers) — or a CSV for table lookup. Evidence
  word lists avoid generic words ("current", "date", "now") that HR briefs use.
- Code chooses chunking/top-k/threshold; the 7B model only writes prose (role, rules,
  refusal line). Threshold is off when must-answer and must-refuse topics overlap.
- Runtime makes 1 model call per question when no tools are attached (no plan step).
- Calculator strips only thousands separators (a plain comma strip broke `round(x, 2)`).
- Windows needs the `tzdata` package for zoneinfo.

## 2026-11-27 — Phase 2

- `forge smoke` on Amit's laptop: both models return valid JSON, but 55 s (qwen) / 26 s
  (llama) for ~115 tokens — cold load and/or model swapping (both don't fit in 8 GB
  together). Mitigations: group calls by model per stage, short prompts, deterministic
  checks before the judge. Asked for `ollama run --verbose` + `ollama ps` numbers.
- Ollama's default context window may truncate long prompts silently; set
  `OLLAMA_CONTEXT_LENGTH=8192` before serving.
- Every prompt starts with a `TASK: <name>` line: makes logs readable and lets the
  scripted test model route on it.
- Answer sources: `requester` (human, including accepting a shown default), `brief`
  (proxy inferred it), `default` (nobody answered). Non-requester answers become
  assumptions in requirements and the final report.
- Guard trusts quote words, not chunk ids: a wrong chunk id with a real quote is corrected;
  a real chunk id with a fabricated quote is rejected.
- Tool topics are not checked against prose; `table_lookup` needs a CSV in the folder.
- If every topic fails the guard, the run stops with GroundingError (no agent generated).

## 2026-11-27 — Phase 1

- Amit's venv is Python 3.13 (no 3.11 on his laptop); CI matrix now 3.11/3.12/3.13.
- JSON mode uses `response_format: json_object` + schema in the system prompt, because
  json_schema support differs across Ollama/Groq models. Validation is always ours.
- Cache only stores temperature-0 calls; cache hits do not count against the budget.
- Chunks keep heading lines inside their text so every chunk is a verbatim span of the
  document; a title line shorter than 8 words is folded into the next section.
- CSV pages are "Row N: col: val; ..." so row citations are quotable.
- `contains_quote` requires every `...`-fragment to have ≥ 3 words (blocks "proof" by
  one-word quotes) and matches on word boundaries (USD 60 ≠ USD 600).
- Embedder: `FORGE_EMBEDDER=fastembed` (default, bge-small, ~130 MB download once) or
  `hash` (tests/CI, offline). Real-embedding test is opt-in: `FORGE_TEST_FASTEMBED=1`.

## 2026-11-27 — Phase 0

**Context.** OceanAI Jr. GenAI Engineer assignment (elimination round). Build ForgeAI,
the agent that builds agents. Deliverables: GitHub repo, 3–8 min demo video, hosted
link. Deadline Sunday 29 Nov 2026 08:00 IST (the PDF says "Monday 29th").

**Decisions locked.**
- Models: Ollama locally — `qwen2.5:7b` generator, `llama3.1:8b` judge (different
  families, against self-preference). Groq free tier for the hosted app. One
  OpenAI-compatible client for both.
- UI: FastAPI + Jinja + HTMX, hand-written CSS; "survey certificate" concept with a
  load-line score (see design.md). Hosted on Hugging Face Spaces (Docker).
- Demos: Run A `hr-shore-staff` (3 HR PDFs, no tools) and Run B `port-call-ops`
  (berth/tariff CSV + 2 port-notice PDFs; CSV lookup, calculator, weather).
- Differentiators: quote anchoring checked in code, dev/holdout split, canary cases to
  measure the judge, typed patches with keep/revert, tool-need contracts.
- No LangChain/LlamaIndex; the loop is owned end to end.
- Artifacts are files under `runs/`, not a database (they are committed evidence).

**Environment.** Amit's laptop: Windows, VS Code, RTX 4060 8 GB, Ollama installed.
Python 3.11+ required locally. Groq key and HF account still to create.

**Phase 0 delivered.** Docs, package skeleton, `forge version|doctor`, `FakeLLM`,
`/healthz`, CI (ruff, mypy strict, pytest on 3.11 + 3.12), Dockerfile.
