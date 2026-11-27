# ForgeAI — Memory (running log)

Newest first. Decisions, bugs, changes, and anything the next session must know.

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
