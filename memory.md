# ForgeAI — Memory (running log)

Newest first. Decisions, bugs, changes, and anything the next session must know.

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
