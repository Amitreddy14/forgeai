# ForgeAI — Tasks

Deadline: **Sunday 29 Nov 2026, 08:00 IST** (assignment says "Monday 29th"; 29 Nov is a
Sunday — planning for the earlier reading). Work one task at a time, top to bottom.
Legend: `[x]` done · `[ ]` open · `(S)` stretch.

## Phase 0 — Scaffold (branch `phase-0-scaffold`)
- [x] Six project docs
- [x] Package layout, settings, `forge` CLI (`version`, `doctor`)
- [x] LLM client protocol + `FakeLLM`
- [x] FastAPI app factory with `/healthz`
- [x] pytest suite, ruff, mypy strict
- [x] GitHub Actions CI, Dockerfile, `.env.example`

## Phase 1 — Foundations (branch `phase-1-foundations`)
- [x] OpenAI-compatible async client: retries, Retry-After, typed errors, JSON mode
- [x] Structured output: schema in prompt, JSON extraction, Pydantic validation, repair retry
- [x] Response cache (SQLite, temperature-0 only) + token budget attributed per stage
- [x] Ingest loaders: PDF, DOCX, CSV (row-as-record), MD, TXT → `Corpus` (skips reported)
- [x] Chunker (size/overlap, page + heading kept, chunks are verbatim document spans)
- [x] Hybrid retrieval: BM25 + fastembed (hash embedder offline), RRF, `min_score`
- [x] `contains_quote` / `normalize` / `extract_numbers`
- [x] `RunStore` (run folder, artifact write/read, traversal-safe names)
- [x] CLI: `forge ingest`, `forge search`, `forge smoke`; CI matrix adds Python 3.13

## Phase 2 — Requirements (branch `phase-2-requirements`)
- [ ] Corpus summary for the interviewer (titles, headings, CSV columns, sizes)
- [ ] Intake: ≤ 5 questions, CLI dialogue, `--answers auto` requester proxy (marked assumed)
- [ ] Requirements schema + Markdown render
- [ ] Groundability guard: retrieve → quote → verify → flag/move to must-refuse
- [ ] (S) Human checkpoint `--approve` (edit in $EDITOR / web form)

## Phase 3 — Agent (branch `phase-3-agent`)
- [ ] Tool registry: `csv_lookup`, `calculator` (AST), `weather` (open-meteo), `current_date`
- [ ] Tool-need contracts + check
- [ ] Agent generator → `AgentDefinition` v0 (prompt, chunking, retrieval, tools, contract)
- [ ] Runtime agent loop (plan → act → answer → check)
- [ ] `forge chat <definition>` CLI runner

## Phase 4 — Evals + judge (branch `phase-4-evals`)
- [ ] Eval generator: answerable / out-of-scope / ambiguous / two-document, ≥ 20, quotas
- [ ] Eval validation (quote verified, facts ⊂ quote, dedupe) + dev/holdout split
- [ ] Deterministic checks (refusal, citations, facts, numbers)
- [ ] LLM judge (different family, length-blind rubric, shuffle) + canaries

## Phase 5 — Optimizer + report (branch `phase-5-optimizer`)
- [ ] Failure classifier (signals first, LLM tiebreak)
- [ ] Critic → typed patches → apply → re-run dev → keep/revert
- [ ] Iteration loop (≤ 3) + changelog
- [ ] Scorecard + `report.md` (holdout headline, open failures, assumptions, judge reliability)
- [ ] (S) Budget early stop · (S) Prompt tournament

## Phase 6 — Demo runs (branch `phase-6-demos`)
- [ ] Synthesis prompts + HR docs (3 PDFs) + port-call docs (CSV + 2 PDFs)
- [ ] Run A end to end on Ollama, commit `runs/hr-shore-staff/`
- [ ] Run B end to end on Ollama, commit `runs/port-call-ops/`
- [ ] Dry run on a third unseen brief (not committed) to check generality

## Phase 7 — Web UI + hosting (branch `phase-7-web`)
- [ ] Builds list, new build (upload + brief), live survey log, certificate page
- [ ] Chat page for any definition
- [ ] Security headers, upload checks, rate limit
- [ ] Deploy to Hugging Face Spaces (Groq key as a Space secret; bake the bge-small model into the image)

## Phase 8 — Ship (branch `phase-8-ship`)
- [ ] README: quick start, architecture, results tables from the committed runs
- [ ] Demo video script (3–8 min) + recording
- [ ] Final check against PRD success criteria; submit
