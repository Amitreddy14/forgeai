# ForgeAI — Architecture

## 1. The pipeline

```
brief.md + docs/ ──► ingest ──► intake ──► guard ──► [checkpoint] ──► generate ──► evalgen
                                                                                     │
                     report ◄── optimize (≤3 iterations: run → judge → classify → patch → keep/revert)
```

Each stage is a pure-ish function: it reads typed artifacts, writes typed artifacts to
`runs/<run_id>/`, and never mutates an earlier artifact. A run can resume from the last
completed stage because every stage's output is on disk.

| Stage | Input | Output | Model? | Independent check |
|---|---|---|---|---|
| ingest | docs folder | `corpus.json` (docs, pages, chunks) | no | file type/size allowlist |
| intake | brief + corpus summary | `intake.json`, `requirements.json/.md` | generator | ≤ 5 questions enforced by schema; auto answers marked `assumed` |
| guard | requirements + corpus | `groundability.json` | generator | quote must be a verbatim substring of the cited document |
| checkpoint | requirements | approved requirements | human | optional (`--approve`) |
| generate | requirements + corpus stats + tool registry | `agent.v0.json` | generator | schema validation; tool-need contract per tool |
| evalgen | corpus + requirements | `evals.json` (dev / holdout) | generator | quote verified; facts must appear in quote; dedupe; category quotas |
| run | agent + evals | `iter_N/results.json` | generator (as the agent) | output contract validated |
| judge | results + evals | `iter_N/judgements.json` | judge (different family) | deterministic checks first; canaries measure the judge |
| optimize | judgements | `iter_N/patches.json`, `agent.vN.json` | generator (as critic) | patch kept only if dev score improves with no regressions |
| report | everything | `scorecard.json`, `report.md`, `changelog.md` | no | numbers computed from per-case results, never from the model |

## 2. Key design decisions

**D1 — Quote anchoring.** Any model claim of the form "the documents say X" must come
with a verbatim quote. Plain code normalises whitespace/case and checks the quote is a
substring of the named document. This is the backbone of the groundability guard, eval
validation, and the judge's grounding check.

**D2 — Dev/holdout split.** Eval cases are split 60/40, stratified by category. The
optimizer only ever sees dev failures. The headline score in the report is the holdout
score, so prompt tuning cannot overfit the number we report.

**D3 — Patches, not regeneration.** The critic emits typed patch operations against the
`AgentDefinition` (e.g. `set retrieval.top_k = 8`, `add_rule "Quote the clause number"`,
`set retrieval.min_score = 0.2`). Each patch names the failure category and case IDs it
targets. Patches are applied, the dev set is re-run, and the patch is reverted if dev
score drops or a previously passing case breaks.

**D4 — Layered judge.** Deterministic checks decide everything they can (refusal flag,
citation validity, numeric facts). The LLM judge only grades semantic correctness, sees
the expected facts (not a reference essay), is told length earns nothing, and comes from
a different model family than the generator. A few canary answers with known-wrong
content are mixed in every run; the judge's canary accuracy is reported.

**D5 — Tool-need contracts.** Each registry tool declares the signals that justify it
(e.g. `weather` needs a location + a forecast/weather topic in must-answer). The
generator proposes tools with a justification pointing to a requirement field; code
rejects any tool whose contract is unmet.

**D6 — One OpenAI-compatible client.** Ollama (`/v1`) and Groq both speak the OpenAI chat
API. One async `httpx` client handles JSON-schema output with validate-and-retry, a
response cache (SQLite, keyed by request hash), and a token budget.

**D7 — Files, not a database.** Run artifacts are JSON/Markdown on disk so they can be
committed as evidence (R9) and diffed in Git. The web app reads the same files.

## 3. Folder structure

```
forgeai/
├── PRD.md  Architecture.md  rules.md  design.md  task.md  memory.md
├── README.md  pyproject.toml  Dockerfile  .env.example
├── .github/workflows/ci.yml
├── src/forgeai/
│   ├── config.py            # Settings (env vars, FORGE_ prefix)
│   ├── cli.py               # `forge` command (Typer)
│   ├── llm/                 # client protocol, OpenAI-compatible client, fake, cache, budget
│   ├── ingest/              # loaders (pdf, docx, csv, md, txt) + chunker
│   ├── retrieval/           # BM25 + embeddings, reciprocal-rank fusion
│   ├── schemas/             # Pydantic models for every artifact
│   ├── stages/              # intake, guard, generate, evalgen, judge, optimize, report
│   ├── tools/               # registry + csv_lookup, calculator, weather, current_date
│   ├── runtime/             # agent loop used by evals, CLI chat and web chat
│   ├── prompts/             # every prompt as a versioned .md template (committed)
│   └── web/                 # FastAPI app, Jinja templates, static CSS
├── data/
│   ├── demos/<run>/         # brief.md, source/*.md, docs/ (PDF + CSV) for each demo
│   ├── examples/hr-mini/    # tiny example for quick real-model checks
│   └── synthesis/           # prompts used to synthesize the demo documents
├── scripts/                 # render_demo_docs.py (Markdown -> PDF)
├── runs/                    # committed artifacts of the two demo runs
└── tests/                   # pytest, all against the fake LLM
```

## 4. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | JD; best ecosystem for this |
| Web/API | FastAPI + Uvicorn, Jinja2 + HTMX | Async, typed; server-rendered UI is one deployable and easy to make distinctive |
| Schemas | Pydantic v2 | Every artifact validated; JSON Schema for structured model output |
| CLI | Typer | `forge build`, `forge chat`, `forge report` |
| HTTP | httpx (async) | LLM calls and the weather tool |
| Retrieval | rank-bm25 + fastembed (bge-small, ONNX CPU) fused with RRF | Hybrid recall without a GPU or vector DB |
| Parsing | pypdf, python-docx, stdlib csv | Small, pure-Python |
| Models | Ollama `qwen2.5:7b` (generator), `llama3.1:8b` (judge); Groq free tier when hosted | Small open models, two families |
| Quality | pytest, pytest-asyncio, ruff, mypy (strict) | CI gates |
| Deploy | Docker → Hugging Face Spaces (free) | Free, Docker-native |

## 5. How the components connect

```
            ┌─────────── cli.py ───────────┐      ┌──────── web/app.py ────────┐
            │ forge build / chat / report  │      │ pages + HTMX partials       │
            └──────────────┬───────────────┘      └──────────────┬─────────────┘
                           └───────────► pipeline.py ◄───────────┘
                                             │ calls stages in order, writes runs/<id>/
          ┌──────────┬──────────┬────────────┼────────────┬──────────┬───────────┐
        ingest    intake      guard       generate      evalgen    optimize    report
          │          │          │            │             │          │
          ▼          └──────────┴───── llm/client ─────────┴──────────┘
     retrieval                          (cache, budget, JSON repair)
          ▲
     runtime/agent  ◄── tools/registry   (used by run, judge-time replay, chat)
```

## 6. Runtime agent loop (what a produced agent does)

1. **Plan** — model reads the question + tool list, returns `{needs_retrieval, tool_calls[]}`.
2. **Act** — retrieve top-k chunks; execute tool calls (validated args, timeouts).
3. **Answer** — model answers in the output contract: `{answer, citations[{doc, chunk_id, quote}], refused, refusal_reason}`.
4. **Check** — code validates the contract, verifies each quote is in the cited chunk, and
   refuses if retrieval confidence is below `min_score` and no tool produced evidence.
   One repair attempt on contract failure.

## 7. Configuration

All settings via environment variables with the `FORGE_` prefix (see `.env.example`):
`FORGE_LLM_BASE_URL`, `FORGE_LLM_API_KEY`, `FORGE_GENERATOR_MODEL`, `FORGE_JUDGE_MODEL`,
`FORGE_RUNS_DIR`, `FORGE_CACHE_PATH`, `FORGE_MAX_TOKENS_PER_RUN`, `FORGE_MAX_ITERATIONS`,
`FORGE_EMBEDDER` (`fastembed` or `hash`).
