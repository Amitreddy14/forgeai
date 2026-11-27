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
- [x] Corpus summary for the interviewer (titles, sections, CSV columns + sample rows, skips)
- [x] Intake: ≤ 5 questions (cap enforced in code), interactive / `--answers auto` / answers file
- [x] Auto answers labelled by source: requester, inferred from brief, interviewer default
- [x] Requirements schema + Markdown render (must-answer with source, must-refuse, gaps, assumptions)
- [x] Groundability guard: retrieve → model claims → quote verified in code → flag + move to must-refuse
- [x] Table-lookup topics need a CSV in the folder; nothing grounded → stop, no agent generated
- [x] (S) Human checkpoint `--approve` (approve, or edit JSON and re-validate)
- [x] Prompts as versioned `.md` templates; `data/examples/hr-mini` for quick real-model runs

## Phase 3 — Agent (branch `phase-3-agent`)
- [x] Tool registry: `csv_lookup` (typed filters), `calculator` (AST, no eval), `weather`
      (open-meteo + marine, fixed hosts), `current_date` (injected clock, tzdata)
- [x] Tool-need contracts: requirement capability AND the human's own words (or a CSV) —
      unjustified proposals rejected, required-but-omitted tools added, all logged
- [x] Agent generator → `AgentDefinition` v0: model writes role prompt / rules / refusal line;
      code sets chunk size (median section), top-k (corpus shape), calibrated refusal threshold
- [x] Runtime loop: retrieve → plan/act (≤3 steps, only if tools) → answer → verify citations
      → one repair → grounded flag; threshold refusal skips the model
- [x] `forge generate`, `forge chat` (any definition; warns when documents changed)
- [x] Ollama Modelfiles pinning an 8k context (`ollama/`)

## Phase 4 — Evals + judge (branch `phase-4-evals`)
- [x] Eval generator: answerable, ambiguous, multi-doc (LLM, from selected passages),
      out-of-scope incl. near misses (LLM), table/arithmetic/weather/date cases (code templates)
- [x] Validation: quote is a verbatim span, every key fact inside the quote, no answer leak,
      near-duplicate removal; rejected candidates logged with reasons; ≥ 20 enforced
- [x] Stratified, deterministic dev/holdout split (60/40)
- [x] Layer 1 deterministic judge: refusal, verified citations, expected source retrieved/cited,
      string/number fact match, numbers absent from all evidence, required tool called
- [x] Layer 2 LLM fact judge: other model family, facts not reference essays, length-blind
- [x] Layer 3 canaries: wrong number, confident omission, padded correct, terse correct;
      judge distrusted below 75% canary accuracy
- [x] Failure classifier (wrong refusal > retrieval miss > hallucination > prompt defect)
- [x] `forge evalgen`, `forge evaluate`; agent calls first, judge calls after (one model swap)
- [x] 4.1 fixes from the first real run: out-of-scope answerability check, fact hygiene,
      context-dependent question filter, similarity-based pairing, judge evidence check

## Phase 5 — Optimizer + report (branch `phase-5-optimizer`)
- [x] Typed patches (top_k, chunk size, retrieval mode, threshold, add rule, add refuse topic),
      each naming its failure category and target cases; versioned definitions
- [x] Critic: mechanical fixes computed from evidence first (retrieval, threshold), model-written
      rules second; rules that contain an expected answer are rejected (anti-overfitting)
- [x] Loop ≤ 3 iterations, ≤ 3 candidates each: keep only if dev rises with zero regressions
- [x] Change log: what, why, targets, fixed/broken cases, dev before/after, tokens
- [x] Report: holdout headline with Wilson 95% interval, before vs after, per-iteration dev
      scorecard, open failures by category with reasons, judge trust, gaps, assumptions, cost
- [x] (S) Budget early stop: stop when the remaining budget cannot cover another evaluation,
      or when an iteration finds no improving patch
- [x] (S) Prompt tournament (`--tournament`)
- [x] `forge build`: the whole factory unattended, with `--resume`

## Phase 6 — Demo runs (branch `phase-6-demos`)
- [x] Synthesis prompts (`data/synthesis/`) + Markdown sources + deterministic PDF renderer
- [x] hr-shore-staff: 3 policy PDFs, 27 sections; brief plants an ungroundable ask (leave balance)
- [x] port-call-ops: 14-row berth CSV + tariff + notices; brief needs lookup, arithmetic, weather
- [x] Fix found on the demo PDFs: headings at a page foot now stay with their text
- [x] Sharper out-of-scope answerability check (own prompt; personal data, other groups,
      future changes and actions are not "answered" by general policy text)
- [x] 6.1 fixes from the first demo runs: tool hints inferred in code, personal-records split,
      similarity backstop for out-of-scope labels, composed two-document cases,
      refuse-topic critic patch
- [ ] Run A end to end on Ollama, commit `runs/hr-shore-staff/`
- [ ] Run B end to end on Ollama, commit `runs/port-call-ops/`
- [ ] Dry run on a third unseen brief (not committed) to check generality

## Phase 7 — Web UI + hosting (branch `phase-7-web`)
- [x] Builds register, certificate page (survey log, load line, conditions, particulars,
      optimizer log, full report, all artifacts), live progress with HTMX polling
- [x] Chat page for any build (verified quotes, tool output, refusals shown as such)
- [x] New build form: upload checks (types, size, count, filename sanitising), one build
      at a time, per-client rate limits, failures written to stage_error.json
- [x] Security headers incl. CSP (no inline scripts/styles; htmx vendored), docs off in prod
- [x] Fix found while building the UI: RunStore wrote lists of models as Python reprs
- [x] Hugging Face Space: `deploy/huggingface/` (Dockerfile clones the GitHub repo,
      bakes bge-small); Groq free tier, generator and judge from different families
- [ ] Create the Space and set secrets (Amit)

## Phase 8 — Ship (branch `phase-8-ship`)
- [ ] README: quick start, architecture, results tables from the committed runs
- [ ] Demo video script (3–8 min) + recording
- [ ] Final check against PRD success criteria; submit
