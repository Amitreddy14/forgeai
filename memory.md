# ForgeAI — Memory (running log)

Newest first. Decisions, bugs, changes, and anything the next session must know.

## 2026-11-28 — Phase 7.1 (first hosted run on Render)

Live at https://forgeai-524s.onrender.com. Pages served; the first chat returned 503 and
the instance then failed twice (Render events "Instance failed"). Two causes fixed:
- Memory: fastembed's default batch of 256 passages builds multi-GB attention tensors on
  first index build; 512 MB instance died. Now batch_size=8, threads=1 (same vectors).
- Groq JSON mode answers malformed JSON with HTTP 400 `json_validate_failed`, which the
  client treated as "endpoint down". The rejected text is now handed back to the
  structured layer's repair loop. Other 400s still raise.
- Chat errors are now logged; a broken-format reply gets its own message (502).
- `HEAD /` answers 200 (Render's uptime probe got 405).

Phase 7.2: the instance failures were really "HTTP health check failed (timed out after
5 seconds)". Building a chat runtime embeds the whole corpus synchronously inside the
async handler; on 0.1 CPU that blocks the event loop long enough for Render to kill the
instance. Runtime builds and per-question retrieval now run in a worker thread (one
lock, built once), and with FORGE_WARM_CHAT=true (set in render.yaml) both demo
runtimes are built at startup. Model failures now name the cause (bad key, unknown
model, HTTP status) instead of a generic "did not answer".

Phase 7.3: that message said 404: Groq shut down llama-3.1-8b-instant for free/dev tiers
on 2026-08-16. Hosted generator is now openai/gpt-oss-20b (Groq's named replacement),
judge qwen/qwen3.8-27b (preview; a different family). gpt-oss reasons before answering
and the reasoning counts against max_tokens, so the client sends reasoning_effort=low
plus 1024 tokens of headroom for gpt-oss models; inline <think> blocks are stripped.
htmx indicator styles disabled via htmx-config (our CSS has them; CSP blocked the inline
copy and left a console error). Committed demos were built locally with qwen2.5-8k, so
hosted chat runs a different model than the certificates measured; README must say so.

Phase 7.4: hosted HR chat verified (24 days, quote verified, 0.1 s). Port weather failed on
Render with "weather service unavailable (HTTPStatusError)" x3 (works locally). Open-meteo's
free tier limits by IP and Render's free egress is shared; the error hid the status, so:
- Errors now name host + HTTP status; 429/5xx retried twice with backoff.
- Fallback: geocoding via OSM Nominatim, forecast via MET Norway (both keyless, need a
  User-Agent, which every request now sends). Output names the source and why.
- Marine wave failure is a note, not a failed forecast.
- The generated refusal message named one topic ("can't give legal or customs advice")
  and was used for every refusal, including the weather outage. Runtime: when every
  tool call failed, the refusal says the service did not respond. Generate prompt: the
  refusal sentence must fit any refusal (future builds; committed demos keep theirs).

## 2026-11-28 — Phase 6.4 (fourth demo runs)

HR: optimizer kept "future financial forecasts" refuse topic (87% -> 93% dev). Port: Kochi
weather now resolves to India (25-31 °C); dev 74%. Two more runtime gaps fixed:
- Citation.chunk_id is optional: a missing id failed schema validation and discarded a
  correct answer; quotes are verified by words anyway.
- Ship evidence or nothing: after the repair, an unrefused answer with no verified quote
  and no tool result is withdrawn and replaced by the refusal message
  (refused_by="grounding").

## 2026-11-28 — Phase 6.3 (third demo runs, merged as PR #7)

Third runs (mostly cache replays): HR dev 93% (14/15), holdout 7/8, trap handled, no
tools (calculator proposed and rejected). Port dev 84% -> 89% (refuse topic kept),
holdout 6/9, exactly csv_lookup/calculator/weather. Two runtime bugs found in the port
artifacts, both also affecting hosted chat:
- c21: correct answer "13500" returned with refused=true. Output contract now checks the
  refusal flag agrees with the text; a contradiction triggers the one repair.
- c22: weather for "Kochi" geocoded to Kochi, Japan (8-20 °C). The tool now takes
  "Kochi, India", filters geocoding results by country name/code, and always prints the
  resolved country; the plan prompt asks for the country.
- Report now states how many model calls were cache replays (tokens count live calls).
- Hosting moved from HF (Docker Spaces now paid) to Render free (render.yaml).

## 2026-11-28 — Phase 7

- Web: FastAPI + Jinja + HTMX, hand-written CSS from design.md. Verified with headless
  Chrome screenshots (desktop + 400 px): fixed a load-line label collision and a mobile
  overflow (grid children need min-width: 0).
- Bug found by the UI: `RunStore.write_json` used `json.dumps(default=str)`, so lists of
  models (changelog.json, canaries.json) were saved as Python reprs. Now converted
  recursively; anything non-JSON raises. Demo runs must be regenerated.
- Hosting: Hugging Face made Docker/Gradio Spaces paid (Amit's account shows it), and
  Koyeb now needs a card, so the app is hosted on Render's free web service (Docker,
  no card, 512 MB, sleeps after 15 idle minutes) via `render.yaml`.
- Groq free tier (generator openai/gpt-oss-20b, judge qwen/qwen3.8-27b since 7.3): hosted
  builds are capped (1 iteration, 150k tokens).
- CSP forbids inline styles, so progress uses <progress>, not a styled width.

## 2026-11-28 — Phase 6.2 (second demo runs)

Second runs: HR dev 82% / holdout 7/8; port dev 84% -> 89% (optimizer kept refuse topic
"future operational schedules"), holdout 7/9. Port agent got exactly csv_lookup,
calculator, weather (current_date not attached): the contract works. c21 showed the
agent loop self-correcting (csv_lookup failed, retried ok, then calculator).
Still not committed; fixes:
- Personal split kept a topic literally named "Remaining leave balance". Now a topic
  whose own name is the personal ask is dropped (refused), not kept.
- Out-of-scope candidates that name a must-answer topic ("HR policies for remote work")
  with no refusal cue (my, next, plans, recent changes, book, status...) are rejected.
- Weather tool returned a forecast and the agent refused anyway: the system prompt now
  states tool results are trusted evidence, and TOOL RESULTS come before PASSAGES.
- agent.v0.prompt.md is now always written.

## 2026-11-27 — Phase 6.1 (fixes from the first two demo runs)

First demo runs: HR dev 75% / holdout 7/7; port dev 67% / holdout 3/7. Not committed,
because both showed factory bugs:
- HR: the model merged "how many leave days are left" into "annual leave entitlement";
  a real entitlement quote passed the guard, so the planted trap was missed. Fix: a
  deterministic personal-records pattern flags such asks; apply_guard SPLITS the topic
  (general topic stays answerable, the personal ask becomes a refusal + gap). The
  requirements prompt now says one ask per topic, personal asks on their own.
- Port: the model marked weather/berths/charges as "documents" topics with no tool
  hints, so no tools were attached, and "port charges estimation" was wrongly flagged.
  Fix: `Topic.tool_hints` (list); code infers hints from the topic's words and the
  folder's CSV (table name, or ≥ 2 column words); topics with hints skip the prose guard.
- Out-of-scope labels: qwen still called in-scope questions "not answered". Backstop:
  an out-of-scope candidate as similar to the documents as the 25th-percentile
  answerable question is rejected.
- Two-document questions written by the model were forced ("compare monsoon draft with
  port dues"). Now composed in code by joining two verified single-document questions
  from different documents ("...? Also, ..."), facts already proven.
- Optimizer had no fix for answering must-refuse questions. New critic patch: a general
  refuse topic (rejected if it copies a test question).
- Split: every category with ≥ 2 cases has at least one holdout case.

## 2026-11-27 — Phase 6

- hr-mini after 4.1: dev 90%, holdout 100%, canaries 100% (was 46% / 40% / 62%).
  `forge build` on hr-mini: 80 s, every dev case passed, so the optimizer had nothing to
  do; hr-mini is too small (11 cases) for the real demos.
- 4.1's out-of-scope check reused the guard prompt and over-rejected true refusals
  ("how much sick leave do I have left" was "supported" by the general sick-leave rule).
  Now a dedicated `evals_answerability` prompt asks whether a passage states the
  specific answer; personal data, other groups, future changes and actions don't count.
- Demo PDFs exposed a chunker bug: a heading at the foot of a page became a 4-word
  chunk and the body on the next page lost its heading. Chunking now streams across
  pages; a chunk is cited by the page it starts on.
- Demo documents: fictional company "Konkan Marine Services"; invented port tariffs;
  every document says it is synthetic on its first line.
- Eval targets raised: answerable 12, ambiguous 3, multi-doc 4, out-of-scope 7, refill 8.

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
