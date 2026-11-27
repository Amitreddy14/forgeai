# ForgeAI — Rules (AI rulebook)

Read this before writing any code in this repo. These rules exist because the whole
product is about not trusting model output. The code must hold itself to the same bar.

## Must do

1. **Validate every model output with Pydantic.** No raw dict from a model travels past
   the function that received it. Invalid output → one repair retry with the validation
   error in the prompt → then a typed failure, never a silent default.
2. **Verify every quote in code.** If a model says a document says something, the quote is
   checked as a normalised substring of that document (`forgeai.textutil.contains_quote`).
3. **Keep prompts in `src/forgeai/prompts/*.md`.** No prompt strings inline in Python
   except one-line system roles. Prompts are versioned in Git and are part of the evidence.
4. **Compute numbers in code.** Scores, pass rates, deltas and counts come from per-case
   results. A model never writes a number into the report.
5. **Write every artifact to the run folder** through `RunStore`, as pretty JSON or
   Markdown, so it can be committed and diffed.
6. **Async for I/O** (LLM calls, HTTP tool). Bound concurrency with a semaphore
   (`FORGE_MAX_CONCURRENCY`) — Ollama on one GPU serialises anyway.
7. **Tests use `FakeLLM`.** No test touches the network or needs an API key. Each stage
   gets tests for its happy path and for the model lying (bad JSON, fake quote, extra tool).
8. **Type everything.** `mypy --strict` passes on `src/`.
9. **Log, don't print**, in library code. CLI output goes through `rich` / Typer echo.

## Must not do

1. Never use `eval`/`exec`. The calculator parses with `ast` and allows only numeric
   literals and `+ - * / ** % ()` plus a small function allowlist.
2. Never put secrets in code, artifacts, logs or the cache. API keys come from env vars;
   `.env` is git-ignored.
3. Never let the HTTP tool call a URL a model wrote. The weather tool builds the
   open-meteo URL itself from validated lat/lon or a geocoded place name.
4. Never regenerate the whole agent in the optimizer. Patches only (D3).
5. Never show the optimizer holdout cases or holdout results (D2).
6. Never let the judge see which model produced an answer, or two answers at once in a
   fixed order.
7. Never claim a result in README or report that a committed artifact does not show.
8. Never add Claude or any AI as a commit co-author.

## Libraries

Allowed: fastapi, uvicorn, pydantic, pydantic-settings, typer, rich, httpx, jinja2,
python-multipart, rank-bm25, fastembed, numpy, pypdf, python-docx.
Dev: pytest, pytest-asyncio, pytest-cov, ruff, mypy, respx.
Do not add LangChain/LlamaIndex: the assignment values understanding what happens
underneath, and the loop is small enough to own.

## Error handling

- Domain errors subclass `forgeai.errors.ForgeError` (`ModelOutputError`,
  `BudgetExceeded`, `GroundingError`, `ToolError`).
- A stage that fails writes `stage_error.json` with the error and stops the run cleanly;
  the report still renders with whatever finished.
- `BudgetExceeded` is not a crash: the optimizer stops, the report says why.
- Web routes return proper status codes (400 bad input, 404 unknown run, 413 too large,
  422 validation, 503 model unreachable) — never a stack trace.

## Security checklist (web)

Upload allowlist (`.pdf .docx .csv .md .txt`), size cap, filename sanitising, run IDs
validated against `^[a-z0-9-]{3,64}$` (no path traversal), security headers, CORS locked,
debug off in production, rate limit on build/chat endpoints.

## Git

Branch per phase (`phase-N-name`), PR to `main`, CI green before merge, tag `phase-N`.
Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`).
