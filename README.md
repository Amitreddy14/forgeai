# ForgeAI

**The agent that builds agents.** Give ForgeAI a one-paragraph brief and a folder of
documents. It interviews you, checks that the documents can support the brief, builds a
grounded agent, writes its own test set, judges the results, fixes what failed (up to 3
times), and hands back the agent with an honest report of what still fails and why.

> Status: Phase 5 (critic-optimizer + honest report). One command runs the whole factory:
>
> `forge build --brief brief.md --docs docs/ --answers auto` See [task.md](task.md) for progress.

## Why it can be trusted

Every stage runs on a small open model, so every stage can be wrong. ForgeAI checks each
one with code that does not depend on the model being honest:

| Risk | Check |
|---|---|
| Brief asks for topics the documents don't cover | Groundability guard: every topic needs a verbatim quote that code finds in a document |
| Test cases with made-up expected answers | Each case's key facts must appear in its verified source quote, or the case is dropped |
| Judge rewards long or confident answers | Deterministic checks first, a judge from a different model family, length-blind rubric, planted canary answers to measure the judge |
| Optimizer overfits the tests it sees | Fixes are tuned on a dev split; the reported score is on a holdout split it never saw |
| "Fixes" that break other things | Each fix is a small patch, kept only if the dev score rises with no regressions |
| Tools the agent doesn't need | Each tool must cite the requirement that needs it |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env               # Windows: copy .env.example .env
ollama pull qwen2.5:7b && ollama pull llama3.1:8b   # then see ollama/README.md
forge doctor                       # checks settings and the model endpoint
forge smoke                        # one structured-JSON call per model, timed
forge ingest path/to/docs          # what was extracted and how it chunks
forge search path/to/docs "how many days of annual leave?"

# Intake + groundability guard on the small example (about 8 model calls)
forge intake --brief data/examples/hr-mini/brief.md --docs data/examples/hr-mini/docs \
  --answers auto --run-id scratch-hr-mini
forge generate --run-id scratch-hr-mini --docs data/examples/hr-mini/docs
forge chat runs/scratch-hr-mini/agent.v0.json
```

For Ollama, create the 8k-context model variants in [`ollama/`](ollama/README.md) so long
prompts are not silently truncated.

Run the web app: `uvicorn forgeai.web.app:app --reload`, then open
http://localhost:8000/healthz.

## Development

```bash
ruff check . && ruff format --check . && mypy && pytest
```

CI runs the same checks on Python 3.11 and 3.12, then builds the Docker image and
smoke-tests it. Tests use a scripted fake model, so CI needs no API key.

## Project docs

[PRD](PRD.md) · [Architecture](Architecture.md) · [Rules](rules.md) ·
[Design](design.md) · [Tasks](task.md) · [Memory](memory.md)
