---
title: ForgeAI
emoji: ⚓
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: The agent that builds agents, and proves they work
---

# ForgeAI (hosted)

The agent that builds agents: give it a one-paragraph brief and a folder of documents,
and it builds a grounded agent, writes its own tests, judges the results, fixes what
failed, and reports honestly what still fails.

Source, documentation and the committed demo runs:
https://github.com/Amitreddy14/forgeai

This Space builds from that repository's `main` branch. Models are served by Groq's
free tier, so chat is quick and new builds are slow and budget-capped. Full builds run
locally with `forge build` on Ollama.
