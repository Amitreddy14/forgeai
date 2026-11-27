# Ollama model variants

Ollama's default context window (2k-4k tokens) can silently cut off long prompts such
as the agent's passages plus schema. These variants pin an 8k context; they share the
base model's weights, so nothing extra is downloaded.

```bash
ollama create qwen2.5-8k  -f ollama/Modelfile.qwen2.5-8k
ollama create llama3.1-8k -f ollama/Modelfile.llama3.1-8k
```

Then set `FORGE_GENERATOR_MODEL=qwen2.5-8k` and `FORGE_JUDGE_MODEL=llama3.1-8k`.
