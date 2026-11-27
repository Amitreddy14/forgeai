# Deploying to Hugging Face Spaces (free)

1. Create a Space at https://huggingface.co/new-space: owner you, name `forgeai`,
   SDK **Docker**, template **Blank**, hardware **CPU basic (free)**, public.
2. Clone it and add the two files from this folder:
   ```bash
   git clone https://huggingface.co/spaces/<your-hf-username>/forgeai hf-forgeai
   copy deploy\huggingface\Dockerfile hf-forgeai\Dockerfile
   copy deploy\huggingface\README.md  hf-forgeai\README.md
   cd hf-forgeai
   git add Dockerfile README.md
   git commit -m "Deploy ForgeAI from GitHub"
   git push
   ```
   When git asks for a password, use a Hugging Face access token with write scope
   (https://huggingface.co/settings/tokens).
3. In the Space's **Settings -> Variables and secrets**, add:

   | Kind | Name | Value |
   |---|---|---|
   | Secret | `FORGE_LLM_API_KEY` | your Groq API key |
   | Variable | `FORGE_LLM_BASE_URL` | `https://api.groq.com/openai/v1` |
   | Variable | `FORGE_GENERATOR_MODEL` | `llama-3.1-8b-instant` |
   | Variable | `FORGE_JUDGE_MODEL` | `openai/gpt-oss-20b` |
   | Variable | `FORGE_MAX_ITERATIONS` | `1` |
   | Variable | `FORGE_MAX_TOKENS_PER_RUN` | `150000` |
   | Variable | `FORGE_MAX_CONCURRENCY` | `1` |

   The judge comes from a different model family than the generator, as locally.
4. The Space rebuilds automatically. Open `https://<your-hf-username>-forgeai.hf.space`.
   To redeploy after pushing to GitHub, use **Settings -> Factory rebuild**.
