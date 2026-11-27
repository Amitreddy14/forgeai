# ForgeAI web app for Hugging Face Spaces (Docker SDK). Port 7860 is what Spaces expects.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORGE_ENV=prod \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1

# Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 forge
WORKDIR /home/forge/app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the embedding model into the image so the Space never downloads it at runtime.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && chmod -R a+rX /opt/fastembed

# The committed demo runs and their documents (chat needs the documents).
COPY --chown=forge:forge runs ./runs
COPY --chown=forge:forge data ./data
RUN mkdir -p /home/forge/app/.forge_cache && chown -R forge:forge /home/forge/app
USER forge

EXPOSE 7860
CMD ["uvicorn", "forgeai.web.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
