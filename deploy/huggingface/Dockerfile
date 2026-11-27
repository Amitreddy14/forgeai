# Hugging Face Space image: builds ForgeAI straight from the public GitHub repository,
# so the Space repo holds only this file and its README.
FROM python:3.11-slim

ARG FORGEAI_REPO=https://github.com/Amitreddy14/forgeai.git
ARG FORGEAI_REF=main

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORGE_ENV=prod \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 forge
WORKDIR /home/forge/app

RUN git clone --depth 1 --branch "$FORGEAI_REF" "$FORGEAI_REPO" . \
    && pip install . \
    && python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && chmod -R a+rX /opt/fastembed \
    && rm -rf .git tests \
    && mkdir -p .forge_cache && chown -R forge:forge /home/forge/app
USER forge

EXPOSE 7860
CMD ["uvicorn", "forgeai.web.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
