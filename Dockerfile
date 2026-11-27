# ForgeAI web app. Port 7860 is what Hugging Face Spaces expects.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORGE_ENV=prod

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 forge
WORKDIR /home/forge/app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY --chown=forge:forge runs ./runs
USER forge

EXPOSE 7860
CMD ["uvicorn", "forgeai.web.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
