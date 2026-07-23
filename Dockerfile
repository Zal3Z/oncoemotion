# oncoemotion API image (baseline mapper + terminology).
# Light CPU image: serves /map, /health, /terminology. The heavy ML stack
# (torch/transformers) for the research endpoints is NOT included here — build a
# separate GPU image for Phases 2-4 (see docs). The research endpoints stay
# disabled by default.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ONCOEMOTION_ENABLE_STEERING=0 \
    ONCOEMOTION_ENABLE_ACTIVATIONS=0

WORKDIR /app

# Install deps first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[api]"

# App content.
COPY terminology ./terminology
COPY configs ./configs
COPY data ./data

# Ensure the PRO-CTCAE terms file exists (regenerate if missing).
COPY scripts ./scripts
RUN python scripts/build_terminology.py || true

EXPOSE 8000

# Non-root user.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Official CTCAE v6.0 / PRO-CTCAE Italian files are license-restricted and NOT
# baked in; mount them at runtime, e.g.:
#   docker run -v $PWD/terminology/official:/app/terminology/official ...
# Without them, the mapper uses the labelled synthetic CTCAE fallback.

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "oncoemotion.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
