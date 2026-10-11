# =============================================================================
# NAWSCL Aircraft Classification - Ollama-client image
#
# Inference runs in HOST Ollama (qwen2.5vl:7b). This container is just the
# thin FastAPI client — no llama-cpp-python / vLLM / CUDA build here.
# Modeled on the pii demo's Dockerfile.ollama.
# =============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements-slim.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

EXPOSE 8093
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8093/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8093"]
