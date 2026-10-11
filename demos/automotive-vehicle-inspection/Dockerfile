# =============================================================================
# Port-of-Entry Vehicle Inspection Analyst - Ollama-client image
#
# Inference runs in HOST Ollama (mistral-small3.2). This container is just the thin
# FastAPI client -- no vLLM, no CUDA, no GPU reservation. Upstream built FROM
# NVIDIA's NGC vLLM image, which is NVIDIA-only and cannot share the GPU with
# Maritime's vLLM; python:3.12-slim is multi-arch, so this builds on aarch64
# and x86_64 alike. Same shape as ballistics-containerized-docker.
# =============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# 8097 is this demo's slot in the Demo Center port map (RUNBOOK.md). Under
# network_mode: host there is no port mapping, so the app binds it directly.
EXPOSE 8097

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8097"]
