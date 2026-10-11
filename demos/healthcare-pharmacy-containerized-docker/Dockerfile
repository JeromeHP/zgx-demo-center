# =============================================================================
# Medical AI Drug Interaction Demo - Docker Container
# Optimized for HP ZGX Nano with NVIDIA GB10 Grace Blackwell Superchip
#
# Inference via Ollama on host (GPU) — container is a thin FastAPI frontend.
#
# Build & run:
#   ./start.sh --build          # First time / after code changes
#   ./start.sh                  # Subsequent runs
#   docker compose down         # Stop
#
# Prerequisites:
#   - Ollama running on host with medical-mixtral model loaded
# =============================================================================

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements-docker.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Set up application directory
WORKDIR /app

# Copy application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python3", "backend/main.py"]
