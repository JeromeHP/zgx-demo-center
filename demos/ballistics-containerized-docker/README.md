# NAWSCL Range Surveillance — Aircraft Classification - Containerized

**Powered by HP ZGX Nano AI Station**

Vision Language Model (VLM) demonstration for range-safety aircraft classification. Upload an image of an aircraft and the system identifies whether it's manned or unmanned, matches it against known platforms, and produces a structured range-safety report. Runs entirely on-premises with no cloud dependency.

> **⚠ Known branding issue (open item):** this demo's frontend and copy use Navy/NAWSCL China Lake framing (`Navy-Emblem.png`, "NAWSCL", "NAF China Lake") inherited from its original source, but the sample aircraft (F-35C Lightning II, MQ-1 Predator) are Air Force platforms. Correct Air Force range-safety designations/branding are a follow-up, not yet applied — see the migration plan's open items.

---

## What This Demo Is

An aircraft classification tool for range-safety operations. Upload an image of an aircraft over a designated range area and the system generates a structured report: manned/unmanned classification, platform identification against known airframes, flight status estimate, mission profile assessment, payload identification, and range-safety recommendations — plus synthetic geolocation data for the selected range area.

The application runs **qwen2.5vl:7b**, a vision-language model served via **Ollama** on the host. A single model handles both image understanding and structured report generation.

Each analysis displays **token consumption metrics** (prompt, completion, and total tokens) — demonstrating exactly what the inference cost would be on a cloud API, and why on-premises inference eliminates those costs entirely.

### Two Example Scenarios

**Manned** — Upload `F-35C_Lightning_II.png` to demonstrate manned-platform classification.

**Unmanned** — Upload `MQ-1_Predator.png` to demonstrate unmanned-platform (UAS) classification.

---

## What It Proves to Customers

1. **Imagery never leaves the platform.** Range-surveillance data stays within the secure environment — no cloud APIs, no network round-trips.

2. **Real-time analysis without connectivity.** Functions in disconnected range environments with zero dependency on uplink.

3. **Vision AI runs locally on HP hardware.** A multimodal model analyzing images in seconds, on hardware the customer owns and controls.

4. **No per-query costs.** The token counter shows exactly how many tokens each analysis consumes. On a cloud API, that's a billable event. On HP hardware, it's free.

---

## Quick Start

```bash
# 1. Pull the model (~6GB)
ollama pull qwen2.5vl:7b

# 2. Start the demo
./start.sh --build
```

The terminal will print a clickable URL with the host IP (e.g., `http://192.168.x.x:8093`).

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| GPU | HP ZGX Nano with NVIDIA GB10 Grace Blackwell, or any NVIDIA GPU with enough VRAM to hold qwen2.5vl:7b alongside whatever else is loaded |
| System Memory | 32GB+ recommended |
| Storage | ~6GB free (model, pulled into Ollama — not part of this image) |
| OS | Ubuntu 22.04 or 24.04 LTS |
| Docker | Docker Engine + Docker Compose |
| Ollama | Running on the host, reachable at `localhost:11434` |

---

## Architecture

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   HTML Frontend   │────▶│  FastAPI Backend   │────▶│  Host Ollama      │
│   (index.html)    │     │  (main.py :8093)   │     │  (qwen2.5vl:7b)   │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
        Both inside one thin Docker container            On the host (GPU)
```

**Frontend** — HTML/CSS/JavaScript interface with image upload, range-area selection, structured report display, and per-analysis token consumption metrics.

**Backend** — FastAPI server that receives image uploads, normalizes them to JPEG (guards against browser-uploaded formats Ollama's decoder rejects, e.g. AVIF/HEIC), base64-encodes them, and sends multimodal prompts to Ollama's `/api/generate` endpoint. Handles classification parsing, synthetic geolocation, report assembly, and token usage tracking.

**Inference** — Ollama serving `qwen2.5vl:7b` on the host GPU, reached over `network_mode: host` at `localhost:11434`. No model weights are baked into this image — the container is a thin FastAPI client only.

---

## Directory Structure

```
ballistics-containerized-docker/
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── ollama_client.py         # Shared Ollama HTTP adapter
│   ├── __init__.py
│   └── requirements-slim.txt    # Runtime dependencies
├── frontend/
│   ├── index.html               # Web interface
│   ├── hp_logo.png              # HP branding
│   └── Navy-Emblem.png          # (see branding note above)
├── sample-images/
│   ├── F-35C_Lightning_II.png   # Manned-platform demo
│   └── MQ-1_Predator.png        # Unmanned-platform demo
├── Dockerfile                    # Thin Ollama-client image
├── docker-compose.yml            # One-command startup, host networking
├── start.sh                      # Launch script with IP detection
├── .dockerignore
├── .gitignore
└── README.md
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/analyze` | POST | Analyze image and generate classification report |
| `/api/health` | GET | Health check (includes Ollama reachability/model status) |
| `/api/ranges` | GET | List available range areas |

### POST /api/analyze

```bash
curl -X POST http://localhost:8093/api/analyze \
  -F "image=@sample-images/F-35C_Lightning_II.png" \
  -F "range_area=baker_range" \
  -F "custom_instructions=Focus on visible payload"
```

The response includes a `token_usage` object and a synthetic `location` object (see the branding note — geolocation data is entirely synthetic, not derived from the image).

---

## Stopping

```bash
docker compose down
```

---

## Troubleshooting

**Docker daemon not running**
```bash
sudo systemctl start docker
```

**Permission denied on Docker commands**
```bash
sudo usermod -aG docker $USER && newgrp docker
```

**`Ollama inference error` on analyze** — Confirm `ollama pull qwen2.5vl:7b` has completed and Ollama is reachable at `localhost:11434` on the host.

**Cannot connect from another machine** — Verify the firewall allows port 8093:
```bash
sudo ufw allow 8093
```

---

## Credits

Vendored from Curtis Burkhalter's `Fed-air-ballistics-identification-containerized`, migrated from a separate `zgx-demos` repo into this kit with the vLLM/llama-cpp backend replaced by a thin Ollama client (same pattern as Pharmacy and PII in this repo).

---

## Security Notice

This demonstration generates **synthetic geolocation data** for illustration purposes. No actual operational coordinates are used or inferred from imagery. The classification banner displays `UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY` to clearly indicate the demo nature of the application.

---

## License

Internal HP demo. Contact the HP ZGX Nano product team for access and distribution questions.
