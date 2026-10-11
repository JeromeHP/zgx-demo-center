# Maritime Surveillance Intelligence Generator - Containerized

**Powered by HP ZGX Nano AI Station**

Vision Language Model (VLM) demonstration for US Navy reconnaissance imagery analysis with synthetic geolocation and structured intelligence report generation. Runs entirely on-premises with no cloud dependency.

---

## What This Demo Is

A fully functional maritime surveillance intelligence tool that analyzes reconnaissance imagery in real time. Upload an aerial or satellite image of a vessel and the system generates a structured intelligence report including vessel classification, threat assessment, synthetic geolocation, and contextual recommendations.

The application runs **Qwen3-VL-8B-Instruct-FP8**, a state-of-the-art vision-language model with FP8 quantization served via **vLLM** for high-performance GPU inference. FP8 quantization reduces the memory footprint to ~9GB with virtually no loss in quality, enabling deployment on a wider range of GPU hardware. A single model handles both image understanding and natural language report generation — no separate text model needed.

Each analysis displays **token consumption metrics** (prompt, completion, and total tokens) — demonstrating exactly what the inference cost would be on a cloud API, and why on-premises inference eliminates those costs entirely.

### Two Example Scenarios

**Non-threatening** — Upload `good-times.png` (aerial photo of a cruise ship) to demonstrate commercial vessel identification with NONE/LOW threat assessment.

**Adversarial** — Upload `navy-ship.jpg` (aerial photo of a naval vessel) to demonstrate military vessel detection with HIGH threat classification and escalated recommendations.

---

## What It Proves to Customers

1. **Classified imagery never leaves the platform.** Reconnaissance data stays within the secure environment — no cloud APIs, no network round-trips.

2. **Real-time analysis without connectivity.** Functions aboard ships at sea, in SCIFs, or in any disconnected environment. Zero dependency on satellite uplink.

3. **Vision AI runs locally on HP hardware.** An 8-billion parameter multimodal model analyzing images in seconds, on hardware the customer owns and controls.

4. **No per-query costs.** The token counter shows exactly how many tokens each analysis consumes. On a cloud API, that's a billable event. On HP hardware, it's free.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/curtburk/navy-surveillance-containerized-docker.git
cd navy-surveillance-containerized-docker

# 2. Download the model (~9GB)
./download_models.sh

# 3. Start the demo
./start.sh
```

The terminal will print a clickable URL with the host IP (e.g., `http://192.168.x.x:8000`). First startup takes 2-3 minutes for model loading.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| GPU | HP ZGX Nano with NVIDIA GB10 Grace Blackwell, or any NVIDIA GPU with 12GB+ VRAM |
| System Memory | 32GB+ recommended |
| Storage | ~20GB free (9GB model + container image) |
| OS | Ubuntu 22.04 or 24.04 LTS |
| Docker | Docker Engine + Docker Compose |
| NVIDIA Container Toolkit | `nvidia-ctk` for GPU passthrough |

### Verify NVIDIA Container Toolkit

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

---

## Architecture

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   HTML Frontend   │────▶│  FastAPI Backend   │────▶│  vLLM Server      │
│   (index.html)    │     │  (main.py :8000)   │     │  (Qwen3-VL :8090) │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
                              All inside one Docker container
```

**Frontend** — Military-themed HTML/CSS/JavaScript interface with image upload, region selection, structured report display, and per-analysis token consumption metrics.

**Backend** — FastAPI server that receives image uploads, base64-encodes them, and sends multimodal prompts to vLLM's OpenAI-compatible API. Handles threat classification, geolocation generation, report assembly, and token usage tracking across VLM calls.

**Inference Engine** — vLLM serving Qwen3-VL-8B-Instruct-FP8 on GPU. FP8 quantization (block size 128) reduces memory usage to ~9GB while maintaining near-identical quality to the BF16 model. Exposes an OpenAI-compatible `/v1/chat/completions` endpoint on internal port 8090. Returns token usage statistics with each completion.

**Containerization** — Based on `nvcr.io/nvidia/vllm:26.01-py3`. The entrypoint script starts vLLM in the background, waits for model loading, then starts the FastAPI application. The model stays on the host and is mounted read-only.

---

## Directory Structure

```
navy-surveillance-containerized-docker/
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── requirements-docker.txt     # Slim runtime dependencies
│   └── entrypoint.sh              # Container startup script
├── frontend/
│   ├── index.html                  # Military-themed web interface
│   ├── hp_logo.png                 # HP branding
│   └── Navy-Emblem.png            # US Navy emblem
├── sample-images/
│   ├── navy-ship.jpg              # Military vessel (HIGH threat demo)
│   └── good-times.png             # Cruise ship (LOW threat demo)
├── models/
│   └── Qwen3-VL-8B-Instruct-FP8/ # Downloaded model (~9GB)
├── Dockerfile                      # Based on NVIDIA vLLM container
├── docker-compose.yml              # One-command startup with GPU
├── start.sh                        # Launch script with IP detection
├── download_models.sh              # Model download from HuggingFace
├── .dockerignore
├── .gitignore
└── README.md
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/analyze` | POST | Analyze image and generate intelligence report |
| `/api/health` | GET | Health check (includes vLLM status) |
| `/api/regions` | GET | List available operating regions |

### POST /api/analyze

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@navy-ship.jpg" \
  -F "region=western_pacific" \
  -F "custom_instructions=Focus on weapons systems"
```

The response includes a `token_usage` object:

```json
{
  "report_id": "SURV-20260325-1234",
  "analysis": "...",
  "token_usage": {
    "prompt_tokens": 1842,
    "completion_tokens": 347,
    "total_tokens": 2189
  }
}
```

---

## Operating Regions

| Region | Coverage | Key Landmarks |
|--------|----------|---------------|
| **Western Pacific** | Philippine Sea, Taiwan Strait | East China Sea, Taiwan Strait |
| **South China Sea** | Spratly/Paracel Islands | Gulf of Tonkin |
| **Arabian Gulf** | Persian Gulf region | Strait of Hormuz, Gulf of Oman |
| **Gulf of Aden** | Horn of Africa | Bab el-Mandeb Strait, Djibouti |
| **Eastern Mediterranean** | Cyprus, Crete | Suez Canal approaches |
| **North Atlantic** | Trans-Atlantic routes | GIUK Gap, Azores |
| **Indo-Pacific** | Southeast Asia straits | Malacca Strait, Singapore Strait |
| **California Coast** | Training region | San Diego, Monterey Bay |

---

## The Customer Conversation

**Opening:** "Let me show you what happens when you put a vision-language AI model on hardware that never needs to phone home."

**During Demo:** Upload the sample ship images. Select operationally relevant AORs. Walk through the 6-section intelligence report — vessel classification, physical characteristics, activity assessment, threat level, confidence, and recommendations. Point out the token count at the bottom of the report.

**Key Messages:**
- "This image was analyzed by an 8-billion parameter multimodal AI, running locally"
- "The imagery never left this machine — no cloud, no API calls, no network required"
- "This works aboard a ship at sea with zero connectivity"
- "See the token count? On a cloud API, that analysis would cost money every time. Here, it's unlimited"

**Closing:** "This is one example. The same hardware runs any AI workload where your data cannot leave your environment."

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

**Model not found at startup** — Ensure `Qwen3-VL-8B-Instruct-FP8/` directory exists in `./models/`. Run `./download_models.sh` if missing.

**vLLM fails to start** — Check GPU memory with `nvidia-smi`. The FP8 model requires ~9GB VRAM. Ensure no other GPU processes are consuming memory.

**Slow first analysis** — Normal. The first image analysis after startup may take longer as vLLM warms up. Subsequent analyses are faster.

**Cannot connect from another machine** — Verify the firewall allows port 8000:
```bash
sudo ufw allow 8000
```

---

## Hardware

This demo is designed for the **HP ZGX Nano AI Station** but the FP8 quantized model runs on any NVIDIA GPU with 12GB+ VRAM, including:

- HP ZGX Nano (NVIDIA GB10 Grace Blackwell, 128GB unified memory)
- NVIDIA RTX 4090 Laptop (16GB)
- NVIDIA RTX 3090/4090 Desktop (24GB)
- NVIDIA RTX 5090 Laptop (24GB)

---

## Security Notice

This demonstration generates **synthetic geolocation data** for illustration purposes. No actual operational coordinates are used or inferred from imagery.

The classification banner displays `UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY` to clearly indicate the demo nature of the application.

---

## License

Internal HP demo. Contact the HP ZGX Nano product team for access and distribution questions.
