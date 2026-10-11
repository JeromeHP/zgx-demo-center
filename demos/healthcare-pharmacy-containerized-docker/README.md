# Medical AI Drug Interaction Demo — Containerized

AI-powered clinical decision support for analyzing drug interactions, running entirely on a single HP ZGX Nano workstation. No cloud connection required. No data leaves the device.

---

## What This Demo Is

A fully functional clinical decision support tool that analyzes drug interactions in real time. A user enters medications and the system instantly returns professional-grade analysis including interaction severity, clinical mechanisms, monitoring recommendations, and alternative medication suggestions.

The application runs a fine-tuned Mixtral-8x7B model (47 billion parameters) using llama.cpp for quantized inference on GPU. The model was trained on authoritative medical datasets from the FDA (DailyMed drug labels), Stanford SNAP (TWOSIDES interaction database), and PubMed Central clinical studies.

### Two Analysis Modes

**Simple Mode** — Enter two drugs and receive interaction severity, clinical mechanism, monitoring recommendations, and alternatives. This is similar to what medical professionals can do today.

**Complex Mode** — Full patient case analysis with multiple medications, age, medical conditions, and lab values. Returns prioritized drug-drug interactions, risk scoring, and clinical action items.

---

## What It Proves to Customers

1. **Enterprise AI runs locally on HP hardware.** The same class of AI that powers cloud services costing thousands per month, running on hardware they can own and control.

2. **Sensitive data stays on-premises.** Healthcare organizations cannot send patient medication lists to external cloud APIs due to HIPAA and data governance requirements. This demo shows AI analysis happening entirely within the customer's environment.

3. **Fine-tuning works.** The model has been trained on authoritative medical datasets, not internet scraping. Customers see that they can customize AI models for their specific domain and data.

The demo translates directly to customer conversations about radiology AI, clinical documentation, medical coding, pathology analysis, and any healthcare AI application requiring data sovereignty.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/curtburk/healthcare-pharmacy-containerized-docker.git
cd healthcare-pharmacy-containerized-docker

# 2. Download the model (~28GB)
./download_models.sh

# 3. Build & run (first time)
./start.sh 

```

The terminal will print a clickable URL with the host IP (e.g., `http://192.168.x.x:8000`). Open it from any browser on the same network.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| GPU | HP ZGX Nano with NVIDIA GB10 Grace Blackwell (or NVIDIA GPU with 24GB+ VRAM) |
| System Memory | 64GB+ unified memory recommended |
| Storage | ~35GB free (28GB model + container image) |
| OS | Ubuntu 22.04 or 24.04 LTS |
| Docker | Docker Engine + Docker Compose |
| NVIDIA Container Toolkit | `nvidia-ctk` for GPU passthrough |

### Verify NVIDIA Container Toolkit

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

If this prints GPU info, you're ready. If not, install the toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

---

## Architecture

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   HTML Frontend   │────▶│  FastAPI Backend   │────▶│  Quantized LLM    │
│   (index.html)    │     │  (main.py)         │     │  (llama.cpp)      │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

**Frontend** — Single-page HTML/CSS/JavaScript application with responsive design for both simple two-drug queries and complex patient case analysis.

**Backend** — FastAPI server providing RESTful endpoints for drug interaction analysis, health checks, and sample query retrieval.

**Inference Engine** — llama.cpp with 4-bit quantized Mixtral-8x7B model. All GPU layers offloaded for maximum inference speed.

**Containerization** — Two-stage Docker build. Stage 1 compiles llama-cpp-python with CUDA support for Blackwell architecture (sm_120). Stage 2 is a slim runtime image. The ~28GB model file stays on the host and is mounted read-only into the container at runtime.

---

## Directory Structure

```
healthcare-pharmacy-containerized-docker/
├── backend/
│   ├── main.py                     # FastAPI application server
│   └── requirements-docker.txt     # Slim runtime dependencies
├── frontend/
│   ├── index.html                  # Web interface
│   └── hp_logo.png                 # HP branding
├── models/
│   └── medical-ft-mixtral-q4       # Quantized model (~28GB, downloaded separately)
├── Dockerfile                      # Two-stage build (CUDA + runtime)
├── docker-compose.yml              # One-command startup with GPU passthrough
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
| `/api/simple_interaction` | POST | Two-drug interaction analysis |
| `/api/complex_interaction` | POST | Multi-drug patient case analysis |
| `/api/health` | GET | Health check and model status |
| `/api/sample_queries` | GET | Pre-built demo examples |

### Simple Interaction

```bash
curl -X POST http://localhost:8000/api/simple_interaction \
  -H "Content-Type: application/json" \
  -d '{"drug1": "warfarin", "drug2": "aspirin"}'
```

### Complex Patient Analysis

```bash
curl -X POST http://localhost:8000/api/complex_interaction \
  -H "Content-Type: application/json" \
  -d '{
    "medications": ["warfarin", "aspirin", "omeprazole", "metoprolol", "amlodipine"],
    "age": 78,
    "conditions": ["atrial fibrillation", "hypertension", "GERD"],
    "lab_values": {"INR": "3.2", "CrCl": "45 mL/min"}
  }'
```

---

## The Customer Conversation

**Opening:** "Let me show you what enterprise AI looks like when it runs entirely on your hardware."

**During Demo:** Enter common drug pairs (warfarin + aspirin, metformin + contrast dye) or use the complex patient scenarios with multiple medications. Point out the response quality and speed.

**Key Messages:**
- "This model has 47 billion parameters, running locally, no cloud required"
- "The training data came from FDA and NIH sources, not internet scraping"
- "Your patient data never leaves this machine"

**Closing:** "This is one example. The same hardware and workflow applies to any AI application where your data cannot leave your environment."

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

**Model not found at startup** — Ensure `medical-ft-mixtral-q4` is in the `./models/` directory. Run `./download_models.sh` if missing.

**Build fails on llama-cpp-python** — The Dockerfile uses `-Wl,--allow-shlib-undefined` to handle the missing `libcuda.so.1` during Docker builds (the NVIDIA driver is injected at runtime by nvidia-container-toolkit). Do not modify the `CMAKE_ARGS` in the Dockerfile.

**Slow first query** — Normal. The model warms up on the first inference. Subsequent queries are faster.

**CUDA out of memory** — Ensure no other GPU processes are running (`nvidia-smi`). The model requires approximately 28GB of GPU memory.

**Cannot connect from another machine** — Verify the firewall allows port 8000:
```bash
sudo ufw allow 8000
```

---

## Model Training Background

The fine-tuned model was created using the following pipeline (scripts not included in this containerized repo):

1. **Data Collection** — Drug interaction data from TWOSIDES (~48,514 interactions), DailyMed FDA drug labels, PubMed Central clinical studies, and DDI Corpus annotations.

2. **Instruction Dataset Preparation** — ~5,000 training samples covering drug-drug interaction analysis, FDA label summarization, clinical study interpretation, polypharmacy scenarios, and comprehensive clinical decision cases.

3. **Fine-Tuning** — Mixtral-8x7B-Instruct-v0.1 with 4-bit QLoRA (LoRA rank 16, alpha 32) targeting all attention and MLP projection layers. Trained with paged AdamW 8-bit optimizer.

4. **Quantization** — Merged LoRA adapters converted to GGUF Q4_K_M format for production inference via llama.cpp.

---

## Hardware

This demo is designed for the **HP ZGX Nano AI Station** featuring:
- NVIDIA GB10 Grace Blackwell Superchip
- Up to 1000 TOPS of AI compute
- 128GB unified memory
- ARM-based (aarch64) architecture

The containerized build targets Blackwell's CUDA compute capability 12.0 (sm_120).

---

## License

Internal HP demo. Contact the HP ZGX Nano product team for access and distribution questions.
