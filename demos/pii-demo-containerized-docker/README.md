# PII Masking Demo - Updated for easier start using a Docker container

# PII Masking Demo - Fine-tuning Comparison

## Overview
This demonstration showcases the effectiveness of fine-tuning large language models for PII (Personally Identifiable Information) detection and masking. The demo compares a base TinyLlama model against a fine-tuned Qwen2.5-32B model, highlighting the dramatic improvement in accuracy and consistency achieved through fine-tuning.

## Prerequisites

- **HP ZGX Nano** (or any system with NVIDIA GPU + 40GB+ VRAM)
- **Docker** with Docker Compose v2
- **NVIDIA Container Toolkit** ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

### First-time Docker setup

Make sure the Docker daemon is running:
```bash
sudo systemctl start docker
```

To have Docker start automatically on boot:
```bash
sudo systemctl enable docker
```

Add your user to the `docker` group so you don't need `sudo` for every command:
```bash
sudo usermod -aG docker $USER
newgrp docker   # applies immediately in current shell
```

Verify your GPU is accessible to Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Quick Start

```bash
# 1. Download model files (~19GB total)
chmod +x download_models.sh
./download_models.sh

### these will download from a HuggingFace repo @ https://huggingface.co/curtburk/pii-masking-demo-models

# 2. Build and run
chmod +x start.sh
./start.sh --build
```

On startup, the terminal will display clickable URLs with the host's IP address:

```
============================================================
🚀 Demo is ready!
============================================================
  Frontend:  http://<your host device IP address>:8000/app/
  API:       http://<your host device IP address>:8000/
  API Docs:  http://<your host device IP address>:8000/docs
============================================================
```

For subsequent runs (no rebuild needed):
```bash
./start.sh
```

To stop:
```bash
Press Ctrl + C
```

## Directory Structure

```
pii-demo-docker/
├── start.sh                      # Launch script (auto-detects host IP)
├── Dockerfile                    # Multi-stage build (CUDA + Python)
├── docker-compose.yml            # GPU passthrough and environment config
├── download_models.sh            # Downloads GGUF models
├── backend/
│   ├── main.py                   # FastAPI server (serves API + frontend)
│   ├── requirements-docker.txt   # Slim Python dependencies
│   └── offline_responses.json    # Fallback responses
├── frontend/
│   └── index.html                # Web interface (you provide this)
└── models/                       # GGUF model files (git-ignored, mounted at runtime)
    ├── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    └── pii_detector_Q4_K_M.gguf
```

## Architecture Changes from Bare-Metal Version

| Aspect | Bare-Metal | Docker |
|--------|-----------|--------|
| Model paths | Hardcoded absolute paths | Environment variables |
| Frontend server | Separate `python3 -m http.server` on port 8080 | Served by FastAPI on port 8000 at `/app/` |
| IP configuration | Manual edit of `start_demo_remote.sh` | Auto-detected by `start.sh` |
| Startup | `./start_demo_remote.sh` | `./start.sh` |
| Dependencies | venv + manual pip installs | Baked into image |
| Models | Stored in project directory | Mounted as Docker volume |

## Configuration

All settings are configurable via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL_PATH` | `/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Path to base model |
| `FINETUNED_MODEL_PATH` | `/app/models/pii_detector_Q4_K_M.gguf` | Path to fine-tuned model |
| `N_GPU_LAYERS` | `-1` (all) | Number of layers to offload to GPU |
| `N_CTX` | `2048` | Context window size |
| `N_BATCH` | `4096` | Batch size for prompt processing |
| `N_THREADS` | `8` | CPU threads for non-GPU work |
| `AUTO_LOAD_MODELS` | `true` | Load models on startup vs. manual trigger |
| `HOST_IP` | Auto-detected | Override the displayed IP address |

You can also override `HOST_IP` at launch time:
```bash
HOST_IP=10.0.0.5 ./start.sh --build
```
### Using the Demo Interface

1. **Load Models**: Click the "Load Models" button first (this takes 1-2 minutes)
2. **Enter Text**: Type or paste text containing PII, or use the sample buttons
3. **Process**: Click "Process & Compare" to see both models' outputs
4. **Compare Results**: Observe the difference between the base and fine-tuned models

## Demo Flow for Presentations

1. **Introduction**: Explain the importance of PII protection in enterprise environments
2. **Show Base Model Performance**: Demonstrate how a generic model struggles with PII detection
3. **Show Fine-tuned Model**: Highlight the accuracy improvement from fine-tuning
4. **Emphasize Local Processing**: Point out that all processing happens locally on the HP ZGX Nano
5. **Performance Metrics**: Show inference time comparisons between models

## Notes for Sales Teams

- Allow 3-5 minutes for initial model loading during setup so it's best to run this a few minutes before you have to present
- Test the demo before customer presentations
- Have sample PII text ready (the interface includes examples)
- The fine-tuned model will consistently outperform the base model
- Emphasize the local processing capability of the HP ZGX Nano
- All processing happens on-device, ensuring data privacy

## Further talking points around the Demo (What, Why and How)

### The Problem We Solve
- **Data Privacy Compliance Challenge**: Enterprises struggle with protecting sensitive customer and employee data (PII/PHI) in documents, emails, and databases while maintaining operational efficiency
- **Cloud Dependency Risks**: Current AI solutions require sending sensitive data to cloud providers for processing, creating potential security vulnerabilities and compliance violations
- **Generic AI Limitations**: Off-the-shelf AI models fail to accurately identify and mask PII consistently, leading to data leaks and regulatory exposure
- **GDPR/HIPAA Compliance**: Organizations face millions in potential fines for mishandling personal data, with regulations becoming increasingly strict globally

### How We Solve It
- **100% Local Processing**: The HP ZGX Nano runs AI models entirely on-premises, ensuring sensitive data never leaves your infrastructure
- **Custom Fine-tuning Capability**: Transform generic AI models into specialized PII detection systems tailored to your specific data formats and requirements
- **Enterprise-Grade Performance**: Process documents with 32B+ parameter models locally, achieving cloud-level accuracy without cloud-level risks
- **Dramatic Accuracy Improvement**: Fine-tuned models show 6x better PII detection accuracy compared to base models on average
- **Real-time Processing**: Instant PII masking without network latency or API rate limits

### Why It's Important
- **Data Sovereignty**: Complete control over where your data lives and how it's processed - critical for regulated industries like healthcare, finance, and government
- **Regulatory Compliance**: Meet stringent data protection requirements (GDPR, HIPAA, CCPA) by ensuring sensitive data never leaves your secure environment
- **Competitive Advantage**: Process sensitive customer data faster than competitors relying on cloud services, without security review delays
- **Cost Predictability**: Eliminate recurring cloud AI API costs - one-time hardware investment with unlimited processing
- **Trust and Reputation**: Demonstrate to customers that their data privacy is paramount by processing everything locally
- **Immediate ROI**: Reduce data breach risk while accelerating AI adoption across the enterprise

### Key Differentiator Message
"While competitors send your sensitive data to the cloud for AI processing, the HP ZGX Nano brings enterprise AI to your data - delivering the same powerful capabilities without ever compromising security or compliance."

## Support
For issues or questions about this demo, please contact Curtis Burkhalter at curtis.burkhalter@hp.com

## Targeting Different GPUs

The Dockerfile defaults to Blackwell architecture (compute capability 120). To target a different GPU, edit the `CMAKE_CUDA_ARCHITECTURES` value in the Dockerfile:

| GPU | Architecture | Value |
|-----|-------------|-------|
| GB10 / GB200 (Blackwell) | sm_120 | `120` |
| RTX 4090 / L40 (Ada Lovelace) | sm_89 | `89` |
| A100 (Ampere) | sm_80 | `80` |

## Troubleshooting

**"Models not found" error**: Verify the `models/` directory contains both `.gguf` files and that the volume mount in `docker-compose.yml` points to the correct location.

**GPU not detected in container**: Run `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` to verify the NVIDIA Container Toolkit is working.

**Build fails on llama-cpp-python**: This usually means the CUDA architecture doesn't match your GPU. Update `CMAKE_CUDA_ARCHITECTURES` in the Dockerfile.

**Port already in use**: `docker compose down` first, or change the port mapping in `docker-compose.yml` (e.g., `"9000:8000"`).

**Wrong IP in startup URLs**: Override with `HOST_IP=your.ip.here ./start.sh`

## Support

Curtis Burkhalter — curtis.burkhalter@hp.com
