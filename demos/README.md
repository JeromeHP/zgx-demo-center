# HP ZGX Demo Center

A multi-demo orchestration tool for the HP ZGX Nano (and similar GB10 / Blackwell ARM systems).

## What this is

Demo Center is a launcher that lets an HP SE walk a customer through several AI demos on a single ZGX Nano without leaving the browser. It boots a static HTML launcher on port 8085 and a small install/launch API on port 8086. Demos themselves run on their own ports (see below) — the launcher knows how to install, start, and link to each one.

All demo source code is vendored directly into the kit. There are no submodules and no runtime clones from external GitHub — once you unzip the bundle, everything you need is on disk.

## Prerequisites

You need an **HP ZGX Nano** (GB10 Grace-Blackwell) — or an NVIDIA DGX Spark, which is the same chip family (SM 12.1 ARM64) — **or an AMD ROCm box** (Strix Halo / Ryzen AI Max, gfx1151). The installer detects which and installs the matching variant; all demos but the Robotic Medical one run on both. See `../docs/HARDWARE_PROFILES.md`. Most of what the installer needs already ships on the machine. The two groups below tell you what's already there versus what you might have to add.

> The installer provisions the Python/conda environment and the AI models for you — you do **not** set those up by hand. (Older docs listed "Ollama running on localhost:11434" and a "conda env named zgx" as prerequisites; they are not. `setup-prereqs.sh` installs Ollama; `install.sh` creates the Python 3.12 / miniforge `zgx` environment and pulls the models during the install run.)

> **Before anything else, complete the first-boot system update (OOBE).** On a freshly reimaged Nano, finish the out-of-box setup / first-boot Ubuntu system update and let it settle before running `setup-prereqs.sh` — otherwise apt repos and snap can be mid-update and the prereq install will fight it.

> **One-time host setup:** `setup-prereqs.sh` (run with `sudo`, then **reboot**) installs the system-level prerequisites — Docker + Compose, the NVIDIA Container Toolkit, **Ollama**, Chromium, and the fullscreen helpers `wmctrl`/`xdotool` — and adds you to the `docker` and `ollama` groups. **Factory image contents vary by build, so always run it** — it detects what's already present and installs only what's missing (idempotent, safe to re-run). `install.sh`'s preflight then **checks** all of them and stops early (pointing you at `setup-prereqs.sh`) if any are missing. `SETUP_COMMANDS.txt` lists the same steps by hand.

### A. Host prerequisites — `setup-prereqs.sh` detects and installs only what's missing

**Do not assume the factory image already has these.** Image contents vary by build — some ZGX Nano images ship Docker and friends, others don't — so **always run `sudo bash setup-prereqs.sh` first** (then reboot). It checks each component and installs only the missing ones, then prints a summary of what was already present vs. installed. `install.sh`'s preflight independently verifies them all and **stops with a clear message naming anything still missing**. (The OS and GPU driver/CUDA come from the IT-provided factory image, not from `setup-prereqs.sh` — if those are missing, contact IT.)

| What it is | Check it's there | If it's missing |
|------------|------------------|-----------------|
| **Ubuntu 24.04 (ARM64)** — the operating system the Nano runs. | `lsb_release -a` | Reimage with the ZGX OS / DGX OS image (IT). |
| **NVIDIA GPU driver + CUDA 13.0** — lets the demos use the Nano's GPU. | `nvidia-smi` | Install the ZGX OS GPU driver stack (IT). |
| **Docker Engine** — runs the containerized demos. | `docker --version` | `sudo bash setup-prereqs.sh` (then reboot). |
| **Docker Compose** — starts the multi-container demos. | `docker compose version` | `sudo bash setup-prereqs.sh` (then reboot). |
| **NVIDIA Container Toolkit** — gives the Docker demos GPU access. | `nvidia-ctk --version` | `sudo bash setup-prereqs.sh` (then reboot). |
| **Ollama** — the local LLM runtime the demos pull their models through. | `ollama --version` | `sudo bash setup-prereqs.sh` (then reboot). |
| **Chromium** — Demo Center opens fullscreen in Chromium. | `chromium --version` (or `snap list chromium`) | `sudo bash setup-prereqs.sh` (then reboot). |
| **wmctrl** — forces the launcher window fullscreen. | `wmctrl -m` | `sudo bash setup-prereqs.sh` (then reboot). |

### B. You may need to install this

- **Disk space — about 150 GB free.** The install downloads a 42 GB AI model plus several demo containers.
  - Check free space in your home folder: `df -h ~`
- **Memory — 128 GB unified (LPDDR5x).** This is built into the ZGX Nano; some demos load 70B-class models that need it. Nothing to install — just be aware the demos assume a full Nano.

## Quick start (HP reps — the bundle)

You install Demo Center from a single download. No git, no tokens, nothing to clone.

1. **Download the bundle.** In a web browser on the ZGX Nano, download `zgx-demo-center-bundle.zip` from **`<your HP download link>`** (ask your team lead for the current link). It's about 110 MB.
2. **Unzip it in your home folder** so it creates a `~/demos` folder:
   ```bash
   cd ~ && unzip ~/Downloads/zgx-demo-center-bundle.zip
   ```
   (If you saved it somewhere other than `Downloads`, point at that location instead.)
3. **Complete the first-boot system update (OOBE), then install the host prerequisites, then REBOOT.** Finish the freshly-reimaged Nano's first-boot Ubuntu update first. Then this one-time step installs Docker + the NVIDIA Container Toolkit, Ollama, Chromium, and the fullscreen helpers (`wmctrl`/`xdotool`), and adds you to the `docker` and `ollama` groups:
   ```bash
   sudo bash ~/demos/setup-prereqs.sh
   sudo reboot
   ```
   The reboot is required — the `docker`/`ollama` group memberships only take effect on a fresh login. (If you can't run the script, `SETUP_COMMANDS.txt` has the same steps as copy-paste commands.) The script is idempotent: if a prerequisite is already present it just confirms it, so it's safe to re-run — and it prints a summary of what was already present vs. what it installed.
4. **Run the installer:**
   ```bash
   cd ~/demos && ./install.sh
   ```
5. **Wait — the first install takes a couple of hours.** It downloads a 42 GB AI model, registers the per-demo models with Ollama, builds the vision engine (~15–20 minutes — this is normal, not a hang), and builds the demo containers. It runs **unattended**, so once it's going you can walk away. (If anything required is missing, preflight stops early with a clear message — pointing you back at step 3 — before the long downloads start.)
6. **Launch it.** When the install finishes, **double-click the "HP ZGX Demo Center" icon on the desktop**. It brings the demos up and opens fullscreen in Chromium (forced fullscreen via `wmctrl`).

### Developer path — running the services without a full install

The bundle above is the **only** way the kit is distributed. This section is for someone who already has the files on the box and wants to start the launcher directly without running the full install.

**The bundle does not have to be unzipped in `$HOME`.** If you unzipped it somewhere else, run the launcher from wherever it landed — `install.sh` and `run_demos.sh` both point `~/demos` at that location on first run:

```bash
cd /wherever/you/unzipped/demos
bash run_demos.sh          # links ~/demos → this copy, then starts :8085/:8086
```

`~/demos` is a fixed convention, not an accident: demos contributed to the kit hardcode `$HOME/demos/<demo>/...` in their launch scripts and Modelfiles, so the link is what lets the files live anywhere. If `~/demos` already exists and points somewhere else, the scripts **stop with an explanation rather than overwrite it** — they never delete a real directory. See `ensure_demos_link.sh`.

Then open `http://<host>:8085/zgx-demo-launcher.html` in a browser. The launcher will detect which demos are installed, which need their models pulled, and which are ready to launch. (Starting the services this way skips the install phases, so the native assets — `yolov8s.onnx`, `zgx_videos/` — won't be staged; run `./install.sh` for that.)

## Architecture

| Port | Service | Source |
|------|---------|--------|
| 8085 | Demo Center launcher (static HTML, served by `python3 -m http.server`) | `zgx-demo-launcher.html` |
| 8086 | Install / launch API (FastAPI + uvicorn, SSE-streamed) | `demo_install_server.py` |
| 8088 | Pipeline Pulse (native, no Docker) | `pipeline-pulse/` |
| 8089 | Spectrum Duel (native, no Docker) | `ew-spectrum-duel/` |
| 8094 | AI Vision (native, no Docker) | `ai-vision/` |
| 8095 | Grant Machine (native, no Docker) | `grant-machine/` |
| 8097 | Port-of-Entry Vehicle Inspection | `automotive-vehicle-inspection/` |
| 8000 | Maritime, Pharmacy, or PII (one at a time — they share the port by design) | their subdirs |
| 8093 | NAWSCL Aircraft Classification | `ballistics-containerized-docker/` |
| 8090 | Robotic Medical Triage | `Robotic-medical-AI-assistant/` |
| 8091 | Construction Safety | `AEC-construction-safety/` |
| 8092 | Document Translation | `document-translation/` |

Both port-8085 and port-8086 services are started by `run_demos.sh` and run in the background. Stop them with `pkill -f "demo_install_server\|http.server"`.

## Included demos

| Demo | Stack | Port | Notes |
|------|-------|------|-------|
| Pipeline Pulse | Native FastAPI + Ollama | 8088 | Live oil-rig dashboard with LLM-written incident briefs (llama3.1:70b). Real GPU telemetry from pynvml. |
| Spectrum Duel | Native FastAPI + WebSocket + TensorRT + Ollama | 8089 | AI jammer vs. AI radar over a synthetic X-band spectrum. 5-class jam classifier (TensorRT FP16, trained on-device at install), reactive counter-EW agents, llama3.1:70b EW briefs. |
| AI Vision | Native FastAPI + WebSocket + TensorRT | 8094 | Real-time multi-camera object detection. YOLOv8s FP16 on GB10. LLM alert engine (llama3.1:70b). |
| Grant Machine | Native FastAPI + SSE + Ollama | 8095 | Agentic grant work in four modes: **Discover** (searches the live federal catalogue for grants matching your uploaded documents), **Explore** (extracts requirements from any uploaded solicitation), **Story** (the built-in NSF 24-584 scenario), **Grade** (scores a proposal you already wrote and cross-checks its figures). Uploaded documents never leave the device; every outbound term is validated and audited. qwen3.6:35b. Optional SearXNG on 8096 for web search. |
| Maritime Surveillance | Docker, vLLM | 8000 | Aerial-imagery vessel intel via Qwen3-VL 8B FP8. |
| Clinical Drug Interaction | Docker, Ollama | 8000 | Mixtral-8x7B with RAG over a 302K-record DDI database (ChromaDB). |
| PII Detection & Masking | Docker, Ollama | 8000 | Side-by-side base Qwen2.5-32B vs. fine-tuned Qwen2.5-32B PII detector (~19GB base + ~20GB fine-tuned). Also supports Word/PDF upload with a downloadable redacted document. |
| NAWSCL Aircraft Classification | Docker, Ollama | 8093 | Aircraft classification (manned/unmanned) for range safety via qwen2.5vl:7b. |
| Port-of-Entry Vehicle Inspection | Docker, Ollama | 8097 | A simulated RoRo vessel discharges Stellantis vehicles through a multi-camera portal: 2–4 views per vehicle, triaged live in one call and routed to Cleared / Needs inspection / Hold, with an inspector queue that shows which camera caught the damage, and a CSV report. Mistral Small 3.2 (24B). 20 vehicles in ~2:45 live, plus a labelled 57-vehicle replay. |
| Construction Safety | Docker, voice-enabled | 8091 | OSHA Q&A with offline fallback for job-site use. |
| Document Translation | Python venv, NLLB-200 | 8092 | 54 language pairs, fully on-device. |
| Robotic Medical Triage | Docker, vLLM + Whisper + Piper | 8090 | Voice triage. Requires a Pollen Robotics Reachy Mini on the same network. |

## First-run notes

- The first time you install a Docker-based demo, the launcher will pull the image and download the model weights. Maritime is ~9 GB, Pharmacy ~28 GB, Robotic ~4 GB. Plan accordingly.
- Model files (`*.gguf`, `*.safetensors`, `*.onnx`, `*.engine`, `*.trt`, `models/` directories) are gitignored. They are not in this repo and never will be — they're downloaded at install time.
- **The installer pulls the global Ollama models for you.** Ollama itself is installed by `setup-prereqs.sh` (the host-prereq step); then during the install run (Phase 3), `install.sh` pulls `llama3.1:70b` and `nomic-embed-text` automatically — you do not run `ollama pull` by hand. (If you're on the developer path and want them ahead of time, the equivalent manual commands are `ollama pull llama3.1:70b` and `ollama pull nomic-embed-text`.)
- **Per-demo models are handled by each demo's install.** Container demos download their own weights when the launcher installs them (see sizes above). The `medical-mixtral` (Pharmacy) and PII's fine-tuned `pii-detector` are downloaded as GGUFs and **registered automatically** by their install steps via `ollama create -f Modelfile` (idempotent — skipped if already registered); PII's base comparison model (`qwen2.5:32b`) is pulled directly from the Ollama library. The global models, Ollama itself, and the conda env are all automated too — nothing model-related is done by hand.

## Constraints and known issues

- **Port 8000 is shared.** Maritime, Pharmacy, and PII all listen on 8000. Only one can be running at a time. The launcher is aware of this and will warn if you try to start a second.
- **GPU memory pressure.** llama3.1:70b alone occupies ~42 GB of unified memory. Before launching a vLLM-based demo (Maritime, Robotic), call `/ollama_release` from the launcher to evict resident Ollama models.
- **INT8 inference is broken on SM 12.1.** The TensorRT calibration scales are silently discarded and the engine falls back to FP16/FP32. Use FP16 baselines only.
- **TensorRT containers 24.12 and 25.02 fail on GB10.** Use `nvcr.io/nvidia/tensorrt:25.05-py3` for any engine builds. Native engines built with host TRT 10.15.1 are not version-compatible with engines built inside the container — pick one and stick with it per demo.

## Credits

Curtis Burkhalter authored the original versions of Maritime, Pharmacy, PII, Ballistics, Construction, Translation, Robotic Triage, and Vehicle Inspection. The versions vendored in this repo include modifications for ZGX Nano / GB10 compatibility:

- Pharmacy: ported from llama-cpp-python (broken on SM 12.1) to Ollama, with a 302K-row DDI ChromaDB
- PII: ported from llama-cpp-python (broken on SM 12.1) to Ollama, with a size-matched base-vs-finetuned comparison and Word/PDF upload + redacted-document download
- Ballistics: vendored from `Fed-air-ballistics-identification-containerized`, ported to a thin Ollama client (same pattern as Pharmacy and PII)
- Construction: port remapped from 8080 to 8091 to avoid launcher conflicts
- Translation: port remapped from 8080 to 8092 for the same reason
- Vehicle Inspection: vendored from `automotive-vehicle-inspection-demo` @ `61f7c46`, ported from in-container vLLM to a thin host-Ollama client (same pattern as Ballistics), port remapped from 8095 (Grant Machine's) to 8097, rebranded from Mercedes-Benz to Stellantis
- All Docker demos: dynamic-IP detection so they bind to whichever interface the Nano is currently using

Pipeline Pulse, AI Vision and Spectrum Duel are native (non-Docker) demos developed in this repo.

## Stability and updates

**Updating means re-downloading the latest `zgx-demo-center-bundle.zip`** from the same link and re-running `./install.sh` — it's safe to re-run, and skips anything already installed. Unzip the new bundle over the old location, then:

```bash
cd ~/demos
./install.sh        # re-stages assets and installs anything new
bash run_demos.sh   # restart launcher + install server
```

The kit is released and tested as a unit — a new bundle may change multiple demos at once. If a demo fails to start after an update, check that its model weights still exist under its `models/` directory; if they were cleared, re-run install for that demo from the launcher.
