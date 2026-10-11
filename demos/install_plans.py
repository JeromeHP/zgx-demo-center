#!/usr/bin/env python3
"""
HP ZGX Nano — Demo Center install plans (shared data module).

Single source of truth for the per-demo install/launch/teardown/discovery
metadata. Imported by both demo_install_server.py (the SSE install server)
and install.py (the install orchestrator) so they never diverge.

This module is PURE DATA — it defines no behavior and intentionally imports
nothing from demo_install_server.py (avoids a circular import).

Two optional keys carry hardware requirements. Both are read with .get(), so
every existing consumer ignores them:

  "requires": {"profiles": [...]}   accelerator profiles this demo can run on
                                    (zgx_hardware.detect()["profile"]). Absent
                                    means "runs anywhere". A demo whose
                                    requirement is unmet is SKIPPED with a
                                    reason by install.py rather than attempted
                                    and failed -- on an AMD box the nvcr.io
                                    NVIDIA containers cannot run at all, and
                                    reporting that as a failure buries the
                                    demos that did install.

  "profiles": {"<profile>": {...}}  per-profile overrides merged over the base
                                    plan (currently "steps").
"""

# ── Install sequences per demo ─────────────────────────────────────────────
INSTALL_PLANS = {
    "maritime": {
        "name": "Maritime Surveillance Intelligence",
        "dir": "Fed-Navy-containerized-docker",
        "steps": [
            {"label": "Source vendored — no clone needed", "cmd": "echo 'Source vendored in monorepo — skipping upstream clone.'"},
            {"label": "Downloading model (~9GB — Qwen3-VL-8B FP8)", "cmd": "cd {dir} && bash ./download_models.sh",
             "heartbeat": {"label": "maritime model", "path": "{dir}/models/Qwen3-VL-8B-Instruct-FP8", "total_gb": 11}},
            {"label": "Building Docker container", "cmd": "cd {dir} && docker compose build"},
        ],
        "launch_cmd": "cd {dir} && bash ./start.sh -d",
        "port": 8000,
        "health_url": "http://localhost:8000/api/health",
        "teardown_strategy": {"type": "docker_compose"},
        "discovery": {"type": "docker_container", "patterns": ["maritime", "navy"]},
        # Explicit, because the fallback in /status is actively wrong here: it
        # checks `any(models_dir.iterdir())`, and the compose bind mount
        # (./models:/models:ro) makes Docker CREATE an empty models/ directory
        # on first launch. On AMD -- where the FP8 checkpoint is deliberately
        # never downloaded -- that empty directory pinned the card at
        # "Needs Setup" permanently, no matter how many times it was installed.
        "readiness_check": {
            "files": [
                "~/demos/Fed-Navy-containerized-docker/backend/main.py",
                "~/demos/Fed-Navy-containerized-docker/models/Qwen3-VL-8B-Instruct-FP8",
            ],
            # vLLM serves the checkpoint in-container; nothing is pulled into Ollama.
            "ollama_model": "",
            "path": "~/demos/Fed-Navy-containerized-docker",
        },
        # Runs anywhere. The default stack is nvcr.io/nvidia/vllm serving an
        # FP8 checkpoint, which is NVIDIA-only twice over (NGC image, and no
        # FP8 path on gfx1151). On other profiles start.sh selects
        # docker-compose.amd.yml, which builds a slim multi-arch image and
        # points backend/main.py at host Ollama -- it already speaks plain
        # OpenAI-compatible /chat/completions, so no application code changes
        # and nothing in the container needs a GPU.
        "profiles": {
            "amd-rocm": {"steps": [
                {"label": "Source vendored — no clone needed", "cmd": "echo 'Source vendored in monorepo — skipping upstream clone.'"},
                # No FP8 download: that checkpoint cannot be served on ROCm.
                {"label": "Pulling vision model (~6GB — qwen2.5vl:7b, shared with the ballistics demo)",
                 "cmd": "ollama pull qwen2.5vl:7b"},
                {"label": "Building Docker container (slim image, external endpoint)",
                 "cmd": "cd {dir} && docker compose -f docker-compose.yml -f docker-compose.amd.yml build"},
            ],
            # No FP8 checkpoint on this path -- the vision model lives in host
            # Ollama instead, so that is what readiness must actually check.
            "readiness_check": {
                "files": ["~/demos/Fed-Navy-containerized-docker/backend/main.py"],
                "ollama_model": "qwen2.5vl:7b",
                "path": "~/demos/Fed-Navy-containerized-docker",
            },
            },
        },
    },
    "medical": {
        "name": "Robotic Medical Triage Voice Agent",
        "dir": "Robotic-medical-AI-assistant",
        "steps": [
            {"label": "Source vendored — no clone needed", "cmd": "echo 'Source vendored in monorepo — skipping upstream clone.'"},
            {"label": "Downloading models (~4GB — Llama 3.1 8B AWQ + Piper TTS)", "cmd": "cd {dir} && bash ./scripts/download_models.sh 2>/dev/null || python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', local_dir='models/Llama-3.1-8B-Instruct-AWQ-INT4')\"",
             "heartbeat": {"label": "medical models", "path": "{dir}/models", "total_gb": 4}},
            {"label": "Building Docker image (~10 min — compiles whisper.cpp CUDA)", "cmd": "cd {dir}/docker && docker build --no-cache -t consent-agent:latest -f Dockerfile.api ."},
        ],
        "launch_cmd": "cd {dir}/scripts && bash ./start_services.sh",
        "port": 8090,
        "note": "Requires Pollen Reachy Mini robot on same network.",
        "health_url": "http://localhost:8090/health",
        "teardown_strategy": {"type": "docker_compose"},
        "discovery": {"type": "docker_container", "patterns": ["consent", "medical"]},
        # nvcr.io/nvidia/vllm, awq_marlin (a CUDA-only kernel) and a whisper.cpp
        # build pinned to CMAKE_CUDA_ARCHITECTURES=121. Out of scope for AMD.
        "requires": {"profiles": ["nvidia-cuda"]},
    },
    "pharmacy": {
        "name": "Clinical Drug Interaction Analysis",
        "dir": "healthcare-pharmacy-containerized-docker",
        "steps": [
            {"label": "Source vendored — no clone needed", "cmd": "echo 'Source vendored in monorepo — skipping upstream clone.'"},
            {"label": "Downloading model (~28GB — Mixtral-8x7B Q4)", "cmd": "cd {dir} && bash ./download_models.sh",
             "heartbeat": {"label": "pharmacy model", "path": "{dir}/models", "total_gb": 28}},
            # Register the downloaded GGUF with Ollama as 'medical-mixtral' (host-level;
            # the backend reads OLLAMA_MODEL=medical-mixtral from host Ollama). Idempotent:
            # skipped if 'ollama list' already shows it. cd {dir} so the Modelfile's
            # relative `FROM ./models/medical-ft-mixtral-q4` resolves.
            {"label": "Registering Ollama model (medical-mixtral)", "cmd": "ollama list | grep -q '^medical-mixtral' || (cd {dir} && ollama create medical-mixtral -f Modelfile.medical)"},
            {"label": "Building Docker image (~15 min — compiles llama.cpp CUDA sm_120)", "cmd": "cd {dir} && docker compose build"},
        ],
        "launch_cmd": "cd {dir} && bash ./start.sh -d",
        "port": 8000,
        "health_url": "http://localhost:8000/api/health",
        "teardown_strategy": {"type": "docker_compose"},
        "discovery": {"type": "docker_container", "patterns": ["pharmacy", "healthcare", "drug"]},
    },
    "pii": {
        "name": "PII Detection & Masking",
        "dir": "pii-demo-containerized-docker",
        "steps": [
            # Download the fine-tuned detector GGUF via the in-repo helper (correct
            # $HOME/demos path). Use the zgx env python (has huggingface_hub from
            # requirements.txt). snapshot_download is itself idempotent.
            {"label": "Downloading fine-tuned PII detector GGUF (~20GB)", "cmd": "cd {dir} && $HOME/miniforge3/envs/zgx/bin/python3 _dl_models.py",
             "heartbeat": {"label": "PII models", "path": "{dir}/models", "total_gb": 20}},
            # Register the fine-tuned model with Ollama (host-level; backend reads
            # BASE/FINETUNED model names from host Ollama). Idempotent guard; cd {dir}
            # so the Modelfile's relative `FROM ./models/*.gguf` resolves.
            {"label": "Registering Ollama model (pii-detector)", "cmd": "ollama list | grep -q '^pii-detector' || (cd {dir} && ollama create pii-detector -f Modelfile.pii-detector)"},
            # Base comparison model is a stock Ollama-library model (size-matched to
            # pii-detector — see backend/main.py's module docstring for why).
            {"label": "Pulling base comparison model (~19-20GB — qwen2.5:32b)", "cmd": "ollama pull qwen2.5:32b"},
            # This step was previously missing entirely, which meant `docker compose
            # up -d` in launch_cmd below just kept reusing whatever image happened to
            # already exist (e.g. a stale build from months earlier) instead of
            # picking up backend code changes — silently, since `up` without `--build`
            # never rebuilds. Always rebuild before launch.
            {"label": "Building Docker container", "cmd": "cd {dir} && docker compose build"},
        ],
        "launch_cmd": "cd {dir} && docker compose up -d",
        "port": 8000,
        "extra_ports": [8083],   # secondary port the launcher already tracks
        "port_path": "/app/",
        "health_url": "http://localhost:8000/api/health",
        "teardown_strategy": {"type": "docker_compose"},
        "discovery": {"type": "docker_container", "patterns": ["pii"]},
    },
    "ballistics": {
        "name": "NAWSCL Aircraft Classification",
        "dir": "ballistics-containerized-docker",
        "steps": [
            {"label": "Pulling vision model (~6GB — qwen2.5vl:7b)", "cmd": "ollama pull qwen2.5vl:7b"},
            {"label": "Building Docker container", "cmd": "cd {dir} && docker compose build"},
        ],
        "launch_cmd": "cd {dir} && bash ./start.sh -d",
        "port": 8093,
        "port_path": "/",
        "health_url": "http://localhost:8093/api/health",
        "teardown_strategy": {"type": "docker_compose"},
        "discovery": {"type": "docker_container", "patterns": ["ballistics"]},
    },
    "spectrum": {
        "name": "Spectrum Duel — AI Electronic Warfare",
        "dir": "ew-spectrum-duel",
        # Native demo: runs under the zgx env python (see start_demo_8089.sh), so
        # its imports must be installed INTO that env. ew_engine/ew_infer need
        # numpy + pycuda + tensorrt; ew_hw.py needs pynvml; ew_demo.py needs
        # httpx. Pins match ai-vision's clean-box validated set deliberately —
        # both native TRT demos share one env, so they must not drift apart.
        #
        # The third step is the interesting one: it TRAINS the jam classifier
        # from scratch on this box (pure numpy, no torch, no ONNX) and then
        # builds the FP16 engine straight through the TensorRT network-definition
        # API. Because it runs under the SAME env python as the runtime, the
        # build-TRT == runtime-TRT invariant holds by construction. Measured on
        # GB10: ~4-6 min cold (a 2-channel net over a training set that now
        # includes straddling tiles), under 2 s warm. Idempotent on TWO tests:
        # the engine must still deserialize under the current TensorRT AND the
        # weights must carry the current ew_train.DATA_VERSION, so a physics or
        # tile-format change forces a retrain instead of silently keeping a
        # model trained on the old world.
        "steps": [
            {"label": "Installing EW deps into zgx env (numpy, pycuda, tensorrt, pynvml, httpx — pycuda compiles from source on aarch64; slow is normal)", "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -m pip install --timeout 60 --retries 10 --extra-index-url https://pypi.nvidia.com numpy==2.4.6 pycuda==2026.1 tensorrt-cu13==10.15.1.29 pynvml httpx"},
            {"label": "Verifying EW imports (numpy, pycuda, tensorrt, pynvml, httpx)", "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -c 'import numpy, pycuda.driver, tensorrt, pynvml, httpx; print(\"ew imports OK — numpy\", numpy.__version__, \"| tensorrt\", tensorrt.__version__)'"},
            {"label": "Training jam classifier on-device + building FP16 TensorRT engine (~4–6 min)", "cmd": "cd {dir} && $HOME/miniforge3/envs/zgx/bin/python3 ew_train.py"},
        ],
        "launch_cmd": "bash $HOME/demos/ew-spectrum-duel/start_demo_8089.sh",
        "port": 8089,
        "port_path": "/",
        "health_url": "http://localhost:8089/api/status",
        "log_file": "ew_spectrum_demo.log",
        # process_pattern (not port_kill): same load-race as AI Vision and
        # Pipeline Pulse — ew_demo.py deserializes the TRT engine and warms
        # llama3.1:70b before it binds :8089, so an Exit-while-loading would
        # orphan a process a port-kill cannot see. The name is unique; port 8089
        # is still added as a fallthrough by both teardown paths.
        "teardown_strategy": {"type": "process_pattern", "patterns": ["ew_demo.py"]},
        "discovery": {"type": "port_probe"},
        # Runs anywhere: ew_infer.py tiers TensorRT -> numpy -> heuristic and
        # labels the live tier in the UI, so a box without CUDA degrades
        # honestly instead of failing.
        "profiles": {
            "amd-rocm": {"steps": [
                {"label": "Installing EW deps into zgx env (numpy, httpx — no TensorRT on ROCm)",
                 "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -m pip install --timeout 60 --retries 10 numpy==2.4.6 httpx"},
                {"label": "Verifying EW imports (numpy, httpx)",
                 "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -c 'import numpy, httpx; print(\"ew imports OK — numpy\", numpy.__version__)'"},
                {"label": "Training jam classifier on-device (numpy tier — no TensorRT engine built)",
                 "cmd": "cd {dir} && EW_FORCE_CPU=1 $HOME/miniforge3/envs/zgx/bin/python3 ew_train.py"},
            ]},
        },
        "readiness_check": {
            "files": [
                "~/demos/ew-spectrum-duel/ew_demo.py",
                "~/demos/ew-spectrum-duel/ew_demo.html",
                "~/demos/ew-spectrum-duel/models/jammer_cls.trt",
                "~/demos/ew-spectrum-duel/models/jammer_cls.npz",
            ],
            "ollama_model": "llama3.1:70b",
            "path": "~/demos/ew-spectrum-duel",
        },
    },
    "construction": {
        "name": "Construction Safety Assistant",
        "dir": "AEC-construction-safety",
        "steps": [
            {"label": "Source vendored — no clone needed", "cmd": "echo 'Source vendored in monorepo — skipping upstream clone.'"},
            {"label": "Installing Python dependencies", "cmd": "cd {dir} && bash ./install.sh"},
        ],
        "launch_cmd": "bash $HOME/demos/AEC-construction-safety/start_demo_8091.sh",
        "port": 8091,
        # The back end on 8191, NOT the 8091 static file server: that one binds in
        # about a second while the model is still loading, so checking it reported
        # "ready" for a demo that could not yet answer a question.
        "health_url": "http://localhost:8191/",
        "log_file": None,
        "extra_ports": [8191],   # back-end API; teardown must free it too
        "teardown_strategy": {"type": "port_kill"},
        "discovery": {"type": "port_probe"},
    },
    "translation": {
        "name": "Private Document Translation",
        "dir": "document-translation",
        "steps": [
            {"label": "Source vendored — no clone needed", "cmd": "echo 'Source vendored in monorepo — skipping upstream clone.'"},
            {"label": "Creating Python virtual environment", "cmd": "cd {dir} && python3 -m venv document-translate-env"},
            {"label": "Installing Python dependencies", "cmd": "cd {dir} && bash ./install.sh"},
            # Fix 5: pre-fetch the translation model at INSTALL time so the demo has
            # NO demo-time internet dependency (the old behavior downloaded ~1.2GB on
            # the first 'Load Models' click — bad on a locked-down CWC network). The
            # backend (backend/main.py) loads facebook/nllb-200-distilled-600M via
            # transformers' from_pretrained() with NO cache_dir and the launcher sets
            # no HF_HOME, so it reads the DEFAULT HuggingFace cache
            # (~/.cache/huggingface/hub). We pre-fetch with the DEMO'S OWN venv python
            # (same huggingface_hub + same default cache resolution as the backend), so
            # the bytes land exactly where 'Load Models' later looks. snapshot_download
            # is idempotent — a re-run is a no-op once cached.
            {"label": "Pre-fetching translation model (~1.2GB — facebook/nllb-200-distilled-600M)",
             "cmd": "cd {dir} && ./document-translate-env/bin/python -c \"from huggingface_hub import snapshot_download; snapshot_download('facebook/nllb-200-distilled-600M')\"",
             "heartbeat": {"label": "translation model", "path": "~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M", "total_gb": 1.2}},
        ],
        "launch_cmd": "bash $HOME/demos/document-translation/start_demo_8092.sh",
        "port": 8092,
        "note": "Model (facebook/nllb-200-distilled-600M, ~1.2GB) is pre-fetched into the HuggingFace cache at install time; the UI 'Load Models' button loads it from disk (no download).",
        # Back end on 8192, not the 8092 static file server — see construction.
        "health_url": "http://localhost:8192/",
        "log_file": None,
        "extra_ports": [8192],   # back-end API; teardown must free it too
        "teardown_strategy": {"type": "port_kill"},
        "discovery": {"type": "port_probe"},
    },
    "pipeline": {
        "name": "Pipeline Pulse — O&G AI Monitoring",
        "dir": "pipeline-pulse",
        # Native demo: runs under the zgx env python (see start_demo_8088.sh), so its
        # imports must be installed INTO that env. pipeline_pulse_v6.py imports httpx
        # (already a global zgx dep) + pynvml (GPU telemetry). Installed here so the
        # orchestrator/launcher provision them; --timeout/--retries ride out slow
        # aarch64 wheel fetches. Then verify the imports actually resolve in-env.
        "steps": [
            {"label": "Installing pipeline deps into zgx env (httpx, pynvml)", "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -m pip install --timeout 60 --retries 10 httpx pynvml"},
            {"label": "Verifying pipeline imports (httpx, pynvml)", "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -c 'import httpx, pynvml; print(\"pipeline imports OK\")'"},
        ],
        "launch_cmd": "bash $HOME/demos/pipeline-pulse/start_demo_8088.sh",
        "port": 8088,
        # A real HTTP check, not a bare TCP connect: the app binds :8088 only
        # after its Ollama warm-up, and /api/status is the cheapest 200 it serves.
        "health_url": "http://localhost:8088/api/status",
        # start_demo_8088.sh redirects the demo's own stdout/stderr here, so this
        # is the only place a startup crash is recorded. /demo_log serves it.
        "log_file": "pipeline_pulse_demo.log",
        # process_pattern (not port_kill): same architecture/load-race as AI Vision
        # — a single nohup python (pipeline_pulse_v6.py) that warms up llama3.1:70b
        # before binding :8088, so Exit-while-loading could orphan it. The name is
        # unique, so pkill-by-name is safe; port: 8088 stays as fallthrough.
        "teardown_strategy": {"type": "process_pattern", "patterns": ["pipeline_pulse_v6.py"]},
        "discovery": {"type": "port_probe"},
        "readiness_check": {
            "files": [
                "~/demos/pipeline-pulse/pipeline_pulse_v6.py",
                "~/demos/pipeline-pulse/pipeline_pulse_v6.html",
            ],
            "ollama_model": "llama3.1:70b",
            "path": "~/demos/pipeline-pulse",
        },
    },
    "vision": {
        "name": "AI Vision — Real-Time Object Detection",
        "dir": "ai-vision",
        # Native demo: runs under the zgx env python (see start_demo_8094.sh), so its
        # imports must be installed INTO that env. vision_engine.py imports cv2 +
        # numpy + pycuda + tensorrt; vision_hw.py uses pynvml. Pins match the clean-box
        # validated set; tensorrt-cu13 comes from NVIDIA's index. opencv is the HEADLESS
        # build only (no GUI/X deps). pycuda compiles from source on aarch64 — SLOW is
        # NORMAL, hence --timeout 60 --retries 10. The engine itself is built AFTER this
        # step (install.py Phase 4b) using THIS env's tensorrt so build==runtime TRT.
        "steps": [
            {"label": "Installing vision deps into zgx env (opencv-headless, numpy, pycuda, tensorrt — pycuda compiles from source on aarch64; slow is normal)", "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -m pip install --timeout 60 --retries 10 --extra-index-url https://pypi.nvidia.com opencv-python-headless==4.13.0.92 numpy==2.4.6 pycuda==2026.1 tensorrt-cu13==10.15.1.29 pynvml"},
            {"label": "Verifying vision imports (cv2, numpy, pycuda, tensorrt, pynvml)", "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -c 'import cv2, numpy, pycuda.driver, tensorrt, pynvml; print(\"vision imports OK — cv2\", cv2.__version__, \"| numpy\", numpy.__version__, \"| tensorrt\", tensorrt.__version__)'"},
        ],
        "launch_cmd": "bash $HOME/demos/ai-vision/start_demo_8094.sh",
        "port": 8094,
        "health_url": "http://localhost:8094/api/status",
        "log_file": "vision_demo.log",
        # process_pattern (not port_kill): the backend loads a TRT engine + 70B
        # model for ~tens of seconds before it binds :8094. A port-kill can't hit a
        # process that hasn't bound its port yet, so Exit-while-loading orphaned the
        # backend (held 1.4 GiB GPU). pkill-by-name catches it regardless of bind
        # state; the declared port: 8094 is still added as a fallthrough by both
        # teardown paths, so this is belt-and-suspenders.
        "teardown_strategy": {"type": "process_pattern", "patterns": ["vision_demo.py", "vision_frontend.py"]},
        "discovery": {"type": "port_probe"},
        # Runs anywhere: vision_backends.py picks TensorRT when an engine and a
        # CUDA context are available, else ONNX Runtime against the same
        # ~/yolov8s.onnx the engine is built from. The UI names the live tier.
        "profiles": {
            "amd-rocm": {
                "steps": [
                    # Split from one chained command. It was written as
                    #   A && B || C
                    # which, if A (opencv+numpy) failed, still ran C -- leaving the
                    # env with onnxruntime but no cv2, and vision_demo.py dying on
                    # `import cv2` with the pip failure long since scrolled away.
                    # Two steps, so a failure is attributable to the right one.
                    {"label": "Installing vision deps into zgx env (opencv-headless, numpy)",
                     "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -m pip install --timeout 60 --retries 10 opencv-python-headless==4.13.0.92 numpy==2.4.6"},
                    # The GPU-accelerated ONNX Runtime, from AMD's own wheel index.
                    #
                    # This was `pip install onnxruntime-rocm || pip install
                    # onnxruntime`, which on a stock pip always took the second
                    # branch: AMD does not publish to PyPI. The demo then ran
                    # detection on CPU while everything reported success.
                    #
                    # The script picks the index matching the installed ROCm and the
                    # right package for that line -- they rename at 7.1 -- and then
                    # checks what onnxruntime ACTUALLY reports, because a wheel can
                    # install cleanly and still leave you on CPU.
                    {"label": "Installing ONNX Runtime for ROCm (GPU detection provider)",
                     "cmd": "bash {dir}/../scripts/install_onnxruntime_rocm.sh"},
                    {"label": "Verifying vision imports + ONNX Runtime providers",
                     "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -c 'import cv2, numpy, onnxruntime as ort; print(\"vision imports OK — cv2\", cv2.__version__, \"| providers\", ort.get_available_providers())'"},
                ],
                # ~/yolov8s.trt is never built on ROCm (install.py Phase 4b skips
                # it deliberately); the ONNX it would have been built from is what
                # actually runs. Requiring the .trt here pinned the card at
                # "Needs Setup" no matter how many times it was installed.
                "readiness_check": {
                    "files": [
                        "~/yolov8s.onnx",
                        "~/zgx_videos/video1.mp4",
                        "~/demos/ai-vision/vision_demo.py",
                    ],
                    "ollama_model": "llama3.1:70b",
                    "path": "~/demos/ai-vision",
                },
            },
        },
        "readiness_check": {
            "files": [
                "~/yolov8s.trt",
                "~/zgx_videos/video1.mp4",
                "~/demos/ai-vision/vision_demo.py",
            ],
            "ollama_model": "llama3.1:70b",
            "path": "~/demos/ai-vision",
        },
    },
    "grant": {
        "name": "Grant Machine — Agentic Grant Writing",
        "dir": "grant-machine",
        # Native demo: runs under the zgx env python (see start_demo_8095.sh).
        # Unlike the other native demos this one needs almost nothing installed —
        # the env already carries 13 of its 14 pyproject deps at the exact required
        # versions (fastapi, uvicorn, httpx, openai, pypdf, rank-bm25, sse-starlette,
        # bs4, jsonschema, psutil, pyyaml, python-multipart), so only markdownify and
        # reportlab are genuinely missing and pydantic is one patch short. reportlab
        # writes the run packet PDF (app/packet.py) — BSD-licensed, pure-Python wheel,
        # so no build step on aarch64 and it clears the demo's own license audit.
        #
        # openai is deliberately NOT bumped to the pyproject floor of 3.x: every API
        # this app uses (chat.completions.create, extra_body, stream_options,
        # parallel_tool_calls, the APITimeoutError/APIConnectionError/APIStatusError
        # hierarchy) exists in the 2.29.0 already installed, and this env is shared
        # with the vision/TRT demos — a major-version bump there is risk with no
        # upside. The floor in pyproject.toml was lowered to match.
        #
        # The third step is the load-bearing one: it proves this box's Ollama
        # actually satisfies the agent's endpoint contract (E1-E12 — tool calls,
        # streaming tool calls, thinking on AND off) before the demo is declared
        # installed. That check is why config sets thinking_control=reasoning_effort:
        # Ollama ignores chat_template_kwargs, so the llama.cpp default silently
        # produces an agent that never shows its reasoning and whose drafter/reviewer
        # profiles think when they must not.
        "steps": [
            # Installs the FULL runtime dependency set from pyproject.toml, not a
            # delta against what happened to be in the env.
            #
            # This used to install only markdownify + pydantic + reportlab, on the
            # reasoning that "the env already carries 13 of its 14 pyproject deps".
            # That was true of the ZGX Nano, whose zgx env had been populated by its
            # factory image and the other demos -- and false of every fresh machine,
            # because install.py Phase 2 builds that env from demos/requirements.txt,
            # which is four packages (fastapi, uvicorn, httpx, huggingface_hub).
            #
            # The result on a clean box was a demo that installed "successfully" and
            # then died on launch with ModuleNotFoundError: No module named
            # 'sse_starlette'. pip skips what is already satisfied, so listing
            # everything costs nothing on a machine that already has them.
            {"label": "Installing Grant Machine deps into zgx env (full runtime set)",
             "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -m pip install --timeout 60 --retries 10 "
                    "'beautifulsoup4>=4.15.0' 'fastapi>=0.141.1' 'httpx>=0.28.1' 'jsonschema>=4.26.0' "
                    "'markdownify>=1.2.3' 'openai>=2.29.0' 'psutil>=7.2.2' 'pydantic>=2.13.5' "
                    "'pypdf>=6.19.0' 'reportlab>=4.0' 'python-multipart>=0.0.32' 'pyyaml>=6.0.3' "
                    "'rank-bm25>=0.2.2' 'sse-starlette>=3.4.11' 'uvicorn[standard]>=0.53.0'"},
            {"label": "Verifying Grant Machine imports",
             "cmd": "$HOME/miniforge3/envs/zgx/bin/python3 -c 'import fastapi, uvicorn, httpx, openai, pydantic, pypdf, rank_bm25, sse_starlette, markdownify, bs4, jsonschema, psutil, yaml, reportlab; print(\"grant imports OK — openai\", openai.__version__, \"| pydantic\", pydantic.VERSION, \"| reportlab\", reportlab.Version)'"},
            # Optional, and deliberately best-effort: SearXNG powers Discover's general
            # web search. Without it search_web reports itself unavailable and the
            # federal catalogue search — which needs no provider — carries the run, so
            # a failure here must not fail the install. Bound to loopback; no API key,
            # no account, nothing routed through a commercial search provider. It is a
            # long-lived side service like Ollama, so teardown does not stop it.
            {"label": "Starting SearXNG for web research (optional — Discover works without it)",
             "cmd": "bash {dir}/scripts/start_searxng.sh",
             # Genuinely best-effort: without it search_web reports itself
             # unavailable and the federal catalogue search carries the run.
             # Flagged so the installer does not fail the demo over it.
             "optional": True},
            {"label": "Verifying Ollama endpoint contract (tool calls, streaming, thinking — E1-E12)",
             "cmd": "cd {dir} && GM_LLM__THINKING_CONTROL=reasoning_effort $HOME/miniforge3/envs/zgx/bin/python3 scripts/endpoint_check.py --base-url http://127.0.0.1:11434/v1 --model qwen3.6:35b --reps 1 --label nano-ollama-install"},
        ],
        "launch_cmd": "bash $HOME/demos/grant-machine/start_demo_8095.sh",
        "port": 8095,
        "port_path": "/",
        # /api/health returns status:"ok" as soon as the process is up, with a
        # separate llm:"ok"|"unreachable" field — "app up, model cold" must not read
        # as a failed launch, and the launcher treating any 200 as live is correct here.
        "health_url": "http://localhost:8095/api/health",
        "log_file": "grant_machine_demo.log",
        # port_kill, not process_pattern: unlike vision/pipeline/spectrum there is no
        # load race to lose — uvicorn binds :8095 in about a second (it builds a BM25
        # index over a 196KB vault at startup and never loads a model; the LLM is
        # contacted lazily, per run). And `-m app.main` is far too generic a pgrep
        # pattern to kill by name safely on a shared box.
        "teardown_strategy": {"type": "port_kill"},
        "discovery": {"type": "port_probe"},
        "readiness_check": {
            "files": [
                "~/demos/grant-machine/app/main.py",
                "~/demos/grant-machine/web/index.html",
                "~/demos/grant-machine/start_demo_8095.sh",
                "~/demos/grant-machine/data/vault",
                "~/demos/grant-machine/data/solicitation",
                # Discover default-denies against this allowlist: without it no search
                # term can leave the device and the mode is inert.
                "~/demos/grant-machine/data/vocabulary/taxonomy.json",
            ],
            "ollama_model": "qwen3.6:35b",
            "path": "~/demos/grant-machine",
        },
    },
    "vehicle": {
        "name": "Port-of-Entry Vehicle Inspection",
        "dir": "automotive-vehicle-inspection",
        # Thin client over host Ollama, the ballistics shape. Upstream ran vLLM
        # in an nvcr.io container against an FP8 checkpoint and published :8095;
        # that image is NVIDIA-only, a second vLLM cannot share the GB10 with
        # Maritime's, and 8095 is Grant Machine. No GPU in the container, so no
        # profile overrides and no `requires`: it runs on every profile.
        "steps": [
            {"label": "Pulling vision model (~15GB — mistral-small3.2, 24B)", "cmd": "ollama pull mistral-small3.2"},
            # Registers mistral-small3.2 with an 8k context as `vehicle-inspector`.
            # Ollama otherwise allocates the model's full 131k context: 36.9 GB
            # resident on the GB10 against 15.8 GB for this. Idempotent, and cd
            # {dir} so the Modelfile resolves -- same pattern as pharmacy/pii.
            {"label": "Registering Ollama model (vehicle-inspector, 8k context)",
             "cmd": "ollama list | grep -q '^vehicle-inspector' || (cd {dir} && ollama create vehicle-inspector -f Modelfile.vehicle)"},
            {"label": "Building Docker container", "cmd": "cd {dir} && docker compose build"},
        ],
        "launch_cmd": "cd {dir} && bash ./start.sh -d",
        # 8097: next free slot after SearXNG (8096). One port only -- inference
        # is host Ollama on 11434, so there are no extra_ports to free.
        "port": 8097,
        "port_path": "/",
        "health_url": "http://localhost:8097/api/health",
        "teardown_strategy": {"type": "docker_compose"},
        "discovery": {"type": "docker_container", "patterns": ["vehicle-inspection"]},
        "readiness_check": {
            "files": [
                "~/demos/automotive-vehicle-inspection/backend/main.py",
                "~/demos/automotive-vehicle-inspection/frontend/index.html",
            ],
            "ollama_model": "vehicle-inspector",
            "path": "~/demos/automotive-vehicle-inspection",
        },
    },
}
