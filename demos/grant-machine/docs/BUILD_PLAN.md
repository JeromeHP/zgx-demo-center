# Grant Machine — Local Agentic AI Demo: Build Plan for Claude Code

**Target (today):** HP ZBook Ultra G1a (Ryzen AI MAX+ PRO 395, Radeon 8060S / gfx1151, 128 GB unified memory), WSL2 Ubuntu 24.04, ROCm + llama.cpp
**Target (later):** HP ZGX Nano (GB10, ARM64, DGX OS) — the app is written to move with a config change only
**Audience:** Public sector (higher ed, K-12, state/local government) marketing demos, presenter-led with a kiosk "explore" mode
**Plan version:** v1 — 2026-09-16

---

## 0. Instructions to Claude Code (read first, follow throughout)

1. **Execute phases in order.** The one exception: Phase 2.1 lets work continue while the model downloads. Each phase ends with an **Acceptance** block. Do not start the next phase until every acceptance item passes. If one cannot pass, stop and report what failed, what you tried, and the root cause you found.
2. **One commit per phase**, each independently revertible. Commit message format: `phase N: <short description>`.
3. **`HUMAN STEP` markers mean stop.** Anything needing `sudo`, a Windows-side change, a password, or a judgment call is marked `HUMAN STEP`. Print the exact commands for Jerome, wait for confirmation, then verify the result yourself before continuing.
4. **Do not touch existing setups.** Jerome already has a working ROCm/llama.cpp install and a `qwen-eval` harness on this machine. Do not `git pull`, rebuild, or reconfigure the existing llama.cpp checkout. If a newer build is needed, build a separate copy (Phase 2.3).
5. **Verify by inspection, not inference.** After every step, run a command that proves the result (e.g., `curl` the endpoint, `grep` the file, run the test). Do not report something as working because an installer printed success.
6. **Portability rules (non-negotiable):**
   - No hardcoded home paths. No `Path.home()`, `expanduser("~")`, `$HOME`, or `/home/<user>` inside `app/`, `web/`, or `tests/`. All paths resolve relative to the repo root or come from config/env.
   - No GPU libraries in the app. No `torch`, `transformers`, `sentence-transformers`, `onnxruntime-gpu`, CUDA, or ROCm packages in the app's dependencies. All GPU work lives in the model server.
   - No external network calls from the app or UI. No CDN scripts, web fonts, analytics, or telemetry. The UI must render with the network unplugged.
   - Only permissive licenses (MIT/BSD/Apache-2.0/ISC/PSF). No GPL/AGPL/LGPL dependencies. In particular, do not use PyMuPDF/fitz (AGPL) or html2text (GPL).
7. **Never invent numbers presented as fact.** Cloud pricing, "human hours" baselines, and statistics must come from config values Jerome sets. Default them to `null` and have the UI say they are not configured.
8. **Keep a running log.** Append to `PROGRESS.md` at the repo root after each phase: date/time, what was done, verification output (trimmed), open issues.
9. **Timeouts:** Local reasoning models are slow on long generations. Set HTTP client timeouts to at least 900 s for model calls and never cap `max_tokens` so low that thinking tokens get truncated (a known failure mode on this machine).

---

## 1. What we are building

**The scenario.** *Harrow Creek Community College* (fictional) is applying to the real, public-domain **NSF Advanced Technological Education (ATE) program solicitation (NSF 24-584)**, Track 2 "Projects" (up to $1,000,000 over 3 years). The deadline recurs on the first Thursday in October, which is **October 1, 2026**, two weeks from today. The grants office is one person.

**The demo.** A local AI agent running entirely on the device:

1. Reads the solicitation and builds a **compliance matrix** of every requirement.
2. Searches the college's private **proposal vault** (past proposals, reviewer comments, budgets, rate agreements, letters, outcomes data), all synthetic.
3. **Runs deterministic checks**, such as budget totals against the track cap and fringe rate against the rate agreement.
4. **Drafts proposal sections** with inline citations to vault documents.
5. **Reviews its own drafts** against NSF merit review criteria and ATE program-specific questions, then revises until they pass or the revision limit is hit.
6. **Verifies citations**: every cited document exists, and every number in the draft appears in the cited document.
7. **Flags planted problems** a human grant writer would need to catch (Spec D.3).

**The four presenter moments** built into the UI:

- **Stopwatch.** AI elapsed time next to a human baseline that Jerome configures.
- **Unplug.** A live network status pill. The presenter disconnects Wi-Fi mid-run and the agent keeps going.
- **Meter.** Tokens generated, tokens per second, and an optional cloud-cost equivalent (only when pricing is configured).
- **Bring your own document.** Explore mode accepts an uploaded solicitation (PDF, HTML, or Markdown). Uploaded files stay in a session folder that Reset wipes.

**Why agentic matters for the local AI story.** The trace panel makes the agent's work visible: every tool call, every search, every self-review loop. The token meter shows how many tokens an agent consumes compared with a single chat. That is the cost argument for flat-cost local hardware, and the vault access is the data-sensitivity argument.

---

## 2. Architecture

```
┌──────────────────────── WSL2 Ubuntu 24.04 on ZBook Ultra ────────────────────────┐
│                                                                                   │
│  llama-server (ROCm, gfx1151)          grant-machine app (Python, no GPU deps)    │
│  Qwen3.6-35B-A3B UD-Q4_K_XL GGUF   ◄── OpenAI-compatible HTTP ──  FastAPI :8095   │
│  127.0.0.1:8080  /v1/chat/completions                              │              │
│                                                                    ├─ agent loop  │
│                                                                    ├─ tools       │
│                                                                    │   (BM25 vault search, budget check,   │
│                                                                    │    citation verify, compliance state) │
│                                                                    ├─ SSE event stream                     │
│                                                                    └─ static web UI (no CDN)               │
└───────────────────────────────────────────────────────────────────────────────────┘
          Windows browser → http://localhost:8095  (WSL2 localhost forwarding)

Later on ZGX Nano: swap llama-server(ROCm) for llama-server(CUDA) or vLLM(NVFP4). App unchanged.
```

**Components:**

| Layer | Choice | License | Notes |
|---|---|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B`, GGUF `unsloth/Qwen3.6-35B-A3B-GGUF` file `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.36 GB) | Apache-2.0 | Text-only use; do not download `mmproj` files |
| Model server | llama.cpp `llama-server` with ROCm backend | MIT | `--jinja` for tool calling, `--reasoning-format deepseek` for `reasoning_content` |
| API | FastAPI + uvicorn + sse-starlette | MIT/BSD | Serves UI at `/`, API under `/api/` |
| LLM client | `openai` Python SDK pointed at `base_url` | Apache-2.0 | Only non-standard params via `extra_body` |
| Retrieval | `rank-bm25` over chunked vault | Apache-2.0 | No embedding model; about 20 documents |
| Parsing | `pypdf`, `beautifulsoup4`, `markdownify` | BSD/MIT/MIT | No PyMuPDF, no html2text |
| Config | `pydantic` v2 + `PyYAML` | MIT | YAML defaults + `GM_*` env overrides |
| Tests | `pytest`, `httpx`, fake OpenAI-compatible server | MIT/BSD | Unit tests need no GPU |
| License audit | `pip-licenses` (dev only) | MIT | Fails the build on GPL/AGPL/LGPL |
| Packaging | Docker, `python:3.12-slim` (multi-arch), `network_mode: host` | — | Matches Demo Center conventions |

**Ports:**

| Port | Service | Notes |
|---|---|---|
| 8080 | llama-server | Bound to `127.0.0.1` |
| 8095 | Grant Machine app | Configurable. Must be checked against the Demo Center launcher registry before integration: 8000 = pii, 8093 = ballistics are known |

---

## 3. Repository layout (create exactly this)

Repo location in WSL: `~/projects/public-sector-demos/grant-machine` (this is the only place `~` appears; it is a human-facing location, not a code path).
Models live outside the repo: `~/models/qwen3.6-35b-a3b/`.

```
grant-machine/
├── README.md
├── PROGRESS.md
├── DATA_SOURCES.md
├── THIRD_PARTY_LICENSES.md          # generated by scripts/license_audit.sh
├── pyproject.toml                   # uv-managed
├── uv.lock
├── .gitignore
├── config/
│   ├── default.yaml                 # committed defaults
│   └── local.yaml.example           # copy to local.yaml (gitignored) for machine overrides
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app, routes, static mount, uvicorn entry
│   ├── settings.py                  # config loading (yaml + env), repo-root path resolution
│   ├── llm_client.py                # OpenAI-compatible client, profiles, reasoning normalization
│   ├── events.py                    # event models + in-memory per-run event bus
│   ├── metrics.py                   # token/tps/elapsed accounting, cost equivalence
│   ├── netstatus.py                 # internet reachability probe + app egress check
│   ├── ingest/
│   │   ├── loaders.py               # pdf/html/md/csv → normalized text with section headings
│   │   └── chunker.py               # heading-aware chunks with stable chunk ids
│   ├── retrieval/
│   │   └── bm25_index.py
│   ├── checks/
│   │   ├── budget.py                # deterministic budget validation
│   │   └── citations.py             # citation + number verification
│   ├── agent/
│   │   ├── orchestrator.py          # tool-calling loop, guards, retries
│   │   ├── tools.py                 # tool schemas + implementations
│   │   ├── state.py                 # run state: compliance matrix, drafts, scores, issues
│   │   └── prompts.py               # system prompts (Spec F)
│   └── replay.py                    # record/replay event streams
├── web/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── data/
│   ├── seeds/harrow_creek.yaml      # canonical facts for synthetic data (Spec D.1)
│   ├── solicitation/
│   │   ├── nsf24-584.html           # fetched original
│   │   ├── nsf24-584.md             # normalized text
│   │   └── requirements.json        # extracted + human-reviewed requirements
│   ├── vault/                       # synthetic institutional documents (committed)
│   │   ├── manifest.json
│   │   └── _planted_issues.json     # NOT indexed; used by tests and presenter notes
│   └── replays/                     # recorded runs (jsonl), committed selectively
├── runtime/                         # gitignored: session uploads, run outputs, logs
├── reports/                         # gitignored except .gitkeep: endpoint checks, e2e results
├── scripts/
│   ├── preflight.sh
│   ├── build_llamacpp_rocm.sh
│   ├── download_model.sh
│   ├── start_model_server.sh
│   ├── endpoint_check.py
│   ├── fetch_solicitation.py
│   ├── extract_requirements.py
│   ├── generate_vault.py
│   ├── validate_vault.py
│   ├── run_e2e.py
│   ├── license_audit.sh
│   └── audit_portability.sh
├── tests/
│   ├── conftest.py
│   ├── fake_llm_server.py
│   ├── test_settings.py
│   ├── test_loaders_chunker.py
│   ├── test_bm25.py
│   ├── test_budget.py
│   ├── test_citations.py
│   ├── test_llm_client_normalization.py
│   ├── test_orchestrator.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── compose.yaml
└── docs/
    ├── ARCHITECTURE.md
    ├── PRESENTER_SCRIPT.md
    ├── PORTING_TO_ZGX_NANO.md
    └── DEMO_CENTER_INTEGRATION.md
```

---

## Phase 0 — Preflight (read-only)

**Goal:** Record the machine's actual state before changing anything.

Create `scripts/preflight.sh`. It makes no changes: it prints each section with a header and writes the full output to `reports/preflight_<timestamp>.txt`. (Create the repo directory and `reports/` first; that is the only write.)

It must check:

```bash
# OS / WSL
cat /etc/os-release | grep -E '^(NAME|VERSION)='
uname -r                                   # expect "microsoft-standard-WSL2"
nproc; free -g; df -h "$PWD"

# GPU passthrough + ROCm
ls -l /dev/dxg                             # WSL GPU device
command -v rocminfo && rocminfo | grep -E 'Name:|Marketing Name' | head -20
command -v hipconfig && hipconfig --version
# Report any line mentioning gfx1151

# Existing llama.cpp (do not modify)
command -v llama-server || true
find "$HOME" -maxdepth 4 -type f -name llama-server 2>/dev/null
# For each llama-server found:
#   <path> --version
#   <path> --list-devices    (confirm a ROCm/HIP device is listed)

# Python tooling
python3 --version
command -v uv && uv --version
command -v git && git --version

# Docker
command -v docker && docker version --format '{{.Server.Version}}' 2>&1
docker info 2>/dev/null | grep -E 'Operating System|Architecture|Server Version'

# Ports
ss -ltnp 2>/dev/null | grep -E ':(8080|8095)\b' || echo "ports 8080/8095 free"

# Disk for the model (need >= 30 GB free on the filesystem that holds ~/models)
df -h "$HOME"
```

**Decision table** (Claude Code evaluates the output and reports):

| Check | Pass condition | If it fails |
|---|---|---|
| WSL2 kernel | `microsoft-standard-WSL2` in `uname -r` | Stop; report |
| GPU device | `/dev/dxg` exists and a llama-server lists a ROCm/HIP device | Stop; `HUMAN STEP`: verify AMD WSL driver/ROCm install |
| Memory | `free -g` total ≥ 48 GB | `HUMAN STEP` in Section 0.A |
| uv | installed | Install in user space (no sudo): `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then re-check |
| Docker | server version prints | Not blocking for Phases 1–11. `HUMAN STEP` before Phase 12 |
| Ports | 8080 and 8095 free | Choose free alternatives, record them in `config/local.yaml` and `PROGRESS.md` |
| Disk | ≥ 30 GB free | Stop; `HUMAN STEP`: free space |

### 0.A `HUMAN STEP` — WSL memory (only if `free -g` < 48 GB)

On Windows, edit `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=64GB
swap=16GB
```

Then in PowerShell run `wsl --shutdown`, reopen the WSL terminal, and have Claude Code re-run `scripts/preflight.sh`.

Known constraint on this machine: WSL2 exposes about **32 GB** of GPU-addressable memory even though the laptop has 128 GB unified memory. The 22.4 GB model plus KV cache fits under that cap. Larger models (e.g., Nemotron 3 Super 120B) will not; that is expected and out of scope.

**Acceptance:**

- [ ] `reports/preflight_*.txt` exists and is summarized in `PROGRESS.md`
- [ ] Every row in the decision table is Pass or has a recorded resolution
- [ ] Commit: `phase 0: preflight report`

---

## Phase 1 — Repository scaffold, config, LLM client

**Goal:** A tested skeleton with the two modules every later phase depends on: `settings.py` (Spec A) and `llm_client.py` (Spec B).

### 1.1 Create the repo

```bash
mkdir -p "$HOME/projects/public-sector-demos/grant-machine"
cd "$HOME/projects/public-sector-demos/grant-machine"
git init -b main
uv init --app --python 3.12 --name grant-machine
uv add fastapi "uvicorn[standard]" sse-starlette openai pydantic pyyaml rank-bm25 pypdf beautifulsoup4 markdownify httpx psutil jsonschema python-multipart
uv add --dev pytest pytest-asyncio pip-licenses
```

Create every directory in Section 3 with a `.gitkeep` where empty. Keep the Phase 0 report in `reports/`. Copy this plan file into the repo as `docs/BUILD_PLAN.md` so it is versioned with the code.

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
runtime/
reports/*
!reports/.gitkeep
!reports/baseline_zbook.json
config/local.yaml
*.gguf
.pytest_cache/
```

### 1.2 Implement

- `config/default.yaml` and `config/local.yaml.example` exactly per Spec A
- `app/settings.py` per Spec A loading rules
- `app/llm_client.py` per Spec B
- `tests/fake_llm_server.py` per Phase 6.4 (build it now; later phases extend its scripts)
- `tests/test_settings.py`: defaults load, local override merges, env override (`GM_LLM__BASE_URL`, `GM_SERVER__PORT` as int), invalid value fails with a readable error, paths resolve relative to the repo root
- `tests/test_llm_client_normalization.py` (against the fake server): `reasoning_content` field, inline `<think>` (non-stream and split across stream chunks), streaming tool-call assembly across fragments, malformed tool args → `parse_error`, missing usage → `estimated=True`, sampling params land in `extra_body` correctly

### 1.3 Minimal app entry

`app/main.py` with only `GET /api/health` returning `{"status":"ok","llm":"unknown"}` for now, plus the `python -m app.main` entry using settings host/port. Verify:

```bash
uv run python -m app.main &   # then:
curl -s http://127.0.0.1:8095/api/health
```

**Acceptance:**

- [ ] `uv run pytest -q` passes
- [ ] Health endpoint responds on the configured port
- [ ] Commit: `phase 1: scaffold, config, LLM client`

---

## Phase 2 — Model server (llama.cpp + Qwen3.6-35B-A3B)

### 2.1 Download the model (start first; it runs in the background)

```bash
uv tool install "huggingface_hub[cli]"      # provides the `hf` command, no sudo
mkdir -p "$HOME/models/qwen3.6-35b-a3b"
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --local-dir "$HOME/models/qwen3.6-35b-a3b"
```

Put this in `scripts/download_model.sh`, with an idempotent check that skips the download if the file exists with size `22360456160` bytes. After downloading, verify:

```bash
stat -c %s "$HOME/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"   # must equal 22360456160
```

The download is about 22 GB and can take a while on a home connection. Start it in the background (`nohup bash scripts/download_model.sh > reports/download.log 2>&1 &`). While it runs, continue with Phase 4.1, 4.3 and the CSV part of 4.4, then Phases 5–7. All of these are testable against the fake LLM server. Return to finish 2.2–2.6 and Phase 3 when the file size verifies.

### 2.2 Decide whether the existing llama.cpp build can serve Qwen3.6

Qwen3.6 uses the `qwen3_5_moe` architecture (hybrid Gated DeltaNet + gated attention MoE), which needs a recent llama.cpp. Test the **existing** binary without modifying it, after the model download from 2.1 completes:

```bash
<existing>/llama-server -m "$MODEL_PATH" -c 4096 -ngl all --port 18080 --host 127.0.0.1 &
# wait until /health returns ok or the process exits; capture the log
```

- If it loads and `/health` returns `{"status":"ok"}`: use the existing binary. Record its path in `config/local.yaml` under `model_server.llama_server_bin`. Kill the test process.
- If it fails with `unknown model architecture` or similar: go to 2.3.

### 2.3 (Only if needed) Build a separate llama.cpp with ROCm

`scripts/build_llamacpp_rocm.sh` builds into `$HOME/opt/llama.cpp-grantmachine` and never touches the existing checkout:

```bash
set -euo pipefail
DEST="$HOME/opt/llama.cpp-grantmachine"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$DEST"
fi
cd "$DEST"
git fetch --tags
git checkout master && git pull --ff-only
git rev-parse HEAD > BUILD_COMMIT.txt

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -j "$(nproc)" --target llama-server

./build/bin/llama-server --version
./build/bin/llama-server --list-devices
```

If `hipconfig` or the HIP toolchain is missing, stop at a `HUMAN STEP`; do not install ROCm packages with sudo on your own. Record the binary path and commit hash in `config/local.yaml` and `PROGRESS.md`.

### 2.4 Launch script

`scripts/start_model_server.sh` (reads optional env vars; defaults shown):

```bash
#!/usr/bin/env bash
set -euo pipefail
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN to the llama-server path}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the GGUF file}"
HOST="${LLM_HOST:-127.0.0.1}"
PORT="${LLM_PORT:-8080}"
CTX="${LLM_CTX:-32768}"
SLOTS="${LLM_SLOTS:-1}"
ALIAS="${LLM_ALIAS:-qwen3.6-35b-a3b}"

exec "$LLAMA_SERVER_BIN" \
  -m "$MODEL_PATH" \
  --alias "$ALIAS" \
  --host "$HOST" --port "$PORT" \
  -c "$CTX" -np "$SLOTS" \
  -ngl all \
  -fa auto \
  --jinja \
  --reasoning-format deepseek \
  --load-mode none \
  --metrics
```

Flag notes for Claude Code:

- `--jinja` is required for OpenAI-style tool calling.
- `--reasoning-format deepseek` puts thinking into `reasoning_content` instead of leaving `<think>` tags in `content`.
- `--load-mode none` (the replacement for the deprecated `--no-mmap`) cut load time on this machine from about 4 minutes to about 21 seconds under WSL2.
- `-c` is total context shared across `-np` slots. With 1 slot and 32K there is room to spare, because most layers are linear attention and the KV cache stays small.

### 2.5 Context sizing test

After the server starts at `CTX=32768`, check headroom. Restart at `65536`. If it loads, `/health` is ok, and a 20K-token prompt completes (generate a synthetic long prompt), keep 65536. Otherwise keep 32768. Record the chosen value and observed load time in `PROGRESS.md`. Do not go above 65536.

### 2.6 Smoke test

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -m json.tool
```

**Acceptance:**

- [ ] GGUF size verified at `22360456160` bytes
- [ ] llama-server serves the model; `/health` ok; `/v1/models` lists alias `qwen3.6-35b-a3b`
- [ ] The smoke completion returns non-empty `content` containing "ready"
- [ ] Binary path, commit (if built), context size, and load time recorded
- [ ] Commit: `phase 2: model server scripts`

---

## Phase 3 — Endpoint compatibility check

**Goal:** One script that proves any OpenAI-compatible endpoint supports everything the agent needs. It runs on the ZBook today and on the ZGX Nano later, and the two reports are compared.

`scripts/endpoint_check.py` — CLI:

```
uv run python scripts/endpoint_check.py --base-url http://127.0.0.1:8080/v1 --model qwen3.6-35b-a3b --reps 3 --label zbook-rocm-llamacpp
```

**Checks** (each rep runs independently; report `passes/reps`):

| ID | Check | Required | How |
|---|---|---|---|
| E1 | Models list | yes | `GET /models` contains `--model` |
| E2 | Plain completion | yes | Non-empty `content`, thinking disabled |
| E3 | Thinking separation | yes | Thinking enabled: final `content` contains no `<think>`. Reasoning is either in `reasoning_content` or strippable by the client normalizer. Record which. |
| E4 | Thinking off | yes | `chat_template_kwargs.enable_thinking=false` → reasoning empty or absent |
| E5 | Single tool call | yes | Tool `search_vault(query: str, k: int)`; prompt requires it; one call; args parse as JSON and validate against the schema (`jsonschema`) |
| E6 | Tool round trip | yes | Send a tool result containing the token `ZX-4471`; the final answer must contain `ZX-4471` |
| E7 | Nested object args | yes | Tool `update_compliance(item: {id: str, status: enum, evidence: [str]})`; args validate |
| E8 | Streaming tool call | yes | `stream=true`; assemble tool-call deltas; result identical in shape to E5 |
| E9 | No spurious tool call | yes | Tools provided, but the question needs none ("What is 2+2?"); expect no tool call |
| E10 | Usage fields | no | `usage.prompt_tokens` and `usage.completion_tokens` present (if absent, the meter falls back to estimates; record it) |
| E11 | Decode speed | no | 400-token non-thinking generation; report completion tokens / generation seconds, and time to first token via streaming |
| E12 | Long context | no | 20K-token synthetic document with a planted fact; answer must contain it |

**Pass rule:** every required check must pass at least 2 of 3 reps (E1, E2 and E4 must pass 3 of 3). Exit code 0 if pass, 1 otherwise.

**Output:**

- Console table
- `reports/endpoint_check_<label>_<timestamp>.json` containing the base URL, model, label, llama/vLLM server version string if obtainable, and per-check results with raw sample responses (truncated to 2,000 characters)
- It must use the same `app/llm_client.py` normalization code the app uses (Spec B), so the check tests the real code path

Sampling for checks uses the profile values from `config/default.yaml` (Spec A).

**Acceptance:**

- [ ] Script passes against the ZBook endpoint; report saved (commit the JSON report as `reports/baseline_zbook.json`, the one report tracked in git)
- [ ] `PROGRESS.md` records E3 mode (`reasoning_content` field vs. inline tags) and E11 tokens/s
- [ ] Commit: `phase 3: endpoint compatibility check`

---
## Phase 4 — Data: solicitation, seed facts, synthetic vault

### 4.1 Fetch the solicitation

`scripts/fetch_solicitation.py`:

1. `GET https://www.nsf.gov/funding/opportunities/ate-advanced-technological-education/nsf24-584/solicitation` and save the raw HTML to `data/solicitation/nsf24-584.html`.
2. **Verify it is still the current ATE solicitation.** Check the ATE program page on nsf.gov for a newer solicitation number. If NSF 24-584 has been superseded, stop at a `HUMAN STEP`, show Jerome the newer solicitation, and ask which to use. Do not silently switch, because the track caps and page limits in Spec D may change.
3. Convert HTML → Markdown with `beautifulsoup4` + `markdownify`: strip nav, header, footer, and scripts, and keep the heading hierarchy. Write `data/solicitation/nsf24-584.md`.
4. Record the URL, retrieval timestamp, SHA-256 of the HTML, and solicitation number in `DATA_SOURCES.md`.

**Facts to verify in the fetched text** (Claude Code confirms each by `grep`, then records the matching line in `PROGRESS.md`):

- Track 2 "Projects": up to $1,000,000, 3 years
- Project Description limit: 15 pages
- Deadline: first Thursday in October annually (→ October 1, 2026)
- Required Project Description elements, including the Evaluation Plan subsection and Results from Prior NSF Support

If any fact does not match, stop and report. The seed file and budget checker depend on these values.

### 4.2 Extract requirements

`scripts/extract_requirements.py` uses the local model (`extractor` profile from Spec A, JSON output) to walk the solicitation Markdown section by section and produce candidate requirements (schema in Spec C). Rules:

- Every requirement has a `source_quote` that **must be an exact substring** of `nsf24-584.md`. Validate this deterministically, and discard and log candidates that fail.
- Deduplicate by normalized text similarity (`difflib.SequenceMatcher` ratio > 0.85 within the same category).
- Write `data/solicitation/requirements.draft.json`.
- Claude Code then reviews every draft requirement against the source text, fixes category/track errors, sets `human_reviewed: true` only on those it verified, and writes `requirements.json`. Target 30–60 requirements. Log counts (candidates, discarded, final).

`HUMAN STEP` (short): Jerome skims `requirements.json` and confirms nothing important is missing before it drives the demo.

### 4.3 Seed facts

Create `data/seeds/harrow_creek.yaml` with the canonical facts from **Spec D.1**, verbatim. Every generated document must be consistent with this file. It is also the ground truth for citation verification tests and planted issues.

### 4.4 Generate the vault

`scripts/generate_vault.py`:

- **Narrative documents** (Spec D.2, types `md`) are generated by the local model with the drafter profile. The prompt includes the seed YAML plus the document's brief, and the model is instructed to use only the provided facts for any number, name, or date. Documents are 600–2,500 words depending on type.
- **Tabular documents** (types `csv`) are generated **deterministically in Python** from the seed file, with no LLM. This includes the budget CSV, which must total exactly the seed's `new_project.budget_total_usd`.
- Each document gets YAML front matter: `doc_id`, `title`, `doc_type`, `date`, `synthetic: true`, `generator: generate_vault.py`, `seed_sha256`.
- Write `data/vault/manifest.json` listing every document.
- Write `data/vault/_planted_issues.json` from Spec D.3. It is excluded from indexing.
- The script is idempotent: `--only <doc_id>` regenerates one document, and `--force` regenerates all.

If the model endpoint is unavailable (download still running), implement the script, generate the CSVs, and come back to the narrative documents after Phase 3 passes.

### 4.5 Validate the vault

`scripts/validate_vault.py` fails on:

- Any document missing front matter or listed in the manifest but absent (or vice versa)
- Any number in Spec D.1 appearing in a document in a form that contradicts the seed. Check the key facts individually: prior award amount, completers target/actual, fringe rates, budget total, dates.
- Budget CSV arithmetic errors (row sums, year sums, grand total ≠ seed)
- The planted issues not being present exactly as specified (e.g., the Northgate letter must be dated 2021-03-15 and reference the prior project)
- Any string in the denylist in `data/seeds/denylist.txt` (real college names in the region, "HP", "NVIDIA", real NSF award numbers), matched case-insensitively. Claude Code seeds this list with obvious collisions it finds by searching for the fictional names used in Spec D.1.

`HUMAN STEP` (short): Jerome skims 3–4 generated documents for tone and realism.

**Acceptance:**

- [ ] Solicitation fetched, verified current, facts confirmed, and `DATA_SOURCES.md` written
- [ ] `requirements.json` exists and validates against the Spec C schema; all `source_quote` values are exact substrings
- [ ] All Spec D.2 documents exist; `validate_vault.py` exits 0
- [ ] Commit: `phase 4: solicitation, seeds, synthetic vault` (the generated documents are committed directly; no LFS)

---

## Phase 5 — Ingestion, retrieval, deterministic checks

### 5.1 Loaders and chunker

- `loaders.py`: `load(path) -> Document(doc_id, title, doc_type, text, sections[])` for `.md` (front matter parsed), `.html` (bs4 + markdownify), `.pdf` (`pypdf`, page-numbered sections), and `.csv` (rendered as a Markdown table plus the raw rows kept for checks). Reject files over 20 MB and PDFs over 200 pages with a clear error.
- `chunker.py`: split on headings, then pack to about 350–500 words with a 50-word overlap. `chunk_id = f"{doc_id}#c{n:03d}"`. Preserve `heading` for display.

### 5.2 BM25 index

- `bm25_index.py`: build from the manifest at startup (and per-session for uploads). The tokenizer is lowercase, splits on non-alphanumerics, and removes a small built-in English stopword list. Numbers are kept as tokens.
- `search(query, k) -> [Hit(doc_id, title, chunk_id, heading, score, snippet)]`. The snippet is 400 characters centered on the best-matching term.

### 5.3 Budget check (`checks/budget.py`)

Deterministic, driven by the seed and config. It returns structured issues with stable codes:

| Code | Rule |
|---|---|
| `BUDGET_OVER_CAP` | Grand total > track cap from `requirements.json` tracks block |
| `FRINGE_RATE_MISMATCH` | Fringe rate applied in the budget CSV ≠ rate in the rate agreement document (parse both) |
| `INDIRECT_ON_EXCLUDED` | Indirect cost applied to participant support or equipment lines |
| `YEAR_SUM_MISMATCH` | Annual totals don't add up to the grand total |
| `DURATION_OVER_LIMIT` | Budget years > track duration |

Output: `{total_usd, cap_usd, over_by_usd, fringe_rate_budget, fringe_rate_agreement, total_if_agreement_fringe_usd, issues: [{code, severity, detail, evidence: [doc_id]}]}`.

`equipment_threshold_usd` comes from config (default 10000, with a comment to verify against current federal Uniform Guidance and institutional policy).

### 5.4 Citation verification (`checks/citations.py`)

- The citation syntax in drafts is `[[doc_id]]` or `[[doc_id#c012]]`.
- **Existence check:** every cited `doc_id`/`chunk_id` exists.
- **Number check:** for each sentence containing a citation, extract numbers (currency, percentages, integers ≥ 10, dates as `YYYY-MM-DD` or `Month YYYY`). Each must appear in the cited document's text after normalization (strip `$`, commas, trailing `.0`). Numbers that appear nowhere in the cited docs are returned as `unsupported_claims`.
- **Uncited numbers:** sentences with numbers but no citation are returned as `uncited_numbers` (a warning, not a failure).

### 5.5 Unit tests

`test_loaders_chunker.py`, `test_bm25.py` (a known query returns the expected doc in the top 3), `test_budget.py` (detects all planted budget issues against the real vault CSV and rate memo; clean fixture produces zero issues), and `test_citations.py` (supported, unsupported, missing doc, uncited cases).

**Acceptance:**

- [ ] `uv run pytest -q tests/test_loaders_chunker.py tests/test_bm25.py tests/test_budget.py tests/test_citations.py` passes
- [ ] `check_budget` on the vault returns exactly the planted `BUDGET_OVER_CAP` and `FRINGE_RATE_MISMATCH` issues and no false positives
- [ ] Commit: `phase 5: ingestion, retrieval, deterministic checks`

---

## Phase 6 — Agent harness

### 6.1 State (`agent/state.py`)

`RunState` holds: `run_id`, `mode` (`story|explore|replay`), `scope`, `requirements`, `compliance` (dict requirement_id → `{status, evidence[], note, updated_step}`), `drafts` (section_id → list of versions `{version, text, citations[], word_count}`), `reviews` (section_id → list), `citation_checks`, `issues` (list; deterministic issues are auto-added from tool results with `source: "check"`, agent-flagged ones with `source: "agent"`), `tool_calls_made` (counter by name), `steps`, `status` (`running|completed|incomplete|cancelled|error`), and `summary`.

Everything is persisted as JSON under `runtime/runs/<run_id>/` at each step, and drafts as `drafts/<section_id>_v<n>.md`.

### 6.2 Tools

Implement exactly the tools in **Spec E**, with JSON Schemas passed to the model and server-side validation (`jsonschema`) before execution. Invalid args → the tool result is `{"error": "invalid_arguments", "detail": "<validation message>"}` and the loop continues.

### 6.3 Orchestrator loop (`agent/orchestrator.py`)

```
messages = [system(orchestrator_prompt(scope, requirements summary)), user(task_brief(scope))]
for step in 1..max_steps:
    if cancelled: status=cancelled; break
    resp = llm.chat(profile="orchestrator", messages, tools=TOOL_SCHEMAS, stream=True,
                    parallel_tool_calls=False)
    emit reasoning deltas (thinking_delta), content deltas, and assembled tool calls
    if resp.tool_calls:
        for call in resp.tool_calls (execute sequentially, even if the model sent several):
            emit tool_call
            result = execute(call)          # validated; exceptions → error result
            emit tool_result (+ domain events: compliance_update, issue_flagged, draft_*, section_scored)
            messages += assistant(tool_calls) + tool(result_for_model(result))
            if call.name == "finish" and result.accepted: status=completed; break loop
    else:
        nudges += 1
        if nudges > 3: status=incomplete; break
        messages += user("Continue the task using the tools. Call finish when every step in scope is complete.")
    compact_history_if_needed(messages)
else:
    status = incomplete (max_steps)
```

**Harness guardrails** (the presenter can call these out: "the harness enforces the policy, not just the prompt"):

1. **`finish` is rejected** unless every tool required by the scope has run: `get_requirements`, `check_budget`, and for each in-scope section `draft_section` → `review_section` → `verify_citations` on its latest version. The rejection result lists what's missing.
2. **`update_compliance` with status `met`** for a requirement tied to an in-scope section is rejected unless that section's latest version passed `verify_citations` with zero unsupported claims.
3. **Revisions are capped** at `agent.max_revisions_per_section`. After the cap, the section is marked `needs_human` automatically, with the reviewer's must-fix list attached.
4. **Repeated identical calls:** after three consecutive identical calls (same name + same args), return an error result telling the model to change approach.
5. **Result size:** tool results sent back to the model are truncated to `agent.max_tool_result_chars` (default 6000). The full result stays in state and the UI.

**History compaction:** estimate tokens as `chars / 3.5`. When the estimate exceeds 65% of the configured context, replace the content of tool results older than the last 6 tool messages with `{"truncated": true, "summary": "<one-line summary>"}`. One-line summaries are generated deterministically per tool (e.g., `search_vault: 5 hits, top doc_id=...`), not by the LLM.

**Sub-calls** (`draft_section`, `review_section`) call the LLM with their own profile and messages and stream `draft_delta` events. Their token usage counts toward the run metrics.

### 6.4 Fake LLM server and orchestrator tests

`tests/fake_llm_server.py`: a FastAPI app exposing `/v1/models` and `/v1/chat/completions` (streaming and non-streaming) that replays a scripted queue of responses per test. Each response can include `content`, `reasoning_content`, inline `<think>…</think>` content, `tool_calls` (including malformed JSON args), and `usage` (present or absent).

`tests/test_orchestrator.py` must cover:

- Happy path, quick scope: a scripted sequence completes with `status=completed` and the expected event order
- Malformed tool args → error result → model corrects → continues
- `finish` called early → rejected with missing-steps list → completes after the missing steps
- Revision cap reached → section `needs_human`
- `max_steps` reached → `incomplete`
- Cancellation mid-run → `cancelled` within one step
- Inline `<think>` responses are normalized the same as `reasoning_content` responses
- Missing `usage` → metrics marked `estimated: true`

**Acceptance:**

- [ ] `uv run pytest -q` passes, with no GPU or model needed
- [ ] Commit: `phase 6: agent harness`

---

## Phase 7 — API and event streaming

Implement **Spec G** in `app/main.py`:

- Serve `web/` at `/` via `StaticFiles(html=True)`, mounted after API routes.
- `uvicorn.run` entry reads host/port from settings; `python -m app.main` starts the server.
- Only one active run at a time. A second `POST /api/runs` returns `409` with the active `run_id`.
- SSE via `sse-starlette`. Clients can reconnect with `Last-Event-ID`, and the server replays from that `seq` using the run's `events.jsonl`.
- `GET /api/health` returns 200 with `{"status":"ok","llm":"ok|unreachable","model":"<alias>","version":"<git short sha>"}`. `llm` is checked with a 2 s timeout call to `/v1/models`, cached for 5 s.

`tests/test_api.py` (using the fake LLM server): health, config-public contains no secrets, run lifecycle, 409 on concurrent run, SSE stream receives `run_started` → `run_finished`, cancel, upload validation (type/size), and reset wipes session uploads.

**Acceptance:**

- [ ] API tests pass
- [ ] `grep -nE "api/health|uvicorn.run|PORT|StaticFiles|app.mount|@app.get|@app.post" app/main.py` output is pasted into `PROGRESS.md` for Demo Center review
- [ ] Commit: `phase 7: API and SSE`

---

## Phase 8 — Presenter UI

Implement **Spec H** in `web/` using plain HTML/CSS/JS (ES modules, no build step, no frameworks, no CDN, no web fonts; use a system font stack).

**Acceptance:**

- [ ] With the fake LLM server running a scripted full-scope run, every panel updates correctly (Claude Code runs the app with `config/local.yaml` pointing at the fake server and verifies with Playwright screenshots at 1920×1080 and 1366×768. Setup: `uv add --dev playwright` then `uv run playwright install chromium`. If Chromium fails to launch because of missing system libraries, stop at a `HUMAN STEP`: Jerome runs `sudo $(uv run which playwright) install-deps chromium`.)
- [ ] With networking disabled for the page (Playwright route-block of all non-localhost requests), the page renders identically. Zero blocked requests must be logged.
- [ ] Keyboard shortcuts work
- [ ] Commit: `phase 8: presenter UI`

---

## Phase 9 — Meters, network status, replay

Implement **Spec I**.

**Acceptance:**

- [ ] Unit tests for cost math (null pricing → `configured: false`), tps calculation, private/loopback/public host classification, and replay timing scaling
- [ ] Manual check (Claude Code, in WSL): with a probe host pointed at an unroutable address (e.g., `10.255.255.1:443`), the status shows offline within 5 s. Restoring it shows online.
- [ ] Record a replay from a fake-server run; replay mode plays it with the REPLAY banner visible at all times
- [ ] Commit: `phase 9: meters, network status, replay`

---

## Phase 10 — Live end-to-end on the ZBook

Prerequisites: Phases 2 and 3 passed and the vault narrative documents are generated.

`scripts/run_e2e.py`:

```
uv run python scripts/run_e2e.py --base-url http://127.0.0.1:8095 --scope quick --reps 3 --label zbook
uv run python scripts/run_e2e.py --base-url http://127.0.0.1:8095 --scope full  --reps 1 --label zbook
```

For each rep: start a run via the API, consume SSE to completion (timeout 30 min), then load the run state and evaluate against `_planted_issues.json`:

| Metric | How measured |
|---|---|
| Completed | `status == completed` |
| P1 over-cap caught | Issue code `BUDGET_OVER_CAP` in state |
| P4 fringe mismatch caught | Issue code `FRINGE_RATE_MISMATCH` in state |
| P2 honest prior outcomes | Latest `results_prior_support` draft contains both `84` and `120` and passed `verify_citations` |
| P3 stale letter caught | Any agent-flagged issue whose evidence includes `letter-northgate-2021` |
| Steps / tool calls | From state |
| Wall time; tokens; tokens/s | From metrics events |
| Revisions per section | From state |
| Unsupported claims (final) | From latest citation checks |

Write `reports/e2e_<label>_<scope>_<timestamp>.json` and append a summary table to `PROGRESS.md`.

**Acceptance (quick scope, 3 reps):**

- [ ] Completed 3/3 without crash or harness error
- [ ] P1 and P4: 3/3
- [ ] P2: ≥ 2/3
- [ ] P3: reported (target ≥ 2/3; if below, improve the orchestrator prompt or the vault letter's clarity, not the check logic, then re-run)
- [ ] Median wall time reported. **If the quick-scope median exceeds 6 minutes, do not change the model or silently disable thinking.** Report the per-phase timing breakdown and propose options to Jerome (e.g., thinking off for the reviewer profile, smaller draft length, fewer requirements in quick scope).
- [ ] Record at least one successful quick-scope run as a replay: `data/replays/quick_zbook_<date>.jsonl` (committed)
- [ ] Commit: `phase 10: live e2e results and replay`

---

## Phase 11 — License and portability audits

`scripts/license_audit.sh`:

```bash
uv run pip-licenses --format=markdown --with-urls --order=license > THIRD_PARTY_LICENSES.md
uv run pip-licenses --format=csv | grep -Ei 'GPL|AGPL|LGPL' && { echo "Copyleft dependency found"; exit 1; } || echo "License audit OK"
```

Also add the model license (Apache-2.0, `Qwen/Qwen3.6-35B-A3B`) and llama.cpp (MIT) to `THIRD_PARTY_LICENSES.md` under "Runtime components not bundled in the app container".

`scripts/audit_portability.sh` (fails on any hit, printing matches):

```bash
grep -rnE 'Path\.home\(\)|expanduser\(|\$HOME|/home/[a-z]' app web tests && exit 1
grep -rnE 'https?://' web app | grep -vE 'localhost|127\.0\.0\.1|nsf\.gov' && exit 1   # no external URLs in app/UI
grep -rniE 'torch|transformers|sentence_transformers|onnxruntime|cupy|pycuda' pyproject.toml app && exit 1
grep -rnE 'fitz|pymupdf|html2text' pyproject.toml app && exit 1
echo "Portability audit OK"
```

(The `nsf.gov` allowance is only for the source attribution link in the footer, which must be rendered as plain text, not a live link, so the page makes no external requests.)

**Acceptance:**

- [ ] Both scripts exit 0
- [ ] Commit: `phase 11: license and portability audits`

---

## Phase 12 — Container

`HUMAN STEP` if Phase 0 showed no Docker: Jerome chooses Docker Desktop WSL integration or Docker Engine in WSL and installs it. Claude Code verifies with `docker run --rm hello-world`.

`docker/Dockerfile`:

```dockerfile
FROM python:3.12-slim
# Claude Code: replace <uv-version> with the exact version printed by `uv --version` in WSL
ARG UV_VERSION=<uv-version>
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir "uv==${UV_VERSION}" \
 && useradd --create-home --uid 10001 demo
WORKDIR /opt/grant-machine
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
COPY web ./web
COPY config ./config
COPY data ./data
RUN mkdir -p runtime reports data/replays && chown -R demo:demo runtime reports data/replays
USER demo
ENV GM_SERVER__HOST=0.0.0.0 GM_SERVER__PORT=8095
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8095/api/health', timeout=2)" || exit 1
CMD ["/opt/grant-machine/.venv/bin/python", "-m", "app.main"]
```

`docker/compose.yaml`:

```yaml
name: grant-machine
services:
  grant-machine:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: grant-machine:local
    network_mode: host
    environment:
      GM_LLM__BASE_URL: ${GM_LLM__BASE_URL:-http://127.0.0.1:8080/v1}
      GM_LLM__MODEL: ${GM_LLM__MODEL:-qwen3.6-35b-a3b}
      GM_SERVER__PORT: ${GM_SERVER__PORT:-8095}
    volumes:
      - grant-machine-runtime:/opt/grant-machine/runtime
    restart: unless-stopped
volumes:
  grant-machine-runtime:
```

Verify:

```bash
docker compose -f docker/compose.yaml up -d --build
curl -s http://127.0.0.1:8095/api/health
docker compose -f docker/compose.yaml logs --tail=50
```

Then run one quick-scope e2e rep against the containerized app.

**Optional today:** if `docker buildx` with ARM64 emulation is available, run `docker buildx build --platform linux/arm64 -f docker/Dockerfile .` to confirm the image builds for ARM64. Do not install QEMU/binfmt with sudo on your own. The authoritative ARM64 build happens on the Nano.

**Acceptance:**

- [ ] Container healthy; one quick-scope e2e rep passes against it
- [ ] Commit: `phase 12: container`

---

## Phase 13 — Documentation

Write:

- `README.md` covers: what it is, the 60-second quickstart (start model server → start app → open `http://localhost:8095`), config overview, and troubleshooting (model endpoint unreachable, slow first token, port conflicts, WSL memory).
- `docs/ARCHITECTURE.md` covers the diagram, component responsibilities, the event flow, and the guardrails list.
- `docs/PRESENTER_SCRIPT.md` follows **Spec J**.
- `docs/PORTING_TO_ZGX_NANO.md` follows **Spec K**.
- `docs/DEMO_CENTER_INTEGRATION.md` records: port (8095, to be confirmed against the launcher registry), health endpoint contract, UI served at `/` (not `/app/`), compose project name `grant-machine`, `network_mode: host`, assets committed to git (no LFS, no install-time downloads), the exact model-server prerequisite, and a draft launcher entry. **Do not modify the `zgx-demo-center` or `zgx-demos` repos as part of this plan.**
- `DATA_SOURCES.md` has a table of every file under `data/`: path, source (URL or "synthetic: generate_vault.py"), license/status ("U.S. Government work — public domain" for the NSF text; "synthetic" otherwise), retrieval/generation date, SHA-256.

**Acceptance:**

- [ ] A fresh shell following only `README.md` can start the demo (Claude Code verifies by stopping everything and following the README literally)
- [ ] Commit: `phase 13: documentation`

---

## Phase 14 — Final verification

Run all of these and paste trimmed results into `PROGRESS.md` under "Final verification":

```bash
uv run pytest -q
bash scripts/license_audit.sh
bash scripts/audit_portability.sh
uv run python scripts/validate_vault.py
uv run python scripts/endpoint_check.py --base-url http://127.0.0.1:8080/v1 --model qwen3.6-35b-a3b --reps 3 --label zbook-final
uv run python scripts/run_e2e.py --base-url http://127.0.0.1:8095 --scope quick --reps 1 --label zbook-final
git status --porcelain   # must be empty
git log --oneline         # one commit per phase
```

Final report to Jerome (in chat, not a file): phases completed, e2e metrics table, known issues, the three most important things to review before showing this to a customer, and anything still marked `HUMAN STEP`.

---

# Reference Specs

## Spec A — Configuration

`config/default.yaml`:

```yaml
llm:
  base_url: "http://127.0.0.1:8080/v1"
  api_key: "not-needed"
  model: "qwen3.6-35b-a3b"
  request_timeout_s: 900
  reasoning_mode: "auto"          # auto | field | inline_tags
  context_tokens: 32768           # set to the value chosen in Phase 2.5 context sizing
  profiles:
    # Sampling values from the Qwen3.6 model card "Best Practices".
    orchestrator:                 # thinking mode, general tasks
      enable_thinking: true
      temperature: 1.0
      top_p: 0.95
      top_k: 20
      min_p: 0.0
      presence_penalty: 1.5
      max_tokens: 8192
    drafter:                      # instruct mode, general tasks
      enable_thinking: false
      temperature: 0.7
      top_p: 0.8
      top_k: 20
      min_p: 0.0
      presence_penalty: 1.5
      max_tokens: 3000
    reviewer:                     # instruct mode, general; JSON output
      enable_thinking: false
      temperature: 0.7
      top_p: 0.8
      top_k: 20
      min_p: 0.0
      presence_penalty: 1.5
      max_tokens: 2000
    extractor:                    # thinking mode for requirement extraction (offline script)
      enable_thinking: true
      temperature: 1.0
      top_p: 0.95
      top_k: 20
      min_p: 0.0
      presence_penalty: 1.5
      max_tokens: 8192

server:
  host: "0.0.0.0"
  port: 8095

agent:
  max_steps: 60
  max_revisions_per_section: 2
  review_pass_threshold: 4.0      # mean criterion score (1–5) to pass
  max_tool_result_chars: 6000
  scopes:
    quick:
      sections: ["results_prior_support"]
      required_tools: ["get_requirements", "check_budget"]
    full:
      sections: ["project_summary", "results_prior_support", "evaluation_plan"]
      required_tools: ["get_requirements", "check_budget"]

budget:
  track: "track_2"
  equipment_threshold_usd: 10000  # verify against current 2 CFR 200 and institutional policy

uploads:
  max_bytes: 20000000
  max_pdf_pages: 200
  allowed_extensions: [".pdf", ".html", ".htm", ".md", ".txt"]

demo:
  device_label: "HP ZBook Ultra G1a (dev)"   # shown in header and replay banner
  human_baseline_minutes:        # null = UI shows "not configured"; set only with a defensible source
    compliance_matrix: null
    budget_review: null
    section_draft: null
  cloud_pricing:                 # null = cost meter hidden; set only from a dated public price list
    input_per_million_usd: null
    output_per_million_usd: null
    label: null                  # e.g., "Frontier API list price, as of <date>"
  connectivity_probe:
    hosts: ["1.1.1.1:443", "9.9.9.9:443"]
    interval_s: 3
    timeout_s: 1.0
```

**Loading rules** (`app/settings.py`):

1. `REPO_ROOT = Path(__file__).resolve().parents[1]`.
2. Load `config/default.yaml`. Deep-merge `config/local.yaml` if present. If `GM_CONFIG` is set, load that file instead of `local.yaml`.
3. Apply env overrides: `GM_` prefix with `__` as the nesting separator (e.g., `GM_LLM__BASE_URL`). Values are parsed as YAML scalars so numbers and booleans work.
4. Validate with pydantic models, and fail fast with a readable error. The top-level `model_server` block is allowed but ignored by the app (scripts read it).
5. All relative paths in config resolve against `REPO_ROOT`.

`config/local.yaml.example` shows `llm.context_tokens`, `server.port`, `demo.device_label`, and a commented `model_server.llama_server_bin` / `model_server.model_path` (used only by scripts, never by the app).

## Spec B — LLM client (`app/llm_client.py`)

```python
class LLMClient:
    def __init__(self, settings): ...        # openai.OpenAI(base_url, api_key, timeout)
    def chat(self, profile: str, messages: list, tools: list | None = None,
             stream: bool = False, json_mode: bool = False,
             on_reasoning=None, on_content=None) -> ChatResult: ...
    def list_models(self) -> list[str]: ...

@dataclass
class ChatResult:
    content: str                  # final answer text, reasoning removed
    reasoning: str                # "" if none
    tool_calls: list[ToolCall]    # id, name, arguments_raw, arguments (dict | None), parse_error (str | None)
    usage: Usage                  # prompt_tokens, completion_tokens, estimated: bool
    timings: Timings              # ttft_s, total_s, decode_tps
    reasoning_source: str         # "field" | "inline_tags" | "none"
```

**Request mapping:** `temperature` and `top_p` go as standard params. `top_k`, `min_p`, and `chat_template_kwargs: {"enable_thinking": bool}` go in `extra_body`. `presence_penalty` is standard. With tools, send `tool_choice="auto"` and `parallel_tool_calls=False`.

**Normalization** (must pass identical tests for both server styles):

- If `reasoning_content` is present (non-stream message or stream deltas), use it.
- Otherwise, if `content` contains `<think>…</think>`, move that text to `reasoning` and strip it from `content`. Handle an unclosed `<think>` in streams by buffering until `</think>`.
- `reasoning_mode` in config forces one path (`field` or `inline_tags`); `auto` detects per response.

**Streaming tool calls:** accumulate `delta.tool_calls[i].function.arguments` fragments by index, then parse JSON once complete. On parse failure, set `parse_error`; do not raise.

**Usage:** if `usage` is missing, estimate `completion_tokens = ceil(len(content + reasoning + tool args) / 3.5)` and set `estimated=True`. For streaming, request `stream_options={"include_usage": True}`, and fall back to the estimate if the server ignores it.

**Retries:** on connection errors or 5xx, 2 retries with 2 s and 5 s backoff. Never retry on 4xx. Timeouts are surfaced as `LLMTimeout` with the elapsed time.

**JSON mode** (reviewer/extractor): append an instruction to return only JSON matching a given schema. Parse it; on failure, send one repair turn with the parse error, then raise `LLMJSONError`. Do not rely on server-side `response_format`, because support differs across servers.

## Spec C — `requirements.json` schema

```json
{
  "solicitation_id": "NSF 24-584",
  "program": "Advanced Technological Education (ATE)",
  "source_url": "https://www.nsf.gov/funding/opportunities/ate-advanced-technological-education/nsf24-584/solicitation",
  "retrieved_at": "2026-09-16T00:00:00Z",
  "source_sha256": "<hash of nsf24-584.html>",
  "tracks": {
    "track_2": { "name": "Projects", "max_total_usd": 1000000, "max_years": 3 }
  },
  "deadline_rule": "First Thursday in October, annually",
  "demo_deadline": "2026-10-01",
  "requirements": [
    {
      "id": "R-001",
      "category": "project_description | merit_review | program_specific | formatting | budget | supplementary_documents | eligibility",
      "text": "Plain-language requirement",
      "source_quote": "Exact substring of nsf24-584.md",
      "source_heading": "Nearest heading in the solicitation",
      "applies_to_tracks": ["track_2"],
      "section_link": "project_summary | results_prior_support | evaluation_plan | null",
      "verification": "agent | deterministic | human",
      "human_reviewed": true
    }
  ]
}
```

Track caps are entered by Claude Code from the verified solicitation text (Phase 4.1), not from model extraction.

## Spec D — Synthetic data

### D.1 Seed facts (`data/seeds/harrow_creek.yaml`)

```yaml
disclaimer: "Fictional institution. All people, organizations, figures, and documents are synthetic."
institution:
  name: "Harrow Creek Community College"
  short_name: "HCCC"
  type: "Public two-year college"
  service_area: "Four rural counties in the Mid-South (fictional)"
  credit_headcount_fall_2025: 6840
  pell_recipient_pct: 52
  first_generation_pct: 41
  president: "Dr. Alan Pruett"
people:
  pi: { name: "Dr. Renee Okafor", title: "Dean of Workforce and Technical Programs" }
  co_pi: { name: "Marcus Lindqvist", title: "Chair, Mechatronics and Industrial Automation" }
  grants_director: { name: "Dana Whitfield", title: "Director of Grants and Sponsored Programs" }
  evaluator: { org: "Bluestem Evaluation Group", lead: "Dr. Priya Raman" }
prior_award:
  id: "HCCC-FICT-ATE-2021"
  title: "Mechatronics Pathways for Rural Manufacturers"
  funder: "NSF ATE (fictional prior award for demo purposes)"
  amount_usd: 598112
  start: "2021-09-01"
  end: "2025-08-31"             # includes a one-year no-cost extension
  outcomes:
    completers_target: 120
    completers_actual: 84
    placement_rate_pct: 78
    placement_respondents: 61
    new_courses: 3
    industry_partners: 11
    dual_credit_agreements: 2
new_project:
  title: "Smart Automation Technician Pathway"
  short_name: "SATP"
  track: "track_2"
  years: 3
  start: "2027-07-01"
  completers_target: 150
  budget_total_usd: 1038450      # planted: exceeds Track 2 cap by 38,450
  fringe_rate_budget_pct: 28.0   # planted: disagrees with rate agreement
rates:
  fringe_rate_agreement_pct: 31.5
  fringe_rate_agreement_doc: "rate-agreement-fy2026"
  indirect_rate_pct: 36.0
  indirect_base: "MTDC"
employers:
  - { name: "Kessler Fluid Systems", letter_doc: "letter-kessler-2026", letter_date: "2026-08-12", references_project: "SATP" }
  - { name: "Tri-County Packaging Cooperative", letter_doc: "letter-tricounty-2026", letter_date: "2026-08-20", references_project: "SATP" }
  - { name: "Northgate Precision Castings", letter_doc: "letter-northgate-2021", letter_date: "2021-03-15", references_project: "Mechatronics Pathways for Rural Manufacturers" }   # planted: stale
demo_context:
  today: "2026-09-16"
  deadline: "2026-10-01"
```

Claude Code: before committing, search the web for each fictional name (institution, the three employers, the evaluator org). If any is a real organization, replace it with another clearly fictional name, update this file, and note the change in `PROGRESS.md`.

### D.2 Vault documents

| doc_id | Title | Type | Generator | Key content |
|---|---|---|---|---|
| `inst-profile-2025` | HCCC Institutional Profile 2025 | md | LLM | Enrollment, demographics, service area, programs |
| `strategic-plan-2025-2030` | Strategic Plan 2025–2030 (excerpt) | md | LLM | Workforce goals; automation priority |
| `concept-note-satp` | SATP Concept Note (faculty draft) | md | LLM | One-page idea for the new project; target 150 completers |
| `prior-proposal-mpm-2021` | Funded Proposal: Mechatronics Pathways (2021) | md | LLM | Project description; promised 120 completers |
| `prior-panel-summary-2021` | NSF Panel Summary (fictional) | md | LLM | Strengths and weaknesses; evaluation critique |
| `prior-annual-report-2025` | Final Outcomes Report: Mechatronics Pathways | md | LLM | 84 completers vs 120 target; 78% placement (n=61); reasons for shortfall (COVID-era enrollment, instructor vacancy) |
| `prior-outcomes-data` | Outcomes Data 2021–2025 | csv | Python | Per-year enrollment, completers (sums to 84), placements |
| `prior-evaluation-report-2025` | External Evaluation Report (Bluestem) | md | LLM | Findings and recommendations |
| `declined-dol-2023` | Declined Proposal: DOL Workforce Grant (2023) | md | LLM | Unfunded; reviewer comments on weak employer engagement |
| `advisory-board-2026` | Industry Advisory Board Roster and Minutes (Aug 2026) | md | LLM | 9 members; discussion of automation skills gaps |
| `labor-market-memo-2026` | Regional Labor Market Memo (synthetic figures) | md | LLM | Clearly labeled synthetic demand estimates |
| `letter-kessler-2026` | Letter of Collaboration: Kessler Fluid Systems | md | LLM | Dated 2026-08-12; references SATP |
| `letter-tricounty-2026` | Letter of Collaboration: Tri-County Packaging | md | LLM | Dated 2026-08-20; references SATP |
| `letter-northgate-2021` | Letter of Support: Northgate Precision Castings | md | LLM | **Dated 2021-03-15; references the prior project, not SATP** |
| `budget-satp-draft` | SATP Draft Budget (3 years) | csv | Python | Line items by year; fringe at 28%; total 1,038,450; participant support and equipment lines excluded from indirect |
| `budget-justification-draft` | SATP Budget Justification (draft) | md | LLM | Narrative consistent with the CSV (including the 28% fringe) |
| `rate-agreement-fy2026` | Fringe and Indirect Rate Memo FY2026 | md | LLM | Fringe 31.5%; indirect 36.0% MTDC |
| `bio-okafor`, `bio-lindqvist`, `bio-raman` | Biographical Sketches | md | LLM | Short, synthetic |
| `facilities-equipment` | Facilities and Equipment | md | LLM | Automation lab description |
| `boilerplate-dissemination` | Dissemination Boilerplate | md | LLM | Reusable text |
| `boilerplate-data-management` | Data Management and Sharing Boilerplate | md | LLM | Reusable text |

The budget CSV columns are `year, category, line_item, amount_usd, fringe_applied, indirect_applied, notes`. Categories: `senior_personnel`, `other_personnel`, `fringe`, `equipment`, `travel`, `participant_support`, `materials_supplies`, `evaluation_contract`, `other_direct`, `indirect`. The Python generator solves the line items so fringe = 28% of personnel, indirect = 36% of the MTDC base (excluding equipment and participant support), and the grand total is exactly 1,038,450, adjusting a `materials_supplies` line as the balancing item. It also records `total_if_agreement_fringe_usd` in its log for test assertions.

### D.3 Planted issues (`data/vault/_planted_issues.json`)

```json
[
  { "id": "P1", "code": "BUDGET_OVER_CAP", "detection": "deterministic",
    "detail": "Draft budget total $1,038,450 exceeds the Track 2 maximum of $1,000,000 by $38,450.",
    "evidence": ["budget-satp-draft"] },
  { "id": "P2", "code": "PRIOR_OUTCOMES_SHORTFALL", "detection": "agent",
    "detail": "Prior award produced 84 completers against a target of 120; Results from Prior NSF Support must report this accurately.",
    "evidence": ["prior-annual-report-2025", "prior-outcomes-data", "prior-proposal-mpm-2021"] },
  { "id": "P3", "code": "STALE_SUPPORT_LETTER", "detection": "agent",
    "detail": "Northgate letter is dated 2021-03-15 and references the prior project, not SATP.",
    "evidence": ["letter-northgate-2021"] },
  { "id": "P4", "code": "FRINGE_RATE_MISMATCH", "detection": "deterministic",
    "detail": "Budget applies 28.0% fringe; FY2026 rate memo specifies 31.5%. Correcting it raises the total further over the cap.",
    "evidence": ["budget-satp-draft", "rate-agreement-fy2026"] }
]
```

## Spec E — Tools

All tools return JSON. Schemas are strict (`additionalProperties: false`).

| Tool | Parameters | Behavior / returns |
|---|---|---|
| `get_requirements` | `category?: enum(Spec C categories)` | Requirements list (id, category, text, section_link). First call also initializes compliance rows as `missing`. Emits `compliance_update` for initialization. |
| `search_vault` | `query: string (3–200 chars)`, `k?: integer 1–8 (default 5)` | BM25 hits: `doc_id, title, chunk_id, heading, score, snippet` |
| `read_document` | `doc_id: string`, `chunk_id?: string`, `max_chars?: integer 500–6000` | Full document text (or chunk), with front matter fields. Unknown id → error listing the closest 3 ids |
| `check_budget` | `track: enum["track_2"]` | Phase 5.3 output. Deterministic issues auto-added to state (`source: "check"`), emitting `issue_flagged` |
| `draft_section` | `section_id: enum[project_summary, results_prior_support, evaluation_plan]`, `evidence: array(1–8) of {doc_id, chunk_id?}`, `guidance: string ≤ 1500` | Sub-call (drafter profile) with the evidence text and linked requirements. Returns `{section_id, version, word_count, citations}`. Streams `draft_delta`. Word targets: summary ≤ 350 words (Overview / Intellectual Merit / Broader Impacts), prior support 400–700, evaluation plan 500–800 |
| `review_section` | `section_id: enum(as above)` | Sub-call (reviewer profile, JSON mode) scoring the latest version against the linked requirements, NSF Intellectual Merit and Broader Impacts, and ATE program-specific questions. Returns `{version, criteria:[{id, score 1–5, rationale, gaps[]}], mean_score, passed, must_fix[]}`. Emits `section_scored` |
| `verify_citations` | `section_id: enum(as above)` | Phase 5.4 on the latest version. Returns `{version, passed, missing_docs[], unsupported_claims[], uncited_numbers[]}` |
| `update_compliance` | `item: {requirement_id: string, status: enum[met, partial, missing, needs_human, not_applicable], evidence: array of string (doc_id or section_id), note: string ≤ 400}` | Updates matrix, with guardrail 2 enforced. Emits `compliance_update` |
| `flag_issue` | `issue: {severity: enum[high, medium, low], title: string ≤ 120, detail: string ≤ 800, evidence: array of doc_id (1–5), recommended_action: string ≤ 400}` | Adds an agent issue; evidence ids must exist. Emits `issue_flagged` |
| `finish` | `summary: string ≤ 1500` | Guardrail 1. Returns `{accepted: bool, missing: []}` |

## Spec F — Prompts (`app/agent/prompts.py`)

Keep prompts in code as constants with `{placeholders}`. Starting drafts (Claude Code may tighten wording, but must keep every rule):

**Orchestrator system prompt:**

```
You are the grants assistant for {institution_name}, working inside a secure, fully local system.
Today is {today}. The proposal is due {deadline} to {solicitation_id} ({program}), {track_name}.

Your job in this run (scope: {scope_name}):
1. Load the solicitation requirements (get_requirements).
2. Run the budget check (check_budget) and flag anything a grants director must fix.
3. For each section in scope — {sections} — gather evidence from the vault (search_vault, read_document),
   draft it (draft_section), review it (review_section), verify citations (verify_citations),
   and revise with draft_section if review fails or citations are unsupported.
4. Review collaboration letters and prior-award materials for problems (outdated, wrong project, overstated outcomes)
   and flag them with flag_issue.
5. Update the compliance matrix as you establish evidence (update_compliance).
6. Call finish with a short summary for the grants director.

Rules:
- Use only facts found in the vault or the solicitation. Never invent numbers, names, dates, or outcomes.
- Report prior results honestly, including shortfalls against targets, with the reasons documented in the vault.
- Everything you produce is a draft for human review. Mark anything uncertain as needs_human.
- Work efficiently: prefer targeted searches, read only what you need, and do not repeat identical calls.
```

**Drafter system prompt:**

```
You write sections of a federal grant proposal for {institution_name}.
Write the "{section_title}" section. Requirements this section must satisfy:
{linked_requirements}

Evidence (the only facts you may use):
{evidence_blocks}   # each block begins with [[doc_id#chunk]] and its title

Rules:
- Cite every factual claim with [[doc_id]] or [[doc_id#cNNN]] immediately after the sentence.
- Use only numbers that appear in the evidence, written exactly as they appear.
- Plain, confident, reviewer-friendly prose. No marketing language. Target {word_target}.
- If the evidence is insufficient for a requirement, write "[NEEDS INPUT: <what is missing>]" rather than guessing.
{revision_guidance}   # present only on revisions: reviewer must_fix list and unsupported claims to remove or fix
Return only the section text in Markdown.
```

**Reviewer system prompt:**

```
You are an experienced NSF ATE panel reviewer. Score the draft section strictly.
Criteria:
{criteria_list}   # linked requirements + Intellectual Merit + Broader Impacts + relevant ATE program-specific questions

For each criterion give score 1–5, a one-sentence rationale, and concrete gaps.
Set must_fix to the smallest set of changes needed to reach a mean score of at least {threshold}.
Penalize any claim that looks unsupported or inconsistent with the evidence list provided.
Return JSON only, matching this schema: {json_schema}
```

**Task brief (user message):** `Start the {scope_name} run now. Sections in scope: {sections}.`

## Spec G — Events and API

**Event envelope:** `{"run_id": str, "seq": int, "ts": ISO8601, "type": str, "data": {...}}`, written to `runtime/runs/<run_id>/events.jsonl` and pushed to SSE subscribers.

| Type | Data |
|---|---|
| `run_started` | `mode, scope, model, base_url_host_class, device_label, started_at` |
| `step_started` | `step` |
| `thinking_delta` | `text` |
| `content_delta` | `text` |
| `tool_call` | `call_id, name, arguments, display` (human-readable label, e.g., "Searching the vault for 'prior outcomes'") |
| `tool_result` | `call_id, name, ok, summary, result` (truncated for UI to 10,000 chars) |
| `compliance_update` | `requirement_id, status, evidence, note` |
| `issue_flagged` | `issue` (with `source`, `code?`) |
| `draft_started` / `draft_delta` / `draft_completed` | `section_id, version, text?/delta?, word_count?, citations?` |
| `section_scored` | `section_id, version, mean_score, passed, criteria, must_fix` |
| `citations_checked` | `section_id, version, passed, unsupported_claims, uncited_numbers` |
| `metrics` | `elapsed_s, prompt_tokens, completion_tokens, tokens_estimated, decode_tps_recent, llm_calls, tool_calls` (emitted at most once per second) |
| `network_status` | `internet: online/offline, llm_endpoint_class: loopback/private/public, app_public_connections` |
| `run_finished` | `status, summary, elapsed_s, totals` |
| `error` | `message, recoverable` |

**Endpoints:**

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | See Phase 7 |
| GET | `/api/config/public` | Device label, model alias, scopes, pricing configured?, baselines configured?, disclaimer text |
| GET | `/api/network` | Latest network status |
| POST | `/api/runs` | Body `{mode: "story" or "explore", scope: "quick" or "full", show_thinking: bool, session_id?: str}` → `{run_id}`; 409 if active |
| GET | `/api/runs/{run_id}/events` | SSE; supports `Last-Event-ID` |
| POST | `/api/runs/{run_id}/cancel` | |
| GET | `/api/runs/{run_id}/state` | Current `RunState` JSON |
| GET | `/api/runs/{run_id}/packet.md` | Downloadable Markdown packet: summary, compliance matrix, issues, drafts with citations, reviewer scores, and a "DRAFT — AI-generated for human review" header |
| POST | `/api/explore/upload` | Multipart, one file; validates extension/size; creates or extends a session under `runtime/sessions/<session_id>/`; for a solicitation upload, runs live requirement extraction (exact-substring rule enforced) and returns counts |
| POST | `/api/reset` | Cancels active run, deletes `runtime/sessions/*`, clears in-memory state |
| GET | `/api/replays` | List of `data/replays/*.jsonl` with metadata |
| POST | `/api/replays/{name}/play` | Body `{speed: 1, 2 or 4}` → `{run_id}` streaming the recorded events with scaled timing, `mode: "replay"` |
| POST | `/api/runs/{run_id}/save-replay` | Copies `events.jsonl` to `data/replays/<name>.jsonl` |

In explore mode, the agent uses uploaded vault documents plus the synthetic vault. If the uploaded solicitation has no track caps it can verify, `check_budget` returns `{skipped: true, reason}` and the guardrail treats it as satisfied.

## Spec H — UI

**Layout at 1920×1080** (CSS grid; degrades to a stacked layout below 1280 px wide):

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Grant Machine · Harrow Creek CC (fictional) · Model: qwen3.6-35b-a3b · Device │
│ METERS: [Agent 03:12 | Human est. —] [Tokens 41,230 · 24.8 tok/s] [Cloud equiv. —]    │
│         [Internet: OFFLINE] [Model: running locally (loopback)]  [App → internet: 0]    │
├───────────────┬──────────────────────────────────────┬───────────────────────────────┤
│ CONTROL (20%) │ AGENT TRACE (40%)                    │ OUTPUTS (40%) — tabs:          │
│ Mode          │ Step cards, newest at bottom,        │ [Compliance] [Issues (4)]      │
│  Story/Explore│ auto-scroll with a "jump to live"    │ [Budget] [Drafts] [Scores]     │
│  /Replay      │ button:                              │                                │
│ Scope         │  [search] Vault: "prior outcomes"    │ Compliance: table with animated│
│  Quick/Full   │     → 5 hits (top: prior-annual-…)   │ status pills                   │
│ [Start] [Stop]│  [read] prior-annual-report-2025     │ Issues: severity-colored cards │
│ [Reset]       │  [draft] Results from Prior Support  │ with evidence chips            │
│ [ ] Show      │     (v1)… live text                  │ Drafts: rendered Markdown;     │
│   reasoning   │  [review] v1: 3.4 / 5 — revise       │ citation chips open a source   │
│ Explore:      │  [verify] Citations OK (v2)          │ drawer with the cited excerpt  │
│  upload zone  │  [issue] Stale support letter        │ Scores: criterion bars per     │
│ Scenario card │ Reasoning (collapsible, dimmed)      │ version (v1 → v2)              │
├───────────────┴──────────────────────────────────────┴───────────────────────────────┤
│ FOOTER: Fictional institution · Synthetic data · Solicitation text: NSF 24-584         │
│ (U.S. Government work) · Not affiliated with or endorsed by NSF · AI output is a draft │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Behaviors:**

- **Issue alerts.** A new issue slides in as a toast (8 s), and the Issues tab badge increments.
- **Run end.** A summary card overlays the trace: "In 4m 12s the agent checked 41 requirements, ran 14 searches, read 9 documents, drafted 1 section (2 versions), verified every citation, and flagged 4 issues." Actions: [Download packet] [Save as replay] [Reset].
- **Meters.** Numbers tick smoothly (requestAnimationFrame easing). Unconfigured meters show "—" with a tooltip "Not configured in config/default.yaml". Never show a fabricated value.
- **Network pill.** Red "Internet: OFFLINE" and green "Model: running locally" are the intended unplug moment; use large, high-contrast text. If the model endpoint classifies as `public`, show a persistent red banner "Model endpoint is not local".
- **Replay mode.** A persistent amber banner: "REPLAY — recorded <date> on <device_label>. Not live." Meters show recorded values.
- **Keyboard.** `S` start, `X` stop, `R` reset (with confirm), `T` toggle reasoning, `F` fullscreen, `1`–`5` output tabs, `?` shortcut help.
- **Theme.** Dark by default for booths, with a light toggle. Minimum body text 16 px; meters 28 px+. WCAG AA contrast. Color is never the only status signal (icons + text).
- **Branding.** Neutral. Put `web/brand/README.md` stating that approved logos go here after Legal/brand review. No HP or NVIDIA logos in this build.
- **Security.** Render Markdown with a small built-in renderer that escapes HTML (no `innerHTML` of model output without escaping).

## Spec I — Metrics, network, replay

- **Elapsed:** from `run_started` to now (or to `run_finished`).
- **Tokens:** sum of `usage` across all LLM calls in the run (orchestrator + sub-calls). If any call is estimated, `tokens_estimated=true` and the UI shows "≈".
- **Decode tokens/s (recent):** rolling 10-second window of completion tokens / generation time from streaming timings.
- **Cloud equivalent:** `prompt_tokens/1e6 × input_price + completion_tokens/1e6 × output_price`, only when both prices are non-null; otherwise `configured: false`. Show `label` under the value.
- **Human baseline:** sum of configured minutes for the steps in scope; hidden if any in-scope value is null.
- **Internet probe:** async TCP connect to each configured host; online if any succeeds within timeout. Emit `network_status` on change and every 10 s.
- **Endpoint class:** resolve the `llm.base_url` host, then classify as loopback, RFC1918/ULA private, CGNAT `100.64.0.0/10` (treat as private, for Tailscale), or public.
- **App public connections:** `psutil.Process().connections(kind="inet")` counting ESTABLISHED remotes whose IP is not loopback/private/CGNAT. Label exactly "App connections to the internet". Do not claim anything about the model server process.
- **Replay:** events stored with original `ts`; playback sleeps `(ts[i] − ts[i−1]) / speed`, capped at 3 s per gap so idle waits don't stall the demo. Emitted with `mode: "replay"` in `run_started`.

## Spec J — Presenter script (`docs/PRESENTER_SCRIPT.md`)

Write a 5-minute talk track with timestamps and on-screen cues. Structure:

1. **0:00 Hook (30 s).** "Grants offices are one or two people. This proposal is due in two weeks. Watch an AI agent do the first pass, entirely on this device."
2. **0:30 Start the quick-scope run.** Point to the trace: requirements loaded, budget check running.
3. **1:00 First catch.** The budget-over-cap and fringe-mismatch issues appear. "Deterministic checks. The agent didn't guess; it computed."
4. **1:30 The unplug.** Disconnect Wi-Fi (on the ZBook: Windows quick settings; on the Nano: pull the cable). Point to the red Internet: OFFLINE pill and the agent still working. "Your proposal vault, budgets, and reviewer comments never leave this box."
5. **2:15 The agent at work.** Searches, reads, drafts. Show one citation chip opening the source excerpt.
6. **3:00 Self-review.** The v1 score fails, the must-fix list appears, v2 revises. "An agent that checks its own work, with guardrails the harness enforces."
7. **3:45 Honesty moment.** Results from Prior Support reports 84 completers against a target of 120, with the documented reasons. "It won't spin your outcomes."
8. **4:15 The meter.** Token count: "An agent used tens of thousands of tokens for one section. That's where per-token cloud bills grow, and why flat-cost local compute matters." (Cost equivalent only if configured.)
9. **4:40 Close.** Summary card, packet download. "Everything is a draft for your grants director. Imagine this for every proposal, every month."

Include: pre-show checklist (model server healthy, endpoint check passed today, Wi-Fi reconnect tested, replay available as a fallback), recovery lines if the run is slow ("While it thinks, let's look at the matrix"), when to switch to replay (and to say so aloud), and the kiosk/explore-mode instructions for booth visitors.

## Spec K — Porting to ZGX Nano (`docs/PORTING_TO_ZGX_NANO.md`)

Write these steps (not executed today):

1. **Preflight on the Nano:** `uname -m` (aarch64), `nvidia-smi`, Docker, and free ports 8080/8095. Stop anything else holding large memory before starting (previous incident: a trading-agent workload caused OOM/unreachability on this node).
2. **Option A (recommended first): llama.cpp CUDA with the same GGUF.** This changes only the hardware, not the server or the model file.
   ```bash
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -j "$(nproc)" --target llama-server
   ```
   Copy or download the same `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, and run `scripts/start_model_server.sh` with the new binary path.
3. **Run the endpoint check** with `--label nano-cuda-llamacpp`, then compare with `scripts/endpoint_check.py --compare reports/baseline_zbook.json <new report>`. Implement `--compare` in Phase 3: it prints per-check pass-count differences, reasoning source, and tok/s.
4. **Option B (performance): vLLM + NVFP4.** Use `nvidia/Qwen3.6-35B-A3B-NVFP4`, preferring NVIDIA's vLLM container for DGX Spark/GB10 if available; the model card recommends vLLM ≥ 0.19.0. Keep the served name identical so app config doesn't change:
   ```bash
   vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
     --served-model-name qwen3.6-35b-a3b \
     --host 127.0.0.1 --port 8080 \
     --max-model-len 65536 \
     --reasoning-parser qwen3 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --language-model-only
   ```
   Re-run the endpoint check with `--label nano-vllm-nvfp4` and compare. Watch E3 (reasoning source), E7/E8 (nested and streaming tool calls), and E10 (usage in streams).
5. **Build the app container natively on the Nano** (`docker compose -f docker/compose.yaml up -d --build`). Set `demo.device_label: "HP ZGX Nano"` in `config/local.yaml`.
6. **Run the e2e quick scope 3 reps** with `--label nano`, and record a new replay `data/replays/quick_nano_<date>.jsonl`.
7. **Demo Center integration** per `docs/DEMO_CENTER_INTEGRATION.md`, including port confirmation against the launcher.

---

## Appendix — Schedule for today

| Block | Phases | Est. time | Notes |
|---|---|---|---|
| Morning setup | 0, 1, 2 (start download) | 45–90 min | The model download runs in the background |
| While downloading | 4 (fetch, seeds, CSVs), 5, 6, 7 | 2–3 h | All testable with the fake LLM server |
| Model online | 2 (finish), 3, 4 (narrative docs + requirements) | 1–1.5 h | Generating about 20 documents locally takes time |
| UI | 8, 9 | 2–3 h | |
| Live runs | 10 | 45–90 min | Includes 3 quick reps + 1 full |
| If time remains | 11, 12, 13, 14 | 1.5–2.5 h | Container and docs can slip to tomorrow without blocking a live demo |

**Minimum demoable today:** Phases 0–10 complete. Phases 11–14 are for handing it to anyone else or moving it to the Nano.
