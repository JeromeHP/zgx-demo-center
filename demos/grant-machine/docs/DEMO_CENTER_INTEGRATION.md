# Demo Center integration

Grant Machine is integrated into the HP ZGX Demo Center as the demo id **`grant`**, running
natively on the ZGX Nano (aarch64 / GB10). This file records what the integration actually
does and, where the shipped build made a different assumption, why that changed.

Earlier revisions of this file were a *draft* written without access to the launcher registry.
The open questions it listed have since been answered on real hardware; the answers are below.

## Port: 8095 — confirmed against the live registry

Checked against `install_plans.py`, the launcher's `PORT_CONFLICTS` / `PORT_PATHS` /
`HEALTH_CHECK_PATHS`, and `RUNBOOK.md`'s port map. In use: 8000 (+8083) maritime/pharmacy/pii,
8085 launcher UI, 8086 install API, 8088 pipeline, 8089 spectrum, 8090 medical, 8091 (+8191)
construction, 8092 (+8192) translation, 8093 ballistics, 8094 vision. **8095 was free and is
now Grant Machine's.**

The 8080/8081 model-server discussion in older revisions of this file no longer applies on the
Nano: the model backend is host Ollama on its existing 11434, so this integration allocates
**no model-server port at all**. `scripts/start_model_server.sh` is kept only for the portable
llama.cpp path.

## Model backend: host Ollama, not a llama.cpp sidecar

The Nano already serves `qwen3.6:35b` (22GB) from Ollama on `127.0.0.1:11434`, so there is no
GGUF download, no llama.cpp build, and no second GPU consumer. It also means Grant Machine
participates in the Demo Center's existing VRAM protocol: the launcher lists `grant` in
`OLLAMA_DEMOS` and calls `/ollama_release` before launching it, the same as the other
Ollama-backed demos.

**This required one real code change.** Ollama's OpenAI-compatible layer differs from
llama-server in two ways that are silent rather than loud — nothing errors, it simply behaves
differently:

1. **It names the separated-reasoning field `reasoning`, not `reasoning_content`** (in both the
   non-streamed message and the streamed delta). `app/llm_client.py` previously read only
   `reasoning_content`, so against Ollama it saw no reasoning at all and the UI's thinking
   panel would have been empty for the whole run. It now reads either name.
2. **It ignores `chat_template_kwargs.enable_thinking` entirely** — and ignores `think` on the
   `/v1` route too. The only knob it honours is `reasoning_effort`, where `"none"` is the sole
   value that actually suppresses thinking (`"low"` still thinks). Left on the llama.cpp
   default, the `drafter` and `reviewer` profiles would think when they must not, and reasoning
   would eat their `max_tokens` budget — the reviewer, which must return JSON at
   `max_tokens: 2000`, can come back with empty content that way.

So `llm.thinking_control` selects the request form: `chat_template` (the default — llama-server,
vLLM) or `reasoning_effort` (Ollama). `start_demo_8095.sh` exports
`GM_LLM__THINKING_CONTROL=reasoning_effort`. The llama.cpp request shape is unchanged by
default, and both forms are covered by tests in `tests/test_llm_client_normalization.py`.

Worth noting for anyone re-running the gate: **`endpoint_check.py`'s E3 does not catch this.**
E3 asserts only that `<think>` tags don't leak into `content`, which a server returning no
reasoning at all satisfies trivially. The signal is E3's `reasoning_source` metadata —
`field` means reasoning was captured, `none` means it was silently lost. Read that value, not
just PASS/FAIL.

## Python runtime: the shared `zgx` conda env, not `uv`

`uv` is not installed on the Nano and the committed `.venv` was an x86 build. Grant Machine
runs under `~/miniforge3/envs/zgx/bin/python3` (Python 3.12.13) like every other native demo.
That env already carried 13 of the 14 runtime dependencies at the exact versions
`pyproject.toml` asks for; the install step adds only `markdownify` and bumps `pydantic`.

`openai` is deliberately left at the 2.29.0 already in the env rather than raised to the
original `>=3.14.1` floor — every API this app uses exists in 2.x, the env is shared with the
TensorRT demos, and the E1–E12 endpoint check passes on it. The floor in `pyproject.toml` was
lowered to match reality.

## Configuration: `GM_*` env vars, not `config/local.yaml`

`start_demo_8095.sh` exports `GM_LLM__BASE_URL`, `GM_LLM__MODEL`, `GM_LLM__THINKING_CONTROL`,
`GM_SERVER__PORT` and `GM_DEMO__DEVICE_LABEL`. Settings load order is
`default.yaml` → `local.yaml` → `GM_<SECTION>__<KEY>`, so these win.

This is on purpose: it means launching the demo never writes to a tracked file. Two other Demo
Center demos `sed -i` their own source at launch time, which is recorded as a defect in
`KNOWN-ISSUES.md`, not a pattern to copy.

## Modes, and what each needs

| Mode | Needs | Notes |
|---|---|---|
| Story | Ollama only | The scripted scenario. Works fully offline. |
| Explore | Ollama only | Uploaded solicitation; extraction is local. Works offline. |
| Grade | Ollama only | Uploaded proposal. Works offline. |
| Discover | Ollama **plus network** | Reaches `api.grants.gov`; falls back to cached results offline. General web search additionally needs the optional SearXNG container. |

So three of the four modes still run on a fully air-gapped box. Only Discover reaches
out, and what it may send is constrained by `app/egress.py` and shown in the UI.

## Optional: SearXNG on 8096

`scripts/start_searxng.sh` starts a loopback-only SearXNG container and writes its own
config (including the `formats: [html, json]` that SearXNG does not enable by default).
It is an install step in `install_plans.py`, deliberately best-effort: a failure prints
a message and exits 0, because a missing search engine must not fail a demo install.
`GM_EGRESS__SEARXNG_URL` in `start_demo_8095.sh` points the app at it.

It is a long-lived side service like Ollama, so the demo's teardown does not stop it.

Note the interaction with the egress guard: the guard refuses non-public hosts, which
would include a loopback SearXNG. Configured endpoints are exempt from **that check
only** — the query terms are still validated, and the request is still audited.

## Health endpoint contract

`GET /api/health` → `200 OK` with
`{"status": "ok", "llm": "ok" | "unreachable", "model": "<alias>", "version": "<git short sha>"}`.

`status` is `"ok"` whenever the app process is up, even when `llm` is `"unreachable"` — a
launcher should read that as "app is up but not demo-ready", not as an app failure. The
launcher's `HEALTH_CHECK_PATHS['grant'] = '/api/health'` treats any 200 as live, which is the
right call: the model is contacted lazily, per run, not at startup.

`version` now reports the *monorepo's* short sha, since the demo is vendored into
`zgx-demo-center` and no longer carries its own `.git`. That is the intended behaviour.

## UI

Served at `/` via `StaticFiles(html=True)`, mounted after all API routes so `/api/*` always
takes precedence. No build step, no CDN, no web fonts.

## Launcher registry entry

The real entry lives in `install_plans.py` under the id `grant`. Teardown is `port_kill` rather
than the `process_pattern` used by vision/pipeline/spectrum: those bind their port only after a
long model load, so a port-kill can miss them, whereas uvicorn here binds `:8095` in about a
second (it builds a BM25 index over a small vault and loads no model). `-m app.main` would also
be far too generic a `pgrep` pattern to kill by name on a shared box.

The launcher's own half of the registry is maintained by hand across roughly fifteen sites in
`zgx-demo-launcher.html` (`PORT_CONFLICTS`, `PORT_PATHS`, `HEALTH_CHECK_PATHS`, `MEM_GB`,
`LOAD_TIMES`, `LAUNCH_BUDGET_S`, `OLLAMA_DEMOS`, `DEMO_IDS`, `STEP_COUNTS`, `DEMO_KEY_BY_ID`,
the card, the install row, and the `DEMOS` modal entry). See `KNOWN-ISSUES.md` — "The port map
is maintained by hand in three places" — before changing the port.

## Container

`docker/compose.yaml` is retained as the portable path and is **not** what the Demo Center
uses; the Nano runs the demo as a host process. If you do use it, `GM_LLM__BASE_URL` must point
at the model endpoint and `network_mode: host` already lets the container reach host Ollama on
loopback.
