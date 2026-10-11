# Grant Machine

A local, agentic AI demo for research offices and grants teams. Three things, one agent,
one model, entirely on-device:

| Mode | What it does |
|---|---|
| **Draft** | Pick one of six institutions — community college, research institute, R1 university, Hispanic-Serving Institution, academic medical centre, rural consortium — and it drafts to that institution's solicitation from its synthetic document vault: deterministic budget checks, cited sections, self-review and revision. Either half can be swapped for your own: upload *any* solicitation (PDF/HTML/Markdown) and it extracts that document's requirements at runtime, keeping only the ones it can quote verbatim; upload your own documents and they become the evidence. Every institution's documents and solicitation are browsable under the picker, or as a full-page brief. |
| **Find Funding** | Upload your institution's documents. It writes a cited capability profile, searches the live federal funding catalogue, and returns a short reasoned list of opportunities you could pursue — with the eligibility concerns a human has to check. |
| **Grade** | Upload a proposal you have already written. It scores every section as a panel would, rules on every requirement, and cross-checks your figures against your own records. |

## What leaves the device

Your documents never do. The agent can research the public web, but a search term is
only transmitted if it matches a published vocabulary of research-field and federal
funding-taxonomy names (`data/vocabulary/`). Anything lifted from your uploads, any
person's name, any award number is refused *before* transmission — and every request,
every refused term and the reason for each is on screen in the **Egress** tab and in
the exported packet.

That boundary is enforced in `app/egress.py`, which is the only code in `app/` allowed
to reach the network; `scripts/audit_portability.sh` fails the build if anything else
tries.

## The guardrails are in the harness, not the prompt

- It cannot finish a writing run until every section is drafted, reviewed and
  citation-verified.
- It cannot recommend a grant the catalogue did not actually return.
- It cannot finish a grading run leaving any requirement unruled — a skipped
  requirement reads as compliant to whoever gets the scorecard.
- It cannot mark a requirement met while the citations behind it carry unsupported
  claims (writing), or without naming the section that satisfies it (grading).

The default institution for Draft mode is *Harrow Creek Community College* (fictional)
applying to the real NSF Advanced Technological Education solicitation (NSF 24-584),
Track 2. All institutional documents, people and figures in `data/vault/` are
synthetic; the solicitation text is the real, public NSF 24-584 (U.S. Government work,
public domain).

The five other scenarios under `data/scenarios/` pair a fictional institution with a
**real, currently posted** federal solicitation, fetched by
`scripts/fetch_scenario_solicitations.py` and verified against the rules each scenario
depends on:

| Scenario | Applies to |
|---|---|
| Ashgrove University Medical Center | NIH PA-25-305, Parent R01 Clinical Trial Required |
| Vantcross State University | NIH PA-25-301, Parent R01 Clinical Trial Not Allowed |
| Calderwick Institute for Coastal Systems | NSF 26-516, GEO Core (Ocean Sciences) |
| Sablecreek State University System | NSF 25-523, EPSCoR E-CORE |
| Tierra Sandoval University | NSF 25-514, S-STEM Track 2 |

Each budget is scaled from comparable real funded awards (`data/anchors/`, pulled by
`scripts/fetch_anchors.py`), and each scenario's planted issues are real failure modes
of *its* call — the NIH salary cap, subaward F&A limits, cost share an E-CORE award does
not allow, S-STEM's scholarship maximum — with the rule quoted from its source in
`solicitation/budget_rules.yaml`. No two scenarios plant the same set, and each drafts
its call's own sections: an academic medical centre writes *Specific Aims*, not
*Results from Prior NSF Support*. See `DATA_SOURCES.md` for where every file under
`data/` came from.

## 60-second quickstart

### On the HP ZGX Nano (the Demo Center path)

Grant Machine is Demo Center demo id `grant` on port **8095**. Install and launch it from the
launcher UI, or directly:

```bash
bash ~/demos/grant-machine/start_demo_8095.sh
```

That script picks the `zgx` conda env interpreter and exports this machine's config as `GM_*`
env vars (host Ollama on 11434, model `qwen3.6:35b`, `thinking_control=reasoning_effort`).
Prerequisite: Ollama running with `qwen3.6:35b` pulled. See
`docs/DEMO_CENTER_INTEGRATION.md`.

### Anywhere else

You need an OpenAI-compatible endpoint with tool calling, then this app.

**1. Point at a model server.** Any OpenAI-compatible `/v1` endpoint that passes
`scripts/endpoint_check.py` works -- Ollama, llama.cpp, vLLM. For llama.cpp:

```bash
export LLAMA_SERVER_BIN=/path/to/llama-server
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
bash scripts/start_model_server.sh          # LLM_PORT to override the port
```

**2. Start the app:**

```bash
uv sync                                     # or: pip install -e . into a 3.12+ env
cp config/local.yaml.example config/local.yaml   # set llm.base_url for your endpoint
uv run python -m app.main
```

**3. Verify the endpoint actually satisfies the agent's contract** before trusting a run --
this catches the differences between servers that otherwise fail silently:

```bash
uv run python scripts/endpoint_check.py --base-url <url> --model <alias> --reps 3 --label <name>
```

**4. Open** `http://localhost:8095`. Pick a mode (Draft/Find Funding/Grade) and scope (Quick/Full) in the
left panel and click **Start**.

To run in a container instead of step 2: `docker compose -f docker/compose.yaml up -d --build`.

**Optional — general web search for Find Funding.** `search_web` needs a self-hosted
SearXNG; without one it reports itself unavailable and the federal catalogue search
(which needs no provider) carries the run.

```bash
bash scripts/start_searxng.sh          # docker, loopback-only, writes its own config
export GM_EGRESS__SEARXNG_URL=http://127.0.0.1:8096
```

There is deliberately no commercial search provider: an API key on a demo box, and
queries routed through a third party, would both undercut the point.

## Configuration

Settings load in this order, each layer overriding the last: `config/default.yaml` (committed,
generic defaults) -> `config/local.yaml` (gitignored, your machine's overrides -- copy
`config/local.yaml.example` to create it) -> `GM_<SECTION>__<KEY>` environment variables (e.g.
`GM_LLM__BASE_URL=http://127.0.0.1:11434/v1`).

Common things to override in `config/local.yaml`:
- `egress.searxng_url` -- self-hosted SearXNG for Find Funding's general web search; empty disables it
- `egress.max_requests_per_run` -- outbound request budget per run
- `llm.base_url` / `llm.model` -- point at your model server
- `llm.thinking_control` -- `chat_template` (llama-server, vLLM) or `reasoning_effort`
  (Ollama, which ignores `chat_template_kwargs` entirely). Getting this wrong is silent, not
  an error: thinking is simply never captured, or never suppressed. See
  `docs/DEMO_CENTER_INTEGRATION.md`
- `llm.context_tokens` -- see Phase 2.5 in `docs/BUILD_PLAN.md` for how to pick this for new hardware
- `server.port` -- if 8095 is taken
- `demo.device_label`, `demo.human_baseline_minutes`, `demo.cloud_pricing` -- presenter-facing values; the UI shows "not configured" rather than a fabricated number until you set these
- `model_server.llama_server_bin` / `model_server.model_path` -- read only by `scripts/*.sh`, never by the app itself

## Troubleshooting

**"Model endpoint unreachable" / `/api/health` shows `"llm": "unreachable"`**
Check the model server is actually running and `llm.base_url` in your config points at it:
`curl -s $(grep base_url config/local.yaml | awk '{print $2}')/models`. The health check is
cached for 5s and uses a 2s timeout, so a slow-to-load model server can transiently show
unreachable right after you start it.

**Slow time-to-first-token / long waits before anything happens**
Normal for local reasoning models -- HTTP client timeouts are set to 900s on purpose (Section
0.9 of the build plan) and thinking tokens are never truncated by a low `max_tokens`. If a
*quick-scope* run's median wall time is far over ~6 minutes on your hardware, that's a known,
documented finding (see `PROGRESS.md`, Phase 10) -- not a bug to chase alone. Use the Replay
mode (a committed example run is under `data/replays/`) for time-boxed live demos.

**A quick-scope run ends at `needs_human` instead of `completed`**
Expected on some scenarios, and worth understanding before you show it. On Ashgrove
(quick scope) the first draft verifies clean -- 20 citations, all resolved -- but the
revision passes then score *worse* than the draft they were revising, so the run exits
via the finish gate as `needs_human` rather than claiming a pass it did not earn.

That is the harness working: it will not mark a run complete on drafts that failed
review. It is a model-behaviour limit on that scenario, not a code fault, so there is no
switch to flip. If you need a guaranteed clean finish for a timed demo, use Harrow Creek
(the scenario the vault and the curated NSF 24-584 requirements were built around) or
Replay mode. Reported honestly here because a demo that argues it does not overstate
results should not overstate its own.

**Port already in use**
`config/default.yaml` assumes the app on 8095 and a model server on 8080. If either is taken
(check with `ss -ltnp | grep <port>`), set an alternate in `config/local.yaml`
(`llm.base_url` / `server.port`) and, for a llama.cpp model server, `LLM_PORT` when running
`scripts/start_model_server.sh`. On the ZGX Nano neither applies: 8095 is reserved for this
demo in the Demo Center's port map, and the model endpoint is host Ollama on 11434.

**WSL2 memory**
If `scripts/preflight.sh` reports less than ~48GB total memory, the model likely won't load
with headroom to spare. Edit `%UserProfile%\.wslconfig` on the Windows side (see Section 0.A of
`docs/BUILD_PLAN.md` for the exact block), then `wsl --shutdown` and reopen your WSL terminal.

**`docker compose version` fails with "unknown command"**
This machine's Docker install may be the plain `docker.io` package without the compose plugin.
On Ubuntu, `sudo apt install docker-compose-v2` (not `docker-compose-plugin` -- that's Docker's
own repo naming, which isn't configured by default on Ubuntu).

## Development

On the Nano, substitute `$HOME/miniforge3/envs/zgx/bin/python3 -m ...` for `uv run python -m ...`
(there is no `uv` on that box).

```bash
uv run pytest -q                        # unit tests, no GPU/model needed
bash scripts/license_audit.sh           # fails on any GPL/AGPL/LGPL dependency
bash scripts/audit_portability.sh       # fails on hardcoded paths, external URLs, GPU deps
uv run python scripts/endpoint_check.py --base-url <url> --model <alias> --reps 3 --label <name>
uv run python scripts/run_e2e.py --base-url http://127.0.0.1:8095 --scope quick --reps 3 --label <name>
```

See `docs/BUILD_PLAN.md` for the full phase-by-phase build plan this repo was built from,
`docs/ARCHITECTURE.md` for how the pieces fit together, `docs/PRESENTER_SCRIPT.md` for the demo
talk track, and `PROGRESS.md` for a running log of what was built, verified, and found along
the way (including honest reporting of what doesn't yet meet target -- see Phase 10).
