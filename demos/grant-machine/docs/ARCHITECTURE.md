# Architecture

```
+-------------------------------- host machine (WSL2/Linux) ---------------------------------+
|                                                                                              |
|  llama-server (ROCm/CUDA/vLLM)              grant-machine app (Python, no GPU deps)          |
|  Qwen3.6-35B-A3B GGUF/NVFP4          <-- OpenAI-compatible HTTP -->   FastAPI :8095          |
|  127.0.0.1:<port> /v1/chat/completions                                   |                   |
|                                                                            +-- agent loop     |
|                                                                            +-- tools          |
|                                                                            |   (BM25 vault    |
|                                                                            |    search,       |
|                                                                            |    budget check, |
|                                                                            |    citation      |
|                                                                            |    verify,       |
|                                                                            |    compliance)   |
|                                                                            +-- SSE event bus  |
|                                                                            +-- static web UI  |
+-----------------------------------------------------------------------------------------------+
                        Browser -> http://localhost:8095 (or wherever server.port is configured)
```

The model server is a separate process the app talks to over plain HTTP (OpenAI-compatible
`/v1/chat/completions` and `/v1/models`) -- swapping hardware or inference stack (llama.cpp on
ROCm today, llama.cpp/vLLM on CUDA for the ZGX Nano later, see `docs/PORTING_TO_ZGX_NANO.md`)
is a config change (`llm.base_url`), not a code change.

## Component responsibilities

| Module | Responsibility |
|---|---|
| `app/settings.py` | Loads `config/default.yaml` -> `config/local.yaml` -> `GM_*` env vars, validates with pydantic, fails fast and readably on bad config. |
| `app/llm_client.py` | The only code that talks to the model server. Normalizes two server styles (`reasoning_content` field vs. inline `<think>` tags) into one `ChatResult` shape; streaming tool-call fragment assembly; usage estimation when the server omits it; retry-with-backoff. |
| `app/events.py` | `Event`/`EventBus`: per-run event log with subscriber fanout, persisted to `runtime/runs/<run_id>/events.jsonl`. |
| `app/metrics.py` | Pure-function cost/human-baseline math (Spec I) -- kept separate from the orchestrator so it's testable without a running agent. |
| `app/netstatus.py` | Internet reachability probe, endpoint host classification (loopback/private/CGNAT/public), `psutil`-based app-egress connection count. Polled by a background thread in `app/main.py` that emits `network_status` events into the active run. |
| `app/ingest/` | `loaders.py` (`.md`/`.html`/`.pdf`/`.csv` -> normalized `Document` with front matter and heading-aware sections), `chunker.py` (sliding-window chunks with stable `{doc_id}#c{NNN}` ids). |
| `app/retrieval/bm25_index.py` | `rank-bm25` search over chunked vault text. No embedding model. |
| `app/checks/budget.py` | Deterministic budget validation (`BUDGET_OVER_CAP`, `FRINGE_RATE_MISMATCH`, `INDIRECT_ON_EXCLUDED`, `YEAR_SUM_MISMATCH`, `DURATION_OVER_LIMIT`) -- parses the actual budget CSV and rate-agreement document text, not the seed file, so it's a genuine independent check on whatever the agent (or a human) puts in the vault. |
| `app/checks/citations.py` | Verifies every `[[doc_id]]`/`[[doc_id#chunk]]` citation exists and every number near a citation appears in the cited document. Reports `citation_count` and any `[[...]]` marker it could not parse, and fails a draft that has either — a check that passes because nothing parsed has verified nothing. |
| `app/agent/state.py` | `RunState`: the compliance matrix, draft versions, reviews, citation checks, issues, tool-call counts. Persisted to `runtime/runs/<run_id>/state.json` after every tool result. |
| `app/agent/tools.py` | The 10 tools the model can call, as strict JSON-Schema-validated functions, plus the harness guardrails (below). |
| `app/agent/orchestrator.py` | The tool-calling loop: one LLM call per step, execute whatever tools come back, feed results back in, repeat until `finish` is accepted or `max_steps` is hit. |
| `app/agent/prompts.py` | System prompts as constants, including the scope-aware "quick" vs. "full" guidance (Phase 10). |
| `app/main.py` | FastAPI routes under `/api/*`, SSE streaming, the run manager (one active run at a time), static UI mount. |
| `app/replay.py` | Copies a finished run's `events.jsonl` into `data/replays/`, and replays one with (capped) scaled timing. |
| `app/egress.py` | **The only code in `app/` that reaches the network.** Default-deny term validation against `data/vocabulary/`, a payload token scan, a session-corpus backstop, transport guards (public hosts only, HTTPS, size cap, per-run budget), and an audit record for every decision. Operator-configured endpoints (a self-hosted SearXNG) are exempt from the host-class check only. |
| `app/discovery/` | `grants_gov.py` (search2 / fetchOpportunity) and `web.py` (SearXNG search, page fetch → Markdown). Fetched pages are written into the session corpus with a `doc_id` so they are cited and verified like any other document. |
| `app/ingest/requirements.py` | Runtime requirement extraction from an uploaded solicitation, shared with `scripts/extract_requirements.py`. Every candidate is kept only if its `source_quote` is verbatim in the source. Cached by content hash. |
| `app/checks/claims.py` | Grading's analogue of citation verification: numeric claims in a submitted proposal are cross-checked against the vault and reported supported / contradicted / unverifiable. |
| `app/packet.py` | The run packet as a paginated PDF — compliance matrix, issues, opportunities, claim findings, and what left the device. |
| `web/` | Plain HTML/CSS/JS (ES modules, no build step, no CDN, no web fonts) consuming the SSE stream. |

## Event flow

1. `POST /api/runs` creates a `RunState` + `EventBus`, starts a background thread running
   `run_orchestrator()`, returns `{run_id}` immediately.
2. The browser opens `GET /api/runs/{run_id}/events` (SSE), which replays any events already
   emitted (`Last-Event-ID` support for reconnects) then streams live ones.
3. Each orchestrator step: one LLM call (`orchestrator` profile, thinking on, streamed) ->
   `thinking_delta`/`content_delta` events as it streams -> if the model called tools, each one
   executes sequentially (`tool_call` then `tool_result` events, plus domain events like
   `compliance_update`, `issue_flagged`, `draft_delta`, `section_scored`, `citations_checked`)
   -> results feed back into the conversation -> next step. A throttled `metrics` event (>=1/s)
   carries elapsed time, token counts, decode tok/s, and cost/baseline math.
4. `finish` is only accepted once the harness guardrails (below) are satisfied; the run ends
   with a `run_finished` event carrying the final status and a summary.
5. Everything is also written to `runtime/runs/<run_id>/state.json` and `events.jsonl` as it
   happens, which is what `save-replay` and the `GET .../state` endpoint read from.

## Playbooks

One orchestrator loop, four run modes. `ToolContext.playbook` selects the system prompt, the
task brief, **which tools the model is offered at all**, and the finish gate:

| Playbook | Modes | Notable exclusions |
|---|---|---|
| `proposal` | Story, Explore | — |
| `discovery` | Discover | no `draft_section` / `check_budget` — there is no solicitation to write to |
| `grade` | Grade | no `draft_section` (an assessor must not be able to rewrite what it scores) and no `verify_citations` (a submitted proposal has no `[[doc_id]]` markers) |

## Harness guardrails (enforced in code, not just prompted)

These are what make the "the harness enforces the policy, not just the prompt" presenter line
(Spec J) actually true:

1. **`finish` is rejected** unless `get_requirements` and `check_budget` have run, and every
   in-scope section has `draft_section` -> `review_section` -> `verify_citations` on its latest
   version. The rejection tells the model exactly what's missing.
2. **`update_compliance` with status `met`** is rejected for any requirement tied to a section
   whose latest version hasn't passed `verify_citations` with zero unsupported claims.
3. **Revisions are capped** at `agent.max_revisions_per_section` (default 2). Once reached, the
   requirement is marked `needs_human` automatically -- and, as of Phase 10's bug fix,
   `draft_section` *refuses outright* on further attempts for that section rather than silently
   letting the model keep trying (an earlier version of this only did the first half, and one
   live run drafted the same section 5 times chasing an unreachable score).
4. **Repeated identical calls** (same tool name + same arguments 3+ times in a row) get an
   error result telling the model to change approach, instead of executing again.
5. **Tool results are truncated** to `agent.max_tool_result_chars` (default 6000) when sent
   back to the model; the full result stays in `RunState` and the UI regardless.
6. **Discovery cannot invent a grant.** `propose_opportunity` refuses any opportunity id the
   catalogue did not return this run, and any rationale without a `[[doc_id]]` citation.
7. **Grading cannot skip a requirement.** `finish` is refused while any requirement has no
   compliance ruling — a requirement nobody looked at reads as compliant to whoever receives the
   scorecard.
8. **A draft with nothing to verify does not pass.** `verify_citations` fails a section whose
   `[[...]]` markers could not be parsed, and one with no citations at all.
9. **The institution's documents cannot leave.** Enforced in `app/egress.py` in three layers, and
   audited whether allowed or refused.
10. **History compaction**: once the estimated token count crosses 65% of `llm.context_tokens`,
   tool results older than the last 6 are replaced with a deterministic one-line summary (not
   LLM-generated) rather than dropped or left to blow the context window.

## Security

Markdown from the model (drafts, citation excerpts) is rendered client-side through a small
built-in renderer that escapes HTML first and only then applies a limited set of Markdown
transforms -- no `innerHTML` of raw model output. The UI makes no external network calls (no
CDN scripts, no web fonts, no analytics); the only non-localhost reference anywhere in `app/`
or `web/` is the plain-text (non-hyperlinked) NSF solicitation attribution in the footer.
