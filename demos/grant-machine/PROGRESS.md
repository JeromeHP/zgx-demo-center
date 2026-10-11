# Progress Log

## Phase 0 — Preflight (2026-09-16)

Ran `scripts/preflight.sh`. Full output: `reports/preflight_20260916T162606Z.txt`.

Decision table:

| Check | Result | Status |
|---|---|---|
| WSL2 kernel | `6.18.33.2-microsoft-standard-WSL2` | Pass |
| GPU device | `/dev/dxg` present; ROCm0 "AMD Radeon(TM) 8060S Graphics" listed by the existing llama-server, 108412 MiB free | Pass |
| Memory | `free -g` total 62 GB, 58 GB free (`.wslconfig`: `memory=64GB`, `swap=16GB`) | Pass — was 27 GB earlier this session; `.wslconfig` has since been updated to 64GB/16GB swap and WSL restarted. No HUMAN STEP needed at this point. |
| uv | `uv 0.12.8` | Pass |
| Docker | server `29.1.3`, Ubuntu 24.04.4, x86_64 | Pass (not blocking pre-Phase 12 anyway) |
| Ports | 8080 occupied, 8095 free | Deviation — 8080 is held by a `docker-proxy` for an unrelated `searxng` container (not llama.cpp). Grant Machine's model server will default to port **8081** instead, recorded in `config/local.yaml` once it exists (Phase 1). |
| Disk | 828 GB free on `/` | Pass |

Existing llama.cpp (not to be modified): `/home/gsept/llama.cpp/build/bin/llama-server`, version `0.3.0-dev (build 10729, commit 458681e1d)`. `--list-devices` confirms it already sees the ROCm0 gfx1151 device, so it's a live candidate for Phase 2.2 (test whether it loads Qwen3.6's `qwen3_5_moe` architecture) — build 10729 is fairly recent, so this may well succeed without needing a separate build.

**Deviation from the plan document:** repo root is `/home/gsept/grant-machine`, not `~/projects/public-sector-demos/grant-machine` (confirmed with Jerome). This doesn't affect portability rules since the app/tests/web code must not hardcode either path anyway. Models still live outside the repo at `~/models/qwen3.6-35b-a3b/` per the doc.

**Acceptance:** all decision-table rows Pass or have a recorded resolution. `git init -b main` done. Committing phase 0.

## Phase 1 — Repository scaffold, config, LLM client (2026-09-16)

- `uv init --app --python 3.12 --name grant-machine`; removed the `src/` layout uv generates by default and set `[tool.uv] package = false` so the app lives directly under `app/` per Section 3's repo layout, not `src/grant_machine/`.
- Added deps: `fastapi uvicorn[standard] sse-starlette openai pydantic pyyaml rank-bm25 pypdf beautifulsoup4 markdownify httpx psutil jsonschema python-multipart`, dev: `pytest pytest-asyncio pip-licenses`.
- `config/default.yaml`, `config/local.yaml.example` per Spec A. `config/local.yaml` (gitignored) created with this machine's real override: `llm.base_url` → `http://127.0.0.1:8081/v1` (port 8080 is an unrelated searxng container, see Phase 0) and the `model_server` paths for this box.
- `app/settings.py`: pydantic v2 models mirroring Spec A exactly (`extra="forbid"` throughout so unknown keys fail loudly), deep-merge of `local.yaml`/`GM_CONFIG`, `GM_<SECTION>__<KEY>` env overrides parsed as YAML scalars, `resolve_path()` helper for repo-root-relative paths, `SettingsError` for readable failures.
- `app/llm_client.py`: `LLMClient`/`ChatResult`/`ToolCall`/`Usage`/`Timings` per Spec B. Handles both `reasoning_content`-field and inline-`<think>` servers (mode `auto`/`field`/`inline_tags`), including think-tag splits across stream chunk boundaries; streaming tool-call fragment assembly with `parse_error` on malformed JSON; usage estimation (`chars/3.5`) with `estimated=True` when the server omits `usage`; retry-with-backoff on connection errors/5xx, never on 4xx; one repair turn for `json_mode`.
- `tests/fake_llm_server.py`: scripted OpenAI-compatible FastAPI app (streaming + non-streaming, configurable chunking) — built now per the doc's Phase 6.4 note, reused going forward.
- `tests/conftest.py`: `make_settings()` fixture for in-memory Settings, `tmp_repo` for file-based settings tests, and a real-socket uvicorn-in-a-thread fixture (`fake_llm_server`) — `httpx.ASGITransport` only implements the async transport interface and `LLMClient` uses the synchronous `openai.OpenAI` client per Spec B, so an in-process ASGI transport can't be wired to it; a loopback socket sidesteps that and exercises the real HTTP/SSE path.
- `app/main.py`: minimal `GET /api/health` → `{"status":"ok","llm":"unknown"}`, `python -m app.main` entry.
- Copied the build plan into the repo as `docs/BUILD_PLAN.md` (versioned with the code, per Phase 1.1).

**Verification:**
```
$ uv run pytest -q
21 passed in 5.02s

$ curl -s http://127.0.0.1:8095/api/health
{"status":"ok","llm":"unknown"}
```

**Acceptance:** pytest passes; health endpoint responds on the configured port. Committing phase 1.

## Phase 2 — Model server (2026-09-16)

**2.1 download:** completed and verified — `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` is exactly `22360456160` bytes at `~/models/qwen3.6-35b-a3b/`.

**2.2 existing llama.cpp compatibility:** tested the **existing** binary at `~/llama.cpp/build/bin/llama-server` (version `0.3.0-dev`, build `10729`, commit `458681e1d`) against the new model, unmodified, on a throwaway port (18080). It loaded and served `/health` -> `{"status":"ok"}` without any `unknown model architecture` error — this build already supports Qwen3.6's `qwen3_5_moe` architecture. **No separate ROCm build (2.3) was needed.** Recorded `model_server.llama_server_bin` in `config/local.yaml`.

**2.4 launch script:** `scripts/start_model_server.sh` per the doc's spec verbatim, with one deviation: the default `LLM_PORT` is `8081`, not `8080` (see Phase 0 -- port 8080 is held by an unrelated `searxng` container on this box, not by Jerome's llama.cpp setup).

**2.5 context sizing:**
| Context | Load time | Result |
|---|---|---|
| 32768 | 18s | `/health` ok, smoke test passed |
| 65536 | 12s | `/health` ok, a real 19,303-token prompt (with a planted fact "ZEBRA-4471" buried in the middle) completed in 83.8s and correctly returned the planted fact |

Kept **65536**. Updated `config/default.yaml`'s `llm.context_tokens` from the placeholder `32768` to `65536` (the actual chosen value, per the config file's own comment instructing this). Both `--load-mode none` load times are well under the ~4 minute figure the doc cites as the pre-`--load-mode-none` baseline on this machine -- confirms that flag's effect here too (12-18s).

**2.6 smoke test:** `/health` ok; `/v1/models` lists alias `qwen3.6-35b-a3b` (`n_params=34.66B`, `n_ctx_train=262144`, quant `Q4_K - Medium`); chat completion with `enable_thinking:false` returned `"ready"` with real `usage` fields present (`prompt_tokens`, `completion_tokens`) -- confirms E10 (usage fields) will pass without falling back to estimates.

**Acceptance:** GGUF size verified; existing llama-server serves the model; `/health` ok; `/v1/models` lists the alias; smoke completion non-empty; binary path/commit/context/load time all recorded above. Committing phase 2.

## Phase 2.1 — Model download started (2026-09-16)

Kicked off `scripts/download_model.sh` in the background (`nohup ... &`, logging to `reports/download.log`) per the plan's own instruction to start the ~22GB download first and work on model-independent phases while it runs. Installed `huggingface_hub[cli]` via `uv tool install` (user-space, no sudo) — this provided the `hf` command.

Continuing with Phase 4.1/4.3/4.4(csv) and Phase 5 while the download runs in the background; will return to finish Phase 2.2 onward once the file size verifies against `22360456160` bytes.

## Phase 4 (partial) — Solicitation, seeds, deterministic vault CSVs (2026-09-16)

Doing this phase in two passes per the plan's own guidance ("if the model endpoint is unavailable... implement the script, generate the CSVs, and come back to the narrative documents after Phase 3 passes"). This entry covers everything that doesn't need the model; a follow-up entry will cover the LLM-generated narrative documents once the model server is confirmed working.

**4.1 — Solicitation fetch:** `scripts/fetch_solicitation.py` fetches `https://www.nsf.gov/funding/opportunities/ate-advanced-technological-education/nsf24-584/solicitation`, strips nav/header/footer/scripts, converts to Markdown with heading hierarchy preserved, and greps the result for the required facts. Before running it, I independently checked both the solicitation page and the ATE program page directly (via a one-off fetch) to confirm NSF 24-584 is still the current, non-superseded solicitation — it is.

Verified present in the fetched text:
- Track 2 award cap $1,000,000 — OK
- Track 2 duration up to three years — OK
- Project Description 15-page limit — OK
- Deadline: first Thursday in October — OK
- Evaluation Plan subsection required — OK
- Results from Prior NSF Support required — OK

Wrote `data/solicitation/nsf24-584.html` (168,678 bytes) and `.md` (98,085 chars), and recorded provenance (URL, retrieval timestamp, SHA-256, U.S. Government work / public domain) in `DATA_SOURCES.md`.

**4.3 — Seed facts:** Before writing `data/seeds/harrow_creek.yaml`, searched the web for each fictional name (institution, the three employers, the evaluator org) per the plan's instruction. No exact-name collisions with real organizations found — closest was "Tri-County Packaging Supply Corp." (Madison Heights, MI), a real but differently-named company, not a match for the seed's "Tri-County Packaging Cooperative." Wrote the seed file verbatim per Spec D.1, unchanged.

**4.4 (CSV half) — Deterministic tabular documents:** `scripts/generate_vault.py --csv-only` generates:
- `data/vault/budget-satp-draft.csv` — solved so fringe = 28% of personnel exactly, indirect = 36% of the MTDC base (excluding equipment/participant_support), and the grand total reconciles to exactly the seed's `budget_total_usd` ($1,038,450) via a final residual correction on the last year's `materials_supplies` line (the designated balancing item, adjusted by -$1 here). Confirmed: over the $1,000,000 Track 2 cap by exactly $38,450 (planted issue P1). `total_if_agreement_fringe_usd` (using the 31.5% rate agreement figure instead of 28%) computes to $1,058,529 — confirms correcting the fringe rate pushes the total further over cap, as intended for P4.
- `data/vault/prior-outcomes-data.csv` — per-year completers (15/22/27/20) sum to exactly 84 against the prior award's target of 120, matching the seed.
- `data/vault/manifest.json` updated with both entries.

**Deviation:** `rate-agreement-fy2026` (Spec D.2 marks this as an LLM-generated narrative doc) has NOT been generated yet — it's needed by Phase 5.3's `FRINGE_RATE_MISMATCH` check to parse the agreement's stated rate from real document text, but generating it requires the model server. Phase 5's own unit tests use inline fixture documents instead of the real vault, so this doesn't block Phase 5. The "`check_budget` against the real vault" acceptance bullet is deferred until this document exists (tracked as a skip in `tests/test_budget.py::test_real_generated_vault_has_exactly_the_planted_budget_issues`).

**Not yet done:** the remaining Spec D.2 narrative documents (institutional profile, strategic plan, prior proposal/panel summary/annual report, evaluation report, declined proposal, advisory board minutes, labor market memo, the three collaboration/support letters, budget justification, rate agreement memo, bios, facilities, boilerplate). These need the model server (Phase 2/3) and will be generated in a follow-up pass, after which Phase 4 will be considered complete and `validate_vault.py` (4.5) will run.

## Phase 5 — Ingestion, retrieval, deterministic checks (2026-09-16)

Built independently of the model server, per the plan's own scheduling (Phases 5-7 run "while downloading").

- `app/ingest/loaders.py`: `load()` dispatches on extension to `.md` (YAML front matter + ATX heading sections), `.html`/`.htm` (bs4 + markdownify, nav/header/footer/script stripped), `.pdf` (pypdf, one section per page), `.csv` (`# key: value` leading comment lines as front matter — CSV has no native front-matter convention — rendered as a Markdown table, raw rows kept in `csv_rows` for `checks/budget.py`). Rejects files > 20MB and PDFs > 200 pages with `LoaderError`.
- `app/ingest/chunker.py`: sliding window over each document's words (attributed to their section heading), ~425 target words, 50-word overlap, stable `{doc_id}#c{NNN}` ids.
- `app/retrieval/bm25_index.py`: `rank-bm25` over chunked text; lowercase/non-alphanumeric-split tokenizer with a small built-in English stopword list (numbers always kept); 400-char snippets centered on the earliest matching query term.
- `app/checks/budget.py`: `check_budget()` returns `BUDGET_OVER_CAP`, `FRINGE_RATE_MISMATCH` (parses the fringe rate out of the rate-agreement document's actual text via regex, not from the seed), `INDIRECT_ON_EXCLUDED`, `YEAR_SUM_MISMATCH`, `DURATION_OVER_LIMIT`, plus `total_if_agreement_fringe_usd` computed generically from the CSV's own `fringe_applied`/`indirect_applied` rate columns (no dependency on the seed file, so it works against any budget CSV shaped like Spec D.2).
- `app/checks/citations.py`: `verify_citations()` parses `[[doc_id]]`/`[[doc_id#chunk_id]]` citations per sentence, checks doc/chunk existence, extracts numbers (currency/percentage/ISO date/Month-YYYY/integers ≥ 10) from citation-stripped sentence text, and checks each appears in the cited document(s) after normalization (strip `$`/commas/trailing `.0`). Uncited numbers are reported as warnings (`uncited_numbers`), not failures. Known limitation documented in-code: the substring-based number match can false-positive when one number is a prefix of another number's decimal expansion (e.g. "84" inside "0.84") — accepted for this demo's scope.
- Tests: `test_loaders_chunker.py`, `test_bm25.py`, `test_budget.py`, `test_citations.py` (49 tests total across the whole suite now).

**Verification:**
```
$ uv run pytest -q tests/test_loaders_chunker.py tests/test_bm25.py tests/test_budget.py tests/test_citations.py
28 passed in 0.84s

$ uv run pytest -q
49 passed in 6.22s
```

`check_budget` against the real `data/vault` directory currently only exercises `BUDGET_OVER_CAP` (confirmed present, correct) since `rate-agreement-fy2026.md` doesn't exist yet; `FRINGE_RATE_MISMATCH` against the real vault is re-verified once Phase 4's narrative docs land.

**Acceptance:** all four Phase 5 test files pass; `BUDGET_OVER_CAP` confirmed on the real vault with no false positives from the other four check codes. Committing phase 5 (Phase 4's partial work is committed separately, just before this).

## Phase 6 — Agent harness (2026-09-16)

Built and fully tested against the fake LLM server; no GPU or model needed.

- `app/events.py`: `Event`/`EventBus` (Spec G envelope: `run_id, seq, ts, type, data`), in-memory with an optional `events.jsonl` log and subscriber fanout. Built now (ahead of its formal Phase 7 slot in the repo layout) because the orchestrator needs somewhere to emit to; Phase 7 wraps this with SSE.
- `app/agent/state.py`: `RunState` (compliance/drafts/reviews/citation_checks/issues/tool_calls_made/steps/status/summary) exactly per 6.1, persisted to `runtime/runs/<run_id>/state.json` on every tool result, drafts also written to `drafts/<section_id>_v<n>.md`.
- `app/agent/prompts.py`: the four Spec F prompt templates verbatim (orchestrator/drafter/reviewer system prompts, task brief), plus the section id/title/word-target tables Spec E's `draft_section` row describes.
- `app/agent/tools.py`: all 10 Spec E tools with strict (`additionalProperties: false`) JSON Schemas, validated via `jsonschema` before execution — invalid args never reach the tool body, matching "the loop continues" requirement. `update_compliance` enforces guardrail 2 (rejects `met` unless the linked section's latest citation check passed with zero unsupported claims) before it ever calls state; `finish` enforces guardrail 1 by checking `get_requirements`/`check_budget` call counts plus draft→review→citation-check recency per in-scope section.
- `app/agent/orchestrator.py`: the loop from Section 6.3's pseudocode, tool calls executed sequentially even when several arrive in one response; repeated-identical-call guard (>3 consecutive same name+args → error result); tool results truncated to `agent.max_tool_result_chars` for the model while the full result stays in state/events; history compaction at 65% of `context_tokens` replaces older tool-result content with a deterministic one-line summary (not LLM-generated), keeping the last 6 tool messages intact.
- `tests/test_orchestrator.py` (8 tests, all against the fake server): happy path to `completed`; malformed tool-call JSON → `invalid_arguments` error result → the model's corrected retry actually executes; `finish` rejected early with a populated `missing` list, then accepted once the gaps are filled; revision cap (set to 0 for the test) → `needs_human` on the linked requirement; `max_steps` exhaustion → `incomplete`; mid-run cancellation (allowed to complete step 1, cancelled before step 2) → `cancelled` at `steps=1`; inline `<think>` reasoning reaches `thinking_delta` events identically to a `reasoning_content` response; missing `usage` → `total_usage.estimated is True`.

**Verification:**
```
$ uv run pytest -q
57 passed in 8.28s
```

**Acceptance:** full suite (including all orchestrator scenarios) passes with no GPU/model. Committing phase 6.

## Phase 7 — API and SSE (2026-09-16)

- `app/events.py` (built in Phase 6) is wrapped with SSE here: `GET /api/runs/{run_id}/events` subscribes a thread-safe queue to the run's `EventBus`, replays `events_since(Last-Event-ID)` first, then streams live events via `asyncio.to_thread(queue.get, ...)` so the blocking queue read never stalls the event loop; stream ends after `run_finished`.
- `app/main.py` implements every Spec G endpoint: `/api/health` (2s-timeout, 5s-cached LLM reachability probe, git short SHA), `/api/config/public` (no secrets — verified by test), `/api/network` (delegates to the new `app/netstatus.py`), the run lifecycle (`POST /api/runs` → 409 with the active `run_id` if one is already running, `GET .../state`, `POST .../cancel`, `GET .../packet.md`, `POST .../save-replay`), `POST /api/explore/upload` (extension/size validated against `settings.uploads`), `POST /api/reset` (cancels the active run, wipes `runtime/sessions/*`), and `GET /api/replays` / `POST /api/replays/{name}/play` (backed by the new `app/replay.py`, which also holds `save_replay()`).
- **Run manager:** a module-level `RunManager` enforces "one active run at a time" via a lock around a single `_active` slot, set synchronously in the `POST /api/runs` handler (before the background thread even starts) so the 409 check is race-free. The orchestrator itself runs in a daemon `threading.Thread` (it's fully synchronous/blocking by design per Spec B), with `run_started`/`error` emitted around the `run_orchestrator()` call and `run_manager.mark_finished()` in a `finally`.
- `app/netstatus.py`: minimal version now (host classification loopback/private/CGNAT/public, and a synchronous TCP-connect probe run via `asyncio.to_thread`). The real Phase 9 work — periodic `network_status` change events and `psutil`-based `app_public_connections` — is not done yet; `app_public_connections` currently always reports `0`.
- `app/replay.py`: `save_replay()` (copy `events.jsonl`), `iter_replay_events()` (scaled timing, gaps capped at 3s per Spec I), `list_replays()`. Playback timing itself will get more scrutiny in Phase 9.
- **Known gaps carried forward:** `POST /api/explore/upload` does not yet run live requirement extraction on an uploaded solicitation (needs `scripts/extract_requirements.py` / the `extractor` profile, Phase 4.2, not yet built) — it currently only validates and stores the file. `_requirements_list()`/`_tracks()` fall back to sensible Track 2 defaults when `data/solicitation/requirements.json` doesn't exist yet (same Phase 4.2 dependency).
- `tests/test_api.py` (9 tests, `fastapi.testclient.TestClient` + the fake LLM server): health, config-public secret-free, full run lifecycle to `completed`, 409 on a concurrent run (made deterministic by monkeypatching `run_orchestrator` to block on a `threading.Event` rather than racing a real run's completion), SSE stream from `run_started` to `run_finished`, cancel-unknown-run 404, upload extension/size rejection, upload success + reset wiping the session directory. The "quick" scope's `sections` list is overridden to `[]` for these tests since full drafting/review/citation behavior is already covered by `test_orchestrator.py`.

**Verification:**
```
$ uv run pytest -q
66 passed in 9.91s

$ grep -nE "api/health|uvicorn.run|PORT|StaticFiles|app.mount|@app.get|@app.post" app/main.py
25:from fastapi.staticfiles import StaticFiles
238:@app.get("/api/health")
252:@app.get("/api/config/public")
268:@app.get("/api/network")
283:@app.post("/api/runs")
377:@app.get("/api/runs/{run_id}/events")
386:@app.post("/api/runs/{run_id}/cancel")
393:@app.get("/api/runs/{run_id}/state")
401:@app.get("/api/runs/{run_id}/packet.md")
413:@app.post("/api/runs/{run_id}/save-replay")
429:@app.post("/api/explore/upload")
448:@app.post("/api/reset")
460:@app.get("/api/replays")
469:@app.post("/api/replays/{name}/play")
504:    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")
509:    uvicorn.run(app, host=settings.server.host, port=settings.server.port)
```

**Acceptance:** API tests pass; grep output recorded above for Demo Center review. Committing phase 7.

## Phase 3 — Endpoint compatibility check (2026-09-16)

`scripts/endpoint_check.py` implements E1-E12 using `app/llm_client.py` directly (the same normalization code the app uses), run against the live model server (port 8081, ctx=65536) started in Phase 2.

```
$ uv run python scripts/endpoint_check.py --base-url http://127.0.0.1:8081/v1 --model qwen3.6-35b-a3b --reps 3 --label zbook-rocm-llamacpp

ID   Name                     Req   Pass/Reps  Result
----------------------------------------------------------------------
E1   Models list              yes   3/3        PASS
E2   Plain completion         yes   3/3        PASS
E3   Thinking separation      yes   3/3        PASS
E4   Thinking off             yes   3/3        PASS
E5   Single tool call         yes   3/3        PASS
E6   Tool round trip          yes   3/3        PASS
E7   Nested object args       yes   3/3        PASS
E8   Streaming tool call      yes   3/3        PASS
E9   No spurious tool call    yes   3/3        PASS
E10  Usage fields             no    3/3        PASS
E11  Decode speed             no    3/3        PASS
E12  Long context             no    3/3        PASS

Overall: PASS
```

**E3 mode:** `reasoning_content` field, 3/3 reps (not inline `<think>` tags) — matches `--reasoning-format deepseek`. `app/llm_client.py`'s `reasoning_mode: "auto"` correctly detects this without configuration.

**E11 decode speed:** ~23.7-23.9 tokens/s sustained decode (479/458/536 completion tokens over 20.0/19.2/22.6s), time-to-first-token ~0.66-0.72s.

**E12 long context:** a real 19,301-token prompt with a planted fact buried in the middle correctly returned it, 3/3 reps.

**E10 usage:** real `usage` fields from the server every time (`estimated=False`) — the token meter (Phase 9) won't need to fall back to character-based estimates against this endpoint.

Server version (via `system_fingerprint`): `b10729-458681e1d` — matches `llama-server --version` from Phase 2.

Full report: `reports/endpoint_check_zbook-rocm-llamacpp_20260916T180642Z.json`. Copied to `reports/baseline_zbook.json` (the one report tracked in git, per `.gitignore`'s explicit allowance).

**Acceptance:** script passes against the ZBook endpoint (12/12 checks, all required at 3/3); `reports/baseline_zbook.json` committed; E3 mode and E11 tok/s recorded above. Committing phase 3.

## Phase 4 (completion) — Narrative vault documents + validation (2026-09-16)

With the model server confirmed working (Phase 2/3), generated the remaining 21 LLM-backed narrative documents from Spec D.2 via `scripts/generate_vault.py` (drafter profile, `max_tokens` bumped to 3800 for this offline script only). All landed within their target word ranges (250-1470 words depending on doc).

**Bug found and fixed before running the full batch:** `write_narrative_doc` was stamping every narrative doc's front-matter `date` with "today" instead of each document's actual date. Added a `date` override to `NarrativeSpec` and set it explicitly for the three dated letters (`letter-kessler-2026` -> 2026-08-12, `letter-tricounty-2026` -> 2026-08-20, `letter-northgate-2021` -> 2021-03-15), then regenerated those three with `--only` after the fix landed.

**Quality issue found by my own read-through and fixed:** the first draft of `declined-dol-2023` named the 2023 DOL proposal itself "Smart Automation Technician Pathway (SATP)" -- a timeline error, since SATP wasn't conceived until the 2026 advisory board meeting. Tightened the brief to forbid that name in the 2023 document (it can only reference SATP prospectively, as the *later* project the lessons fed into) and regenerated. The doc's own Phase 4.5 HUMAN STEP ("Jerome skims 3-4 generated documents for tone and realism") is still open -- I did a broader read-through myself as part of building this, but a real skim by Jerome before the demo is still worthwhile.

**Spot-checked critical planted-issue documents directly** (not just via automated checks): `letter-northgate-2021` correctly dated 2021-03-15, references only the old "Mechatronics Pathways" project, never mentions SATP. `prior-annual-report-2025` honestly states 84 completers against a target of 120, with 78% placement among 61 respondents. `budget-justification-draft` states 28.0% fringe (matching the CSV) and 36.0% indirect. `rate-agreement-fy2026` states 31.5% fringe and 36.0% indirect, and even calls out that "previous budget templates may have utilized a lower estimate (such as 28.0%)" unprompted.

Wrote `data/vault/_planted_issues.json` (Spec D.3, verbatim) and `data/seeds/denylist.txt` (HP/NVIDIA plus a few real Mid-South community colleges as a regression guard, per Phase 4.3's earlier web-search results -- no actual collisions were found).

**`scripts/validate_vault.py`** checks: manifest/file-set consistency both directions; front matter present with all required keys on every document; the letters' front-matter dates match the seed exactly; the Northgate letter never mentions SATP; completers 84/120 appear in both the annual report and evaluation report; the agreement/budget fringe and indirect rates appear in their respective documents; budget CSV grand total matches the seed's `budget_total_usd` exactly and year sums reconcile; every planted issue's evidence doc(s) exist; both deterministic planted codes (`BUDGET_OVER_CAP`, `FRINGE_RATE_MISMATCH`) are actually produced by `check_budget` against the real vault; and no denylisted string appears anywhere (word-boundary matched -- a naive substring check on "HP" originally false-positived on "throughput", fixed to `\bHP\b`).

**Verification:**
```
$ uv run python scripts/validate_vault.py
Vault validation OK.

$ uv run pytest -q
66 passed in 11.54s
```

**Acceptance:** solicitation fetched/verified/facts confirmed (4.1, done earlier); `requirements.json` **not yet built** (Phase 4.2, `scripts/extract_requirements.py`, still open -- see gaps below); all Spec D.2 documents exist (23 total: 2 CSV + 21 md); `validate_vault.py` exits 0. Committing phase 4 (completes the partial commit from earlier in this log).

**Known remaining gap (closed below):** Phase 4.2 (`scripts/extract_requirements.py`, `data/solicitation/requirements.json`) was not built in this pass. `app/main.py`'s `_requirements_list()`/`_tracks()` fall back to empty/default values without it, and the orchestrator's `get_requirements` tool currently has nothing real to load. This blocks a true end-to-end Phase 10 run until it exists -- flagging it now rather than discovering it mid-Phase-10.

## Phase 4.2 — Requirement extraction and curation (2026-09-17)

WSL restarted overnight (background processes, including the model server, were killed; nothing was lost since everything through Phase 4 was already committed). Restarted the model server (same ctx=65536, port 8081) and built the missing piece.

`scripts/extract_requirements.py` walks `nsf24-584.md` section-by-section (splitting on `##`/`###`/`####` headings, skipping pure boilerplate like "About NSF" and "Privacy Act"), asks the local model (`extractor` profile) for candidate requirements as JSON per section, validates every `source_quote` as an exact substring of that section's text (discarding anything that fails), and deduplicates within category by text-similarity ratio.

**First attempt was far too slow to be practical:** the `extractor` profile's default `enable_thinking: true` had the model spending 2-6+ minutes of reasoning per section (some sections generated 4,000-5,700+ thinking tokens) for what is fundamentally a mechanical extract-and-quote task -- projecting to 80+ minutes for all 23 sections. Every candidate is validated deterministically afterward regardless, and I hand-review the whole draft anyway, so thinking mode's accuracy benefit here didn't justify the cost. Overrode `enable_thinking: false` and `max_tokens: 4096` for this offline script only (not a change to `config/default.yaml` or the live `extractor` profile) and reran -- finished in a few minutes, 179 candidates from 23 sections, 11 discarded on bad source_quote/category, 179 remaining after dedup (the 0.85 similarity threshold clearly wasn't catching everything, given the target is 30-60 final).

**Human curation** (delegated to a fork since it was read-heavy work I didn't need to keep in my own context, but the actual judgment calls were made by reading the draft and the solicitation directly, per the plan's instruction that this step is Claude Code's own review, not another script): curated 179 candidates down to **54 final requirements** in `data/solicitation/requirements.json`. Category breakdown: project_description 12, budget 11, eligibility 8, supplementary_documents 8, merit_review 6, program_specific 5, formatting 4. `section_link` set on 17 of 54 (12 -> `project_summary`, 3 -> `evaluation_plan`, 2 -> `results_prior_support`); the rest are eligibility/budget/formatting/reporting items that don't map to a section the demo drafts, so `section_link: null`.

Dropped: everything specific to Tracks 1/3/4 (this demo is Track 2 only), pure Research.gov/Grants.gov submission-mechanics steps, and eight near-identical "NSF funds may not be used for X" budget-exclusion rules merged into one consolidated requirement. **One real gap found and fixed by hand:** none of the 179 auto-extracted candidates stated the Track 2 $1,000,000/3-year cap explicitly, even though it's the demo's central planted issue (P1). Added it manually (`R-009`), sourced from the verified exact quote "Track 2: ATE Projects: Maximum budget of $1,000,000 over three years.", `verification: "deterministic"` since `check_budget` already enforces it independent of the LLM.

All 54 entries: `human_reviewed: true`, sequential `R-001`..`R-054`, every `source_quote` re-verified as an exact substring of `nsf24-584.md`.

**Verification:**
```
$ uv run python scripts/validate_vault.py
Vault validation OK.

$ uv run pytest -q
66 passed in 11.32s
```

**Acceptance:** `requirements.json` exists, validates against the Spec C schema, all `source_quote` values are exact substrings, 54 requirements (within the 30-60 target), all human-reviewed. This closes the Phase 4.2 gap noted above -- Phase 4 is now fully complete. Committed as `732b357`.

## Phase 8 — Presenter UI (2026-09-17)

Built `web/index.html`, `web/styles.css`, `web/app.js` (plain ES module, no build step, no CDN, no web fonts) per Spec H, plus `web/brand/README.md`.

**Pulled forward from Phase 9 (same reasoning as `app/events.py` in Phase 6):** the UI's meters panel needs a `metrics` event with real numbers to render, which nothing was emitting yet. Added `app/metrics.py` (`cost_equivalent()`, `human_baseline_minutes()` -- pure functions, unit tested in `tests/test_metrics.py`) and wired a throttled (>=1/s, plus a final forced emit) `metrics` event into `app/agent/orchestrator.py`, sourced from two new counters on `ToolContext` (`llm_call_count`, `last_decode_tps`, updated via a new `note_llm_call()` alongside every `add_usage()` call, including inside `draft_section`/`review_section`'s sub-calls). Real Phase 9 work still open: push-based `network_status` events (the UI currently polls `GET /api/network` every 5s instead) and the `psutil`-based `app_public_connections` count (still hardcoded to 0 in `netstatus.py`).

**One endpoint added beyond Spec G's table:** `GET /api/vault/{doc_id}` (optional `?chunk_id=`). Spec H requires citation and evidence chips to open a source drawer with the cited excerpt, and nothing in the original endpoint table provides a read path for that -- without it the behavior simply can't exist. Read-only, reuses `app/ingest/loaders.load` and `chunk_document`.

**Two real bugs found and fixed while building the Playwright verification, not UI-specific:**
1. **`app/llm_client.py` streaming truncation.** In streaming mode with no `reasoning_content` field present (the common case for `reasoning_mode: "auto"` once a response has no thinking), content deltas are routed through `_InlineThinkStreamState` to detect inline `<think>` tags split across chunks. That state machine deliberately withholds a short tail at each step, released via `flush_remainder()` at stream end -- but `flush_remainder()`'s tail was only added to the internal accumulator used for the final `ChatResult.content`, never passed to the caller's `on_content`/`on_reasoning` callbacks. Every streamed draft was missing its last few characters client-side (e.g. a citation's closing `]]` and doc id) even though the final result and word count were correct. Fixed by calling the callbacks on the flushed tail too; added a regression test (`test_content_deltas_streamed_to_callback_without_reasoning_field`) using realistic draft-length content, since the bug only manifests with content long enough to matter.
2. **Compliance table only showed individually-touched requirements.** `get_requirements`'s bulk-init `compliance_update` event uses `requirement_id: "*"` as a marker (per its own design -- the event doesn't carry all 54 initial rows), and the client was simply discarding it, so the table only ever showed rows that later got an explicit `update_compliance` call. Fixed by having the client fetch `GET /api/runs/{run_id}/state` on that marker event and populate the full compliance map from the authoritative server state (already exposed by Phase 7) rather than only tracking incremental updates.

**UX gap found and fixed:** the run-end summary overlay (Spec H's overlay with Download packet / Save as replay / Reset) had no way to dismiss it while keeping the run's results on screen -- a presenter finishing a run would be forced to either reset (losing the panels) or get stuck. Added a close (X) button and backdrop-click dismissal that just hides the overlay.

**`scripts/verify_ui_playwright.py`** (installed via `uv add --dev playwright` + `uv run playwright install chromium`; hit the exact `libasound.so.2` missing-system-library failure the doc anticipated -- **HUMAN STEP**, Jerome ran `sudo .venv/bin/playwright install-deps chromium`): starts the fake LLM server and a real app instance in-process (env-var config override, no changes to `config/local.yaml`), scripts a full-scope run (get_requirements, check_budget, draft/review/verify x3 sections, a flag_issue, an update_compliance, finish), drives it through a real Chromium browser at both 1920x1080 and 1366x768, and checks: every output tab populates (compliance rows, issue cards + badge, budget summary, 3 draft blocks with citation chips, 3 score bars), a citation chip opens the source drawer with the real cited document text, and keyboard shortcuts (`1`-`5`, `T`, `?`) work. Separately reloads the page with all non-localhost requests blocked via Playwright route interception and confirms it renders identically with zero requests actually blocked (the app makes none -- no CDN, no web fonts, no external calls anywhere).

**Verification:**
```
$ uv run python scripts/verify_ui_playwright.py
Starting fake LLM server...
Starting app server...
Running browser checks (1920x1080, 1366x768)...
Running network-disabled check...

UI verification OK. Screenshots in reports/ui_screenshots

$ uv run pytest -q
74 passed in 11.49s
```

**Acceptance:** every panel updates correctly at both viewports (screenshots in `reports/ui_screenshots/`, gitignored -- verification artifacts, not committed); network-disabled rendering identical with zero blocked requests; keyboard shortcuts verified. Committing phase 8.

## Phase 9 — Meters, network status, replay (2026-09-17)

Most of Spec I's meters work landed already in Phase 8 (pulled forward: `app/metrics.py`, the `metrics` event). This phase closes the remaining gaps it explicitly deferred.

- **`app/netstatus.py`:** `count_app_public_connections()` now does the real thing -- `psutil.Process().net_connections(kind="inet")` (with a fallback to the older `connections()` name for psutil <6.0), counting `ESTABLISHED` remotes whose IP isn't loopback/private/CGNAT. Labeled exactly "App connections to the internet" in the UI, and says nothing about the model server process, per spec.
- **Background network poller** (`app/main.py`, wired via a proper `lifespan` context manager, not the deprecated `@app.on_event`): polls `netstatus.get_status()` every `connectivity_probe.interval_s` (3s default), caches the result for `GET /api/network` to serve instantly, and emits a real `network_status` event into whichever run is currently active on change or at least every 10s -- both per Spec I. `GET /api/network` now serves the cache when fresh and only falls back to a synchronous probe if the poller hasn't run yet (e.g. right at startup).
- **Manual check (done directly, not just described):**
  ```
  $ uv run python -c "... probe_internet(['10.255.255.1:443'], timeout_s=1.0) ..."
  unroutable host -> online=False, elapsed=1.01s
  real hosts -> online=True, elapsed=0.03s
  ```
  Offline detected in ~1s (well under the 5s bar), online detected immediately once restored to reachable hosts.
- **Replay check, extended into `scripts/verify_ui_playwright.py`** (reuses its running fake-server + app instances): runs a scripted full-scope fake run, saves it via `POST /api/runs/{run_id}/save-replay`, plays it back through a real browser, and confirms the amber REPLAY banner is visible from the start of playback through to the run-complete summary card. Screenshot: `reports/ui_screenshots/replay_banner.png` (gitignored, verification artifact).

**Two more real bugs found while building the replay check (not the backend -- both client-side):**
1. **Start button was unusable in replay mode.** `#mode-select`'s change handler set `$("#btn-start").hidden = true` whenever mode was "replay" and never un-hid it for that mode, even though the button's own click handler already correctly dispatches to `playReplay()` vs `startRun()` based on `state.mode`. The button should never have been hidden at all. Fixed by removing the hide-on-replay logic entirely.
2. **My own check script's timing bug** (not app code, but worth recording): it started asserting the REPLAY banner's visibility in the same tick as `page.click("#btn-start")`, before the async `playReplay()` (POST + SSE connect) had a chance to run, so it failed immediately every time regardless of whether the app worked. Fixed by waiting for the banner to first appear before monitoring that it stays visible.

**New unit tests:** `tests/test_netstatus.py` (host classification loopback/private/CGNAT/public, `probe_internet` true/false including a real bound-but-listening socket to make the "true" case deterministic rather than depending on real internet access, `count_app_public_connections` smoke test), `tests/test_replay.py` (gap timing from real recorded timestamps, speed scaling, the 3s gap cap, blank-line tolerance, event payload preservation, `save_replay`/`list_replays`), plus three new `decode_tps`/`ttft_s` tests in `tests/test_llm_client_normalization.py` (the "tps calculation" and timing pieces of this phase's unit-test requirement that Phase 8's `test_metrics.py` didn't cover).

**Verification:**
```
$ uv run pytest -q
93 passed in 13.47s

$ uv run python scripts/verify_ui_playwright.py
... (all checks, including the new replay check) ...
UI verification OK.
```

**Known simplification, documented rather than silently shipped:** `decode_tps_recent` in the `metrics` event is the most recently completed LLM call's tokens/s, not a true rolling 10-second window across a call in progress. For this demo's single-slot local server, one call's measurement already reflects current throughput; a proper sliding window would need buffering per-token timestamps mid-stream for marginal benefit. Revisit if a future scope needs mid-generation tok/s smoothing.

**Acceptance:** unit tests for cost math, tps, host classification, and replay timing all pass; manual offline/online check done directly (not just described); a replay recorded from a fake-server run plays back with the REPLAY banner visible throughout. Committing phase 9.

## Phase 10 — Live end-to-end on the ZBook (2026-09-17)

`scripts/run_e2e.py` starts a real run via the API, consumes SSE to completion, loads final state, and evaluates against `_planted_issues.json` (P1-P4), writing `reports/e2e_<label>_<scope>_<timestamp>.json`.

### Quick scope, 3 reps -- three iterations, documented honestly

**Iteration 1 (before any fixes):** Completed 3/3. P1 3/3, P4 3/3, P2 3/3, **P3 1/3** (below the >=2/3 target). **Median wall time 1577s (~26 min)** -- far over the 6-minute target.

Digging into *why* it was slow (not just accepting it) surfaced a real harness bug: `max_revisions_per_section` (default 2) was enforced in `review_section` (marks the section `needs_human`) but nothing stopped `draft_section` from being called again anyway. One rep drafted `results_prior_support` **5 times** chasing a review score that review itself had already given up on -- 60 steps (the max_steps ceiling), 1.16M cumulative prompt tokens, 1719s for that rep alone. Fixed: `draft_section` now refuses outright once the cap is reached (commit `47c10fb`).

Separately, reproducing a `review_section` call directly against the real model surfaced a second real bug: the reviewer prompt showed the model an abstract JSON Schema and asked it to "match this schema" -- the model echoed the schema's `{"type":"object","properties":{...}}` wrapper back instead of producing a filled-in instance. Valid JSON, wrong shape, so `json_mode`'s parse check never caught it -- every review had silently been scoring `mean_score: 0.0` with empty `criteria`. Fixed by replacing the schema with a concrete filled-in example, the pattern LLMs follow far more reliably (commit `6b12a6e`).

**Iteration 2 (after both bug fixes):** Completed 3/3. P1 3/3, P4 3/3, **P2 2/3** (down from 3/3 -- see below), **P3 0/3** (worse). **Median wall time 1131s (~18.9 min)** -- the revision-cap fix alone cut median time by ~28%, but nowhere near the 6-minute target.

**Root-cause investigation (with Jerome) into the remaining ~19 minutes:** average ~27s/step across 32-43 steps/run. Cumulative prompt tokens are large (500K-720K) but this reflects the *growing conversation*, not per-step reprocessing -- llama-server's single-slot KV cache reuse means most of that isn't being reprocessed every step. The ~27s/step figure lines up much better with **decode time**: the `orchestrator` profile has `enable_thinking: true` and runs on *every single step* (32-43 times/run), unlike `drafter`/`reviewer` (thinking off, 2-5 sub-calls/run). This is a real, measured finding, not a guess -- but the plan's Section 0 explicitly says not to change the model or silently disable thinking without Jerome's input, so I stopped and asked rather than just flipping the setting.

**Jerome asked me to try a non-model, non-thinking fix first.** Investigating further: the orchestrator's system prompt instructed steps 4-5 (collaboration-letter review, compliance-matrix updates) unconditionally, regardless of scope -- so "quick" scope only narrowed *which sections get drafted*, not the open-ended full-vault exploration and up-to-54-requirement compliance sweep that was actually the bulk of the step count. Made "quick" scope explicitly target the three known collaboration letters and limit compliance updates to in-scope requirements + the budget check, instead of open-ended exploration (commit `45a4bd8`, no model or thinking changes).

**Iteration 3 (after the scope-targeting fix), the current code state:** Completed 3/3. P1 3/3, P4 3/3, **P2 1/3** (down further), **P3 2/3** (now meets target). **Median wall time 1313s (~21.9 min)** -- essentially flat vs. iteration 2, slightly worse.

**Honest result: the scope-targeting fix achieved its secondary goal (P3 now meets the >=2/3 target) but did not move the needle on wall time.** That's a real, measured result against my working hypothesis, not a success I'm inflating -- narrowing exploration wasn't the dominant time cost; per-step thinking-mode decode time is, exactly as the iteration-2 investigation indicated. Digging into the P2 drop across iterations 2-3: it's not the agent reporting outcomes dishonestly -- `verify_citations` is failing on the final draft (19 and 2 unsupported claims in two reps of iteration 3, vs 0 in the passing rep), a plausible side effect of the reviewer-JSON fix actually working now: real must_fix feedback pushes the drafter to address substantive gaps, and some of that added detail isn't grounded in the evidence provided. This is a genuine, separate quality issue from the speed question, left open below rather than patched under time pressure.

**Decision on the remaining ~19-22 minute quick-scope time (discussed with Jerome, not decided unilaterally):** left as-is for now. Likely real fix is hardware, not software: the target-later HP ZGX Nano (GB10/Blackwell) is a purpose-built AI accelerator vs. the ZBook's laptop APU, and the plan's own Section K already recommends an NVFP4-quantized build on that hardware specifically for performance -- plausibly enough of a decode-speed jump to reach the 6-minute target without touching thinking mode at all. This can't be verified until Nano hardware is available (Phase 13, not done). For live ZBook demos in the meantime, the presenter script's existing replay fallback ("while it thinks, let's look at the matrix" / switch to a recorded replay) is the practical answer, not a workaround -- Spec J already designed for exactly this.

**Official measured numbers for this phase's acceptance (iteration 3, current code):**

| Metric | Result | Target | Met? |
|---|---|---|---|
| Completed | 3/3 | 3/3 | Yes |
| P1 (over cap) | 3/3 | 3/3 | Yes |
| P4 (fringe mismatch) | 3/3 | 3/3 | Yes |
| P2 (honest prior outcomes) | 1/3 | >=2/3 | **No** -- open issue, see above |
| P3 (stale letter) | 2/3 | >=2/3 | Yes |
| Median wall time | 1313s (~21.9 min) | <=360s (6 min) | **No** -- open issue, see above |

Full report: `reports/e2e_zbook_quick_20260917T180059Z.json`.

**Open items for Jerome, not resolved in this pass:**
1. Quick-scope median wall time (~19-22 min) is well over the 6-minute target. Root cause identified (thinking-mode decode time across many orchestrator steps) but the only remaining lever is touching thinking settings, which the plan gates behind Jerome's decision. Current call: leave it, rely on the replay fallback for live demos, revisit once ZGX Nano hardware is available.
2. P2 (honest prior-outcomes reporting) is below target due to citation-verification failures on revised drafts, not dishonest content. Needs its own investigation (likely a drafter-prompt or revision-guidance tightening) separate from the speed question.

### Full scope, 1 rep

The doc's own SSE-consumption timeout guidance (30 min) turned out to be too short for full scope in practice: my script's client-side timeout fired at 1800.2s while the run itself was still healthy server-side, at step 59/60 -- not a hang, just a scope where 3 sections' worth of drafting genuinely needs more than 30 minutes at current per-step speed. Polled the server directly past the script's timeout rather than treating that as a failure.

**Actual result: `status: incomplete`** -- the run hit `max_steps` (60, shared with quick scope) without ever successfully calling `finish`. Given quick scope (1 section) typically uses 30-45 steps, and full scope (3 sections) is proportionally heavier per tool-call category (5x `get_requirements`, 9x `draft_section`, 6x `review_section`, 13x `read_document`), 60 shared steps is very plausibly just not enough headroom for 3 sections to reliably reach `finish` -- a distinct, separate finding from the quick-scope speed question, and not one this pass attempts to fix (would mean bumping `max_steps` for `full` specifically, a reasonable config-only follow-up, not attempted here per "move on" direction from Jerome mid-investigation).

Despite not finishing, the substantive work done was solid: P1 and P4 both caught (`BUDGET_OVER_CAP`, `FRINGE_RATE_MISMATCH`); `results_prior_support`'s draft contains both `84` and `120` with citation checks passing (`[True, True]`) -- P2's honest-reporting bar was actually met here, even though the run overall didn't reach `finish`; 7 agent-flagged issues total, including one about "an outdated employer support letter referenced in project materials" (evidence: `declined-dol-2023`, `prior-evaluation-report-2025`) that's conceptually the P3 stale-letter problem but doesn't cite `letter-northgate-2021` directly, so it doesn't count as a strict P3 catch under this script's exact-evidence-match rule.

Report: `reports/e2e_zbook_full_20260917T184905Z.json` (recorded the script's premature timeout, not the server's actual final state -- the numbers above are from polling `GET /api/runs/{run_id}/state` directly afterward).

### Replay

Saved a successful, fully-caught (P1/P2/P3/P4 all true) quick-scope run as the committed replay: `data/replays/quick_zbook_20260917.jsonl` (9,725 events, run summary: 84/120 completers reported honestly, Northgate's stale 2021 letter flagged high-severity, both budget issues caught, `results_prior_support` v3 passed citation verification). This is the version of the demo Jerome should lean on for live presentations given the current wall-time situation -- plays back at a natural pace with the REPLAY banner, no 20-minute live wait.

### Open items for Jerome, not resolved in this pass (added to)

3. Full scope's `max_steps` (60, shared with quick) appears insufficient for reliable completion with 3 sections -- one rep hit the ceiling at `status: incomplete`, not `completed`, despite the actual section work (P1/P2/P4-equivalent) being done correctly. Likely fix: a higher `max_steps` for the `full` scope specifically in `config/default.yaml`, not attempted in this pass.

**Verification:**
```
Quick scope (iteration 3, current code): 3/3 completed, P1 3/3, P4 3/3, P2 1/3, P3 2/3, median 1313s
Full scope (1 rep): incomplete (max_steps), P1/P4 caught, P2 criteria met, P3 conceptually flagged but not exact-matched
Replay saved: data/replays/quick_zbook_20260917.jsonl
```

**Acceptance:** completed 3/3 (quick); P1 and P4 3/3; P2 and the 6-minute wall-time target are **not** met, both investigated, documented, and left as open items with Jerome's explicit input rather than papered over; P3 reported at 2/3 (meets target after the scope-targeting fix); a successful quick-scope run recorded and committed as a replay. Committing phase 10.

## Phase 11 — License and portability audits (2026-09-17)

`scripts/license_audit.sh` and `scripts/audit_portability.sh` per the doc's spec, both exiting 0 with no dependency-tree changes needed -- the project was already clean.

**Two real script-correctness bugs found and fixed while actually running these (not just writing them):**
1. **`audit_portability.sh` false-positived on compiled bytecode.** The grep for hardcoded home paths matched inside `.pyc` files under `__pycache__/` (Python embeds source file paths in compiled bytecode for tracebacks) -- these are gitignored build artifacts, not shipped code. Added `--exclude-dir=__pycache__ --exclude='*.pyc'` to every grep.
2. **`license_audit.sh` picked up an unrelated system package.** `THIRD_PARTY_LICENSES.md` initially listed `amdsmi` (AMD's ROCm system-management library) as a project dependency. It isn't -- it's not in `pyproject.toml` or `uv.lock` -- but this machine has a system-wide `PYTHONPATH=/opt/rocm-wsl/share/amd_smi:/opt/rocm-wsl/libexec/amdsmi_cli` set for Jerome's existing ROCm/GPU tooling, which leaks into `pip-licenses`'s import scan despite `uv run`'s venv isolation (`PYTHONPATH` entries get added to `sys.path` regardless of which venv is active). `unset PYTHONPATH` before running `pip-licenses` fixed it -- confirmed `amdsmi` no longer appears and the report now reflects only this project's real 60-ish dependencies, all MIT/BSD/Apache-2.0/MPL-2.0/PSF, no copyleft.

Added the model (Apache-2.0) and llama.cpp (MIT) to `THIRD_PARTY_LICENSES.md` under "Runtime components not bundled in the app container", per the doc.

**Verification:**
```
$ bash scripts/license_audit.sh
Wrote THIRD_PARTY_LICENSES.md
License audit OK

$ bash scripts/audit_portability.sh
Portability audit OK

$ uv run pytest -q
94 passed in 14.03s
```

**Acceptance:** both scripts exit 0. Committing phase 11.

## Phase 12 — Container (2026-09-17)

`docker/Dockerfile` and `docker/compose.yaml` per the doc's templates, `UV_VERSION` pinned to `0.12.8` (this machine's actual `uv --version`).

**HUMAN STEPs needed (Docker itself was already present from Phase 0, but compose wasn't):** this machine's Docker install is the native Ubuntu `docker.io` package, not Docker Desktop -- the `docker compose` subcommand wasn't installed, and the `docker-compose`/`docker-compose-plugin` names that exist as broken symlinks on this system point at a Docker Desktop WSL mount that isn't present here. `sudo apt install docker-compose-plugin` (Docker's own repo naming) failed with "Unable to locate package" since Docker's official apt repo isn't configured -- the correct Ubuntu-repo package is `docker-compose-v2`. Jerome ran `sudo apt install docker-compose-v2`; verified with `docker compose version` (2.40.3) afterward.

**Unexplained file-loss incident, noted for the record:** between creating `docker/Dockerfile` + `docker/compose.yaml` and running `docker compose up` the first time (while waiting on the human step above), the entire `docker/` directory disappeared -- not a WSL restart (uptime was continuous, unlike the earlier Phase 4.2 incident), and `git status` showed a perfectly clean tree at the Phase 11 commit, meaning the files were never committed and something removed them before that happened. No root cause identified. Recreated both files and `git add`ed them immediately (before doing anything else) so they'd be safe in git's object store even before a commit, then proceeded -- no recurrence.

**Verification:**
```
$ docker compose -f docker/compose.yaml up -d --build   # GM_LLM__BASE_URL=http://127.0.0.1:8081/v1 (this machine's port, see Phase 0)
 grant-machine  Built
 Container grant-machine-grant-machine-1  Started

$ curl -s http://127.0.0.1:8095/api/health
{"status":"ok","llm":"ok","model":"qwen3.6-35b-a3b","version":"unknown"}
```
(`version: "unknown"` is expected -- `.git` isn't copied into the image, so `git rev-parse` has nothing to read inside the container. Not a bug.)

```
$ docker compose -f docker/compose.yaml logs --tail=50
grant-machine-1  | INFO:     Application startup complete.
grant-machine-1  | INFO:     Uvicorn running on http://0.0.0.0:8095
```

```
$ uv run python scripts/run_e2e.py --base-url http://127.0.0.1:8095 --scope quick --reps 1 --label zbook-docker
[zbook-docker/quick] rep 1/1: status=completed wall=1085.3s P1=True P4=True P2=True P3=False
```

One quick-scope e2e rep against the containerized app completed successfully (`status=completed`), same behavior as the native app (P1/P4/P2 caught, P3 missed this particular rep -- consistent with the run-to-run variance already documented in Phase 10, not a container-specific regression).

`docker buildx` (ARM64 cross-build check) is not installed on this machine either -- same underlying gap as compose (native `docker.io`, no Desktop). Per the doc, this is explicitly optional today and the authoritative ARM64 build happens on the Nano; not installed, since the doc says not to install QEMU/binfmt via sudo on my own initiative.

Left the container running after verification (`docker compose ps` shows it healthy) -- will need to be stopped before Phase 13's "follow the README from a clean state" check.

**Acceptance:** container healthy; one quick-scope e2e rep passes against it. Committing phase 12.

## Phase 13 — Documentation (2026-09-17)

Wrote `README.md`, `docs/ARCHITECTURE.md`, `docs/PRESENTER_SCRIPT.md`, `docs/PORTING_TO_ZGX_NANO.md`, `docs/DEMO_CENTER_INTEGRATION.md`, and extended `DATA_SOURCES.md` to cover every file under `data/` (31 rows: the solicitation, both seed files, both requirements files, all 23 vault documents, the manifest and planted-issues files, and the committed replay -- each with its source, license/status, date, and SHA-256).

**One real documentation bug found and fixed by actually following the README, not just writing it:** `config/local.yaml.example` still overrode `llm.context_tokens` to `32768`, the Phase 2 *placeholder* value -- but `config/default.yaml` has carried the real Phase 2.5 result (`65536`) since that phase landed. A fresh user copying the example as instructed would have silently *reduced* their context size below what this project actually validated. Fixed the example to explain when `context_tokens` needs overriding at all (rarely) instead of unconditionally shipping a stale number.

**Presenter script honesty note:** `docs/PRESENTER_SCRIPT.md` is written as the original 5-minute live talk track (Spec J), but opens with an explicit, prominent correction pointing at Phase 10's real measured timing (~15-22 min live on this hardware) and recommends Replay mode as the default for live audiences, with the committed `data/replays/quick_zbook_20260917.jsonl` as the concrete artifact to use. Did not pretend the original 5-minute framing still describes a live run on this hardware.

**README verification (the doc's own acceptance bar -- actually done, not assumed):** stopped the app server and the container (both were running from Phases 10/12), removed `config/local.yaml`, then followed `README.md` literally: `cp config/local.yaml.example config/local.yaml` (plus substituting the placeholder `/home/USER/` paths, same as any real user would), `uv sync`, confirmed the model server health check line works, started the app with `uv run python -m app.main`, and confirmed `/api/health` and `/` both respond correctly. The model server itself was left running throughout (already verified extensively in Phases 2/3/10) rather than burning another ~15s reload cycle to prove something already proven.

**Verification:**
```
$ cp config/local.yaml.example config/local.yaml   # + path substitution
$ uv sync
Resolved 58 packages in 5ms
Checked 55 packages in 3ms

$ curl -s http://127.0.0.1:8081/health
{"status":"ok"}

$ uv run python -m app.main &
$ curl -s http://localhost:8095/api/health
{"status":"ok","llm":"ok","model":"qwen3.6-35b-a3b","version":"cbf0dda"}

$ curl -s http://localhost:8095/ | head -5
<!doctype html>
...
<title>Grant Machine</title>

$ uv run pytest -q
94 passed in 13.53s
```

**Acceptance:** a fresh shell following only `README.md` starts the demo -- verified directly, not assumed; one real doc/config bug found and fixed in the process. Committing phase 13.

## Phase 14 — Final verification (2026-09-17)

```
$ uv run pytest -q
94 passed in 14.58s

$ bash scripts/license_audit.sh
Wrote THIRD_PARTY_LICENSES.md
License audit OK

$ bash scripts/audit_portability.sh
Portability audit OK

$ uv run python scripts/validate_vault.py
Vault validation OK.

$ uv run python scripts/endpoint_check.py --base-url http://127.0.0.1:8081/v1 --model qwen3.6-35b-a3b --reps 3 --label zbook-final
E1-E9 (required): 3/3 each, PASS
E10-E12 (optional): 3/3 each, PASS
Overall: PASS

$ uv run python scripts/run_e2e.py --base-url http://127.0.0.1:8095 --scope quick --reps 1 --label zbook-final
status=completed wall=1210.8s P1=True P4=True P2=False P3=True
(consistent with the Phase 10 finding: P1/P4/P3 reliably caught, P2 is the known open citation-verification issue -- no new regression)

$ git status --porcelain
(empty)

$ git log --oneline
20 commits: one per phase (0-13) plus three targeted, individually-justified bug/chore
commits found mid-build (reviewer JSON-shape bug, revision-cap enforcement, quick-scope
targeting, .claude/ gitignore, a Phase 4.2 progress-log addendum) -- each documented in
its own commit message and in PROGRESS.md at the point it was found.
```

**Acceptance:** all six verification commands pass; working tree clean; commit history one-per-phase plus documented exceptions. Grant Machine build complete through Phase 14.

---

## Phase 15 -- Demo Center integration on the HP ZGX Nano (2026-09-18)

Ported this build from the x86 ZBook it was written on into the `zgx-demo-center` monorepo as
demo id **`grant`**, port **8095**, running natively on the Nano (aarch64 / GB10). This is the
step `docs/DEMO_CENTER_INTEGRATION.md` deliberately left to whoever owned that repo.

### Transfer damage found and repaired first

The tree arrived with the classic Windows/x86 hand-off damage, caught by diffing against the
nested `.git` before it was removed:

- **CRLF line endings** introduced into exactly three files: `data/solicitation/nsf24-584.html`
  (949 lines), `data/vault/budget-satp-draft.csv` (31), `data/vault/prior-outcomes-data.csv` (6).
  Not cosmetic -- `app/checks/budget.py` parses the budget CSV and the rate-agreement text
  directly, which is the whole basis of the deterministic budget check.
- **Exec bits lost** on five `scripts/*.sh` (mode 100755 -> 100644).

Both were restored from the committed objects (`git checkout -- .`), verified clean, and only
then was the nested `.git` removed. The 22-commit build history is archived as a git bundle at
`~/grant-machine-build-history.bundle` (612 KB); it is not committed.

Also purged: the x86 `.venv` (287 MB, `home = /usr/bin`), `runtime/` (25 MB of ZBook run
state), `reports/*` except the baselines, `.pytest_cache`, `__pycache__`, and the ZBook's
`config/local.yaml` (hardcoded `/home/gsept/...` paths). 104 files / 3.2 MB now vendored.

### The one real code change: Ollama is not llama-server

The Nano already served `qwen3.6:35b` from Ollama on 11434, so no llama.cpp build and no GGUF
download -- and, more importantly, no second GPU consumer outside the Demo Center's
`/ollama_release` VRAM protocol. But Ollama's OpenAI-compatible layer differs from
llama-server's in two ways that **fail silently rather than loudly**:

1. It names the separated-reasoning field `reasoning`, not `reasoning_content` (in both the
   non-streamed message and the streamed delta). `app/llm_client.py` read only the latter, so
   against Ollama it captured no reasoning at all.
2. It ignores `chat_template_kwargs.enable_thinking` entirely -- and `think` on the `/v1`
   route. The only knob it honours is `reasoning_effort`, and `"none"` is the sole value that
   actually suppresses thinking (`"low"` still thinks).

Item 2 is the more dangerous one. Left unfixed, the `drafter` and `reviewer` profiles think
when they are configured not to, and the reasoning consumes their `max_tokens` budget -- the
reviewer, which must return JSON at `max_tokens: 2000`, comes back with empty content. That is
a correctness bug, not a speed one; it was caught by probing the endpoint directly rather than
by the test suite or the endpoint check.

Fix: `llm_client.py` now reads either field name, and a new `llm.thinking_control` setting
selects `chat_template` (default -- llama-server/vLLM request shape byte-identical to before)
or `reasoning_effort` (Ollama). Five tests added in `tests/test_llm_client_normalization.py`
covering both field names and both request shapes; 99 pass.

**Honest note on the gate that did not catch this:** `scripts/endpoint_check.py` E3 reported
PASS against Ollama *before* the fix, with the app seeing zero reasoning. E3 asserts only that
`<think>` tags do not leak into `content`, which a server emitting no reasoning satisfies
trivially. The real signal was `checks.E3.samples[].meta.reasoning_source` -- `none` before the
fix, `field` after. The weak assertion is logged in the Demo Center's `KNOWN-ISSUES.md`; it has
not been tightened here.

### Endpoint check -- `--label nano-ollama`, 3 reps (`reports/baseline_nano.json`)

E1-E12 all PASS, 3/3 each. Against `reports/baseline_zbook.json`:

| | ZBook (ROCm llama.cpp) | Nano (Ollama, GB10) |
|---|---|---|
| E3 reasoning source | `field` | `field` |
| E11 decode | 23.8 tok/s | **77.2 tok/s (3.2x)** |
| E11 TTFT | ~0.69 s | ~0.10 s |
| E10 usage | present, not estimated | present, not estimated |

### E2E -- the Phase 10 open issue is resolved on this hardware

`scripts/run_e2e.py --scope quick --reps 3 --label nano` (`reports/e2e_baseline_nano.json`):

| | ZBook (Phase 10, iter 3) | Nano | Target |
|---|---|---|---|
| Completed | 3/3 | 3/3 | 3/3 |
| P1 budget over cap | 3/3 | 3/3 | >=2/3 |
| P4 fringe mismatch | 3/3 | 3/3 | >=2/3 |
| P2 honest prior outcomes | 1/3 | **3/3** | >=2/3 |
| P3 stale letter | 2/3 | 2/3 | >=2/3 |
| **Median wall time** | **1313 s (~21.9 min)** | **189.9 s (~3.2 min)** | **<=360 s** |

Individual reps: 190.7 s, 189.9 s, 112.1 s. **Median wall time meets the <=360 s target for the
first time** -- 6.9x faster than the ZBook, against a 3.2x raw decode speedup, the remainder
coming from fewer orchestrator steps per run (23-26 here). Phase 10's stated root cause
(per-step thinking-mode decode time) and its stated remedy (faster hardware, not a thinking-
settings change) both hold up.

P2 moving from 1/3 to 3/3 is the other notable change, and I am **not** claiming the port fixed
it. Phase 10 diagnosed P2 as a citation-verification quality issue, and nothing in this phase
touched drafting, review or citation logic. Plausible contributors are the different quant
(Ollama's `qwen3.6:35b` vs. the UD-Q4_K_XL GGUF) and n=3 sampling noise on a metric that was
already unstable across Phase 10's own iterations (3/3 -> 2/3 -> 1/3). Treat it as encouraging,
not as evidence of a fix.

### Integration surface

- `install_plans.py`: new `grant` entry -- 3 install steps (deps, import verification, and the
  E1-E12 endpoint check as an install-time gate), `port_kill` teardown, `port_probe` discovery,
  readiness check on files + `qwen3.6:35b`.
- `install.py`: `grant` appended to `WEB_INSTALL_IDS`.
- `zgx-demo-launcher.html`: 15 hand-maintained registration sites (card, install row, `DEMOS`
  modal entry, and 12 JS maps).
- `start_demo_8095.sh`: new, modelled on the spectrum launcher including its `ZGX_PYTHON`
  guard. Config passed as `GM_*` env vars, never written to a tracked file.
- Runtime: the shared `zgx` conda env (Python 3.12.13), which already had 13 of the 14
  dependencies at the exact required versions. Only `markdownify` was installed and `pydantic`
  bumped 2.13.4 -> 2.13.5. `openai` deliberately left at the env's 2.29.0 and the
  `pyproject.toml` floor lowered to match -- every API this app uses exists in 2.x, the env is
  shared with the TensorRT demos, and E1-E12 passes on it.

### A second vacuous pass, found while verifying: `license_audit.sh`

`scripts/license_audit.sh` reported **"License audit OK" while auditing zero packages**, and
truncated a tracked file on the way. It ran `uv run pip-licenses > THIRD_PARTY_LICENSES.md`
unconditionally; with no `uv` on the Nano the redirect emptied the file to the 7 lines of its
own appended heredoc, and the final `if uv run pip-licenses --format=csv | grep -Ei 'GPL'`
likewise produced nothing, so the `else` branch printed a pass. `set -uo pipefail` without
`-e` let every failure through.

Fixed three ways: the script now refuses outright when `uv` is absent (exit 2, with the reason),
writes to a temp file and only `mv`s it into place after the tool has succeeded *and* produced
non-empty output, and treats an empty copyleft scan as an error rather than a pass. It
deliberately has **no non-uv fallback**: pip-licenses reports whatever is installed in the
environment it runs under, so under the shared `zgx` env it would list several hundred
unrelated packages (torch, azure, the TensorRT demos' deps) as this project's dependencies --
a worse compliance artifact than none. `THIRD_PARTY_LICENSES.md` was restored from the archived
build-history bundle and carries a scope note explaining the `openai` version difference.

This is the same class of defect as the E3 finding above, in a different gate: a check that
cannot distinguish "verified clean" from "verified nothing". Both were found by looking at what
the check actually measured rather than at its PASS line.

### Verification

`pytest -q` 99 passed; `scripts/audit_portability.sh` OK; `scripts/validate_vault.py` OK;
endpoint check PASS 3/3 (`--reps 3`); e2e 3/3 as above; install API `/status/grant`,
`/install/grant` (all 3 steps, `complete`), `/health_check/grant`, `/running`,
`/demo_log/grant`, `/start/grant`, `/stop/grant` all correct; launcher JS parses
(`node --check`); shared-env regression clean (`cv2`, `tensorrt`, `pycuda`, `pynvml` still
import). `scripts/license_audit.sh` cannot run here by design -- see above.

---

## Phase 16 -- v2 Phase 0: the egress boundary (2026-09-18)

Grant Machine v2 adds Discover / Intake / Grade. Discover needs the agent to research the
open web, which breaks the v1 claim that nothing leaves the box. The replacement claim is
asymmetric and stronger: **the agent may research the public web; the institution's own
material never leaves the device.** That is only worth saying if it is enforced, so the
boundary was built and attacked before any feature was built on top of it.

### `app/egress.py` -- one way out, three layers

Every outbound byte goes through `EgressGuard.request()`. Layers, in the order they run:

1. **Default-deny term validation.** A search term leaves only if it matches the published
   vocabulary in `data/vocabulary/`. The question asked is not "is this sensitive?" (a
   judgment call) but "is this on the public list?" (not one).
2. **Payload token scan.** Every word in the outbound query values must be vocabulary
   language. Layer 1 should guarantee this, so a hit here means a caller bypassed
   `validate_terms` -- which is exactly what it is for.
3. **Session corpus scan.** The payload is checked against n-grams of the uploaded documents,
   counting only n-grams that carry a non-vocabulary token. Unreachable if layer 2 holds; it
   exists so the red-team test is exercising a real backstop rather than a tautology.

Transport is guarded separately: `netstatus.classify_host()` is reused to refuse anything not
classified `public`, so a crafted URL cannot reach the local model endpoint, the Demo Center
install API on 8086, or anything else on the operator's network. HTTPS only, response size
capped, per-run request budget. Every decision -- allowed or refused -- becomes an
`EgressRecord`.

### The design mistake the tests caught

The first implementation checked verbatim-from-upload *before* the vocabulary, on the
reasoning that a term appearing in a user's document is user data. The test suite failed
immediately: `robotics` was refused, because the institution's strategic plan says "robotics".
That rule bans every field the institution actually works in -- precisely the terms discovery
must search for -- while providing no protection, since "robotics" is public language either
way.

The vocabulary now decides first and a match wins outright. The verbatim and digit checks still
run, but only to choose the most *informative* refusal reason for a term that default-deny has
already refused: "this came out of your own document" is what the audit panel should say, and
it is a far better answer on stage than "not on the list". Recorded here because the wrong
version looked more secure and was strictly worse.

### Verification

`tests/test_egress.py`, 26 tests, all passing (suite total now 125). The one that matters is
`test_red_team_no_upload_content_reaches_the_wire_by_any_path`: it plants a distinctive phrase,
a PI name and an award number in an upload, then tries six routes out -- as a term, nested in a
JSON body, in a URL query string, blended into a legitimate query -- and asserts against a
mock transport that **no request was transmitted at all** and that all six attempts are on the
audit record.

The portability audit gained a chokepoint rule: any file in `app/` other than the four
explicitly allowlisted (`egress.py`, `llm_client.py`, `main.py`, `netstatus.py` -- the last
three touch only the local model endpoint or a payload-free TCP probe) that imports `httpx`,
`socket`, `urllib.request` or `requests` fails the build. **The rule was verified by making it
fail**: a temporary `app/_audit_probe.py` calling `httpx.get` was caught and the audit exited 1.
After two vacuous-pass gates found in Phase 15 (E3, `license_audit.sh`), a new gate does not get
to be trusted because it printed OK.

### Live end-to-end

Against the real grants.gov `search2` API from the Nano, with a simulated institutional upload
in the corpus:

| Attempted term | Outcome |
|---|---|
| `Zephyr Cascade Fabrication Protocol` (from the upload) | refused -- `verbatim_from_upload` |
| `Semiconductor Advanced Technician Pathway` (project name) | refused -- `verbatim_from_upload` |
| `Marisol Thackeray-Vance` (PI name) | refused -- `verbatim_from_upload` |
| `award 8814402` (identifier) | refused -- `verbatim_from_upload` |
| `semiconductor manufacturing` | sent -- 145 live opportunities |
| `technician education` | sent -- 433 live opportunities |
| `robotics` | sent -- 28 live opportunities |

Three requests left the box; three terms left the box; four attempts were refused and logged.
Note the loop closing on itself: `technician education` returns **24-584, Advanced Technological
Education** -- the very solicitation the rest of the demo is built around.

### Vocabulary

`data/vocabulary/field_terms.txt` (372 hand-authored generic field names) plus the grants.gov
facet taxonomy fetched by `scripts/build_vocabulary.py` (923 agencies, categories, eligibilities
and funding instruments) = 1,295 allowlist entries. Provenance in `DATA_SOURCES.md`. The
vocabulary is a security artifact: widening it is the only edit that changes what can be
transmitted, and `build_vocabulary.py --offline` rebuilds without network.

### Not yet done in this phase

The UI half -- replacing the `pill-app-conn` "App connections to the internet: 0" pill (which
becomes meaningless once the agent researches the web) with "Your documents: on this device
only" and a live "Agent web access" counter opening the audit panel -- lands with Phase 1,
where there is a run that actually performs egress to display.

---

## Phase 17 -- the packet becomes a real document (2026-09-18)

**Reported:** "when trying to save as packet it loads just a plain text webpage vs pdf doc
that is readable."

Correct, and it was the obvious reading of the code: `GET /api/runs/{id}/packet.md` returned
`PlainTextResponse(..., media_type="text/markdown")`, and the UI's Download packet button did
`window.open()` on it. Browsers render `text/markdown` as unstyled plain text, so the hand-off
artifact -- the thing a grants director is supposed to file or email -- arrived as a wall of
raw Markdown with `|---|---|` table pipes in it.

### `app/packet.py`

New `GET /api/runs/{id}/packet.pdf` returning a paginated PDF with
`Content-Disposition: attachment`, so it downloads as a named file rather than opening as yet
another tab of text. `packet.md` stays for machine consumption and diffing.

The document carries: a metadata block (institution, solicitation, run, model, device,
timestamp), the run summary, issues ordered by severity with their codes and recommended
actions, the compliance matrix as a real table with colour-coded status and a repeating header
row across pages, and each drafted section with its reviewer score, citation-check result and
must-fix list. Every page carries `DRAFT — AI-GENERATED, FOR HUMAN REVIEW` in the header and
the on-device generation stamp in the footer.

Two constraints carried over from the rest of the app rather than re-decided:

- **Nothing may read as more finished than it is.** Reviewer scores, unsupported-claim counts
  and the revision-cap `needs_human` markings travel *with* the drafts. A packet that printed
  only the prose would be a better-looking lie.
- **Model text is never trusted.** ReportLab's `Paragraph` parses a mini-HTML dialect, so draft
  bodies are XML-escaped *before* the small Markdown subset is applied -- the same
  escape-then-transform order the web UI uses (`docs/ARCHITECTURE.md`, "Security"). A test
  feeds a draft full of hostile markup (`<para>`, unclosed tags, bare `&`) and asserts the
  document still builds.

ReportLab was chosen over the alternatives on licence and portability: BSD-3, pure-Python
wheel (`py3-none-any`), so it needs no compilation on aarch64 and clears
`scripts/license_audit.sh`. `fpdf2` was excluded outright -- it is LGPL, which that audit fails
on by design. Added to `pyproject.toml` and to the Demo Center install step in
`install_plans.py`, with the import-verification line extended to cover it.

### A second bug found while verifying it

The packet endpoints resolved runs from `run_manager`, which is in-memory only. Restart the app
-- as the Demo Center's own `/start` and `/stop` do routinely -- and every past run's packet
became a 404 while its `state.json` sat on disk beside it. Since a packet is most often wanted
*after* the run, that was the more likely failure in practice than the formatting one.

Added `RunState.from_dict()` (symmetrical with the existing `to_dict()`, tolerant of unknown and
missing keys so an older `state.json` still loads) and a `_state_for_packet()` helper that
returns the in-memory run if present and otherwise rehydrates from disk. Verified by restarting
the app and pulling the packet for a run created before the restart: HTTP 200, 8 pages. An
unknown run id still 404s.

### Verification

`tests/test_packet.py`, 8 tests: output really is a PDF (`%PDF-` magic, `%%EOF` trailer), an
empty/cancelled run still renders rather than raising, DRAFT provenance is present, escaping
happens before Markdown transformation, hostile markup still builds, citations survive into the
document, and a round-tripped state produces the same packet. Suite total 133.

Rendered and read end to end, not just byte-checked: a live quick run (222 s, 32 steps) produced
an 8-page packet, rasterised with `pdftoppm` and inspected. One layout defect found and fixed
that way -- `needs_human` in the narrow Status column wrapped mid-word as "needs_hum / an",
since underscores give the wrapper no break opportunity. Status is now rendered with spaces and
the column is wider. The remaining `needs_human` in the text is inside the model's own summary
prose, which is its words and stays as written.

---

## Phase 18 -- v2 Phase 1: Discover (2026-09-18)

Upload an institution's own documents; the agent reads them, writes a cited profile of what
they actually do, searches the live federal funding catalogue, and comes back with a short,
reasoned list of opportunities worth pursuing -- without any of their material leaving the box.

New run mode `discover`, selected in the UI alongside Story / Explore / Replay.

### Playbooks rather than one growing tool list

`ToolContext.playbook` is `"proposal"` or `"discovery"`, and `schemas_for(playbook)` decides
which tools the model is even offered. A discovery run has no solicitation, so handing it
`draft_section` / `check_budget` / `get_requirements` would be inviting it to wander into
writing a proposal it has no requirements for. The orchestrator loop, guardrails, history
compaction and metrics are all shared unchanged; only the prompt, the brief, the tool set and
the finish gate differ.

Six new tools: `draft_profile`, `search_grants`, `get_opportunity`, `fetch_url`, `search_web`,
`propose_opportunity`. `search_vault`, `read_document`, `verify_citations` and `flag_issue` are
reused as-is.

### The guardrail that makes it worth demoing

`propose_opportunity` refuses any opportunity id the catalogue did not actually return this
run, and refuses a rationale with no `[[doc_id]]` citation. The discovery analogue of the
citation gate: **the agent cannot recommend a grant it never retrieved**, and cannot justify one
without pointing at evidence. Enforced in `tool_propose_opportunity`, not prompted.

The profile is stored as a normal draft under section id `institution_profile`, so
`verify_citations` applies to it with no special-casing, and the finish gate requires it to have
been verified with zero unsupported claims before anything can be proposed.

**One thing the gate deliberately does NOT require is a minimum number of proposals.** The first
version did, and it was wrong in a way worth recording: the prompt tells the agent that a short
list of genuinely suitable opportunities beats a padded one, and then the gate would have
refused to let it finish having honestly found nothing -- pressuring it to pad. Integrity here
comes from proposals corresponding to real retrieved opportunities, not from a quota.

### Fetched pages are citable documents

Anything `fetch_url` retrieves is converted to Markdown (the same bs4 + markdownify path
`scripts/fetch_solicitation.py` already used), written into the session corpus with a `web-<host>-<hash>`
doc id, and cited exactly like a vault document -- so `verify_citations` covers it. Without
that, open-web content would be the one source in this app the citation gate could not see.
The text is also folded into the egress guard's corpus, so material pulled off the web cannot
then be echoed back out as a search term.

### Four live runs, three real bugs

Run 1 (364 s, hit the 60-step cap): proposed four opportunities, containment perfect -- only
`advanced manufacturing`, `robotics`, `technician education`, `workforce development` left the
device. But **every `fetchOpportunity` call was refused**: the payload is `{"opportunityId": "324369"}`,
an id the catalogue had just returned, and the layer-2 token scan treated those digits as
unknown. The agent burned its whole step budget retrying.

*Fix:* echo-back. The guard records tokens from responses it receives, and a token the server
itself supplied may go back out -- **unless it also appears in the institution's documents**, in
which case it is their data whoever else publishes it. An award number the institution cites
stays put even if grants.gov happens to mention it.

Run 2 (151 s, did nothing at all): 60 steps, zero searches. `search_vault` returned no hits for
any query. **BM25's IDF goes to zero -- negative, in rank_bm25's formulation -- for a term that
occurs in every document.** With a two-document upload every query scored <= 0, the `scores[i] <= 0`
filter dropped everything, and the agent stared at an apparently empty vault while the files sat
right there. The shipped 54-chunk vault has positive scores throughout, so this never surfaced
in a year of proposal runs; it only appears on the small corpora Discover is built for.

*Fix:* when the score floor would drop a chunk, keep it if it shares real tokens with the query.
Ranking still works, and a query with no overlap still returns nothing -- the fallback relaxes
the floor, it does not invent relevance. Regression test in `tests/test_bm25.py`.

Run 3 (300 s, `incomplete`): profile written and cited, three searches, 53 opportunities seen,
one proposal -- then **twelve consecutive `finish` calls**, each refused with "verify_citations
has not been run on the latest institution_profile". The agent had tried exactly that and been
told `'institution_profile' is not one of ['project_summary', 'results_prior_support',
'evaluation_plan']`: the tool's schema enum was never widened when the profile was made a
citable draft. It was being held to a gate it had no way to satisfy.

Also refused in run 3: `https://www.nsf.gov/funding/pgm_summ.jsp?prs=ATE` -- a link NSF's own
page had just handed it. Scanning the query string of a published link rejected ordinary
navigation.

*Fixes:* `VERIFIABLE_SECTION_IDS` adds the profile to the enum; and a URL that appeared verbatim
in a response we already received is treated as a link to follow rather than a message to scan.
A URL we were never given is still scanned in full -- covered by a test that a constructed URL
carrying upload text is still refused.

### Run 4 -- the one that works

**Completed in 105 s, 20 steps.** Three searches, 34 opportunities retrieved, 8 outbound
requests, **zero refusals**. Two recommendations:

| Confidence | Opportunity | Note |
|---|---|---|
| HIGH | NSF 24-584, Advanced Technological Education | correctly identified as the best fit; flagged the two-year IHE eligibility check and the Oct 2026 deadline |
| MEDIUM | NSF 24-564, IUSE/PFE: RED | upgraded from LOW in an earlier run after fetching the programme page and finding Track 4 explicitly invites two-year partnerships |

Containment audited against the run's own egress log: institution name, PI name, project name,
award number, enrollment and lab size -- **none reached the wire**. The complete list of what
left is six generic field terms and eight requests to `api.grants.gov` and `www.nsf.gov`.

### UI

`Discover` mode with multi-file upload into one session (the upload endpoint became additive --
it previously minted a fresh session per file, so only the last document was ever visible),
an **Opportunities** tab ranked by confidence with concerns listed, and an **Egress** tab showing
every request and every refused term with its reason.

The `pill-app-conn` pill, which read "App connections to the internet: 0" as a proof of zero
egress, had to change meaning: Discover deliberately reaches the catalogue, so a counter that
meant "clean" would now mean nothing. It is replaced by "Your documents: on this device only"
plus a live, clickable "Agent web access: N requests · M refused" that opens the audit panel.

### Verification

156 tests (up from 133): `tests/test_discovery.py` covers playbook tool isolation, term
refusal with reasons, the anti-fabrication gate, the citation requirement on rationales, the
finish gate including the honest-nothing-found path, fetched pages becoming citable, and
graceful degradation when no web-search provider is configured. `tests/test_egress.py` gained
echo-back and follow-links tests. Portability audit extended: `web/` must still contain zero
external URLs, while `app/` allows them only in the guarded discovery clients -- both halves
verified by making them fail.

### Not done

`search_web` needs a self-hosted SearXNG to do anything; without one it reports itself
unavailable and `search_grants` carries the run. Deliberately no commercial default -- an API
key on a demo box, and queries routed through a third party, would both undercut the point.

---

## Phase 19 -- v2 Phase 2: Intake (2026-09-18)

Feed it any solicitation -- PDF, HTML or Markdown -- and the agent works to *that* document's
requirements and *that* document's sections, instead of the built-in NSF 24-584 set.

### This was a latent bug, not only a new feature

`docs/BUILD_PLAN.md`, `docs/PRESENTER_SCRIPT.md` and the UI label all describe Explore mode as
accepting an uploaded solicitation. The code made the upload the **vault**, while
`_requirements_list()` went on loading `data/solicitation/requirements.json` regardless. Explore
mode has never done what three places say it does: you could upload a solicitation and the agent
would keep working NSF 24-584's 54 requirements. Intake is Explore's correct implementation, so
it is not a fourth mode.

### `app/ingest/requirements.py`

The extraction logic moved out of `scripts/extract_requirements.py` so the build-time script and
the runtime path share one implementation rather than two that drift. The script now imports it.

Three properties carried over deliberately:

- **Quotes are validated, not trusted.** A candidate survives only if its `source_quote` appears
  character-for-character in the section it came from. On the live NIH run, **16 of 94 candidates
  were discarded** for failing that check -- the anti-hallucination guard visibly working.
- **Thinking off.** Mechanical extract-and-quote work. Measured here at ~18 s/section with it
  off; the original build measured 2-6 minutes per section with it on for marginal gain.
- **Cached by content hash** (`sha256` of text + model + prompt version), so re-running the same
  solicitation is instant.

Added: `ExtractionProgress` streamed per section, modest concurrency, and `propose_sections()`,
which asks the model what narrative sections *this* solicitation calls for rather than assuming
NSF's three.

### Measured before building, as the plan required

Sequential extraction of the full 23-section NSF solicitation: **~175 s** -- not the 10-20
minutes feared. 4-way concurrency measured **1.29x** (GPU-bound, not request-bound), so it is
used but not tuned. Both numbers taken on the Nano before any UI was written.

### De-hardcoding

Sections were a JSON-schema `enum` of the three NSF ATE names. That broke twice -- once when the
institution profile became a citable draft in Phase 18, and again here -- and both times the
model asked for exactly the right thing and was told it "is not one of [...]" with no way to
discover what it should have said. Replaced with `_check_section()`, which validates against the
run's actual sections and names them in the error.

Also now solicitation-derived rather than assumed: the reviewer persona (an uploaded call for
proposals is not reviewed by an NSF ATE panel), the merit criteria (Intellectual Merit /
Broader Impacts are NSF's specifically, and scoring somebody else's proposal against criteria
their solicitation never mentions is simply wrong), and the scope guidance (the shipped version
names Harrow Creek's three employer letters, which would send the agent hunting for documents
that do not exist).

### A bug the NSF document could never have found

First live NIH run died with `'list' object has no attribute 'get'`. `extract_from_section`
assumed the model returns `{"requirements": [...]}`; NIH's document provoked a bare top-level
array, and `AttributeError` was not in the caught set. Latent in the original script since
Phase 4 -- invisible because NSF 24-584 always produced the wrapped form. Now both shapes are
accepted, non-dict entries are discarded rather than fatal, and an intake failure degrades to
the shipped requirements instead of ending a run the operator has already waited minutes for.

### Live: PAR-24-119, an NIH NIDCD R21

A real 68 KB NIH announcement -- different agency, structure and vocabulary from anything this
build has seen.

| | |
|---|---|
| Sections scanned | 9 |
| Requirements extracted | 78 |
| Discarded, unquotable | 16 |
| Sections derived | "Specific Aims", "Project Strategy" -- NIH's own structure, not NSF's |
| Reviewer criteria | "Alignment with NIDCD Mission", "Scientific Rigor & Methodology" |
| Intellectual Merit / Broader Impacts in output | **absent**, correctly |
| Run | completed, 28 steps, 316 s |

The first NIH run scored 1.0-2.5 across sections, for an honest reason: it was drafting a
hearing-research R21 out of Harrow Creek's advanced-manufacturing vault. The code was right and
the pairing was useless. So an Explore session can now carry `kind=documents` alongside its
solicitation; given a matching lab profile the same solicitation scored **3.8 / 3.2**, and the
drafted Specific Aims cited that lab's own pilot data.

Worth noting: section planning varies between runs on the same document ("Specifications /
Significance and Innovation / Approach" in one, "Specific Aims / Project Strategy" in another).
Both are plausible NIH structures, but it is model variance, not a stable derivation.

### Verification

169 tests (up from 156). `tests/test_requirements_extraction.py` covers boilerplate-heading
skipping, the quote guard (an invented requirement with a fabricated quote is dropped),
category/text validation, near-duplicate collapsing, the output shape the app already consumes,
per-section progress, content-hash caching including a corrupt cache entry, section derivation
and its fallback, and tolerance of every JSON response shape including the bare array that
caused the outage above.

---

## Phase 20 -- v2 Phase 3: Grade (2026-09-18)

Upload a proposal that has already been written and get it marked: every section scored as a
panel would, every requirement ruled on, the budget checked, and -- the part that matters most --
its figures cross-checked against the institution's own records.

New run mode `grade`. The submission's own headings become sections, seeded into `state.drafts`
as v1 drafts, so `review_section` and the compliance matrix apply with no special-casing. A flat
document with no headings becomes one section rather than nothing, since a PDF export frequently
loses heading structure.

### Two guardrails, and one that had to change meaning

**The grader cannot rewrite what it is assessing.** `draft_section` is not in the grade tool
set. An agent asked to assess work should not be able to quietly improve it and then score its
own version.

**It cannot finish leaving a requirement unruled.** Every requirement needs a status, including
`not_applicable`. A requirement nobody looked at reads as compliant to whoever receives the
scorecard, which is worse than a bad score.

**`update_compliance`'s "met" gate had to change.** In the proposal playbook, marking a
requirement met requires the linked section to have passed `verify_citations`. A *submitted*
proposal carries no `[[doc_id]]` markers -- it was written by a human for a funder -- so that
gate cannot apply and would have blocked every requirement. In grade mode "met" instead requires
naming the section of the submission that satisfies it, and that section must actually exist.

### The gap the first live run exposed

The test proposal was written with planted flaws: a budget over cap, a wrong fringe rate,
marketing puffery, a vague evaluation plan, and one factual overstatement -- "graduating 120
completers against a target of 120" where the institution's own data records 84.

It caught everything except the overstatement. The reviewer scored that section **3/4** and
praised it for "exceeding graduation targets". Removing `verify_citations` from grading was
correct, but it left the submission's facts entirely unexamined -- and checking claims against
the record is the single most valuable thing a grader can do.

`app/checks/claims.py` is the grading analogue: pull numeric claims out of a section, retrieve
the evidence that should support each from the vault, and judge supported / contradicted /
unverifiable, with the evidence in hand rather than from recall. A contradiction raises a
`CLAIM_CONTRADICTED` issue at high severity, and the finish gate now requires every section to
have been claim-checked as well as scored.

**Its own test killed the first version.** I had a deterministic shortcut: if every number in a
claim appears in the retrieved evidence, mark it supported without a model call. For the claim
"graduated 120 completers against a target of 120", the record "Completers: 84 against a target
of 120" *does* contain 120 -- as the target. The shortcut declared it supported and never asked.
Presence of a figure is not support for the claim it was lifted into, and this is the one check
whose entire purpose is catching numbers that look right and are not. The shortcut is gone, and
the comment saying why is in the file so nobody re-adds it as an optimisation.

### Batching, because bookkeeping was eating the run

The first grade run spent **39 of 60 steps** calling `update_compliance` once per requirement --
54 of them for NSF 24-584 -- and never reached `finish` at all. `update_compliance` now takes an
`items` array, applying the guardrail to each item individually. The next run finished in 20
steps, and the final one in 11.

### Live result

| | |
|---|---|
| Sections seeded from the submission | 5 |
| Run | completed, 11 steps, 165 s |
| Budget | `BUDGET_OVER_CAP` ($38,450 over) and `FRINGE_RATE_MISMATCH` (28.0% vs 31.5%), both deterministic |
| project_summary | **1.36/5** -- the puffery scored as a panel would score it |
| evaluation_plan | 2.20/5 |
| Compliance | 54/54 ruled: 9 met, 15 partial, 17 missing, 12 needs_human, 1 n/a |
| Claim check | `CLAIM_CONTRADICTED` on the Project Summary's "400 students, 95% placement" against the vault's actual 84 completers / 78% placement |

**Honest limit:** the mechanism fires and demonstrably catches overstatement, but recall is not
complete. In this run the specific "120 completers" sentence came back `unverifiable` rather
than `contradicted` -- BM25 retrieval did not surface the prior-outcomes chunk for that sentence.
It caught a different inflated claim in the same proposal. Worth improving retrieval for claim
checking; not worth claiming it catches everything.

### Verification

193 tests (up from 169). `tests/test_grade.py` covers the tool surface (draft_section out of
reach), the evidence gate on "met" including evidence naming a section that does not exist, the
finish gate across sections-unscored / claims-unchecked / requirements-unruled / no-submission,
batch updates applying the guardrail per item, and that the proposal playbook's citation gate is
unchanged. `tests/test_claims.py` covers numeric-claim extraction, the contradiction case, the
absence of the shortcut, and degradation to `unverifiable` when judging fails or returns a
verdict string that is not one of the three.

---

## Phase 21 -- finishing for a full test (2026-09-18)

Closing the gaps that stood between "the features work" and "somebody else can sit down and
drive it".

### A UI bug that only a browser would have found

`app/main.py` and the tools were tested by API throughout. The UI was not, and it had a real
break: `pill-app-conn-text` was still referenced in `app.js` after Phase 18 removed that element,
so `applyNetworkStatus()` threw a `TypeError` on **every** `network_status` event -- a timer that
fires for the whole run. The network pills would have been dead on arrival.

`tests/test_ui_wiring.py` now fails the build if any `$("#id")` in `app.js` has no matching
element, if an inline handler names a function that does not exist, if a mode in the selector has
no control group, if an output tab has no panel, or if any external URL appears in the UI. Static
checks, but they cover exactly the class of error that is invisible until someone opens the page.

The pill's replacement does something more useful than the line that broke: it compares the
socket count `netstatus` observes against the requests the egress guard actually recorded, and
says so if they disagree. Connections this app did not make through the guard are precisely what
would matter.

### SearXNG, so Discover's web search is not inert

`scripts/start_searxng.sh` starts a loopback-only container and writes its own config, including
the `formats: [html, json]` that SearXNG does not enable by default -- without which
`/search?format=json` returns 403 and the tool looks broken rather than unconfigured. It is an
install step, deliberately best-effort: it exits 0 even when it cannot start anything, because a
missing search engine must not fail a demo install.

**This exposed a conflict in my own boundary.** The egress guard refuses non-public hosts so a
crafted URL cannot reach the model endpoint or the Demo Center's install API -- which also
blocked a loopback SearXNG the operator deliberately stood up. Resolved by distinguishing
operator-configured infrastructure from model-chosen targets: a configured endpoint is exempt
from the host-class check **only**. Query terms are still validated and the request is still
audited, so what may be sent is unchanged. Tests cover that trusting one local endpoint does not
trust the rest of the host, and that a lookalike (`127.0.0.1:8096.evil.test`) does not inherit
trust through prefix matching.

### The packet carries the new modes

It was still a writing-run artifact. It now includes Discover's ranked shortlist with the
concerns a human must check, Grade's contradicted and unverifiable claims, and a "What left this
device" section listing every term transmitted and every host contacted. A grading packet says
"Submitted sections, as assessed" rather than claiming the agent drafted them.

One deliberate omission: the egress section lists what was **sent**, never what was refused.
Reprinting a refused term would put the institution's own language into a document meant to be
shared -- the exact thing the boundary stopped. There is a test for that.

### Docs

`README.md` rewritten around the four modes and what leaves the device. `PRESENTER_SCRIPT.md`
gained a talk track for each of Discover, Explore and Grade, with the line that lands for each
and the timings -- plus what not to promise: section planning under Explore varies between runs,
and claim checking is not exhaustive. `ARCHITECTURE.md` documents the playbook mechanism and the
new modules. `DEMO_CENTER_INTEGRATION.md` records which modes need network (one of four) and the
SearXNG interaction with the guard.

208 tests.

---

## Phase 22 -- sample inputs, real funder data, and typed profiles (2026-09-19)

Reported after hands-on testing: the modes work but there is almost nothing to test them
*with*. Accurate, and structural — Story reads the shipped `data/vault/`, while Discover,
Explore and Grade all read `runtime/sessions/`, which ships empty. Every input used to validate
those three during development was a file written by hand and never committed.

### Samples

`data/samples/manifest.json` plus `GET /api/samples` and `POST /api/samples/{id}/load`, and a
"Load a sample" picker in each mode. Six sets: Harrow Creek's institutional documents for
Discover, three real solicitations for Explore, and two contrasting proposals for Grade.

The manifest **references** files already in the repo rather than copying them, so there is one
copy of each document and one provenance row. Loading two samples into one session combines
them, matching how uploads already behave.

### Real funder data

`scripts/fetch_samples.py`, following `fetch_solicitation.py`'s discipline — fetch, convert,
verify by grep, record source URL, retrieval date and sha256. Ships a real current NIH
announcement, a real posted grants.gov opportunity, and a real *funded* NSF award abstract.

The first NIH document fetched was **expired** — NIH serves retired announcements at the same
URL with an "Expired" title prefix. That is exactly the "nobody should present a dead
opportunity as live" risk, so the fetcher now refuses an expired document rather than labelling
it, and the sample was moved to a current announcement.

### Typed profiles

A form in Explore and Discover writing `POST /api/session/profile`, which renders the fields as
Markdown into the session's `documents/`. Deliberately a **document**, not run configuration:
BM25 indexes it, the agent cites it as `[[profile]]`, `verify_citations` and `check_claims`
apply to claims resting on it, and the egress guard's `SessionCorpus` picks it up — so typed
research descriptions are contained exactly as uploaded ones are.

Real institution names are accepted here and the UI says so. That is not a contradiction of the
synthetic-institution rule: `validate_vault.py` enforces that over `data/vault/` at build time
and never sees session data. Verified with a real university name in the form.

### The vocabulary was too small for actual research

Typing a materials lab's profile showed `neutron characterization` being refused — a public
field name, blocked because the hand lexicon only had 372 broad terms. Discover would have been
useless to exactly the people it is for.

`build_vocabulary.py` now also harvests NIH RePORTER's project index terms and NSF programme
names: **1,295 entries to 28,981**. Recall on a spread of real research language went to 6/6.
Containment was re-verified, because expanding an allowlist 22x is precisely when it would
break: all 35 egress tests pass, and institution names, PI names, project names and award
numbers are still refused.

### Two bugs the samples found

**An unquoted front-matter date killed an entire run.** YAML parses `date: 2026-09-19` into a
`datetime.date`, which travelled as far as the tool result and failed with "Object of type date
is not JSON serializable". The vault's own documents quote their dates; an uploaded or typed one
cannot be relied on to. Front matter is now coerced to JSON-safe scalars in the loader — a
visitor's document must not be able to end a run by being ordinary Markdown.

**Grading checked a submission against the wrong institution's records.** Running Grade on the
real SUNY Binghamton award, the corpus was still Harrow Creek's vault, so `check_claims`
compared a real award's figures against a fictional college's documents and reported "wrong
institution cited for prior award" as a contradiction — confident nonsense about the demo's own
unrelated scenario. Grade now reads session documents when they are supplied, and the flawed
SATP sample ships the six institutional records its claims are checked against.

That fix also resolved the claim-check recall gap logged in Phase 20. Against the full
24-document vault, BM25 never surfaced the prior-outcomes chunk for the "120 completers"
sentence and it came back `unverifiable`. Against the six records that actually belong to the
submission it lands every time:

> *"We graduated 120 completers against a target of 120"* — **contradicted**: the prior
> evaluation report explicitly documents only 84 completers.

Recall for claim checking depends on the corpus being the *right records*, not a large vault.

### Verified

Discover driven from the sample button alone, no hand-made files: 240 s, 7 searches, 62
opportunities retrieved, 4 proposed across NSF/EPA/USDA, profile citing 7 of the 8 sample
documents, 8 generic terms out, and institution, PI, award id and project name all contained.
Grade on the flawed sample: all three planted contradictions caught plus both deterministic
budget findings. 226 tests.

---

## Phase 23 -- a research scenario (2026-09-19)

Harrow Creek is a two-year technician college that does no research, so the one shipped
scenario could never demonstrate "find grants for our research group" — the thing Discover and
Grade are for. `data/scenarios/meridian/` is the counterpart: the **Calderwick Institute for
Coastal Systems**, a fictional coastal-science institute inside a fictional public university.

### Generators parameterised rather than forked

`generate_vault.py` and `validate_vault.py` gained `--seed` / `--vault`, defaulting to today's
values so nothing about Harrow Creek moves (verified: `git status` shows no change under
`data/vault/`, and its validation still passes).

A seed can now declare its **own document set** (`documents:`), its **CSV document ids**
(`csv_documents:`), the **facts that must survive regeneration** (`key_facts:`) and the
**phrases that must not appear** (`must_not_contain:`). Adding a scenario is a data change.

Three couplings surfaced while doing it, each of which would have broken any second scenario:

- `write_outcomes_csv` was shaped entirely around a workforce programme — enrolled, completers,
  placement respondents. A research institute reports papers, students and datasets, so a
  scenario without `completers_actual` now gets a generic metric table.
- `check_budget` hardcoded `budget-satp-draft.csv` and `rate-agreement-fy2026.md` as defaults.
  That is not just a validation problem: **the agent's own budget tool would have failed
  outright on any other vault.** Both now fall back to finding the document by shape.
- The budget CSV carried Harrow Creek's evaluator name and project title as literals, so the
  research scenario's first budget was quietly contaminated by the scenario it was modelled on.
  Both now derive from the seed, and a new `check_scenario_isolation` fails the build if any
  scenario contains another's institution, people or project names.

### The planted issue that was not planted

The seed declared a PI over-committed across current and pending support. The generated
biosketch reported a healthy **12% total** with "significant availability" — a planted issue
that existed only in the seed's own description of itself, which is worse than no planted issue
at all, because the scenario reads fine.

Fixed by making the brief explicit about what the table must show, and by adding the
`key_facts` / `must_not_contain` mechanism so drift is caught rather than discovered. Verified
by making it fail: appending "maintains significant availability" to the biosketch now fails
validation. The regenerated table commits 30/20/25/20/15% — 110%, stated plainly, with no
comment. A reviewer is meant to notice it, not be told.

### Two bugs that were letting real overstatement through

Grading the research proposal, the deterministic budget findings fired but the claim check
returned **unverifiable** on "produced 14 peer-reviewed publications and supported 6 graduate
students" against a record of 9 and 4. Two causes, both in code shared with citation checking:

- **`_NUMBER_RE` ignores bare single digits.** Correct for citation verification, where a stray
  numeral near a citation is noise — and exactly wrong for claim checking, where "6 students"
  against a record of 4 *is* the overstatement. Claim checking now has its own pattern.
- **The evidence handed to the judge was a 400-character excerpt** centred on the first query
  token, which for a metrics table stopped immediately before the value. The judge answered
  honestly about a table whose relevant cell had been trimmed away. `Hit` now carries the whole
  chunk and the judge sees that.

Both were invisible against the first scenario because its overstatement (120 completers vs 84)
uses three-digit numbers.

### Result

Grading the research proposal now catches every planted issue, plus one nobody planted
explicitly:

| | |
|---|---|
| `BUDGET_OVER_CAP`, `FRINGE_RATE_MISMATCH` | deterministic |
| 14 publications / 6 students | **contradicted** — the final report records 9 and 4 |
| "All 8 planned field campaigns were executed successfully" | **contradicted** — only 5 of 8 completed |
| 28.0% fringe | **contradicted** by the rate agreement's 31.5% |

It also noticed the proposal does not fit the solicitation it was being judged against — with
no solicitation loaded it grades against the built-in NSF ATE requirements, and reported
"wrong project type" and "institution mismatch: four-year vs two-year applicant". Correct, and
now said plainly in the sample's description.

232 tests. Both scenarios validate; Harrow Creek is byte-identical.

---

## Phase 24 -- four more institution types (2026-09-21)

Two scenarios was not enough to show this to higher education. A community college and a
research institute cannot represent an R1, a minority-serving institution, a medical centre or
a system office, and those are different customers with different failure modes — not the same
demo with the names changed.

Four new fictional scenarios, each at the depth Calderwick set:

| Scenario | Type | What it is for |
|---|---|---|
| **Vantcross State University** | Public R1 | NIH R01 **resubmission**: subawards, core facilities, effort across a lab, 58% F&A |
| **Tierra Sandoval University** | Hispanic-Serving Institution | NSF capacity-building, where **eligibility is the whole ballgame** |
| **Ashgrove University Medical Center** | Academic medical centre | NIH clinical trial: IRB, DSMB, enrolment feasibility |
| **Sablecreek State University System** | Four-campus system office | Consortium: subaward structure, cost share, per-campus commitment |

53 documents generated, no failures. Planted issues are native to each type rather than the
same three repeated — subaward F&A applied to the whole subaward instead of the first $25,000;
a resubmission with no introduction answering the prior summary statement; an enrolment
percentage the registrar's census contradicts, on which eligibility rests; an IRB approval that
expired in March; a data safety monitoring board that exists only as a proposal; cost share
described as confirmed when half is pending; two of four campuses with no current letter.

### Three validation failures that were my checks, not the documents

`validate_vault.py` failed three scenarios, and in all three cases the document was right:

- The IRB status document states its expiry as "March 31, 2026". My `key_facts` demanded the
  ISO form. A date has many renderings and the check compares literal strings.
- The cost share register says "**not** fully committed" — exactly the planted defect — and my
  `must_not_contain` banned the positive phrase, so it flagged the negation of it.
- `must_not_contain: ["SSI"]` fired inside the ordinary words "mission" and "missions".

The third is the one worth keeping: **a short project acronym banned as a substring is a
landmine for every future scenario**, so the check now matches whole words. Verified by
planting a genuine whole-word mention and watching it fail.

### The gap the medical centre exposed

Grading the clinical trial caught every numeric overstatement — 450 participants against a
record of 288, seven sites against four, and an accrual projection its own feasibility analysis
contradicts — and **missed both signature regulatory failures**.

The cause: `check_claims` only extracted sentences containing numbers. "IRB approval is current
and in good standing" and "an independent data safety monitoring board is in place" contain no
digits, so the checker never saw them — and those are precisely the claims a research
administrator needs checked.

Claim extraction now also captures **status assertions**: a short, explainable list of
approval/commitment phrasings (`is current`, `is in place`, `has been approved`, `is confirmed`,
`all X were executed`), with matching guidance to the judge. Deliberately narrow — scoring prose
is what the section review is for, and treating every sentence as a checkable claim would spend
model calls on opinions. Tests assert both halves: the status claims are caught, and
"this approach is novel" is not.

Re-graded, it now reports:

> *"Approval is current and in good standing."* — **contradicted**: IRB approval for the parent
> protocol expired on March 31, 2026 and no renewal has been issued.
>
> *"An independent data safety monitoring board is in place..."* — **contradicted**: the DSMB
> has not yet been constituted.

And the HSI scenario catches the figure its designation depends on: *"As a Hispanic-Serving
Institution with 31% Hispanic enrollment"* — **contradicted**, the official census records 26.4%.

### State

Six scenarios, all validating. 16 samples (6 Discover, 9 Explore, 9 Grade). 235 tests. Harrow
Creek and Calderwick byte-identical.

**Known limit, unchanged:** with no solicitation loaded, every Grade run is scored against the
built-in NSF ATE requirements, which produces real but irrelevant findings for a clinical trial
or a coastal-science proposal ("missing required ATE partnerships"). Pair a Grade sample with a
solicitation sample to fix it. Worth making the UI nudge toward that pairing.

## Phase 20 — the scenario picker, and three bugs it uncovered

The six scenarios were reachable from Explore, Discover and Grade, and **not from Story** —
the one mode built around them. Story read `data/vault/` and NSF 24-584 unconditionally. A
scenario needs both halves: an institution *and* something to apply to, and drafting a clinical
trial against a technician-education solicitation is incoherent.

`scripts/build_scenario_solicitation.py` gives each generated scenario its own synthetic
solicitation, written from its seed and labelled synthetic in its own front matter, with
requirements extracted at build time and committed (85–93 each). Harrow Creek keeps the real,
hand-curated NSF 24-584 it was built around. `GET /api/scenarios` lists them; `scenario` is
threaded through run creation, vault resolution, requirements, tracks, institution name, the
packet and `RunState`, and the UI grew a picker.

The first live Ashgrove run then found three things, in ascending order of seriousness.

### 1. `verify_citations` was the third vacuous-pass gate

The draft cited `[[aumc-institute-profile-2026#aumc-institute-profile-2026#c000]]` — the model
was shown both a `doc_id` and a `chunk_id`, and a chunk id already contains its doc id. The
two-segment citation pattern did not match that at all, so the run recorded `citations: []`
**and the check passed**, because with nothing parsed there was nothing left to object to.

A pass over an empty set is not a pass. `verify_citations` now reports `citation_count` and
`malformed_citations`, and fails a draft that has either an unreadable `[[...]]` marker or no
citations at all. The UI step card names the count for the same reason: "Citations OK" over
nothing verified is invisible from the room.

That is **the third gate in this project that printed a pass while checking nothing** — after
the E3 endpoint check and `license_audit.sh`. The practice that catches them is the same one
each time: verify a new gate by making it fail.

### 2. Chunk-level citations had never worked

Following the doubled form back: `[[doc#c000]]` — the exact syntax the prompt asks for — was
parsed as chunk id `c000` and compared against the real chunk ids, which are `doc#c000`. So
*every* precise citation the agent made was reported as a missing document, and only whole-doc
citations could ever pass. The citation is now rebuilt into canonical form before lookup, and
all three forms resolve.

### 3. The quick-scope prompt named another institution's partners

`QUICK_SCOPE_GUIDANCE` hardcoded Harrow Creek's three industrial employers. So a quick Ashgrove
run — an academic medical centre — was told to go and read the Kessler Fluid Systems letter,
searched a hospital's vault, found nothing, and filed **three fabricated "missing partner
letter" issues** into the scorecard. Not wasted steps: inventions, in the one artifact this
demo claims contains none.

Quick guidance is now built from the scenario's own seed (`employers` or `collaborators`), with
a generic fallback, and a test asserts no scenario can inherit another's partner names.

Story sections are also scenario-derived now: `propose_sections` runs at build time and is
committed, so Ashgrove drafts *Specific Aims* rather than *Results from Prior NSF Support*.
And `propose_sections` no longer swallows a dead endpoint into a silent generic fallback —
build scripts pass `strict=True`, because a build against an unreachable model server used to
commit exactly what a successful one did.

### State

Six scenarios, selectable in every mode. 243 tests. Verified live end to end: an Ashgrove Story
run failed citation verification on v1 (a `[[NEEDS INPUT: ...]]` placeholder the old gate could
not see), revised, and passed v2 with 16 verified citations.

## Phase 21 — the page was the scroll container

Two reports: the Institution picker "does nothing", and the view is dragged back down to
the newest reasoning step, so the token counter cannot be read while the agent works.

The second one is a real bug with a single cause. `#control`, `#trace` and `#outputs`
all declared `overflow-y: auto`, but nothing bounded their height — so none of them
ever scrolled and the **document** grew instead. Every consequence followed from that:

- `card.scrollIntoView()` scrolls *every* scrollable ancestor, the window included, so
  each new trace card yanked the viewport down to the tail.
- The header went with it, which is why the token counter was unreachable.
- The "at the bottom?" check listened to `#trace`, which never scrolled — so
  `traceAutoScroll` was permanently true and **Jump to live never appeared**. The escape
  hatch had been there all along and could not fire.

The desktop layout is now pinned to the viewport (`body` flex column, `overflow: hidden`,
columns `min-height: 0`), so the header and footer stay put and each column scrolls
itself; below 1280px it reverts to one column and normal document scrolling. Trace
autoscroll moves `#trace.scrollTop` and nothing else. Verified against a live streaming
run: scrolled to the top of the trace, then 30 seconds and seven cards later the trace
was still at the top, the window had not moved, Jump to live was showing, and the token
counter was visible and still counting (55,085 → 109,706).

Three static gates now guard it, each verified by reintroducing the bug and watching
them fail: the columns must be bounded, `.scrollIntoView(` must not appear in `app.js`,
and every autoscroll call must be guarded by `traceAutoScroll`.

### The picker

It works. Driven in a real headless browser — select Vantcross, press Start — the run is
created with `scenario: "vantcross"` and its 85 requirements. What was missing was any
*confirmation*: the scenario card and note changed, but the page-scroll bug meant the
control column scrolled away, and the footer went on attributing NSF 24-584 whatever was
selected — a false attribution under a synthetic solicitation, not just a stale string.

Now the footer follows the scenario, and the note says what the choice governs **in the
current mode**: Story uses the vault and the solicitation; Discover and Grade use the
scenario's vault only until documents are uploaded, after which the uploads are the
corpus and the scenario supplies the requirements alone. That last case is where the
picker really did do almost nothing, and it now says so.

## Phase 22 — the browser was running a different build

Both fixes from Phase 21 were reported as not working. They were in the repo, in the
served files, and verified in a headless browser against the same process the operator
was using. The server access log settled it:

```
100.106.26.18 - "GET /?v=1790011416442"     200
100.106.26.18 - "GET /api/config/public"    200
100.106.26.18 - "GET /api/network"          200
100.106.26.18 - "GET /api/samples"          200
```

No `GET /api/scenarios`. No `GET /app.js`. The operator's browser was running an
`app.js` from **before the scenario picker existed** — it fetched `/api/samples` because
that code was in its copy, and never fetched `/api/scenarios` because that code was not.

The Demo Center loads `/?v=<timestamp>`, so index.html was always fresh. `app.js` and
`styles.css` are referenced as bare relative paths and were served with **no cache
headers at all**, so the browser reused them without revalidating. The markup had the
dropdown; the code that fills it did not. Every symptom followed: an inert picker, and
a trace that still called `scrollIntoView` and dragged the viewport down.

From the server the two builds were indistinguishable, which is the part worth fixing:

- The UI is served `no-store` now — a handful of small files over loopback or a LAN,
  where caching buys nothing and costs a whole class of phantom bug.
- `GET /` stamps the asset URLs with a build id (git sha + newest UI mtime), so a copy
  cached before that header existed cannot be reused either.
- `GET /api/build` reports it, and the footer shows the build **this browser is
  running** — turning red with "STALE, reload" when it disagrees with the server.

Three tests, the first verified by turning caching back on and watching it fail.

The lesson is not about caching. Every check ran against files on disk or a browser
started fresh for the test, so none of them could see what the operator's browser was
actually executing — the same shape as the vacuous gates in Phase 20: a check that
passes because it is looking at the wrong thing.

## Phase 25 — Draft mode, the institution's documents, and scenarios built on real calls (2026-09-29 → 2026-10-06)

Two pieces of work, reported from the stage: the UI hid what each institution was working
from, and the six scenarios felt like one scenario with the names changed.

### The UI

**Modes.** Story and Explore were one drafting pipeline with a different solicitation, and
"Explore" next to "Discover" read as the same word twice. The list is now **Draft / Find
Funding / Grade / Replay**. Draft has two choices — the institution's solicitation or an
uploaded one, the institution's documents or uploaded ones — and sends them explicitly
(`use_uploaded_solicitation`, `use_uploaded_documents`), so a file uploaded earlier cannot
override "this institution's" after the presenter switches back. `"draft"` resolves to
`"story"` or `"explore"` server-side, so packets, state and saved replays only ever meet
the two values they knew. Find Funding keeps the API value `discover`.

**The institution's documents.** A Documents panel under the picker lists the scenario
overview, the solicitation, and every document grouped (People, Letters, Budget & rates,
Prior work, Institution, Plans & compliance). "Open brief" lays the same out full-page.
A presenter-notes toggle flags the documents holding planted issues and lists them —
hidden by default so an audience watches the agent find them. Each scenario has a short
overview (`data/scenario_summaries.yaml`), kept out of every vault so the agent can never
cite it.

**The reader.** Documents and citations opened in a 420px drawer: fine for one line,
miserable for a document. It is now a centred reader with a readable measure, and pipe
tables (every budget CSV) render as tables instead of rows of bars.

**A citation bug the panel exposed.** `/api/vault/{doc_id}` only looked in `data/vault/` —
Harrow Creek's — so every citation in a run for any other institution opened "Document
not found". It now searches the session's uploads, the named scenario, then every vault.

**The Find Funding report.** A Find Funding run now ends in a report a research office can
use: per opportunity, why it fits (cited), what of the institution's it would serve
(`aligned_with`, new on `propose_opportunity`), what a person must check, award ceiling,
cost sharing, closing date, and links to Grants.gov and the agency's full announcement.
Printable; reopenable from the Opportunities tab. Found while testing it live:

- grants.gov's `synopsis.agencyName` is sometimes the *contact person* ("Joseph Perez,
  Program Analyst"); the agency now comes from `agencyDetails`.
- Converting "Dec 31, 2026 12:00:00 AM EST" to a Central browser's local time showed the
  deadline as Dec 30. Dates are cut, never converted.
- NIH announcements from 2026 on are not at `/grants/guide/pa-files/<n>.html` (PAR-26-040
  is a 404 there), so a constructed link would have been dead on stage. The fallback is
  the NIH Guide search, which always resolves.
- A later thin search hit overwrote the detail record for the same opportunity, dropping
  the announcement link. `opportunities_seen` now merges.

### The scenarios

Every scenario but Harrow Creek shared the same four planted issues (over-cap budget,
fringe mismatch, stale letter, claim contradicted — 13 instances of the last), the same
**$1,038,450** budget at 28% fringe — `build_budget_rows` hard-coded one budget, so a
clinical trial budgeted "Automation lab equipment" — and the same synthetic $1M call.

Each now applies to a **real, currently posted** solicitation, fetched and verified by
`scripts/fetch_scenario_solicitations.py`:

| Scenario | Call | Budget | Planted (★ deterministic) |
|---|---|---|---|
| Ashgrove | NIH PA-25-305 (R01 CT Required) | $3.71M / 5 yr | ★ salary over NIH cap · IRB lapsed · DSMB not constituted · accrual unsupported · ClinicalTrials.gov results overdue |
| Vantcross | NIH PA-25-301 (R01 CT Not Allowed) | $4.24M / 5 yr | ★ $500K prior approval · ★ subaward F&A over limit · effort 110% · no A1 introduction · papers overstated |
| Calderwick | NSF 26-516 (GEO Core) | $1.03M / 3 yr | ★ equipment below threshold · no Mentoring Plan · campaigns overstated · effort 115% |
| Sablecreek | NSF 25-523 (EPSCoR E-CORE) | $9.64M / 4 yr | ★ committed cost share prohibited · ★ F&A on equipment · letter of support not allowed · campuses uncommitted |
| Tierra Sandoval | NSF 25-514 (S-STEM Track 2) | $1.75M / 6 yr | ★ scholarship over $15K · ★ F&A on scholarships · HSI enrolment overstated · no external evaluator |

Budgets are sized from real comparable awards (`scripts/fetch_anchors.py` → `data/anchors/`,
NIH RePORTER and NSF Award Search; ids and amounts only, no names) and built line by line
by `app/budget_builder.py`. Each call's money rules are curated with quotes and sources in
`solicitation/budget_rules.yaml` and checked by new codes in `app/checks/budget.py`:
`SALARY_OVER_CAP`, `SUBAWARD_FA_OVER_LIMIT`, `EQUIPMENT_BELOW_THRESHOLD`,
`COST_SHARE_PROHIBITED`, `PARTICIPANT_SUPPORT_BELOW_MINIMUM`, `SCHOLARSHIP_OVER_MAX`,
`LARGE_BUDGET_PRIOR_APPROVAL`, and `INDIRECT_ON_EXCLUDED` split into
`INDIRECT_ON_EQUIPMENT` / `INDIRECT_ON_PARTICIPANT_SUPPORT`. Harrow Creek declares no rules
and is checked exactly as before (test-pinned).

**Two rules the primary sources corrected.** The plan said to move Vantcross's subaward
limit to the revised 2 CFR 200 $50,000 — but both NIH NOFOs state that under the FY2026
appropriations act the 45 CFR 75 indirect-cost provisions *still apply to NIH*, so it
stays **$25,000** for NIH and is $50,000 only for the NSF scenarios. And the NIH salary cap
is not in the NOFO; it is the Executive Level II figure on NIH's salary-cap page,
**$228,000** for Jan–Sep 2026.

### The answer key was in the prompt

The worst finding. `generate_vault.py` pasted the **entire seed** into the drafting
prompt — `planted_issues`, the `*_claimed_in_draft` numbers, and every other document's
brief ("a reviewer is supposed to notice this"). The model obligingly wrote the answers
into the documents: one rate agreement noted that "active proposals have incorrectly
utilized a fringe rate of 28.0%". `_facts_for_prompt` now withholds all of that and gives a
budget summary instead of lines whose flags mark the errors; every narrative document in
the four rebuilt scenarios was regenerated.

Seed *facts* leak too, which `must_not_contain` now guards per document:

- Ashgrove's seed recorded the ClinicalTrials.gov results due date; the model wrote
  "Despite the passage of this deadline…". The date is now a comment, and the registry
  status is a bare four-row table.
- Tierra Sandoval's `people.evaluator` named an external evaluator; eight documents then
  described the new programme's evaluation as external — cancelling the planted
  "no external evaluator" issue. The entry is gone; only the prior award's report names
  that firm, as history.

### Smaller fixes

- Generated scenarios dated every letter "today" — stale 2021 letters included. The
  generator now takes letter dates from the seed's collaborators, and the validator checks
  them. The four retired stale letters (Brightmoor, Vanlee, Pellamy, Ocotillo, plus
  Vantcross's Tessaly) are gone from their vaults.
- The model's proposed draft sections for the NIH clinical-trial NOFO included "Table of
  Page Limits" and two form pages. A curated `solicitation/sections.yaml` now wins.
- The budget check reported one finding per budget year — five identical salary-cap cards.
  It merges to one per line, naming the years.
- **Find Funding wrote fetched web pages into shipped vaults.** `session_dir=vault_dir if
  vault_dir != data/vault` meant "a session's uploads", but is also true of every built-in
  institution except Harrow Creek, so pages (one a CDC "Page Not Found") became institution
  documents for every later run. They now go to `runtime/runs/<id>/web/`; the reader still
  finds them. Thirteen strays were moved out to `runtime/stray-web-pages/`.

### Verification

- `validate_vault.py` passes for all six vaults; it now also fails on a budget issue nobody
  planted, a planted deterministic issue that does not fire, and a wrong letter date.
- 298 tests pass, including: every generated scenario applies to a real call with sourced
  rules; no planted issue type in more than two scenarios; each budget yields exactly its
  planted deterministic codes; no scenario carries the old $1,038,450.
- One live Draft (Quick) run per scenario on qwen3.6:35b: every scenario's deterministic
  issues fired, none of the retired shared issues appeared, each drafted its call's real
  sections. Agent-found: Vantcross's program-officer email, Sablecreek's letter of support.
  The document-based issues need a Full run or the scenario's Grade sample (rewritten to
  match).
- A live Find Funding run on Vantcross's documents: 4 real opportunities, each with
  `aligned_with`, ceiling, cost sharing and working links.

### Not done

- Full-scope and Grade runs of the four rebuilt scenarios.
- Harrow Creek still has the original four issues by design: it is the baseline the saved
  replays are built on.
