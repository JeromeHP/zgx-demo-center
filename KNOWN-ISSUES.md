# Known Issues — HP ZGX Demo Center

A standing register of bugs, gaps and gotchas that are **real but not currently being
worked**. It exists so a problem found once does not have to be rediscovered from
scratch the next time something breaks on stage.

## How to use this file

- One `###` entry per issue under **Open**, newest first.
- **Delete the entry when it is fixed.** Git history keeps it; a growing "Resolved"
  section makes this file useless within a month.
- Anything actively being worked belongs in a plan or a sprint, not here. Move it out
  when work starts, and delete it when work lands.
- Every entry needs a **Symptom**. An issue nobody can recognise cannot be triaged, and
  the point of this file is that the next person matches what they are seeing against
  this list.
- `Cause: unknown` is a legitimate entry. Recording an unexplained symptom is worth more
  than recording nothing.

Entry template:

```markdown
### Short title
- **Severity:** blocker | degrades-demo | cosmetic | latent
- **Area:** launcher | install | env | docs | <demo name>
- **Found:** YYYY-MM-DD
- **Symptom:** what an operator actually sees.
- **Cause:** the mechanism, with `file:line` if known. "Unknown" is valid.
- **Workaround:** what to do on demo day, or "none".
- **Fix:** the intended change, or "not scheduled".
```

Severities: **blocker** — a demo cannot run. **degrades-demo** — it runs but embarrasses
you. **cosmetic** — wrong text, right behaviour. **latent** — harmless today, will bite
after some future change.

---

## Open

### Vehicle Inspection's open photo set has no minor damage, and few damaged brands
- **Severity:** degrades-demo
- **Area:** vehicle
- **Found:** 2026-10-07
- **Symptom:** in a live run the **Needs inspection** lane stays empty, and the **Hold** lane
  only ever shows older Fiat, Abarth, Opel, Citroën, Peugeot and Ram vehicles. No damaged
  Jeep, Dodge, Chrysler, Alfa Romeo, Maserati, DS, Lancia or Vauxhall appears.
- **Cause:** every image must be openly licensed and publicly available, and each vehicle
  needs 2–4 views of the same physical car. Only 11 damaged vehicles met that bar, and they are
  mostly 1970s–2000s cars; damaged Chargers and Challengers exist only as single photos. A
  Jeep CJ set was dropped because it auto-cleared on both passes: rust and missing parts, not
  visible collision damage. Openly licensed photos of *minor* damage (scuffs, door dings) on a
  verifiable Stellantis car were not found at all. With the multi-view prompt, 46 of 46 clean
  vehicles auto-cleared, so nothing currently lands in Needs inspection.
- **Workaround:** the inspector story runs on the Hold lane, where every vehicle needs a human
  decision. Say plainly that the photo set is public-domain and Creative Commons material.
- **Fix:** not scheduled. Openly licensed multi-angle sets of minor and modern damage would fill
  both gaps; add them with `vehicles.json` + `CREDITS.md`, then re-record
  (`scripts/record_vessel_replay.py`).

### Vehicle Inspection's Spot Check still calls some clean cars MINOR
- **Severity:** cosmetic
- **Area:** vehicle
- **Found:** 2026-10-06 (reduced 2026-10-07)
- **Symptom:** a single photo of a clean car comes back MINOR with near-identical boilerplate:
  "light scratches and scuffs on the hood and front bumper, approximately 2–3 inches". The
  model's own first-look line ("What it saw") claims the same scratches.
- **Cause:** a real misperception on one photo, not a self-contradiction. Spot Check and the
  lane's detailed report now use the lane's grounding (undamaged is the normal case, and a
  PHOTO line before the verdict). Measured on the 57 primary photos: clean cars reading NONE
  went from **27/46 to 41/46**; false flags at MODERATE or MAJOR went from 6 to **0**; damaged
  cars stayed 11/11 MAJOR+; time per photo was unchanged (12.6 s). The remaining 5 are all
  MINOR. The multi-view lane triage clears all 46.
- **Workaround:** demo from the lane. In Spot Check, a MINOR on a clean car routes to the
  touch-up bay, not a hold.
- **Fix:** not scheduled. Candidates: a second view of the same car (the lane gets these
  right), or a rubric line that MINOR needs a specific, visible mark.

### Running a native demo's start script before the Demo Center creates `~/demos` breaks the stack
- **Severity:** latent
- **Area:** install
- **Found:** 2026-09-18
- **Symptom:** `run_demos.sh` refuses to start with "`~/demos` is an existing directory, not a
  link to this checkout", and neither 8085 nor 8086 comes up. Happens on a box where the Demo
  Center stack has never been started but a demo was launched by hand.
- **Cause:** every native start script opens with `mkdir -p "$HOME/demos/logs"`
  (`ew-spectrum-duel/start_demo_8089.sh:7`, and the same two lines in pipeline, vision and
  grant). The `~/demos` symlink is created by `ensure_demos_link.sh`, which only runs from
  `run_demos.sh` / `start_demo_center.sh`. Run a start script first and `mkdir -p` silently
  creates `~/demos` as a **real directory**; `ensure_demos_link.sh` then correctly refuses to
  replace it, and the whole stack is blocked by a directory containing one log file.
- **Workaround:** the error message prints the exact fix. Preserve any logs first:
  `mv ~/demos/logs/*.log ~/zgx-demo-center/demos/logs/ && rm -rf ~/demos && ln -s ~/zgx-demo-center/demos ~/demos`
- **Fix:** have the start scripts resolve `LOG_DIR` from their own location
  (`$(cd "$(dirname "$0")/.." && pwd)/logs`) rather than through `$HOME/demos`, or have them
  fail loudly when `~/demos` is not a symlink instead of creating it.

### `endpoint_check.py` E3 passes for a server that returns no reasoning at all
- **Severity:** latent
- **Area:** grant
- **Found:** 2026-09-18
- **Symptom:** `scripts/endpoint_check.py` reports E3 "Thinking separation" PASS and an overall
  PASS, but the demo's thinking panel stays empty for an entire run.
- **Cause:** `grant-machine/scripts/endpoint_check.py` `check_e3` asserts only that `<think>`
  tags do not leak into `content`. A server that emits no reasoning whatsoever satisfies that
  trivially, so the check cannot distinguish "reasoning correctly separated" from "reasoning
  silently absent". Hit for real when first pointing the demo at Ollama, which names the field
  `reasoning` rather than `reasoning_content` — E3 passed while the app saw nothing. The signal
  is in the report JSON: `checks.E3.samples[].meta.reasoning_source`, which is `field` when
  reasoning was captured and `none` when it was lost.
- **Workaround:** read `reasoning_source` in the JSON report, not just PASS/FAIL. The app side
  of this specific bug is fixed (it now reads either field name); the weak assertion is not.
- **Fix:** make `check_e3` require `result.reasoning` to be non-empty and
  `reasoning_source != "none"`, not just the absence of tags.


### Running the pharmacy demo modifies a tracked database file
- **Severity:** degrades-demo
- **Area:** pharmacy
- **Found:** 2026-09-04
- **Symptom:** after launching the drug-interaction demo, `git status` shows
  `demos/healthcare-pharmacy-containerized-docker/ddi_chromadb/chroma.sqlite3` modified,
  and `git pull` can then refuse to fast-forward.
- **Cause:** `docker-compose.yml` mounts `./ddi_chromadb` read-write (`- ./ddi_chromadb:/app/ddi_chromadb`)
  and ChromaDB writes to `chroma.sqlite3` on open. The file is committed to the repo, so
  every launch dirties the working tree. Contrast the `./backend` mount right above it,
  which is correctly `:ro`.
- **Workaround:** `git checkout -- demos/healthcare-pharmacy-containerized-docker/ddi_chromadb/`
  after running the demo.
- **Fix:** either mount it `:ro` (if ChromaDB tolerates a read-only store here) or move
  the database out of the repo and rebuild it at install time with `build_ddi_db.py`.

### A native demo's launch stream never ends
- **Severity:** latent
- **Area:** launcher
- **Found:** 2026-09-04
- **Symptom:** none visible. `GET /start/<native demo>` streams `⏳ still working —
  Nm0Ns elapsed (no output yet)` heartbeats forever and never emits `complete`, even
  though the demo started fine seconds earlier.
- **Cause:** `zgx_demo_lib.py` `stream_command()` reads the child's stdout until EOF.
  The native launch scripts (`start_demo_8088.sh`, `_8089.sh`, `_8094.sh`, `_8095.sh`) background a
  `nohup`'d process, and although that process redirects its own stdout/stderr to a log
  file, the pipe's write end stays open long enough that the read never reaches EOF —
  so the loop falls through to the heartbeat branch indefinitely. Containerised demos
  are unaffected (`docker compose up -d` exits cleanly).
- **Workaround:** none needed. Readiness comes from the client's health poll, not from
  `complete`, and the launcher now closes the EventSource itself once a demo is ready,
  fails, or times out — so nothing leaks on the client. The `error` path is unaffected:
  a launch script that exits non-zero before backgrounding anything still reports in
  about two seconds.
- **Fix:** race `proc.stdout.readline()` against `proc.wait()` in `stream_command()` and
  stop once the launch command itself has exited and a short drain window passes. Not
  done yet because that read loop is shared with `/install`, where long silent Docker
  builds depend on exactly the current heartbeat behaviour — it needs its own test pass.

### `~/miniforge3/envs/zgx` is a symlink to `dpv`, not a real env
- **Severity:** latent
- **Area:** env
- **Found:** 2026-09-04
- **Symptom:** `conda env list` shows both `dpv` and `zgx` pointing at the same prefix.
  Nothing misbehaves.
- **Cause:** On 2026-08-18 someone ran `conda rename -n zgx dpv`
  (see `~/miniforge3/envs/dpv/conda-meta/history`), which broke every native demo at
  once. The repair was `ln -s dpv ~/miniforge3/envs/zgx` — chosen over renaming back
  because `conda rename` is a clone-and-remove that copies several GB of pip-installed
  torch/tensorrt/pycuda and can break their recorded paths.
- **Workaround:** none needed; the symlink is functionally complete.
- **Fix:** `conda rename -n dpv zgx` once there is time to re-verify that
  `import cv2, tensorrt, pycuda.driver, torch` still work afterwards.

### The port map is maintained by hand in three places
- **Severity:** latent
- **Area:** launcher
- **Symptom:** none today — all three copies agree. (They did not until 2026-10-06: the
  `DEMOS` entries for construction and translation still said `port:8080`, so their
  Details buttons read "Launch — :8080" while opening 8091/8092.)
- **Found:** 2026-09-04
- **Cause:** `zgx-demo-launcher.html` holds `PORT_CONFLICTS` / `PORT_PATHS` /
  `HEALTH_CHECK_PATHS` near the top and a second copy of port+path per card in the
  `DEMOS` array further down; `install_plans.py` holds a third, and `RUNBOOK.md`'s table a
  fourth. Changing a port in one place silently breaks either readiness polling or the URL
  the Launch button opens.
- **Workaround:** `tests/test_port_map.py` now fails when any copy disagrees with
  `install_plans.py`, when two demos claim a port outside the declared :8000 group, or when
  a demo lands on 8085/8086/8096/11434. Run it after touching any port. Grep the whole repo
  for the old number as well — prose and comments are not covered.
- **Fix:** serve one map from the install API and have the launcher read it at load.

### `/running` only probes a demo's primary port
- **Severity:** latent
- **Area:** launcher
- **Found:** 2026-09-04
- **Symptom:** a demo whose front end is down but whose back end is still up reports as
  stopped.
- **Cause:** `demo_install_server.py` `/running` resolves `discovery.type: port_probe`
  against `plan["port"]` alone. Teardown now also frees `extra_ports`
  (8083 for pii, 8191/8192 for construction/translation), but discovery does not read
  that key, so its view of "running" is still single-port.
- **Workaround:** none needed — teardown is the path that mattered, and it is fixed.
- **Fix:** have `port_probe` discovery consider `extra_ports` as well.

### Launching construction or translation rewrites tracked source files
- **Severity:** degrades-demo
- **Area:** construction, translation
- **Found:** 2026-09-04
- **Symptom:** `git status` can go dirty simply because a demo was launched; after a
  `git pull` a demo may come up on the wrong port.
- **Cause:** `AEC-construction-safety/start_demo_8091.sh:20-24` and
  `document-translation/start_demo_8092.sh:20-24` `sed -i` the back-end port into
  `backend/main.py` and the API URL into `frontend/index.html` on every run; translation
  rewrites again from its exit trap. Both are no-ops today only because the files are
  already in their patched state — a pull that restores an original value re-arms them.
- **Workaround:** `git diff` after launching if you are about to commit.
- **Fix:** read `BACKEND_PORT` from the environment in both back ends and stop patching
  source at launch time.

### AEC ships three start scripts that disagree on ports
- **Severity:** latent
- **Area:** construction
- **Found:** 2026-09-04
- **Symptom:** running the wrong script starts the demo where the launcher cannot find
  it, or against a machine that is not this one.
- **Cause:** `start_demo.sh` targets 8000; `start_demo_remote.sh:10-11,53` targets
  8000/8080 with a hardcoded fallback IP `192.168.10.117`; only `start_demo_8091.sh`
  matches the 8091/8191 scheme the launcher and `install_plans.py` expect.
- **Workaround:** only ever use `start_demo_8091.sh`, which is what the launcher calls.
- **Fix:** delete the other two.

### The robot demo cannot install as documented
- **Severity:** blocker (for that demo, which is disabled pending hardware)
- **Area:** medical
- **Found:** 2026-09-04
- **Symptom:** installing `medical` appears to succeed, then `start_services.sh` aborts
  with "Models directory not found".
- **Cause:** `install_plans.py:35` invokes `scripts/download_models.sh`, which does not
  exist in the checkout, with stderr suppressed — so it silently falls through to a
  Python fallback that fetches only the Llama model, never the Piper TTS voice that
  `docker/Dockerfile.api:78` expects. Separately
  `Robotic-medical-AI-assistant/scripts/start_services.sh:108-116` requires a `models/`
  directory that is absent.
- **Workaround:** none. The demo is gated on a Reachy Mini being present anyway.
- **Fix:** add the missing `download_models.sh`, drop the `2>/dev/null`, and let the step
  fail loudly.

### Pipeline Pulse hides its own warm-up failure
- **Severity:** degrades-demo
- **Area:** pipeline
- **Found:** 2026-09-04
- **Symptom:** the demo starts normally, then the first question in front of an audience
  takes tens of seconds instead of being instant.
- **Cause:** `pipeline-pulse/pipeline_pulse_v6.py:205-217` blocks FastAPI startup on a
  90 s Ollama warm-up wrapped in a bare `except: pass`. If the warm-up times out or
  errors, the server still binds :8088 and reports healthy, but `llama3.1:70b` is cold.
- **Workaround:** ask one throwaway question before the audience is watching.
- **Fix:** record the warm-up result and expose it in `/api/status` so readiness reflects
  it.

### The iframe `postMessage` bridge does not check origin
- **Severity:** latent
- **Area:** launcher
- **Found:** 2026-09-04
- **Symptom:** none.
- **Cause:** `zgx-demo-launcher.html` accepts a `demo-shell-return` message from any
  framed origin without verifying `event.origin`. Recorded so it reads as an accepted
  trade-off on a closed, single-operator box rather than an oversight.
- **Workaround:** n/a.
- **Fix:** check `event.origin` against the launcher's own host before acting.

### Two install strings describe software that is not there
- **Severity:** cosmetic
- **Area:** install
- **Found:** 2026-09-04
- **Symptom:** install-time estimates and GPU expectations are wrong, which misleads
  anyone planning a demo around them.
- **Cause:** `install_plans.py:58` labels the pharmacy build "~15 min — compiles
  llama.cpp CUDA sm_120", but `healthcare-pharmacy-containerized-docker/Dockerfile`
  builds no llama.cpp at all (it is a thin `python:3.12-slim` Ollama client).
  `document-translation/install.sh` advertises a 7B Seed-X model (~14 GB) while
  `document-translation/backend/main.py:4,20` loads
  `facebook/nllb-200-distilled-600M` (~1.2 GB).
- **Workaround:** trust the Dockerfile and the backend, not the labels.
- **Fix:** correct both strings.

### The Chromium kiosk path cannot work on this box
- **Severity:** degrades-demo
- **Area:** launcher
- **Found:** 2026-09-04
- **Symptom:** `start_demo_center.sh` prints "done" and exits 0, but no window ever
  appears. `logs/demo_center_launch.log` shows
  `Missing X server or $DISPLAY` (recorded 2026-08-18 and 2026-09-03).
- **Cause:** the ZGX Nano is run headless over SSH/tailnet; `start_demo_center.sh`
  unconditionally launches Chromium in app mode and only warns that the window "did not
  map within 15s".
- **Workaround:** use `run_demos.sh` and open the launcher from another device — this is
  what `RUNBOOK.md` already prescribes.
- **Fix:** detect a missing `$DISPLAY` up front and say so, or drop the wrapper on this
  device.
