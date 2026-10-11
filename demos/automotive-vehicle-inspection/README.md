# Port-of-Entry Vehicle Inspection Analyst

VLM-powered damage triage for finished vehicles discharged from RoRo carriers at a port of
entry. A simulated vessel drives its vehicles through a **multi-camera portal**. The device
triages each one automatically from all of its camera views, and only the ones that need a
person reach an inspector.

Demo Center id **`vehicle`** · port **8097** · Mistral Small 3.2 (24B, vision) via host
Ollama, registered as `vehicle-inspector`. Branded for **Stellantis**.

> **Compliance by Architecture:** imagery, inference and inspector output stay on the
> device. Zero cloud dependency.

## The inspection lane (default tab)

Press **Start discharge** and 20 vehicles roll through the portal. Each is photographed by
**2–4 portal cameras**, and every view goes to the model in **one call**. The model writes one
line per camera, then a verdict, and the vehicle is routed:

| Triage severity | Lane | Who acts |
|---|---|---|
| NONE | **Cleared**: released automatically | nobody; an inspector may still open any card |
| MINOR, MODERATE | **Needs inspection** | an inspector confirms |
| MAJOR, CRITICAL | **Hold** | an inspector decides; goes into the exception report |
| unreadable answer | **Needs inspection** ("model output incomplete") | an inspector |

**Only an explicit NONE ever auto-clears** (`inspection.route`, unit-tested). An answer that
contradicts itself ("MINOR" with "No exterior damage observed") goes to a person, and the
reason says so.

**The portal** shows the vehicle's cameras as a grid ("FRONT-LEFT CAM", "REAR-RIGHT CAM"…).
They "capture" in sequence as the car arrives. Once triaged, each tile shows that camera's
one-line finding, and the cameras the model says show damage (`DAMAGE_VIEWS`) are outlined in
red.

Click any card to open the **review drawer**:
- a strip of every camera view; it opens on the first one that saw damage;
- that camera's finding and the photo's own credit;
- the AI triage, including which cameras saw the damage;
- **Generate detailed report**: the full eight-field report over all views, streamed live;
- five decision buttons (the AI's suggestion is outlined) and a note.

A decision is recorded next to the AI's call, and the stats strip counts agreements and
overrides. **Inspection report (CSV)** lists every vehicle with:
- the simulated VIN, consignment (new/used/classic) and cameras;
- the AI severity and disposition, and which cameras saw damage;
- the lane and reason;
- the human decision and note;
- per-view photo credits.

The run lives **server-side** (`backend/lane.py`), so a page reload, or the launcher closing
and reopening the iframe, picks it up again from `GET /api/lane/state`.

### Two-tier inference: why a live run fits in under three minutes

On the GB10 the cost is output tokens, not images. One call with four views measured 5.7 s at
768 px, against ~4.5 s for one view. So the lane asks for a short answer: one line per camera,
then severity, panels, one finding, `DAMAGE_VIEWS` and disposition. The full eight-field report
(130–200 tokens, 9–16 s) runs only when an inspector asks for it. Every prompt embeds the same
`RUBRIC` constant (`backend/inspection.py`), so the tiers and the "do not downgrade" rules
cannot drift apart.

| Measured on the GB10, 2026-10-07 | |
|---|---|
| Triage per vehicle, all views in one call | mean 7.5 s (5.0–13.0 s), 57 vehicles × 2 passes |
| Live run, 20 vehicles (43–47 camera views) | 2:37, 2:51 and 2:42 over three runs |
| Live run of 24 (the earlier size) | would be ~3:15, over the 3-minute limit, so the live size is 20 |
| Sustained throughput shown | ~500 vehicles/hour from one device, every camera inspected |
| Parallel requests | no help: 4 concurrent took 43.7 s vs 48.3 s sequential, because Ollama serves one at a time |

### Replay

**Replay** plays the full 57-vehicle vessel recorded on this device (`data/replays/`) at 3×,
under a permanent **"REPLAY — recorded … Not live"** banner.
- It makes **no model calls**, so the chrome's token and GPU meters honestly stay idle.
- The stats show the **recorded** model time per vehicle, not the accelerated playback.
- Routing still runs through the same code as live mode.
- Detailed reports and decisions still work. A detailed report in replay is live inference,
  and is labelled as such.

**Re-record whenever the triage prompt, the model or a photo changes**:

```bash
cd ~/demos/automotive-vehicle-inspection          # stop any live run first
~/miniforge3/envs/zgx/bin/python3 scripts/record_vessel_replay.py   # ~15 min
```

It runs the whole vessel twice, sending each vehicle's views exactly as the live lane does, and
writes:
- the replay, from pass 1, with a header recording the device, model, prompt hash and a hash
  of every view;
- each vehicle's `observed_lane` in `vehicles.json`: the lane it took if both passes agree,
  `unstable` if not.

`observed_lane` only balances a live run's mix (`lane.draw_live`). It is never shown and never
routes anything. `tests/test_vehicle_lane.py` fails if:
- the recording no longer matches the prompt, the model or any photo; or
- a damaged vehicle auto-cleared in it.

**Recorded 2026-10-07:**
- **46 of 46 clean vehicles cleared automatically and 11 of 11 damaged vehicles were held**,
  with 0 unstable.
- One damaged set, a rusted, stripped Jeep CJ, cleared on both passes. It was removed, because
  its photos show a derelict rather than visible collision damage.
- Nothing currently lands in Needs inspection. See `KNOWN-ISSUES.md`.

## Spot Check (second tab)

The original single-photo flow: pick a sample (each vehicle's primary view) or upload a
photo, and get the full report. It uses the lane's grounding: the prompt says undamaged is
the normal case, and the model writes what it sees (`PHOTO_1`, shown as "What it saw")
before it grades. Measured on the 57 primary photos:

| Spot Check | before grounding | grounded |
|---|---|---|
| Clean cars reading NONE | 27 / 46 | **41 / 46** |
| False flags at MODERATE or MAJOR | 6 | **0** |
| Damaged cars graded MAJOR+ | 11 / 11 | 11 / 11 |
| Time per photo | 12.5 s | 12.6 s |

The remaining 5 false flags are MINOR (`KNOWN-ISSUES.md`). The lane's detailed report is
grounded the same way, with one line per camera.

## Model

`vehicle-inspector` is `mistral-small3.2` with an 8k context (`Modelfile.vehicle`, created at
install). Ollama otherwise allocates the model's full 131k context: **36.9 GB resident**
against **15.8 GB** for this, the same weights. `MODEL_LABEL` keeps "Mistral Small 3.2 (24B)"
on screen.

Rules this demo depends on, learned the hard way:

- **A non-thinking model only.** Ollama's `qwen3-vl:8b` is the thinking variant, and on
  Ollama 0.34 no flag stopped it (`reasoning_effort: "none"`, `think: false` on `/v1` or
  `/api/chat`). The reasoning ate the token budget and every report came back empty.
- **Temperature 0** (`INFERENCE_TEMPERATURE`). Upstream used 0.3, and the same showroom-clean
  photo came back NONE on one run and MAJOR on the next.
- **Ground the multi-view prompt, or it recites the rubric.** The first multi-view draft
  said "damage visible in any one photo counts" and showed an example `DAMAGE_VIEWS` of
  "1, 3". It graded **all 12 clean probe vehicles MAJOR, every one "cracked headlight lens",
  photo 1**: the rubric's most emphasised example, recited rather than seen. The model could
  see the photos (asked plainly, it described each one correctly). The fix was neutral wording
  ("most vehicles arrive with no damage…") plus one `PHOTO_n` line per camera before the
  verdict: 8/8 clean NONE and 6/6 damaged MAJOR+ on the probe, then 46/46 and 11/11 on the full
  vessel. `tests/` pins both parts.
- **The same grounding fixed Spot Check:** clean cars reading NONE went from 27/46 to 41/46
  on single photos, with no MODERATE or MAJOR false flags left. See "Spot Check" above.
- **Borderline photos are sensitive to re-encoding.** Re-saving a photo at 1024 px moved one
  clean Ram 1500 from NONE to MINOR in the full report. Re-record the replay after touching
  any photo.

Comparison that chose the model, 31 photos inspected twice with the full report:

| | `qwen3-vl:8b-instruct` | `mistral-small3.2` |
|---|---|---|
| Verdicts that changed between identical runs | 2 of 31 | **0 of 31** |
| Damage photos graded MAJOR or CRITICAL | 17 of 17 | 17 of 17 |
| Time per full report | ~6 s | ~14 s |

## Run

From the launcher: **Install** (pull, register `vehicle-inspector`, build), then **Load Demo**.
By hand:

```bash
ollama pull mistral-small3.2
cd ~/demos/automotive-vehicle-inspection
ollama create vehicle-inspector -f Modelfile.vehicle
docker compose build
./start.sh -d                  # http://<host>:8097
docker compose down            # stop
```

`frontend/`, `sample-images/` and `data/` are bind-mounted read-only, so page edits, photos
and a new recording take effect without a rebuild. Backend changes need
`docker compose up -d --build`.

## Code

| File | What |
|---|---|
| `backend/inspection.py` | rubric, single-photo and multi-view prompts, parsers (incl. `DAMAGE_VIEWS`, per-photo findings), routing, image prep (pure) |
| `backend/lane.py` | vehicle manifest, live draw, run state, worker, replay, decisions, CSV (inference injected) |
| `backend/vin.py` | simulated VINs: real Stellantis WMIs, valid check digit, labelled SIMULATED |
| `backend/main.py` | FastAPI app, multi-image streaming model calls, routes |
| `scripts/record_vessel_replay.py` | records the replay and refreshes `observed_lane` |

## Endpoints

- `GET  /api/health`: `service`, `status` (`healthy` only when Ollama answers **and** the
  model is registered), `model` (label) and `ollama_model`.
- `GET  /api/metrics`: session token totals for the shared chrome.
- `GET  /api/lane/state`: full snapshot.
- `GET  /api/lane/events`: server-sent events: `run_started`, `vehicle_scanning`, `token`,
  `vehicle_routed`, `decision`, `full_token`, `full_report`, `run_finished`, `run_stopped`
  and `error`.
- `POST /api/lane/start` `{port, vessel, mode: live|replay}` · `POST /api/lane/stop`.
- `POST /api/lane/{vid}/decision` `{action, note}` · `POST /api/lane/{vid}/full_report`.
- `GET  /api/lane/report.csv`.
- `POST /api/inspect` and `POST /api/inspect_sample`: Spot Check.

## Samples: the vessel

`sample-images/` holds **57 vehicles and 125 camera views, all Stellantis brands**, one folder
per vehicle (`<id>/<angle>.jpg`, 2–4 views of the **same physical car**). The brands are
Abarth, Alfa Romeo, Chrysler, Citroën, Dodge, DS, Fiat, Jeep, Lancia, Maserati, Opel, Peugeot,
Ram and Vauxhall. `lane.STELLANTIS_BRANDS` is the allowlist, and a test rejects anything else.

- **46 clean vehicles**, mostly current models:
  - Dodge Charger 2019/2021, Charger Daytona 2024, Challenger 2018 and 2020 Widebody,
    Hornet, Durango; Ram 1500 Rebel and Tradesman; Pacifica.
  - Wrangler, Gladiator, Avenger, Grand Cherokee and Grand Cherokee L.
  - Peugeot 208, 2008, 408, 3008; Citroën C3, Berlingo, C5 Aircross; Opel and Vauxhall
    Corsa, Mokka, Grandland, Astra.
  - Giulia and Stelvio Quadrifoglio, Junior; MC20, Grecale, Levante; DS 7; Ypsilon; 500e,
    600e, Grande Panda.
  - **Classics:** 1968 and 1969 Charger R/T, 1969 Alfa Spider and GT Junior, 1986 Peugeot 205.
- **11 damaged vehicles across 6 brands:**
  - Fiat 500 (crash-tested), Fiat Uno, Abarth 1000 TC
  - Citroën C3 and Saxo
  - Peugeot 205, 205 GTi, 206 rollover, and a Peugeot hatchback (front close-ups)
  - Ram 1500 under a fallen tree, and a race-prepared Opel Kadett

  No openly licensed multi-angle set of a damaged Jeep, Dodge, Chrysler, Alfa, Maserati, DS,
  Lancia or Vauxhall was found. Damaged Chargers and Challengers exist only as single photos.

**The rule for every image: openly licensed and publicly available on the web.**
- **Allowed licences:** CC0, public domain, CC BY or CC BY-SA. Never NC (non-commercial) or ND
  (no derivatives; resizing is a derivative).
- **Public source:** each view links to its page on Wikimedia Commons or Flickr (found via
  Openverse). Tests enforce the licence allowlist and an `http(s)` source on every view.
- **Attribution:** CC BY and BY-SA require it. The UI shows each view's author, licence and
  source, and `sample-images/CREDITS.md` lists them all.
- **Resized** to 1024 px; the lane sends 768 px to the model.

How sets were curated, so new ones can be added the same way:
1. Find same-session frames: neighbouring file numbers, Front/Rear name pairs, and the same
   author and date.
2. Check by eye that every view is the **same car** (paint, wheels, plate, trim), and confirm
   the make and model. Titles were wrong several times: an "Alfa Tonale" sold as a Dodge
   Hornet, a Fiat 600 tagged Abarth, and "sets" that mixed two cars.
3. Label the angles: if the front is on the left of the image it is `front_left`; if the rear
   is on the left it is `rear_right`.
4. **Excluded on principle:** fatal or injury scenes, war or riot damage, and burned-out cars.
5. Then re-record the replay.

## Swapping the OEM

1. Put the customer's approved mark at `frontend/oem-logo.png`. The page hides the slot when
   the file is absent; it never shows a placeholder.
2. Change the `alt="Stellantis"` text on that `<img>` in `frontend/index.html`.
3. Replace the vehicles in `sample-images/` (same folder layout), rewrite `vehicles.json` and
   `CREDITS.md`, add the brands' WMIs to `backend/vin.py`, and re-record the replay.

## Origin

Vendored from [curtburk/automotive-vehicle-inspection-demo](https://github.com/curtburk/automotive-vehicle-inspection-demo)
@ `61f7c463679a28be084b164d2e786aaf6bbe5847` (Curtis Burkhalter). The Demo Center's rules
take precedence over upstream's wherever they disagree:

| Upstream | Here | Why |
|---|---|---|
| vLLM in an NGC container, FP8 checkpoint | Thin `python:3.12-slim` client over host Ollama | NGC vLLM is NVIDIA-only, and a second vLLM cannot share the GPU with Maritime's |
| Host port 8095, container port 8000, internal vLLM on 8090 | **8097**, bound directly under `network_mode: host` | 8095 is Grant Machine, 8000 the shared Maritime/Pharmacy/PII port, 8090 Medical. Published ports would also bypass ufw (`tools/lockdown-tailnet.sh`) |
| `start.sh` prompts if :8091 is busy | No prompt | 8091 is Construction here, and a prompt would hang the launcher's SSE stream |
| Own header with HP chip, Google Fonts, "Powered by" a named device | Shared ZGX chrome, system fonts, device label from `/sysinfo` | `demos/ui/README.md`, air-gap, `scripts/audit_hardware_strings.sh` |
| One photo at a time, 4 Mercedes samples | Multi-camera vessel lane with human review; 57 Stellantis vehicles, 125 open-licence views | Customer demo |
| `restart: unless-stopped`, bare `curl -f` healthcheck | `restart: "no"`, healthcheck greps `vehicle-inspection` | A crash loop hides the error; a bare probe can be answered by another service |

**The port map lives in [`RUNBOOK.md`](../../RUNBOOK.md), not here.** `tests/test_port_map.py`
fails if the launcher, `install_plans.py` and the RUNBOOK disagree.

## Prompt notes (from upstream's `tasks/lessons.md`)

`RUBRIC` in `backend/inspection.py` carries three load-bearing rules:

1. **Each severity tier lists explicit examples**, including "ANY cracked or broken lighting
   assembly" under MAJOR. Without them, the model called out a cracked headlight and still
   graded it MODERATE.
2. **A `SEVERITY ASSIGNMENT RULES` block makes the rubric a hard floor:** a finding that
   matches tier X means severity is at least X, and a single severe finding outranks many
   minor ones.
3. **CRITICAL uses photo-visible anchors** (crumpled sheet metal, a deformed bumper, more than
   25% of the area, 3+ panels from one impact) and causal language ("consistent with impact").

`INSPECTOR_SUMMARY` forbids boilerplate negative findings ("no broken glass observed"). The
output is **labeled plain-text lines, not JSON**. That is more reliable for VLM output, and
`parse_vlm_output` tolerates drift in casing, order and markdown emphasis.

## License

Apache 2.0 (upstream).
