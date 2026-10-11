# Spectrum Duel — AI Electronic Warfare

**HP ZGX Nano (GB10) · native, no Docker · port 8089**

A synthetic X-band engagement in which two AI agents fight over the spectrum. A
**jammer** picks its technique and target frequency; a **TensorRT FP16 CNN**
classifies that technique off the live waterfall; a **defender** picks its
countermeasure from the classifier's verdict alone. `llama3.1:70b` writes the EW
officer's brief over the top.

It runs autonomously so it stays alive on an unattended table, and a rep can
escalate it live with the operator buttons.

## What you are looking at

The screen leads with **the loop**, not with the spectrum:

```
① ATTACK  →  ② SENSE  →  ③ DECIDE  →  ④ RESULT
what the      the 64×64    the counter   did the
jammer did    tile + the   chosen from   datalink
              verdict      the verdict   survive
```

Each stage lights as it fires, and a plain-language strip underneath narrates
what just happened ("Jammer up — a repeater copying our signal at 9.5 GHz" →
"Network read it as a repeater in 0.28 ms" → "Changing our waveform so the copy
stops matching — correct counter"). Nobody needs to be told how to read it.

The waterfall sits below that, and it **labels its own features** — your link in
cyan, the threat in red, a DRFM replica in orange, with a caption naming what
each one is. It earns its place as the *evidence* that the classification is
real rather than a random label generator, but it is no longer the thing you
have to decode before the demo means anything.

The thumbnail in stage ② is the literal 64×64 tile handed to the network,
shipped from the server — not a downscale of the display, which would drop a
3-bin spot tone and make the claim false.

**The classifier reads two channels, not one.** Channel 0 is the spectrum.
Channel 1 is *our own transmit plan* — which bins we are occupying, row by row,
drawn in cyan over the thumbnail. This is not a convenience: a spot tone that
tracks the victim sits on the victim's own carrier, and after the 4:1 decimation
the two share a tile bin. From power alone the network called 41% of tracking
spot tones "CLEAR" and held FIXED, leaving the tone sitting on us. A real EW
suite has never had that problem — it knows its own emissions — and giving the
network the same knowledge took tracking-spot recall from 59% to 87%.

**The most valuable panel is "Last Misread."** When the network names a threat
wrongly, the counter it picks is wrong and the link drops. That moment is frozen
— the tile it got wrong, what it said, what it should have said, and what the
mistake cost in seconds and Mbps — instead of scrolling past in three seconds.

## Why it needs a Nano

Three AI workloads run concurrently on one disconnected box:

| Workload | Where | Cost |
|---|---|---|
| 5-class jam classifier | TensorRT FP16 on GB10 | ~0.29 ms / tile, host to host |
| Jammer + defender policies | numpy, CPU | microseconds |
| EW officer brief | llama3.1:70b via Ollama | ~42 GB resident |

Plus live GPU telemetry from `nvidia-smi` / `pynvml`.

## What is real and what is simulated

The demo is explicit about this — turn on **Annotation Mode** in the header and
hover anything; every tooltip carries a green `REAL — measured on this device`
or amber `SIMULATED — synthetic RF` badge.

- **Simulated**: the RF physics, the emitters, the engagement geometry. All
  numpy, all on CPU.
- **Real**: the waveform classification (a genuine CNN, genuinely on the GPU),
  the LLM briefs, the inference latency, and the GPU telemetry.

## The classifier is load-bearing

The defender **never sees ground truth** — it reads only the CNN's output. Each
jam class has exactly one cheap right answer:

| Threat | Counter | Why |
|---|---|---|
| `BARRAGE` | Burn-through | Brute force answers brute force; no waveform trick helps |
| `SPOT` | Frequency agility | Hop out from under the tone faster than it can re-aim |
| `SWEPT` | Adaptive null / blanking | Blank each pass before it saturates the front end |
| `DRFM` | Waveform change | A coherent replica can only be beaten by changing shape |
| `CLEAR` | Hold | Nothing to counter — don't burn agility for free |

So a misclassification produces a genuinely wrong countermeasure and the link
pays for it, live and visibly. That is the most instructive moment in the demo,
and it happens on its own every few engagements — almost always in the ~3 s
after the jammer switches, while the classifier's 64-row tile still straddles
two techniques. The Last Misread card says which of the two it was, so nobody
has to guess whether the model had an excuse.

That table is not asserted, it is **derived from the physics**. Re-derive it:

```bash
$HOME/miniforge3/envs/zgx/bin/python3 tools/counter_matrix.py
```

It measures link uptime for every (technique × countermeasure) pair and fails if
any disagrees with `COUNTER_FOR` in `ew_engine.py`. If you change the RF model,
run this, then retrain.

## The model — trained on this box, at install, in about five minutes

There is no PyTorch in the `zgx` env, no ONNX, and `*.trt`/`models/` are
gitignored, so nothing can be shipped pre-built. `ew_train.py` therefore:

1. generates synthetic spectrograms from `ew_engine.py` — the *same* code that
   drives the live demo, and both sides build the array through the one
   `stack_tile()` helper, so train and serve cannot drift on channel order,
   normalisation or decimation;
2. trains a small 2-channel CNN in **pure numpy** (im2col convolutions,
   hand-written backprop, Adam) — 4–5 min depending on load;
3. hands the weights straight to the **TensorRT network-definition API** and
   serializes an FP16 engine — ~20 s.

Because the build runs under the same env python as the runtime,
build-TRT == runtime-TRT by construction. It is idempotent, and idempotent on
the *right* test: the weights carry a `DATA_VERSION` stamp, and the step
retrains whenever `ew_engine.py`'s physics or the tile format has moved past it.
"The engine still deserializes" alone was not enough — a box that had installed
once kept serving a model trained on superseded physics forever, silently, and
those are exactly the boxes that get demoed most. **Bump `DATA_VERSION` in
`ew_train.py` whenever you change the RF model or the tile.**

Four things in the training set exist because the live distribution has them
and the first version of the trainer did not:

- **30% straddling tiles** (`P_TRANSITION`), labelled with the *newest*
  technique. The window is 3.2 s and the jammer moves every 7 s, so 28–37% of
  everything the network ever sees contains two techniques. Trained without
  them it scored 43% on those and stayed confident while wrong, so the
  confidence gate did not catch it either.
- **Mid-window countermeasure changes** (`P_MODE_CHANGE`) — the defender
  re-decides every 0.75 s, so our own carrier moves inside a single tile.
- **Non-square pooling**, 2 then 4 in time rather than 4 then 4. With 4×4 twice
  the time axis collapsed to four 0.8 s cells and max-pooling let one bright old
  row hide sixteen quiet recent ones; the network could not represent "it
  stopped three rows ago" no matter how it was trained.
- **Tracking spot tones oversampled** (`P_TRACK`, and a countermeasure draw
  biased toward FIXED when the tone tracks). A misread tracking spot traps the
  defender in FIXED, which keeps the tone hidden on our own carrier, which
  causes the next misread — a feedback loop the live demo sits inside and an
  independently-drawn training mode never visits at the same rate. Live SPOT
  recall was 57% against 79% on the independent mix; it is now 81%.

Typical result: **~90% test accuracy**, 5×5 confusion matrix printed at install,
plus the two numbers that actually matter, reported separately:

```
settled tiles   :  89.3%      one technique fills the 64-row window
straddling tiles:  90.7%      the window still contains the previous technique
```

Deliberately not 99%. The residual is real and identifiable, not noise: a spot
tone that *tracks* the victim lands on our own carrier, and even with the
transmit-plan channel the two are close at long range. Range on its own is not
the difficulty — measured accuracy is flat from 8 km to 40 km — so do not claim
it is.

The live demo grades itself against ground truth in the **Classifier Scorecard**
panel, split the same two ways, so nothing has to be taken on faith from this
paragraph.

## It never dies on stage

Three tiers, and the active one is always shown in the header:

| Tier | When | Shown as |
|---|---|---|
| `tensorrt-fp16` | engine loads, CUDA OK | TensorRT FP16 · GPU |
| `numpy-cpu` | no engine / no GPU, weights present | CPU fallback · numpy |
| `heuristic` | weights missing too | DEGRADED · heuristic |

Force a tier to demo it: `EW_FORCE_CPU=1`, or `EW_ENGINE_PATH=/nonexistent`.
If Ollama is unreachable the briefs fall back to a canned template and the duel
keeps running.

## Files

```
ew_engine.py          RF synthesis, SINR, both agents, tile format (one source of truth)
ew_infer.py           classifier facade + 3 fallback tiers
ew_train.py           INSTALL-TIME: numpy training + TRT engine build
ew_demo.py            FastAPI app, sim thread, WebSockets, brief streaming
ew_demo.html          self-contained frontend (no CDN, no chart library)
ew_hw.py              GPU telemetry (verbatim copy of ai-vision/vision_hw.py)
tools/counter_matrix.py   re-derives the counter table from the physics
start_demo_8089.sh    launcher
models/               generated at install — gitignored
```

## Run it

```bash
bash ~/demos/ew-spectrum-duel/start_demo_8089.sh
curl -s localhost:8089/api/classifier          # active tier, latency, TRT version
tail -f ~/demos/logs/ew_spectrum_demo.log
```

Endpoints: `GET /`, `WS /ws/spectrum` (binary waterfall rows, ~5 KB/s, and the
full 20 s backlog on connect so a fresh tab is never staring at a black canvas),
`WS /ws/state` (JSON @10 Hz, ~30 KB/s — carries state, narration, the rolling
classifier scorecard, the engagement log, the live classifier tile at 2 Hz, and
the misread card when one is captured),
`POST /api/scenario/{drfm|barrage|second_emitter|reset|autonomy}`,
`GET /api/status`, `GET /api/classifier`, `GET /api/events`.

Environment: `EW_PORT` (8089), `EW_OLLAMA_MODEL` (llama3.1:70b),
`EW_FORCE_CPU`, `EW_ENGINE_PATH`.
