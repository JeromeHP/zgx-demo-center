"""
Spectrum Duel — FastAPI backend.

HP ZGX Nano · AI-vs-AI electronic warfare demo · port 8089

  GET  /                    → ew_demo.html (Cache-Control: no-store)
  WS   /ws/spectrum         → binary waterfall rows (1 byte n_rows + n*256 uint8)
  WS   /ws/state            → JSON state @10 Hz, inbound {"cmd": ...} for buttons
  POST /api/scenario/{name} → drfm | barrage | second_emitter | reset | autonomy
  GET  /api/status          → full snapshot (curl-testable)
  GET  /api/classifier      → active tier, latency, engine path, TRT version
  GET  /api/events          → last 100 engagement events

Threading: one simulation thread owns the RF scene, both agents, and the CUDA
context (inference is inline — ~0.28 ms end-to-end, host-to-host, against a
50 ms frame budget, so a queue would only add staleness). The event loop
only ever reads lock-guarded snapshots, so it never blocks. The LLM brief runs
as a single asyncio task shared by every connected client, so two open browsers
still mean exactly one call into the 70B.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

import ew_engine as E
from ew_infer import Classifier
from ew_hw import GpuPoller

from zgx_metrics import TokenLedger  # noqa: E402

LEDGER = TokenLedger()


HERE = Path(__file__).resolve().parent
HTML_FILE = HERE / "ew_demo.html"
PORT = int(os.environ.get("EW_PORT", 8089))

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("EW_OLLAMA_MODEL", "llama3.1:70b")

HIST_ROWS = 400                 # 20 s of waterfall history
CLASSIFY_EVERY = 4              # frames between classifications (5 Hz)
DEFENDER_EVERY = 15             # 0.75 s — the defender turns inside...
# ...the jammer's decision loop. This MUST stay comfortably longer than the
# classifier's 64-row (3.2 s) tile window: if the jammer switches faster than
# the window, every tile straddles two techniques and the classifier — trained
# on homogeneous tiles — confidently reports nonsense. 7 s also happens to be
# the right dramatic pacing: the defender gets ~9 decisions per attack, so an
# audience sees the counter land and the link recover before the next move.
JAMMER_EVERY = 140              # 7 s
UPTIME_WINDOW = 1200            # 60 s of frames for the scoreboard
BRIEF_MIN_GAP_S = 14.0
# Rolling classifier scorecard: 600 classifications at 5 Hz = the last 2 min.
# The demo has ground truth, so it can grade itself on stage instead of asking
# the room to take a training-time number on faith.
ACC_WINDOW = 600
# A tile is "settled" once the whole 64-row window post-dates the last technique
# change. Before that it straddles two techniques and is genuinely hard — that
# distinction is the honest way to report accuracy, and it is also the entire
# explanation for every error the classifier makes.
SETTLED_AFTER = E.TILE_ROWS

# Plain-language names. The screen says "DRFM"; the narration says what that
# actually means to the person watching.
PLAIN = {
    "CLEAR":   "nothing — the emitter has gone quiet",
    "BARRAGE": "barrage noise across the whole band",
    "SPOT":    "a spot tone on our frequency",
    "SWEPT":   "a tone sweeping the band",
    "DRFM":    "a repeater copying our signal",
}
# Compact back-reference. The "Jammer up" line carries the full explanation, so
# every later mention in the same engagement can be short — which keeps the
# most important line of all ("wrong call") from being ellipsised away.
SHORT = {
    "CLEAR":   "a clear band",
    "BARRAGE": "barrage noise",
    "SPOT":    "a spot tone",
    "SWEPT":   "a sweeping tone",
    "DRFM":    "a repeater",
}
COUNTER_PLAIN = {
    E.FIXED:    "Holding — nothing to counter",
    E.HOP:      "Hopping to a new frequency",
    E.NOTCH:    "Blanking the sweep as it passes",
    E.WIDEBAND: "Changing our waveform so the copy stops matching",
    E.BURN:     "Turning up power to burn through",
}
# The model is told what these mean. Left to itself, llama3.1 expands the class
# names into plausible-sounding nonsense — it produced "SPOT (Saturate Pulse
# Overload Technique)" on this box — and an EW audience reads an invented
# acronym as the whole demo being invented.
GLOSS = {
    "CLEAR":   "no jamming — the band is clear",
    "BARRAGE": "barrage noise jamming: broadband noise radiated across the "
               "entire band at once",
    "SPOT":    "spot jamming: a single narrowband tone parked on the victim's "
               "operating frequency",
    "SWEPT":   "swept-spot jamming: a narrowband tone swept repeatedly across "
               "the band, hitting the victim on each pass",
    "DRFM":    "DRFM repeater jamming: digital radio frequency memory stores "
               "the victim's waveform and retransmits a coherent copy of it",
}

ENGAGEMENT_WIN_S = 3.0          # counter must land within this to be a DEF win.
                                # Measured restore times cluster between 0.5 s
                                # and 4 s, so this is the threshold that actually
                                # discriminates; at 5 s the defender won every
                                # engagement and the scoreboard said nothing.
DENIAL_GRACE_S = 6.0            # if an attack has not denied the link by now,
                                # it simply failed — do not score it as a win


class DuelState:
    """Everything the sim thread produces and the event loop reads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rows = np.zeros((HIST_ROWS, E.NBINS), np.uint8)
        # Our own transmit plan, row by row — classifier channel 1. Two small
        # arrays rather than a second 400x256 raster: the carrier is a centre
        # bin and a width, and the mask is cheaper to rebuild than to store.
        self.f_bin = np.zeros(HIST_ROWS, np.int16)
        self.f_w = np.full(HIST_ROWS, E.REF_WIDTH, np.uint8)
        self.row_count = 0
        self.sinr = 0.0
        self.mbps = 0.0
        self.sinr_hist = deque(maxlen=UPTIME_WINDOW)
        self.up_hist = deque(maxlen=UPTIME_WINDOW)
        self.cls = {"label": "CLEAR", "confidence": 0.0, "probs": [0.0] * E.NCLASS,
                    "backend": "", "backend_label": "", "latency_ms": 0.0}
        self.truth = "CLEAR"
        self.jammer = {"technique": "CLEAR", "target": "CENTER", "bin": 0,
                       "explored": False, "confidence": 0.0}
        self.defender = {"mode": E.FIXED, "label": E.COUNTER_LABEL[E.FIXED],
                         "switches": 0, "correct": True}
        self.events = deque(maxlen=100)
        self.event_seq = 0
        self.score = {"def_wins": 0, "jam_wins": 0, "uptime_pct": 100.0,
                      "last_counter_s": None, "median_counter_s": None,
                      "engagement_open": False}
        # Every measured time-to-counter this session. A win/loss tally against
        # a fixed 3 s threshold is brittle in both directions — it read 9-15
        # against the defender before the classifier was fixed and 6-0 for it
        # after, and neither number told anyone how fast the system actually is.
        # The median does, and it does not depend on where a threshold is drawn.
        self.counter_times = deque(maxlen=64)
        self.jam_range_km = 12.0
        # Where the interesting things are in the band, so the waterfall can
        # label its own features instead of being a texture nobody can read.
        self.features = {"friendly_bin": 0, "friendly_width": 3, "jam_bin": 0,
                         "jam_kind": "CLEAR", "replica_bin": None, "sweep_bin": None}
        # Plain-language narration of the decision loop, for the viewer who has
        # never seen a spectrogram. The technical log lives in `events`.
        self.tile_b64 = ""          # classifier channel 0, for the UI thumbnail
        self.tile_mask_b64 = ""     # channel 1, bit-packed (512 B, not 4 KB)
        self.tile_seq = 0
        self.narration = deque(maxlen=6)
        self.narration_seq = 0
        # The most recent misidentification that actually cost us the link,
        # frozen so a presenter can point at it. Includes the very tile the
        # model got wrong.
        self.last_miss = None
        self.miss_seq = 0
        self.miss_count = 0          # misreads that actually cost link time
        self.misread_count = 0       # every misread, costly or not
        self.miss_cost_s = 0.0       # cumulative seconds of link lost to them
        # Rolling self-grading: (truth, pred, settled?) per classification.
        self.cls_hist = deque(maxlen=ACC_WINDOW)
        self.acc = {"n": 0, "overall": None, "settled": None,
                    "transition": None, "recall": {}, "support": {}}
        self.second_emitter = False
        self.autonomy = True
        self.override_until = 0.0
        self.frames = 0

    def say(self, text):
        """Add a plain-language narration line (always under the lock)."""
        self.narration_seq += 1
        self.narration.append({"seq": self.narration_seq, "ts": time.time(), "text": text})

    def push_event(self, kind, text):
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "ts": time.time(),
                            "kind": kind, "text": text})


STATE = DuelState()
CLASSIFIER = Classifier()
# Scenario commands from the UI. The sim thread owns the RFScene, so the request
# handler may not touch it directly — it queues here and the sim thread applies.
_SCENARIO_Q: deque = deque(maxlen=64)
HW = GpuPoller()
_stop = threading.Event()
_SIM_THREAD: threading.Thread | None = None

# Set by the sim thread when something worth narrating happens; consumed by the
# brief task. Only the newest matters — a stale trigger is dropped, not queued.
_brief_trigger: dict | None = None
_brief_lock = threading.Lock()

BRIEF = {"id": 0, "seq": 0, "text": "", "running": False,
         "model": OLLAMA_MODEL, "online": True}


def _tile_from_history(st: DuelState) -> np.ndarray:
    """Newest TILE_ROWS rows as the (2, 64, 64) array the network is handed.

    Channel 0 is the received power exactly as the waterfall shows it — the
    uint8 the display already quantised, so the thumbnail really is the input
    and not a lookalike. Channel 1 is our own transmit plan over the same rows.
    """
    n = min(st.row_count, E.TILE_ROWS)
    idx = (st.row_count - np.arange(n, 0, -1)) % HIST_ROWS
    block = st.rows[idx].astype(np.float32) / 255.0
    mask = E.friendly_mask(st.f_bin[idx], st.f_w[idx])
    if n < E.TILE_ROWS:
        pad = np.zeros((E.TILE_ROWS - n, E.NBINS), np.float32)
        block = np.vstack([pad, block])
        mask = np.vstack([pad, mask])
    return np.stack([E.decimate(block), E.decimate(mask)]).astype(np.float32)


def _ghz(bin_i):
    return E.BAND_LO_GHZ + bin_i * E.BIN_MHZ / 1000.0


def _scorecard(hist) -> dict:
    """Grade the classifier against ground truth over the rolling window.

    Split settled vs straddling deliberately. A single blended number is the
    least informative thing we could put on screen: it hides that essentially
    every error happens in the seconds right after the jammer changes
    technique, which is the one fact about this model worth explaining.
    """
    if not hist:
        return {"n": 0, "overall": None, "settled": None, "transition": None,
                "recall": {}, "support": {}}
    ok = [t == p for t, p, _ in hist]
    st = [t == p for t, p, s in hist if s]
    tr = [t == p for t, p, s in hist if not s]
    recall, support = {}, {}
    for c in E.CLASSES:
        sel = [t == p for t, p, s in hist if t == c]
        support[c] = len(sel)
        recall[c] = round(100.0 * sum(sel) / len(sel), 1) if sel else None
    pct = lambda v: round(100.0 * sum(v) / len(v), 1) if v else None
    return {"n": len(hist), "overall": pct(ok), "settled": pct(st),
            "transition": pct(tr), "n_settled": len(st), "n_transition": len(tr),
            "recall": recall, "support": support}


def _features(scene) -> dict:
    """Where the readable structures are, so the waterfall can caption itself."""
    replica = sweep = None
    if scene.technique == "DRFM" and len(scene._hist) > scene.drfm_delay:
        replica = int((scene._hist[-1 - scene.drfm_delay] + scene.drfm_offset) % E.NBINS)
    elif scene.technique == "SWEPT":
        span = E.SWEEP_SPAN
        tri = abs(((scene.sweep_phase / span) % 2.0) - 1.0)
        sweep = int(tri * (span - 1)) % E.NBINS
    return {
        "friendly_bin": int(scene.friendly_bin),
        "friendly_width": int(scene.friendly_width),
        "jam_bin": int(scene.jam_bin),
        "jam_kind": scene.technique,
        "replica_bin": replica,
        "sweep_bin": sweep,
    }


def sim_thread():
    """20 Hz: advance the RF scene, classify, let both agents act, score."""
    CLASSIFIER.start()                      # CUDA context belongs to THIS thread
    scene = E.RFScene(seed=None)
    jammer = E.JammerAgent()
    defender = E.DefenderAgent()
    st = STATE
    with st.lock:
        st.push_event("system", f"Classifier online — {CLASSIFIER.info()['backend_label']}")

    range_target = E.JAM_REF_KM
    period = 1.0 / E.ROW_HZ
    next_t = time.perf_counter()
    engage_start = None
    engage_tech = None
    denial_start = None
    last_tech_change = -10 ** 9      # frame index of the last technique change
    pending_miss = None
    res_tile = [""]
    res_mask = [""]
    res_probs = [[0.0] * E.NCLASS]
    res_latency = [0.0]
    pending_reward = []

    while not _stop.is_set():
        # One bad frame must not take the demo down. Without this a single
        # exception kills the sim thread silently: /ws/state keeps streaming
        # the last snapshot at 10 Hz, the waterfall stops, and the UI shows
        # frozen numbers with no indication anything is wrong.
        try:
            out = scene.step()
            row_u8 = (E.normalize_rows(out["row_dbm"]) * 255).astype(np.uint8)

            with st.lock:
                slot = st.row_count % HIST_ROWS
                st.rows[slot] = row_u8
                st.f_bin[slot] = out["friendly_bin"]
                st.f_w[slot] = out["friendly_width"]
                st.row_count += 1
                st.frames += 1
                st.sinr = out["sinr_db"]
                st.mbps = out["mbps"]
                st.sinr_hist.append(out["sinr_db"])
                st.up_hist.append(1.0 if out["sinr_db"] >= E.LINK_UP_DB else 0.0)
                st.truth = scene.technique
                st.features = _features(scene)
                st.jam_range_km = scene.jam_range_km
                frames = st.frames
            pending_reward.append(out["sinr_db"])

            # ── jammer platform manoeuvres slowly between engagements ──────────
            if frames % 1200 == 0:                       # new intent every 60 s
                range_target = float(np.random.uniform(E.JAM_MIN_KM, E.JAM_MAX_KM))
            scene.jam_range_km += np.clip(range_target - scene.jam_range_km, -0.02, 0.02)

            # ── operator overrides (queued by the request handlers) ────────────
            while _SCENARIO_Q:
                cmd = _SCENARIO_Q.popleft()
                if cmd == "drfm":
                    scene.set_technique("DRFM", scene.friendly_bin)
                elif cmd == "barrage":
                    scene.set_technique("BARRAGE")
                elif cmd == "second_emitter":
                    scene.extra_jammers = 0 if scene.extra_jammers else 1
                    with st.lock:
                        st.push_event("operator", "Second emitter "
                                      + ("removed" if not scene.extra_jammers else "on station"))
                elif cmd == "reset":
                    scene.extra_jammers = 0
                    scene.set_technique("CLEAR")
                    jammer.__init__()
                    defender.__init__()
                    engage_start = denial_start = engage_tech = None
                    last_tech_change = frames
                    pending_miss = None
                    with st.lock:
                        st.score["engagement_open"] = False
                        st.second_emitter = False
                        st.cls_hist.clear()
                        st.acc = _scorecard(st.cls_hist)
                        st.miss_count = st.misread_count = 0
                        st.miss_cost_s = 0.0
                        st.last_miss = None
                        st.miss_seq += 1
                if cmd == "second_emitter":
                    with st.lock:
                        st.second_emitter = bool(scene.extra_jammers)
                if cmd in ("drfm", "barrage"):
                    last_tech_change = frames
                    engage_start, engage_tech, denial_start = time.time(), scene.technique, None
                    with st.lock:
                        st.score["engagement_open"] = True
                        st.jammer = {"technique": scene.technique, "target": "OPERATOR",
                                     "bin": scene.jam_bin, "explored": False,
                                     "confidence": 0.0}

            # ── classify (5 Hz) ────────────────────────────────────────────────
            if frames % CLASSIFY_EVERY == 0:
                with st.lock:
                    tile = _tile_from_history(st)
                res = CLASSIFIER.classify(tile)
                # Keep the exact tile the model just judged. If this turns out
                # to be a misidentification we want to SHOW the input rather
                # than describe it — "here is what it saw" is the whole point
                # of the replay card.
                res_tile[0] = base64.b64encode(
                    (np.clip(tile[0], 0, 1) * 255).astype(np.uint8).tobytes()).decode()
                res_mask[0] = base64.b64encode(
                    np.packbits(tile[1] > 0.5, axis=1).tobytes()).decode()
                res_probs[0] = list(res["probs"])
                res_latency[0] = res["latency_ms"]
                settled = (frames - last_tech_change) >= SETTLED_AFTER
                with st.lock:
                    st.cls = res
                    st.tile_b64 = res_tile[0]
                    st.tile_mask_b64 = res_mask[0]
                    st.tile_seq += 1
                    st.cls_hist.append((scene.technique, res["label"], settled))
                    st.acc = _scorecard(st.cls_hist)

            # ── defender acts (0.75 s) ─────────────────────────────────────────
            if frames % DEFENDER_EVERY == 0:
                with st.lock:
                    pred, conf, truth = st.cls["label"], st.cls["confidence"], st.truth
                mode, switched = defender.act(pred, conf)
                scene.set_mode(mode)
                correct = E.COUNTER_FOR.get(truth, E.FIXED) == mode
                # Two different failures, and the screen used to conflate them:
                #  · misread  — the network named the wrong threat
                #  · lag      — it named the right one but the track-before-
                #               switch gate has not committed the counter yet
                # Only the first is a classifier error, and only the first
                # belongs in the misread count.
                misread = pred != truth
                with st.lock:
                    st.defender = {"mode": mode, "label": E.COUNTER_LABEL[mode],
                                   "switches": defender.switches, "correct": correct,
                                   "misread": misread, "predicted": pred}
                if switched:
                    with st.lock:
                        st.push_event(
                            "defender",
                            f"Classified {pred} ({conf:.0%}) → {E.COUNTER_LABEL[mode]}"
                            + ("" if correct else f"  ⚠ actual threat is {truth}"),
                        )
                        st.say(f"Network read it as {SHORT[pred]} in {res_latency[0]:.2f} ms.")
                        if correct:
                            st.say(f"{COUNTER_PLAIN[mode]} — correct counter.")
                        else:
                            st.say(f"Wrong call — it is really {SHORT[truth]}, "
                                   f"so that counter will not work.")
                    _trigger_brief("counter", pred=pred, truth=truth, mode=mode,
                                   conf=conf, correct=correct)
                # Capture on the CONDITION, not on the switch. Tying capture to
                # `switched` meant a misread whose wrong counter happened to
                # already be in force was never recorded — the header could read
                # "MISREADS 1" while stage ③ was showing a wrong counter, which
                # is precisely the contradiction a technical customer spots.
                if misread and not correct and pending_miss is None:
                    with st.lock:
                        st.misread_count += 1
                    pending_miss = {
                        "ts": time.time(), "pred": pred, "conf": float(conf),
                        "truth": truth, "counter": mode,
                        "counter_label": E.COUNTER_LABEL[mode],
                        "right_counter": E.COUNTER_FOR.get(truth, E.FIXED),
                        "right_label": E.COUNTER_LABEL[E.COUNTER_FOR.get(truth, E.FIXED)],
                        "probs": list(res_probs[0]),
                        "tile_b64": res_tile[0],
                        "tile_mask_b64": res_mask[0],
                        "settled": (frames - last_tech_change) >= SETTLED_AFTER,
                        "since_switch_s": round((frames - last_tech_change) / E.ROW_HZ, 1),
                        "min_mbps": 1e9, "down_s": 0.0,
                    }

            # ── jammer acts (7 s) ──────────────────────────────────────────────
            if frames % JAMMER_EVERY == 0:
                with st.lock:
                    overridden = time.time() < st.override_until
                    autonomy = st.autonomy
                if pending_reward:
                    # Only credit the bandit for outcomes it actually caused —
                    # rewarding it for an operator-forced technique poisons the
                    # Q-table exactly when a presenter is using the buttons.
                    if not overridden:
                        jammer.reward(float(np.mean(pending_reward)))
                    pending_reward = []
                if not overridden and autonomy:
                    tech, bin_, targ, explored = jammer.act(scene.mode, scene.friendly_bin)
                    if tech != scene.technique:
                        last_tech_change = frames
                        # say() mutates the narration deque and its sequence
                        # counter, so it has to be inside the lock like every
                        # other writer — it was not, and two of these landing in
                        # the same tick could hand the UI a duplicate seq and
                        # silently drop a line.
                        with st.lock:
                            if tech == "CLEAR":
                                st.push_event("jammer", "Jammer ceased transmission — band clear")
                                st.say("Jammer has stopped transmitting — the band is clear.")
                            else:
                                st.push_event("jammer", f"Jammer → {tech} at bin {bin_} ({targ})")
                                st.say(f"Jammer up — {PLAIN[tech]} at {_ghz(bin_):.2f} GHz.")
                        if tech == "CLEAR":
                            # An emitter that has gone quiet is not an
                            # engagement — do not open one and do not score it.
                            engage_start = engage_tech = denial_start = None
                            with st.lock:
                                st.score["engagement_open"] = False
                        else:
                            engage_start, engage_tech, denial_start = time.time(), tech, None
                            with st.lock:
                                st.score["engagement_open"] = True
                    scene.set_technique(tech, bin_)
                    with st.lock:
                        st.jammer = {"technique": tech, "target": targ, "bin": bin_,
                                     "explored": explored,
                                     "confidence": jammer.confidence()}

            # ── misidentification capture ──────────────────────────────────────
            # Track what the mistake actually cost, then freeze it. This moment is
            # the most instructive thing in the demo and it otherwise scrolls past
            # in about three seconds.
            if pending_miss is not None:
                with st.lock:
                    mbps_now, sinr_now = st.mbps, st.sinr
                    still_wrong = not st.defender["correct"]
                    still_misread = st.defender.get("misread", False)
                pending_miss["min_mbps"] = min(pending_miss["min_mbps"], mbps_now)
                if sinr_now < E.LINK_UP_DB:
                    pending_miss["down_s"] += period
                if (not still_wrong or not still_misread
                        or time.time() - pending_miss["ts"] > 30.0):
                    if pending_miss["down_s"] >= 0.5:      # it actually cost us something
                        pending_miss["cost_s"] = round(pending_miss["down_s"], 1)
                        pending_miss["min_mbps"] = round(max(0.0, pending_miss["min_mbps"]), 1)
                        with st.lock:
                            st.miss_seq += 1
                            st.miss_count += 1
                            st.miss_cost_s += pending_miss["down_s"]
                            pending_miss["seq"] = st.miss_seq
                            st.last_miss = pending_miss
                            st.say(f"That misread cost {pending_miss['cost_s']:.1f}s of link.")
                    pending_miss = None

            # ── engagement scoring ─────────────────────────────────────────────
            # An engagement only counts once the attack has ACTUALLY denied the
            # link. Scoring every technique change as a defender win — including
            # attacks that never landed — is how you end up with a 10-0 scoreboard
            # that nobody in the room believes.
            if engage_start is not None:
                with st.lock:
                    up_now = st.sinr >= E.LINK_UP_DB
                    recent = list(st.up_hist)[-10:]
                if denial_start is None:
                    if not up_now:
                        denial_start = time.time()          # the attack landed
                        with st.lock:
                            st.score["engagement_open"] = True
                    elif time.time() - engage_start > DENIAL_GRACE_S:
                        with st.lock:
                            st.push_event("score", f"{engage_tech} ineffective — link never dropped")
                            st.score["engagement_open"] = False
                        engage_start = None
                else:
                    if len(recent) == 10 and all(recent):
                        dt = time.time() - denial_start
                        with st.lock:
                            st.score["last_counter_s"] = round(dt, 1)
                            st.counter_times.append(dt)
                            st.score["median_counter_s"] = round(
                                float(np.median(st.counter_times)), 1)
                            st.score["engagement_open"] = False
                            if dt <= ENGAGEMENT_WIN_S:
                                st.score["def_wins"] += 1
                                st.push_event("score", f"DEFENDER holds — link restored in {dt:.1f}s")
                                st.say(f"Link back at {st.mbps:.0f} Mbps — {dt:.1f}s to counter.")
                            else:
                                st.score["jam_wins"] += 1
                                st.push_event("score", f"JAMMER wins — link denied {dt:.0f}s")
                                st.say(f"Jammer held the link down {dt:.0f}s — that one got through.")
                        engage_start = denial_start = None
                    elif time.time() - denial_start > 45.0:
                        with st.lock:
                            st.score["jam_wins"] += 1
                            st.score["engagement_open"] = False
                            st.push_event("score", f"JAMMER wins — {engage_tech} held the link down")
                        engage_start = denial_start = None

            with st.lock:
                if st.up_hist:
                    st.score["uptime_pct"] = round(100.0 * float(np.mean(st.up_hist)), 1)

            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))

        except Exception as exc:
            with st.lock:
                st.push_event('system', f'sim fault: {type(exc).__name__}: {exc}')
            print(f'  [sim] fault: {exc!r}', flush=True)
            time.sleep(0.1)

    CLASSIFIER.close()


def _trigger_brief(kind, **kw):
    global _brief_trigger
    with _brief_lock:
        _brief_trigger = {"kind": kind, **kw}


# ── LLM brief (single shared stream) ───────────────────────────────────────
def _prompt_for(trig, snap):
    """Build the brief prompt.

    The counter actually in force is ALWAYS included, even for a
    technique-change brief. Without it the model cheerfully invents a
    countermeasure ("activating frequency hopping") that the system never took,
    and the brief then contradicts the countermeasure shown on screen — which is
    exactly the kind of detail an EW audience notices.
    """
    counter = E.COUNTER_LABEL.get(snap["mode"], snap["mode"])
    pred, truth = trig["pred"], trig["truth"]
    if trig["correct"]:
        tail = ""
    elif truth == "CLEAR":
        # "the actual threat is CLEAR" reads as a threat named CLEAR and the
        # model duly wrote a paragraph about what CLEAR does to us.
        tail = (" That identification is WRONG — there is in fact no jammer on "
                "the band, so we are holding a countermeasure we do not need.")
    else:
        tail = (f" That identification is WRONG — the actual threat is {truth}, "
                f"so the countermeasure now in force is the wrong one and the "
                f"link will suffer until it is corrected.")
    gloss = "\n".join(f"  {c} = {GLOSS[c]}" for c in dict.fromkeys((pred, truth))
                      if c in GLOSS)
    return (
        "You are an electronic warfare officer on watch in a ship's combat "
        "information center. Be terse and technical.\n\n"
        "These terms have fixed meanings. Use them as given and never invent an "
        f"expansion for an abbreviation:\n{gloss}\n\n"
        f"The classifier identified the threat as {pred} at "
        f"{trig['conf']:.0%} confidence.{tail}\n"
        f"Countermeasure currently in force: {counter}.\n"
        f"Protected link SINR: {snap['sinr']:.1f} dB. Throughput: {snap['mbps']:.0f} Mbps. "
        f"Link uptime last 60s: {snap['uptime']:.0f}%.\n\n"
        "Reply in exactly three lines, each labelled. In COUNTER-EW you must "
        "report the countermeasure listed above as the action taken — do not "
        "invent a different one.\n"
        "**THREAT:** <technique and what it does to us, one sentence>\n"
        "**ASSESSMENT:** <effect on the protected link, one sentence>\n"
        "**COUNTER-EW:** <the countermeasure in force and why it works, one sentence>"
    )


async def brief_task():
    global _brief_trigger
    import httpx

    last = 0.0
    while not _stop.is_set():
        await asyncio.sleep(0.4)
        with _brief_lock:
            trig, _brief_trigger = _brief_trigger, None
        if not trig or time.time() - last < BRIEF_MIN_GAP_S:
            continue
        last = time.time()
        with STATE.lock:
            snap = {"sinr": STATE.sinr, "mbps": STATE.mbps,
                    "uptime": STATE.score["uptime_pct"],
                    "mode": STATE.defender["mode"]}
        BRIEF.update(id=BRIEF["id"] + 1, seq=0, text="", running=True)
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream("POST", OLLAMA_URL, json={
                    "model": OLLAMA_MODEL,
                    "prompt": _prompt_for(trig, snap),
                    "stream": True,
                    "options": {"num_predict": 220, "temperature": 0.3},
                }) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        tok = chunk.get("response", "")
                        if tok:
                            BRIEF["text"] += tok
                            BRIEF["seq"] += 1
                        if chunk.get("done"):
                            LEDGER.record(chunk, fallback_text="")
                            break
            BRIEF["online"] = True
        except Exception:
            # Ollama down: canned brief, muted pill, duel keeps running.
            BRIEF["online"] = False
            BRIEF["text"] = (
                f"**THREAT:** {trig.get('tech', trig.get('pred', 'Unknown'))} jamming detected.\n"
                f"**ASSESSMENT:** Link SINR {snap['sinr']:.1f} dB, {snap['mbps']:.0f} Mbps.\n"
                "**COUNTER-EW:** Countermeasure applied. (LLM offline — canned brief.)"
            )
            BRIEF["seq"] += 1
        finally:
            BRIEF["running"] = False


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Spectrum Duel")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.on_event("startup")
async def _startup():
    HW.start()
    global _SIM_THREAD
    _SIM_THREAD = threading.Thread(target=sim_thread, name="ew-sim", daemon=True)
    _SIM_THREAD.start()
    asyncio.create_task(brief_task())
    asyncio.create_task(_warm_ollama())


async def _warm_ollama():
    """Warm the 70B so the first brief is not cold (pipeline-pulse pattern)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            await client.post(OLLAMA_URL, json={"model": OLLAMA_MODEL,
                                                "prompt": "Ready.", "stream": False})
    except Exception:
        pass


@app.on_event("shutdown")
async def _shutdown():
    _stop.set()
    t = globals().get("_SIM_THREAD")
    if t is not None:
        t.join(timeout=2.0)          # let CLASSIFIER.close() pop the CUDA context


def _snapshot():
    st = STATE
    with st.lock:
        sinr_series = [round(v, 1) for v in list(st.sinr_hist)[-120:]]
        return {
            "sinr_db": round(st.sinr, 1),
            "mbps": round(st.mbps, 1),
            "link": ("UP" if st.sinr >= E.LINK_UP_DB
                     else "DOWN" if st.sinr < E.LINK_DOWN_DB else "DEGRADED"),
            "classifier": dict(st.cls),
            "truth": st.truth,
            "jammer": dict(st.jammer),
            "defender": {**st.defender,
                         # recompute against the truth as of NOW
                         "correct": E.COUNTER_FOR.get(st.truth, E.FIXED)
                                    == st.defender["mode"],
                         # Recomputed against the truth as of NOW, like
                         # `correct`, so the two can never disagree on screen.
                         "misread": st.cls["label"] != st.truth},
            "score": dict(st.score),
            "events": list(st.events)[-8:],
            "sinr_series": sinr_series,
            "autonomy": st.autonomy,
            "override": time.time() < st.override_until,
            "frames": st.frames,
            "features": dict(st.features),
            "narration": list(st.narration),
            "miss_seq": st.miss_seq,
            "miss_count": st.miss_count,
            "misread_count": st.misread_count,
            "miss_cost_s": round(st.miss_cost_s, 1),
            "accuracy": dict(st.acc),
            "second_emitter": st.second_emitter,
            "override_s": max(0.0, round(st.override_until - time.time(), 1)),
            "map": {"jam_range_km": round(st.jam_range_km, 1)},
            "band": {"lo_ghz": E.BAND_LO_GHZ, "hi_ghz": E.BAND_HI_GHZ,
                     "bins": E.NBINS, "link_up_db": E.LINK_UP_DB},
        }


@app.get("/", response_class=HTMLResponse)
async def index():
    if not HTML_FILE.exists():
        return HTMLResponse("<h1>ew_demo.html not found</h1>", status_code=404)
    return HTMLResponse(HTML_FILE.read_text(),
                        headers={"Cache-Control": "no-store"})


@app.get("/api/metrics")
async def api_metrics():
    """Session token totals for the shared chrome."""
    return LEDGER.snapshot()


@app.get("/api/status")
async def api_status():
    snap = _snapshot()
    snap["hw"] = HW.get()
    snap["brief"] = {k: BRIEF[k] for k in ("id", "seq", "running", "model", "online")}
    with STATE.lock:
        m = STATE.last_miss
    if m:
        snap["last_miss"] = {k: v for k, v in m.items()
                             if k not in ("tile_b64", "tile_mask_b64")}
    return JSONResponse(snap)


@app.get("/api/classifier")
async def api_classifier():
    return JSONResponse(CLASSIFIER.info())


@app.get("/api/events")
async def api_events():
    with STATE.lock:
        return JSONResponse({"events": list(STATE.events)})


VALID_SCENARIOS = {"drfm", "barrage", "second_emitter", "reset", "autonomy"}


def _apply_scenario(name: str) -> dict:
    st = STATE
    with st.lock:
        if name == "reset":
            st.score.update(def_wins=0, jam_wins=0, last_counter_s=None,
                            median_counter_s=None)
            st.counter_times.clear()
            st.events.clear()
            st.push_event("system", "Engagement reset by operator")
        elif name == "autonomy":
            st.autonomy = not st.autonomy
            st.push_event("system", f"Autonomy {'on' if st.autonomy else 'held'}")
        elif name == "second_emitter":
            # A second emitter changes the ENVIRONMENT, not the jammer's
            # decisions — it must not freeze the agent the way a forced
            # technique does. The sim thread logs the actual toggle state.
            pass
        else:
            st.override_until = time.time() + 20.0
            st.push_event("operator", f"OPERATOR OVERRIDE — {name} forced for 20s")
        return {"ok": True, "scenario": name, "autonomy": st.autonomy}


@app.post("/api/scenario/{name}")
async def api_scenario(name: str):
    if name not in VALID_SCENARIOS:
        return JSONResponse({"ok": False, "error": f"unknown scenario {name}"},
                            status_code=400)
    res = _apply_scenario(name)
    _SCENARIO_Q.append(name)
    return JSONResponse(res)


@app.websocket("/ws/spectrum")
async def ws_spectrum(ws: WebSocket):
    """Binary waterfall. Message = [n_rows][n_rows * NBINS bytes]."""
    await ws.accept()
    with STATE.lock:
        # Start from the OLDEST row still held, not from "now". Starting at now
        # meant every page load — and every reconnect — opened on twenty seconds
        # of black canvas while the buffer refilled, which is most of the time a
        # visitor spends at the table. The history is already in memory.
        cursor = max(0, STATE.row_count - HIST_ROWS)
    try:
        while True:
            await asyncio.sleep(0.1)
            with STATE.lock:
                total = STATE.row_count
                # A client that stalls for more than HIST_ROWS/ROW_HZ (20 s) has
                # fallen off the back of the ring buffer — the slots it still
                # wants have been overwritten. Snap it to the oldest row we
                # actually still hold rather than shipping recycled garbage.
                if total - cursor > HIST_ROWS:
                    cursor = total - HIST_ROWS
                # 8 rows is the steady-state tick (20 Hz over a 0.1 s loop);
                # the larger cap only bites while a fresh client catches up on
                # the backlog, which it then does in two frames. The length byte
                # is one octet, so this must stay under 256.
                n = min(total - cursor, 200)
                if n > 0:
                    idx = (np.arange(cursor, cursor + n)) % HIST_ROWS
                    payload = STATE.rows[idx].tobytes()
                    cursor += n
            if n > 0:
                await ws.send_bytes(bytes([n]) + payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await ws.accept()
    sent_brief = (-1, -1, False)
    sent_miss = -1
    tick = 0

    async def pump():
        nonlocal sent_brief, sent_miss, tick
        while True:
            snap = _snapshot()
            snap["hw"] = HW.get()
            key = (BRIEF["id"], BRIEF["seq"], BRIEF["running"])
            if key != sent_brief:
                snap["brief"] = {"id": BRIEF["id"], "seq": BRIEF["seq"],
                                 "text": BRIEF["text"], "running": BRIEF["running"],
                                 "model": OLLAMA_MODEL, "online": BRIEF["online"]}
                sent_brief = key
            tick += 1
            if tick % 5 == 0:
                with STATE.lock:
                    snap["tile_b64"] = STATE.tile_b64
                    snap["tile_mask_b64"] = STATE.tile_mask_b64
            with STATE.lock:
                miss, mseq = STATE.last_miss, STATE.miss_seq
            if mseq != sent_miss:
                snap["last_miss"] = miss
                sent_miss = mseq
            await ws.send_json(snap)
            await asyncio.sleep(0.1)

    async def recv():
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            cmd = data.get("cmd")
            if cmd in VALID_SCENARIOS:
                _apply_scenario(cmd)
                _SCENARIO_Q.append(cmd)

    tasks = [asyncio.create_task(pump()), asyncio.create_task(recv())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                raise exc
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
