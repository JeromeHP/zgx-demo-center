"""
Spectrum Duel — RF scene synthesis and the two duelling agents.

This module is the single source of truth for the RF model. It is imported by
BOTH the live demo (ew_demo.py) and the install-time trainer (ew_train.py), so
the jam classifier is always trained on exactly the distribution it will see at
runtime. If you change the physics here, re-run ew_train.py.

Nothing in this file touches the GPU — synthesis and the agents are numpy/CPU.
The classifier lives in ew_infer.py.

Everything modelled here is SIMULATED. What is real in this demo: the waveform
classification (TensorRT FP16), the LLM briefs (Ollama), and the GPU telemetry.
The UI labels that split explicitly (Annotation Mode).
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

# ── Band / raster geometry ──────────────────────────────────────────────────
NBINS       = 256          # frequency bins across the displayed band
BAND_LO_GHZ = 8.0
BAND_HI_GHZ = 12.0
ROW_HZ      = 20.0         # waterfall rows per second
TILE_ROWS   = 64           # classifier tile height (time)
TILE_BINS   = 64           # classifier tile width (freq) — NBINS decimated 4:1
DECIM       = NBINS // TILE_BINS

# The classifier input has TWO channels:
#   0 — received power, normalised (what the antenna hears)
#   1 — our own transmit plan: which bins WE are occupying, row by row
#
# Channel 1 is not a cheat, it is the thing that makes the problem well posed. A
# spot tone that tracks the victim lands on the victim's own carrier, and after
# the 4:1 max-pooled decimation the tone and the carrier occupy the same tile
# bin. From power alone the two are separable only by amplitude, and at 30 km a
# 46 dB spot arrives at about the same level as our own 38 dB carrier — so a
# power-only network called 41% of tracking spot tones "CLEAR" and held FIXED,
# leaving the tone sitting on us. It was not a training failure; the information
# was not in the input.
#
# A real EW suite has never had this problem: it knows its own emissions. Giving
# the network the same knowledge lets it learn the correct rule — excess energy
# in bins we are already using is somebody else's.
TILE_CH = 2

CLASSES = ["CLEAR", "BARRAGE", "SPOT", "SWEPT", "DRFM"]
NCLASS  = len(CLASSES)
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# ── Power model (dBm). Contributions sum in LINEAR power, not in dB. ────────
NOISE_DBM   = -96.0
NOISE_SIGMA = 2.5
FRIENDLY_DB = 38.0          # dB above the thermal floor
REF_WIDTH   = 3             # data bandwidth in bins (the un-spread carrier)
PASSBAND_PAD = 4            # extra bins each side the front end still sees (AGC)
NOTCH_THRESH = 10.0         # linear J/N above which a notch fires on a bin
SWEEP_SPAN  = NBINS         # a swept-spot jammer sweeps the WHOLE band — that is
                            # the entire point of it. You cannot hop out from
                            # under a sweeper, so frequency agility is not the
                            # answer here; blanking each pass is.
# Front-end desensitization. A high-power sweep passing through the passband
# saturates the receiver AGC, which then takes several frames to recover — so a
# transient jammer costs you far more than the dwell it actually occupies.
# Blanking (NOTCH) is precisely the countermeasure for this: you throw away the
# pass and keep your gain. This is what makes NOTCH the right answer to SWEPT
# rather than just a second-best HOP.
AGC_SAT_DB     = 25.0       # in-band J/N above which the front end saturates
AGC_RECOVER_N  = 14         # frames of degraded sensitivity after saturation
AGC_PENALTY_DB = 32.0       # effective noise rise while recovering — deep enough
                            # that an unblanked pass genuinely drops the link
JAM_DB = {"BARRAGE": 30.0, "SPOT": 46.0, "SWEPT": 44.0, "DRFM": 40.0}

# Jammer geometry. Received jam power falls off with range, so a distant emitter
# is both less effective AND harder to classify — which is what keeps the
# classifier honest. A model that is never wrong reads as fake to an EW
# audience, and it would remove the demo's best moment: watching a
# misclassification produce the wrong counter and cost the link.
JAM_REF_KM = 12.0
JAM_MIN_KM, JAM_MAX_KM = 8.0, 40.0

LINK_UP_DB   = 10.0         # SINR at or above this = link usable
LINK_DOWN_DB = 3.0          # below this = link down

DISPLAY_LO_DBM = -105.0     # waterfall colour-scale floor
DISPLAY_HI_DBM = -35.0      # waterfall colour-scale ceiling

# Defender counter modes
FIXED, HOP, NOTCH, WIDEBAND, BURN = "FIXED", "HOP", "NOTCH", "WIDEBAND", "BURN"
BURN_DB = 10.0              # burn-through: raise transmit power against barrage

BAND_MHZ = (BAND_HI_GHZ - BAND_LO_GHZ) * 1000.0
BIN_MHZ  = BAND_MHZ / NBINS

# How well a DRFM replica still correlates with the victim waveform under each
# countermeasure. A stored-replica jammer is coherent against a predictable
# carrier and nearly useless against one that changes shape every pulse.
DRFM_COHERENCE = {FIXED: 1.0, HOP: 0.80, NOTCH: 1.0, WIDEBAND: 0.03, BURN: 1.0}

# How the defender answers each PREDICTED jam class. Keyed by the classifier's
# verdict, never by ground truth — a misclassification therefore produces a
# genuinely wrong counter, which is the most instructive moment in the demo.
# Five jam classes, five genuinely different right answers — which is what makes
# the classifier load-bearing rather than decorative. Each of these is the
# measured best counter for its class (see the counter matrix in README.md), not
# an assertion: run tools/counter_matrix.py to re-derive it from the physics.
COUNTER_FOR = {
    "CLEAR":   FIXED,      # nothing to counter — don't burn agility for free
    "BARRAGE": BURN,       # brute force answers brute force; no waveform trick helps
    "SPOT":    HOP,        # frequency agility straight out from under the tone
    "SWEPT":   NOTCH,      # null the sweeper on each pass
    "DRFM":    WIDEBAND,   # a coherent replica can only be beaten by changing shape
}
COUNTER_LABEL = {
    FIXED:    "Hold — no countermeasure",
    HOP:      "Frequency agility",
    NOTCH:    "Adaptive null / notch",
    WIDEBAND: "Waveform change — spread spectrum",
    BURN:     "Burn-through — raise transmit power",
}


def _mw(dbm):
    return 10.0 ** (np.asarray(dbm, dtype=np.float64) / 10.0)


def _dbm(mw):
    return 10.0 * np.log10(np.maximum(mw, 1e-18))


def normalize_rows(rows_dbm: np.ndarray) -> np.ndarray:
    """dBm → 0..1, the scale used by BOTH the waterfall and the CNN input."""
    x = (rows_dbm - DISPLAY_LO_DBM) / (DISPLAY_HI_DBM - DISPLAY_LO_DBM)
    return np.clip(x, 0.0, 1.0)


def decimate(rows_norm: np.ndarray) -> np.ndarray:
    """NBINS → TILE_BINS by max-pooling.

    Max (not mean) on purpose: a SPOT tone is 3 bins wide out of 256, and mean
    pooling washes it into the floor — exactly the feature the CNN needs most.
    """
    r = np.atleast_2d(rows_norm)
    return r.reshape(r.shape[0], TILE_BINS, DECIM).max(axis=2)


def friendly_mask(bins, widths) -> np.ndarray:
    """Rows of 0/1 over NBINS marking the bins WE occupy — classifier channel 1.

    Decimated by the same max-pool as the power channel, so a carrier narrower
    than the decimation factor still survives as a 1.
    """
    rows = np.zeros((len(bins), NBINS), np.float32)
    for i, (b, w) in enumerate(zip(bins, widths)):
        half = int(w) // 2
        rows[i, np.arange(int(b) - half, int(b) + half + 1) % NBINS] = 1.0
    return rows


class RFScene:
    """One friendly emitter and one jammer, rendered a row at a time.

    step() advances the scene by a single waterfall row and returns
    (row_dbm[NBINS], sinr_db). SINR is a genuinely computed quantity — friendly
    power over in-band interference plus noise — not a scripted prop.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0

        # Friendly emitter
        self.friendly_bin = int(self.rng.integers(48, NBINS - 48))
        self.mode = FIXED
        self.hop_period = 6
        self.friendly_width = 3
        self._hist = deque(maxlen=32)       # past friendly bins, for DRFM replay

        # Jammer
        self.technique = "CLEAR"
        self.jam_bin = int(self.rng.integers(32, NBINS - 32))
        self.sweep_rate = float(self.rng.choice([-7.0, -5.5, 5.5, 7.0]))
        self.sweep_phase = float(self.rng.integers(0, NBINS))
        self.drfm_delay = int(self.rng.integers(1, 4))
        self.drfm_offset = int(self.rng.choice([-16, -13, -10, 10, 13, 16]))
        self.jam_range_km = JAM_REF_KM      # drives received jam power (and the map)
        self.extra_jammers = 0              # scenario button: 2nd emitter
        self._agc_recover = 0               # frames left of AGC desensitization

    # ── configuration the agents drive ─────────────────────────────────────
    def set_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        self.friendly_width = 11 if mode == WIDEBAND else 3
        if mode in (HOP, WIDEBAND):
            self._rehome()

    def set_technique(self, tech, target_bin=None):
        self.technique = tech
        if target_bin is not None:
            self.jam_bin = int(np.clip(target_bin, 4, NBINS - 5))
        if tech == "SWEPT":
            self.sweep_rate = float(self.rng.choice([-7.0, -5.5, 5.5, 7.0]))
        if tech == "DRFM":
            self.drfm_delay = int(self.rng.integers(1, 4))
            self.drfm_offset = int(self.rng.choice([-16, -13, -10, 10, 13, 16]))

    def _rehome(self):
        """Move the friendly carrier somewhere the jammer is not."""
        for _ in range(12):
            cand = int(self.rng.integers(24, NBINS - 24))
            if abs(cand - self.jam_bin) > 24:
                self.friendly_bin = cand
                return
        self.friendly_bin = int(self.rng.integers(24, NBINS - 24))

    def jam_power_offset_db(self) -> float:
        """Free-space spreading loss relative to the reference range."""
        return -20.0 * np.log10(max(self.jam_range_km, 1.0) / JAM_REF_KM)

    # ── one waterfall row ──────────────────────────────────────────────────
    def step(self):
        """Advance one waterfall row.

        Returns a dict: row_dbm, sinr_db, mbps, friendly_bin, jam_bin.
        """
        n = NBINS
        power = _mw(NOISE_DBM + self.rng.normal(0.0, NOISE_SIGMA, n))
        noise_ref = _mw(NOISE_DBM)

        if self.mode == HOP and self.t % self.hop_period == 0:
            self._rehome()

        # Friendly emitter. TOTAL radiated power is held constant, so spreading
        # across more bins genuinely trades per-bin SNR for jam resistance —
        # WIDEBAND is a real tradeoff, not a free win.
        half = self.friendly_width // 2
        fb = np.arange(self.friendly_bin - half, self.friendly_bin + half + 1) % n
        per_bin_db = FRIENDLY_DB - 10.0 * np.log10(self.friendly_width / REF_WIDTH)
        if self.mode == BURN:
            per_bin_db += BURN_DB
        f_power = _mw(NOISE_DBM + per_bin_db)
        power[fb] += f_power
        self._hist.append(self.friendly_bin)

        # Jammer. `jam` is what the receiver SEES (drives the waterfall);
        # `jam_eff` is what actually degrades the demodulator. They differ for
        # DRFM: the replica keeps radiating at full strength — you can still see
        # it — but once the victim changes waveform it stops correlating, so it
        # stops hurting. That is why "switch to spread spectrum" makes the ghost
        # stay on screen while the link recovers.
        jam = np.zeros(n)
        jam_eff = np.zeros(n)
        jam_coherent = np.zeros(n)    # DRFM — indistinguishable from us, unnotchable
        techs = [self.technique] + (["BARRAGE"] if self.extra_jammers else [])
        for tech in techs:
            if tech == "CLEAR":
                continue
            lvl = _mw(NOISE_DBM + JAM_DB[tech] + self.jam_power_offset_db())
            if tech == "BARRAGE":
                jam += lvl
                jam_eff += lvl
            elif tech == "SPOT":
                idx = np.arange(self.jam_bin - 1, self.jam_bin + 2) % n
                jam[idx] += lvl
                jam_eff[idx] += lvl
            elif tech == "SWEPT":
                self.sweep_phase += self.sweep_rate
                span = SWEEP_SPAN
                tri = abs(((self.sweep_phase / span) % 2.0) - 1.0)   # triangle 0..1
                c = int(tri * (span - 1)) % n
                # Footprint must exceed the per-frame step or a fast sweep
                # would alias straight over a narrow victim without ever
                # touching it.
                idx = np.arange(c - 4, c + 5) % n
                jam[idx] += lvl
                jam_eff[idx] += lvl
            elif tech == "DRFM":
                if len(self._hist) > self.drfm_delay:
                    src = self._hist[-1 - self.drfm_delay]
                    c = (src + self.drfm_offset) % n
                    idx = np.arange(c - 1, c + 2) % n
                    coh = lvl * DRFM_COHERENCE.get(self.mode, 1.0)
                    jam[idx] += lvl
                    jam_eff[idx] += coh
                    jam_coherent[idx] += coh

        power += jam

        # SINR over the occupied bins. NOTCH excises the worst offenders at the
        # demodulator — the energy stays on the display, it just stops counting.
        # Two distinct mechanisms, because they have two distinct counters:
        #
        #  1. POWER. Noise-like jamming (barrage/spot/swept) competes with the
        #     signal inside the occupied bins. Spreading buys processing gain
        #     against the narrowband cases; a wider front-end passband is what
        #     the AGC sees and can be saturated by.
        #  2. DECEPTION. A DRFM replica is a copy of our own waveform, so the
        #     correlator locks onto it wherever it sits in the band — its harm
        #     does not depend on spectral overlap at all, and no notch can
        #     separate it from us. Only changing the waveform (so the stored
        #     replica stops matching) defeats it, which is why WIDEBAND is the
        #     only counter that works on DRFM.
        pb = np.arange(self.friendly_bin - half - PASSBAND_PAD,
                       self.friendly_bin + half + PASSBAND_PAD + 1) % n
        interf_fb = jam_eff[fb]
        coh_fb = jam_coherent[fb]
        occupied = np.ones(len(fb), bool)
        notched = np.zeros(len(fb), bool)

        if self.mode == NOTCH:
            # A notch excises a band, but it cannot separate a coherent replica
            # from the victim it is copying, and any notch landing on our own
            # occupied bins takes our signal with it. Hence: great against a
            # transient sweep, useless against a tone parked on top of us.
            notched = (interf_fb - coh_fb) > noise_ref * NOTCH_THRESH

        live = occupied & ~notched
        sig_bins = int(np.count_nonzero(live))
        deception = jam_coherent.sum()          # already coherence-weighted

        if sig_bins == 0:                       # we notched ourselves off the air
            sinr = -20.0
        else:
            kept = np.where(notched, coh_fb, interf_fb)
            pg = REF_WIDTH / len(fb)

            # AGC saturation / recovery. NOTCH blanks the offending bins before
            # the front end sees them, so it never enters recovery.
            raw_jn = jam_eff[pb].sum() / max(noise_ref * len(pb), 1e-18)
            if self.mode != NOTCH and 10.0 * np.log10(max(raw_jn, 1e-18)) > AGC_SAT_DB:
                self._agc_recover = AGC_RECOVER_N
            noise_eff = noise_ref * REF_WIDTH
            if self._agc_recover > 0:
                self._agc_recover -= 1
                noise_eff *= 10.0 ** (AGC_PENALTY_DB / 10.0)

            sinr = 10.0 * np.log10(
                (f_power * sig_bins)
                / max(kept.sum() * pg + deception + noise_eff, 1e-18)
            )

        # Shannon capacity over the DATA bandwidth (not the spread bandwidth —
        # spreading does not buy you throughput). This is the number that makes
        # SINR legible to a non-RF audience: "the link just fell from 590 Mbps
        # to 6".
        mbps = REF_WIDTH * BIN_MHZ * np.log2(1.0 + 10.0 ** (min(sinr, 40.0) / 10.0))

        self.t += 1
        return {
            "row_dbm": _dbm(power),
            "sinr_db": float(sinr),
            "mbps": float(max(mbps, 0.0)),
            "friendly_bin": int(self.friendly_bin),
            "friendly_width": int(self.friendly_width),
            "jam_bin": int(self.jam_bin),
        }


# ── Training-data generation (same physics, used by ew_train.py) ────────────
# Fraction of training tiles that STRADDLE a technique change, labelled with the
# technique occupying the newest rows.
#
# This is not a nicety, it is the single largest error source at serve time. The
# tile is 64 rows = 3.2 s of history, but the jammer changes technique every 7 s
# and the operator buttons change it instantly — so for the ~3 s after every
# switch the live tile contains two techniques. Trained on homogeneous tiles
# only, the network scored 99% on settled tiles and 43% on straddling ones, and
# straddling tiles are 28% of everything it ever sees. Worse, it stayed
# confident while being wrong (mean 0.63 on errors), so the defender's
# confidence floor did not filter them out and it committed to a wrong counter
# right when the attack was freshest.
#
# Labelling the mixed tile with the NEWEST technique teaches the recency bias
# the defender actually needs: react to what the jammer is doing now, not to the
# average of the last three seconds.
P_TRANSITION = 0.30
# The defender re-decides every 0.75 s (15 rows), so a mode change inside a
# 64-row window is routine at serve time and must appear in training too.
P_MODE_CHANGE = 0.25
TRANS_MIN_ROWS, TRANS_MAX_ROWS = 20, 44      # rows the NEW technique occupies


# Probability the jammer is aimed AT the victim rather than elsewhere. Higher
# for the two narrowband techniques because that is what the live bandit does
# with them, and because a tone parked on our own carrier is the case the
# classifier most needs practice on.
P_TRACK = {"SPOT": 0.62, "DRFM": 0.62}
P_TRACK_DEFAULT = 0.5


def _rand_target(sc, rng, track: bool) -> int:
    """Where the jammer is aimed. TRACK puts it on the victim, +/- ESM error.

    Without the TRACK case the classifier never sees a SPOT tone parked on the
    carrier, and confidently calls it DRFM at serve time.
    """
    if track:
        t = sc.friendly_bin + int(rng.integers(-2, 3))
    else:
        t = int(rng.integers(16, NBINS - 16))                   # CENTER / RANDOM
    return int(np.clip(t, 4, NBINS - 5))


def _rand_mode(rng, bias_fixed: bool = False) -> str:
    """A defender posture. Every one the defender can actually hold must appear
    in training — BURN included. It raises the carrier by BURN_DB, which is a
    visibly brighter line the classifier would otherwise never have seen, and
    the defender enters it on every barrage.

    bias_fixed exists because of a feedback trap the live system falls into and
    an independently drawn mode does not reproduce. Misread a tracking spot tone
    as CLEAR and the defender holds FIXED; holding FIXED leaves the tone sitting
    on our carrier, where it is hardest to see; so the next tile is misread the
    same way. The running demo spends far more time in (narrowband, tracking,
    FIXED) than an independent draw ever visits — measured 57% spot recall live
    against 79% on the independent mix — so that state has to be sampled at
    something nearer the rate it actually occurs.
    """
    p = ([0.50, 0.18, 0.10, 0.10, 0.12] if bias_fixed
         else [0.32, 0.26, 0.14, 0.14, 0.14])
    return str(rng.choice([FIXED, HOP, NOTCH, WIDEBAND, BURN], p=p))


def make_tile(cls_name: str, rng: np.random.Generator,
              transition: bool = False) -> np.ndarray:
    """One 64x64 normalized classifier tile labelled `cls_name`.

    With transition=True the window opens on a DIFFERENT technique and switches
    to `cls_name` partway down, so the label describes the newest rows only.
    """
    sc = RFScene(seed=int(rng.integers(0, 2**31 - 1)))
    track = rng.random() < P_TRACK.get(cls_name, P_TRACK_DEFAULT)
    hard = track and cls_name in P_TRACK
    # Use set_mode() rather than poking .mode so the width/rehome side effects
    # match the live path exactly. Mode first: HOP and WIDEBAND rehome the
    # carrier, and a TRACK aim point has to be computed after that move.
    sc.set_mode(_rand_mode(rng, bias_fixed=hard))
    sc.jam_range_km = float(rng.uniform(JAM_MIN_KM, JAM_MAX_KM))

    n_new = TILE_ROWS
    if transition:
        others = [c for c in CLASSES if c != cls_name]
        prev = str(rng.choice(others))
        sc.set_technique(prev, target_bin=_rand_target(
            sc, rng, rng.random() < P_TRACK.get(prev, P_TRACK_DEFAULT)))
        n_new = int(rng.integers(TRANS_MIN_ROWS, TRANS_MAX_ROWS + 1))

    for _ in range(12):                       # warm-up so DRFM history exists
        sc.step()

    steps = []
    if transition:
        steps += [sc.step() for _ in range(TILE_ROWS - n_new)]
    sc.set_technique(cls_name, target_bin=_rand_target(sc, rng, track))

    # A defender mode change lands mid-window at serve time; it must here too.
    flip_at = (int(rng.integers(4, n_new - 4))
               if rng.random() < P_MODE_CHANGE and n_new > 12 else -1)
    for i in range(n_new):
        if i == flip_at:
            sc.set_mode(_rand_mode(rng, bias_fixed=hard))
        steps.append(sc.step())

    return stack_tile([o["row_dbm"] for o in steps],
                      [o["friendly_bin"] for o in steps],
                      [o["friendly_width"] for o in steps])


def stack_tile(rows_dbm, f_bins, f_widths) -> np.ndarray:
    """(TILE_CH, TILE_ROWS, TILE_BINS) — the exact array the network is handed.

    Both the trainer and the live server go through here, so train and serve
    cannot disagree about channel order, normalisation or decimation.
    """
    power = decimate(normalize_rows(np.stack(rows_dbm)))
    mine = decimate(friendly_mask(f_bins, f_widths))
    return np.stack([power, mine]).astype(np.float32)


def make_dataset(n: int, rng: np.random.Generator):
    """Returns (X, y, is_transition) — the flag lets the trainer score the two
    regimes separately instead of hiding the hard one inside one average."""
    y = rng.integers(0, NCLASS, n)
    trans = rng.random(n) < P_TRANSITION
    X = np.stack([make_tile(CLASSES[int(c)], rng, transition=bool(t))
                  for c, t in zip(y, trans)])
    return X.astype(np.float32), y.astype(np.int64), trans


# ── The two agents ─────────────────────────────────────────────────────────
class JammerAgent:
    """Epsilon-greedy contextual bandit over (technique x targeting strategy).

    Context is what a real ESM receiver could observe about the victim: whether
    the link is holding still, hopping, notching, or spread. Reward is how far
    the action drove the victim's SINR down, so the jammer genuinely learns
    which technique beats which countermeasure over the course of a demo.
    """

    # CLEAR is an action, not just an initial condition. A jammer that never
    # stops radiating is not a jammer anyone in an EW audience recognises —
    # emission control is the first thing you do when your jamming is not
    # working and you would rather not be geolocated. It also has two practical
    # effects on the demo: the CLEAR class stops being decorative (it was 0.3%
    # of live frames, so the "five threats, five counters" claim was really
    # four), and the link visibly HEALS between engagements instead of being
    # permanently under attack.
    TECHS   = ["BARRAGE", "SPOT", "SWEPT", "DRFM", "CLEAR"]
    TARGETS = ["TRACK", "CENTER", "RANDOM"]
    # BURN belongs here. It was missing, so every burn-through posture was
    # folded into the FIXED context and the bandit could not learn the one thing
    # it most needs to know — that barrage fails against burn-through. BURN is
    # the defender's most common posture by a wide margin, so this was the
    # majority of its experience being filed under the wrong key.
    CONTEXTS = [FIXED, HOP, NOTCH, WIDEBAND, BURN]

    # Cost of RADIATING each technique, subtracted from the reward: transmit
    # power burned, and how loudly it advertises the platform to our own ESM.
    # Without this, barrage is strictly dominant — it is the widest and (as
    # modelled) the highest total-power option — and the bandit collapses onto
    # it, which is exactly what it did: 68% of live frames were barrage and a
    # rep watching for two minutes never saw three of the five threats.
    # CLEAR is costed too, as the jammer's opportunity cost: a jammer that is
    # not jamming is failing its mission. Without that it was strictly free, so
    # once the defender started countering everything the bandit learned to shut
    # down permanently — correct behaviour, and a blank screen. Priced at 0.08 it
    # still beats futile barrage (0.22) but loses to cheap jamming, so the
    # emitter goes quiet when it is being beaten and comes straight back.
    TECH_COST = {"BARRAGE": 0.22, "SWEPT": 0.10, "CLEAR": 0.08,
                 "DRFM": 0.06, "SPOT": 0.04}
    # Repetition penalty. Costs alone just move the monopoly around: whichever
    # technique is cheapest wins every tie, and when the defender is countering
    # everything every technique ties at "didn't work". Barrage owned 68% of the
    # run before costs; spot owned 57% after them. A jammer that never varies is
    # also trivially predicted, so doctrine rotates — this prices that in, and
    # keeps all five signatures on screen for the rep talking over it.
    REPEAT_PENALTY = 0.06
    REPEAT_DECAY = 0.72

    def __init__(self, seed=None, eps=0.18):
        self.rng = np.random.default_rng(seed)
        self.eps = eps
        self.n_act = len(self.TECHS) * len(self.TARGETS)
        self.Q = np.zeros((len(self.CONTEXTS), self.n_act))
        self.N = np.zeros((len(self.CONTEXTS), self.n_act))
        self.last = None
        self._was_silent = False
        self._recent = {t: 0.0 for t in self.TECHS}     # decayed usage

    def _ctx(self, victim_mode):
        return self.CONTEXTS.index(victim_mode) if victim_mode in self.CONTEXTS else 0

    def act(self, victim_mode, friendly_bin):
        c = self._ctx(victim_mode)
        if self.rng.random() < self.eps or self.N[c].sum() < 3:
            a = int(self.rng.integers(self.n_act))
            explored = True
        else:
            rep = np.array([self._recent[self.TECHS[i // len(self.TARGETS)]]
                            for i in range(self.n_act)])
            a = int(np.argmax(self.Q[c] - self.REPEAT_PENALTY * rep))
            explored = False
        tech = self.TECHS[a // len(self.TARGETS)]
        # One epoch of EMCON at a time. The reward already discourages endless
        # silence, but a bandit is free to sit in a local optimum for minutes and
        # a demo table cannot afford that — this caps quiet stretches at a single
        # 7 s beat, which is also the pacing that lets an audience see the link
        # heal before the next attack lands.
        if tech == "CLEAR" and self._was_silent:
            a = int(self.rng.integers(self.n_act - len(self.TARGETS)))
            tech = self.TECHS[a // len(self.TARGETS)]
            explored = True
        self._was_silent = tech == "CLEAR"
        for t in self._recent:
            self._recent[t] *= self.REPEAT_DECAY
        self._recent[tech] += 1.0
        targ = self.TARGETS[a % len(self.TARGETS)]
        if targ == "TRACK":
            bin_ = friendly_bin + int(self.rng.integers(-2, 3))
        elif targ == "CENTER":
            bin_ = NBINS // 2
        else:
            bin_ = int(self.rng.integers(16, NBINS - 16))
        self.last = (c, a)
        return tech, int(np.clip(bin_, 4, NBINS - 5)), targ, explored

    def reward(self, sinr_db):
        """Lower victim SINR ⇒ higher reward, minus what the technique costs."""
        if self.last is None:
            return
        c, a = self.last
        tech = self.TECHS[a // len(self.TARGETS)]
        r = float(np.clip((20.0 - sinr_db) / 40.0, 0.0, 1.0)) - self.TECH_COST[tech]
        self.N[c, a] += 1
        self.Q[c, a] += (r - self.Q[c, a]) / self.N[c, a]

    def confidence(self):
        if self.last is None:
            return 0.0
        c, a = self.last
        return float(np.clip(self.Q[c, a], 0.0, 1.0))


class DefenderAgent:
    """Chooses a countermeasure from the CLASSIFIER's verdict.

    It never sees ground truth. When the CNN calls a DRFM replica a SPOT tone,
    the defender hops instead of changing waveform, the replica follows it, and
    SINR stays in the floor until the classifier recovers — visible, honest,
    and the best thing in the demo to talk over.
    """

    def __init__(self, confirm=2, min_conf=0.45):
        self.mode = FIXED
        self.last_pred = "CLEAR"
        self.last_conf = 0.0
        self.switches = 0
        self.confirm = confirm          # agreeing looks needed before committing
        self.min_conf = min_conf
        self._streak_cls = "CLEAR"
        self._streak_n = 0

    def act(self, pred_class, confidence):
        """Track-before-switch.

        A single noisy frame must not be able to flip the countermeasure — at
        20 Hz with a ~94% classifier that would thrash visibly during quiet
        periods. Require `confirm` consecutive agreeing looks above a confidence
        floor before committing, which is also what a real ESM system does
        before it acts on a track.
        """
        self.last_pred, self.last_conf = pred_class, confidence
        if pred_class == self._streak_cls:
            self._streak_n += 1
        else:
            self._streak_cls, self._streak_n = pred_class, 1

        if confidence < self.min_conf or self._streak_n < self.confirm:
            return self.mode, False

        want = COUNTER_FOR.get(pred_class, FIXED)
        if want != self.mode:
            self.mode = want
            self.switches += 1
            return want, True
        return want, False
