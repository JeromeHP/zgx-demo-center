"""Re-derive the counter matrix in ew_engine.COUNTER_FOR from the physics.

The objective is LINK UPTIME (fraction of frames with SINR >= LINK_UP_DB),
not mean SINR. Burn-through raises SINR against everything, so on raw SINR it
would always "win"; what actually matters operationally is whether the link is
usable, with the cheaper countermeasure preferred when several keep it up.

Run:  $HOME/miniforge3/envs/zgx/bin/python3 tools/counter_matrix.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ew_engine as e

# Operational cost of holding each posture (agility churn, power, exposure).
COST = {e.FIXED: 0.0, e.NOTCH: 0.04, e.HOP: 0.06, e.WIDEBAND: 0.10, e.BURN: 0.35}
MODES = [e.FIXED, e.HOP, e.NOTCH, e.WIDEBAND, e.BURN]


# The jammer re-aims on its own decision epoch with ESM error — it does NOT
# track the victim frame-by-frame. That lag is the whole reason frequency
# agility works: the defender hops every 6 frames, the jammer can only re-aim
# every 30. Modelling a perfect tracker would make hopping useless and a static
# aim would make it invincible; this is the honest middle.
REAIM_FRAMES = 30
ESM_ERROR_BINS = 2


def uptime(tech, mode, seeds=8, rows=400, warm=60):
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        sc = e.RFScene(seed=seed)
        sc.set_technique(tech, 128)
        sc.friendly_bin = 128
        sc.set_mode(mode)
        vals = []
        for i in range(rows):
            if i % REAIM_FRAMES == 0:
                err = int(rng.integers(-ESM_ERROR_BINS, ESM_ERROR_BINS + 1))
                sc.jam_bin = int(np.clip(sc.friendly_bin + err, 4, e.NBINS - 5))
            vals.append(sc.step()["sinr_db"])
        out.append(np.mean([v >= e.LINK_UP_DB for v in vals[warm:]]))
    return float(np.mean(out))


def main():
    print(f"link uptime %% (SINR >= {e.LINK_UP_DB:.0f} dB), best = highest uptime, "
          f"ties broken by lower cost\n")
    print("  technique  " + "".join(f"{m[:5]:>9}" for m in MODES) + "      best   declared")
    bad = 0
    for tech in e.CLASSES:
        up = {m: uptime(tech, m) for m in MODES}
        best_up = max(up.values())
        best = min((m for m in MODES if up[m] >= best_up - 0.02), key=lambda m: COST[m])
        declared = e.COUNTER_FOR[tech]
        ok = best == declared
        bad += not ok
        print(f"  {tech:9s} " + "".join(f"{100*up[m]:8.0f}%" for m in MODES)
              + f"   {'OK ' if ok else 'MISMATCH '}{best:9s} {declared}")
    print()
    if bad:
        print(f"{bad} class(es) disagree with ew_engine.COUNTER_FOR — fix one or the other.")
    else:
        print("COUNTER_FOR matches the physics for all classes.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
