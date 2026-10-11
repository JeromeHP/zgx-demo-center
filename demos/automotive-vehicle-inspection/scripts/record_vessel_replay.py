#!/usr/bin/env python3
"""
Record a vessel run for Replay mode, and refresh each photo's observed lane.

Runs every vehicle in sample-images/vehicles.json through the triage prompt --
all of its portal views in one call, exactly as the live lane does -- against
host Ollama, twice. Then:

  data/replays/vessel_<date>.jsonl   pass 1, verbatim: what the model said, how
                                     long it took, how many tokens. Replay mode
                                     plays this back under a "Not live" banner.
                                     Older recordings are removed.
  vehicles.json `observed_lane`      the lane a vehicle took, if both passes agree;
                                     "unstable" if they did not. Used only to
                                     balance the live draw (lane.draw_live), never
                                     shown and never used to route.

Re-run it whenever the triage prompt, the model, or a photo changes:
tests/test_vehicle_lane.py fails until you do, because a stale recording would
show the audience results a live run no longer produces.

    cd ~/demos/automotive-vehicle-inspection
    ~/miniforge3/envs/zgx/bin/python3 scripts/record_vessel_replay.py

Takes ~12 minutes for ~58 vehicles (two passes at ~5-6 s each). Stop any live demo
first: Ollama serves one request at a time and the timings would be wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # demos/, for zgx_hardware

from backend.inspection import (LANE_IMAGE_MAX, TRIAGE_MAX_TOKENS, format_prompt,  # noqa: E402
                                prepare_image_for_vlm, prompt_hash, triage_prompt)
from backend.lane import load_vehicles, triage_from_text  # noqa: E402

SAMPLES = HERE / "sample-images"
REPLAYS = HERE / "data" / "replays"


def photo_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def device_label() -> str:
    try:
        import zgx_hardware
        return zgx_hardware.detect().get("device_label") or "unknown device"
    except Exception:
        return "unknown device"


def triage(client: httpx.Client, base: str, model: str, vehicle: dict, port: str) -> tuple[str, float, dict]:
    """Same request the live lane makes: every view, portal order, then the prompt."""
    content = []
    for w in vehicle["views"]:
        with Image.open(SAMPLES / w["file"]) as im:
            b64 = prepare_image_for_vlm(im, max_dim=LANE_IMAGE_MAX)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": format_prompt(triage_prompt([w["angle"] for w in vehicle["views"]]), port)})
    body = {"model": model, "max_tokens": TRIAGE_MAX_TOKENS, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    t0 = time.time()
    r = client.post(f"{base}/chat/completions", json=body)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip(), time.time() - t0, data.get("usage") or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="vehicle-inspector")
    ap.add_argument("--model-label", default="Mistral Small 3.2 (24B)")
    ap.add_argument("--port", default="baltimore")
    ap.add_argument("--seed", type=int, default=20261006, help="discharge order of the recorded vessel")
    args = ap.parse_args()

    vehicles = load_vehicles(SAMPLES)
    order = vehicles[:]
    random.Random(args.seed).shuffle(order)
    passes: list[dict[str, dict]] = []
    with httpx.Client(timeout=300) as client:
        triage(client, args.base_url, args.model, order[0], args.port)  # load the model
        for n in (1, 2):
            results = {}
            for i, v in enumerate(order, 1):
                text, secs, usage = triage(client, args.base_url, args.model, v, args.port)
                tri, lane, _ = triage_from_text(text, secs, usage, [w["angle"] for w in v["views"]])
                results[v["id"]] = {"text": text, "seconds": round(secs, 2), "usage": usage, "lane": lane,
                                    "severity": tri["severity"], "damage_views": tri["damage_views"]}
                print(f"pass {n} [{i:2d}/{len(order)}] {v['id']:44s} {v['group']:6s} {lane:8s} "
                      f"{tri['severity']:9s} {len(v['views'])}v {secs:5.1f}s {','.join(tri['damage_views'])}", flush=True)
            passes.append(results)

    first, second = passes
    unstable = [f for f in first if first[f]["lane"] != second[f]["lane"]]

    REPLAYS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    out = REPLAYS / f"vessel_{stamp.strftime('%Y%m%d')}.jsonl"
    header = {
        "type": "header", "recorded_at": stamp.isoformat(timespec="seconds"), "device": device_label(),
        "model": args.model, "model_label": args.model_label, "prompt_hash": prompt_hash(args.model),
        "port": args.port, "vehicles": len(order), "unstable_between_passes": unstable,
        "photos": {w["file"]: photo_hash(SAMPLES / w["file"]) for v in order for w in v["views"]},
    }
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for v in order:
            r = first[v["id"]]
            fh.write(json.dumps({"type": "vehicle", "id": v["id"], "text": r["text"],
                                 "seconds": r["seconds"], "usage": r["usage"]}, ensure_ascii=False) + "\n")
    for old in REPLAYS.glob("vessel_*.jsonl"):
        if old != out:
            old.unlink()

    manifest_path = SAMPLES / "vehicles.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for v in manifest["vehicles"]:
        if v["id"] in first:
            v["observed_lane"] = "unstable" if v["id"] in unstable else first[v["id"]]["lane"]
            v["observed_with"] = f"{args.model} · {stamp.date().isoformat()}"
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    from collections import Counter
    counts = Counter(first[f]["lane"] for f in first)
    secs = [first[f]["seconds"] for f in first]
    by_group = Counter((v["group"], first[v["id"]]["lane"]) for v in order)
    print("by group/lane:", dict(by_group))
    print(f"\nwrote {out.relative_to(HERE)}  lanes={dict(counts)}  unstable={len(unstable)} {unstable}")
    print(f"triage seconds: mean {sum(secs)/len(secs):.2f}  min {min(secs):.2f}  max {max(secs):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
