"""
The inspection lane: a simulated vessel discharging vehicles through a
drive-through camera portal, each one triaged automatically into

    Cleared            the model affirmatively saw no damage -> auto-release
    Needs inspection   minor/moderate finding, or unreadable output -> a human
    Hold               major/critical damage -> a human, urgently

and an inspector queue where a person makes the call the model only proposes.

Runs server-side so a page reload, or the launcher closing and reopening the
iframe, picks the run back up from GET /api/lane/state. One run at a time,
one worker: Ollama on the GB10 serves one request at a time (four concurrent
inspections measured 1.1x faster than sequential), so parallelism here would
only reorder the queue, not shorten it.

Two modes:
  live    every vehicle is a real model call, streamed token by token.
  replay  a run recorded on this device (data/replays/*.jsonl), played back
          faster than real time under a "REPLAY ... Not live." banner. It
          never calls the model, so the chrome's token and GPU meters stay
          honestly idle. Routing still goes through route(), so a replay can
          never disagree with how live mode would treat the same output.

Inference is injected (LaneManager(infer_triage=..., infer_report=...)), so
this module imports no HTTP client and the tests drive it with fakes.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

try:  # noqa: E402 -- see zgx_metrics import in main.py for why both styles exist
    from inspection import (DISPOSITION_PLAYBOOK, LANE_CLEARED, LANE_HOLD, LANE_REVIEW, LANES, PORTS,
                            classify_severity_checked, parse_area_percent, parse_damage_views,
                            parse_photo_findings, parse_vlm_output, resolve_disposition, route)
    from vin import WMI, simulated_vin
except ImportError:  # pragma: no cover
    from backend.inspection import (DISPOSITION_PLAYBOOK, LANE_CLEARED, LANE_HOLD, LANE_REVIEW, LANES,
                                    PORTS, classify_severity_checked, parse_area_percent,
                                    parse_damage_views, parse_photo_findings, parse_vlm_output,
                                    resolve_disposition, route)
    from backend.vin import WMI, simulated_vin

# Every vehicle on the vessel must be a Stellantis brand. tests/ enforce it
# against vehicles.json, so a stray photo cannot ship.
STELLANTIS_BRANDS = frozenset(WMI)

# A live run must take 2-3 minutes at most. Multi-view triage (2-4 portal views
# in one call) averaged 7.5 s per vehicle on the GB10 (5-13 s; damaged cars
# write longer findings), so 24 vehicles ran ~3:15 and the live vessel is 20.
LIVE_SIZE = 20
# Composition of a live run by the lane each vehicle was OBSERVED to take on
# this device (vehicles.json `observed_lane`). Used only to keep the lanes
# busy in a short run -- never shown, and never used to route. A lane with an
# empty pool (no vehicle currently reads "review") is topped up from the rest.
LIVE_MIX = {LANE_CLEARED: 13, LANE_REVIEW: 2, LANE_HOLD: 5}
# Replays play at 3x the recorded per-vehicle time.
REPLAY_SPEED = 3.0
# Pause between vehicles so the portal animation reads; not counted as
# inspection time anywhere.
PORTAL_GAP_S = 0.6

DEFAULT_VESSEL = "M/V Atlantic Meridian"  # fictional

Emit = Callable[[str], Awaitable[None]]
# infer([(path, angle), ...], port_key, on_token) -> (text, seconds, usage)
# Every portal view of one vehicle goes to the model in a single call.
Infer = Callable[[list[tuple[Path, str]], str, Emit], Awaitable[tuple[str, float, dict]]]


# -- Manifest ----------------------------------------------------------------

def load_vehicles(samples_dir: Path) -> list[dict]:
    """The vessel: one entry per vehicle, each with 2-4 views of the same car."""
    data = json.loads((Path(samples_dir) / "vehicles.json").read_text(encoding="utf-8"))
    return [v for v in data["vehicles"] if v.get("brand") in STELLANTIS_BRANDS and v.get("views")]


def draw_live(samples: list[dict], rng: random.Random, size: int = LIVE_SIZE,
              mix: dict[str, int] = LIVE_MIX) -> list[dict]:
    """A live vessel: `size` vehicles with every lane represented.

    Draws `mix[lane]` from each observed-lane pool, tops up from whatever is
    left, shuffles, and then makes sure a flagged vehicle appears within the
    first five so the audience sees routing early rather than a run of
    clean cars."""
    pools = {lane: [s for s in samples if s.get("observed_lane") == lane] for lane in mix}
    picked: list[dict] = []
    for lane, n in mix.items():
        pool = pools[lane][:]
        rng.shuffle(pool)
        picked += pool[:n]
    rest = [s for s in samples if s not in picked]
    rng.shuffle(rest)
    picked += rest[: max(0, size - len(picked))]
    picked = picked[:size]
    rng.shuffle(picked)
    early = (LANE_HOLD, LANE_REVIEW)
    for lane in early:
        if any(s.get("observed_lane") == lane for s in picked[:5]):
            continue
        j = next((i for i, s in enumerate(picked) if s.get("observed_lane") == lane), None)
        # Only displace a vehicle whose lane is not itself wanted up front, or
        # placing the review car could evict the hold car placed a moment ago.
        slots = [i for i in range(1, 5) if picked[i].get("observed_lane") not in early]
        if j is not None and slots:
            k = rng.choice(slots)
            picked[k], picked[j] = picked[j], picked[k]
    return picked


# -- Replay files ------------------------------------------------------------

def latest_replay(replay_dir: Path) -> Path | None:
    files = sorted(Path(replay_dir).glob("vessel_*.jsonl"))
    return files[-1] if files else None


def read_replay(path: Path) -> tuple[dict, list[dict]]:
    header, rows = {}, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "header":
            header = obj
        elif obj.get("type") == "vehicle":
            rows.append(obj)
    return header, rows


# -- Run state ---------------------------------------------------------------

@dataclass
class Vehicle:
    vid: str
    idx: int
    sample: dict                   # the vehicles.json entry
    vin: str
    state: str = "queued"          # queued -> scanning -> routed
    triage: dict | None = None
    lane: str | None = None
    reason: str | None = None
    decision: dict | None = None
    full_report: dict | None = None

    def to_dict(self) -> dict:
        s = self.sample
        return {
            "vid": self.vid,
            "idx": self.idx,
            "vehicle_id": s["id"],
            "file": s["views"][0]["file"],          # primary view, for thumbnails
            "views": [{"file": w["file"], "angle": w["angle"],
                       "credit": {"author": w.get("author"), "license": w.get("license"),
                                  "license_url": w.get("license_url"), "source": w.get("source")}}
                      for w in s["views"]],
            "consignment": s.get("consignment", "new"),
            "brand": s["brand"],
            "vehicle": s["vehicle"],
            "vin": self.vin,
            "vin_simulated": True,
            "state": self.state,
            "triage": self.triage,
            "lane": self.lane,
            "lane_label": LANES.get(self.lane) if self.lane else None,
            "reason": self.reason,
            "decision": self.decision,
            "full_report": self.full_report,
        }

    def view_paths(self, samples_dir: Path) -> list[tuple[Path, str]]:
        return [(Path(samples_dir) / w["file"], w["angle"]) for w in self.sample["views"]]

    @property
    def angles(self) -> list[str]:
        return [w["angle"] for w in self.sample["views"]]


@dataclass
class Run:
    run_id: str
    mode: str
    port: str
    vessel: str
    vehicles: list[Vehicle]
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    status: str = "running"        # running | finished | stopped | error
    replay: dict | None = None     # header of the replay being played
    error: str | None = None


def triage_from_text(text: str, seconds: float, usage: dict | None,
                     angles: list[str] | None = None) -> tuple[dict, str, str]:
    """Model text -> (triage record, lane, reason). Shared by live and replay.
    `angles` maps DAMAGE_VIEWS photo numbers back to the cameras that took them."""
    parsed = parse_vlm_output(text)
    severity, explicit = classify_severity_checked(parsed)
    disposition = resolve_disposition(parsed, severity)
    lane, reason = route(severity, explicit)
    # Seen on real clean photos: "DAMAGE_SEVERITY: MINOR" followed by panels
    # NONE, "No exterior damage observed" and RELEASE. The lane still sends it
    # to a person -- only an explicit NONE clears -- but the reason should say
    # why, because "the model contradicted itself, so it asked" is the point.
    if explicit and severity != "NONE" and (
            parsed["affected_panels"].strip().upper().startswith("NONE")
            or "no exterior damage" in parsed["damage_observations"].lower()):
        reason = f"Inconsistent answer: graded {severity} but reported no damage; a human decides"
    triage = {
        "severity": severity,
        "severity_explicit": explicit,
        "affected_panels": parsed["affected_panels"],
        "observations": parsed["damage_observations"],
        "damage_views": parse_damage_views(parsed, angles or []),
        "photo_findings": parse_photo_findings(text, angles or []),
        "disposition": disposition["key"],
        "disposition_label": disposition["label"],
        "raw": text,
        "seconds": round(seconds, 2),
        "tokens": (usage or {}).get("completion_tokens"),
    }
    return triage, lane, reason


class LaneManager:
    def __init__(self, samples_dir: Path, replay_dir: Path, infer_triage: Infer,
                 infer_report: Infer, model_label: str = ""):
        self.samples_dir = Path(samples_dir)
        self.replay_dir = Path(replay_dir)
        self.infer_triage = infer_triage
        self.infer_report = infer_report
        self.model_label = model_label
        self.run: Run | None = None
        self._task: asyncio.Task | None = None
        self._subs: set[asyncio.Queue] = set()
        # One model at a time: the worker and an inspector's on-demand full
        # report share it. asyncio.Lock is FIFO, so a report requested mid-run
        # goes right after the vehicle currently being scanned.
        self._model = asyncio.Lock()
        self._report_tasks: dict[str, asyncio.Task] = {}

    # -- pub/sub ---------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    async def _emit(self, event: dict) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # a stalled client; drop it, it re-syncs from /state
                self._subs.discard(q)

    # -- queries ---------------------------------------------------------
    def replay_info(self) -> dict | None:
        path = latest_replay(self.replay_dir)
        if not path:
            return None
        header, rows = read_replay(path)
        return {"file": path.name, "recorded_at": header.get("recorded_at"),
                "device": header.get("device"), "model": header.get("model"),
                "model_label": header.get("model_label"), "port": header.get("port"),
                "vehicles": len(rows)}

    def stats(self) -> dict:
        r = self.run
        if not r:
            return {}
        routed = [v for v in r.vehicles if v.lane]
        secs = [v.triage["seconds"] for v in routed if v.triage]
        flagged = [v for v in routed if v.lane in (LANE_REVIEW, LANE_HOLD)]
        decided = [v for v in routed if v.decision]
        agree = sum(1 for v in decided if v.decision["action"] == v.triage["disposition"])
        avg = sum(secs) / len(secs) if secs else None
        end = r.finished_at or time.time()
        return {
            "total": len(r.vehicles),
            "inspected": len(routed),
            "cleared": sum(1 for v in routed if v.lane == LANE_CLEARED),
            "review": sum(1 for v in routed if v.lane == LANE_REVIEW),
            "hold": sum(1 for v in routed if v.lane == LANE_HOLD),
            "awaiting_human": sum(1 for v in flagged if not v.decision),
            "decided": len(decided),
            "agree": agree,
            "override": len(decided) - agree,
            "elapsed_s": round(end - r.started_at, 1),
            # Recorded seconds in replay, live seconds otherwise -- either way
            # the real per-vehicle model time, never the accelerated playback.
            "avg_s": round(avg, 2) if avg else None,
            "per_hour": int(3600 / avg) if avg else None,
            "auto_clear_pct": round(100 * sum(1 for v in routed if v.lane == LANE_CLEARED) / len(routed))
            if routed else None,
        }

    def snapshot(self) -> dict:
        r = self.run
        base = {"replay_available": self.replay_info(), "live_size": LIVE_SIZE,
                "model_label": self.model_label, "ports": PORTS, "default_vessel": DEFAULT_VESSEL}
        if not r:
            return {**base, "run": None}
        return {**base, "run": {
            "run_id": r.run_id, "mode": r.mode, "port": r.port, "port_name": PORTS[r.port]["name"],
            "vessel": r.vessel, "status": r.status, "error": r.error, "replay": r.replay,
            "started_at": r.started_at, "finished_at": r.finished_at,
            "vehicles": [v.to_dict() for v in r.vehicles], "stats": self.stats(),
        }}

    def _vehicle(self, vid: str) -> Vehicle:
        for v in (self.run.vehicles if self.run else []):
            if v.vid == vid:
                return v
        raise KeyError(vid)

    # -- control ---------------------------------------------------------
    async def start(self, port: str, vessel: str, mode: str) -> dict:
        await self.stop(emit=False)
        if port not in PORTS:
            port = "baltimore"
        vessel = (vessel or DEFAULT_VESSEL).strip()[:60]
        rng = random.Random()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        replay_header = None
        replay_rows: list[dict] = []
        if mode == "replay":
            path = latest_replay(self.replay_dir)
            if not path:
                raise ValueError("No recorded replay on this device. Record one with scripts/record_vessel_replay.py.")
            replay_header, replay_rows = read_replay(path)
            by_id = {v["id"]: v for v in load_vehicles(self.samples_dir)}
            chosen = [by_id[row["id"]] for row in replay_rows if row["id"] in by_id]
            port = replay_header.get("port", port)
        else:
            mode = "live"
            chosen = draw_live(load_vehicles(self.samples_dir), rng)
        vehicles = [
            Vehicle(vid=f"{run_id}-{i:03d}", idx=i + 1, sample=s,
                    vin=simulated_vin(s["brand"], int(s.get("model_year") or 2025), rng))
            for i, s in enumerate(chosen)
        ]
        self.run = Run(run_id=run_id, mode=mode, port=port, vessel=vessel, vehicles=vehicles,
                       replay={"file": path.name, **{k: replay_header.get(k) for k in
                               ("recorded_at", "device", "model", "model_label")}} if mode == "replay" else None)
        await self._emit({"type": "run_started", "snapshot": self.snapshot()})
        worker = self._replay(replay_rows) if mode == "replay" else self._live()
        self._task = asyncio.create_task(worker)
        return self.snapshot()

    async def stop(self, emit: bool = True) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        if self.run and self.run.status == "running":
            self.run.status = "stopped"
            self.run.finished_at = time.time()
            for v in self.run.vehicles:
                if v.state == "scanning":
                    v.state = "queued"
            if emit:
                await self._emit({"type": "run_stopped", "stats": self.stats()})

    async def _finish(self) -> None:
        self.run.status = "finished"
        self.run.finished_at = time.time()
        await self._emit({"type": "run_finished", "stats": self.stats()})

    async def _scan(self, v: Vehicle) -> None:
        v.state = "scanning"
        await self._emit({"type": "vehicle_scanning", "vid": v.vid, "idx": v.idx})

    async def _routed(self, v: Vehicle, text: str, seconds: float, usage: dict | None) -> None:
        v.triage, v.lane, v.reason = triage_from_text(text, seconds, usage, v.angles)
        v.state = "routed"
        await self._emit({"type": "vehicle_routed", "vehicle": v.to_dict(), "stats": self.stats()})

    async def _live(self) -> None:
        r = self.run
        try:
            for v in r.vehicles:
                await self._scan(v)

                async def on_token(chunk: str, vid=v.vid) -> None:
                    await self._emit({"type": "token", "vid": vid, "text": chunk})

                async with self._model:
                    text, seconds, usage = await self.infer_triage(
                        v.view_paths(self.samples_dir), r.port, on_token)
                await self._routed(v, text, seconds, usage)
                await asyncio.sleep(PORTAL_GAP_S)
            await self._finish()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surface it; never leave the UI spinning
            r.status, r.error, r.finished_at = "error", str(e), time.time()
            await self._emit({"type": "error", "message": str(e), "stats": self.stats()})

    async def _replay(self, rows: list[dict]) -> None:
        by_id = {row["id"]: row for row in rows}
        try:
            for v in self.run.vehicles:
                row = by_id[v.sample["id"]]
                await self._scan(v)
                text = row["text"]
                play_s = max(0.4, row["seconds"] / REPLAY_SPEED)
                words = text.split(" ")
                step = play_s / max(1, len(words))
                for i, w in enumerate(words):
                    await self._emit({"type": "token", "vid": v.vid, "text": w + (" " if i < len(words) - 1 else "")})
                    await asyncio.sleep(step)
                await self._routed(v, text, row["seconds"], row.get("usage"))
                await asyncio.sleep(PORTAL_GAP_S / 2)
            await self._finish()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.run.status, self.run.error, self.run.finished_at = "error", str(e), time.time()
            await self._emit({"type": "error", "message": str(e), "stats": self.stats()})

    # -- the human ---------------------------------------------------------
    async def decide(self, vid: str, action: str, note: str = "") -> dict:
        v = self._vehicle(vid)
        if v.state != "routed":
            raise ValueError("This vehicle has not been inspected yet.")
        if action not in DISPOSITION_PLAYBOOK:
            raise ValueError(f"Unknown action {action!r}")
        v.decision = {"action": action, "label": DISPOSITION_PLAYBOOK[action]["label"],
                      "note": (note or "").strip()[:500],
                      "agrees_with_ai": action == v.triage["disposition"],
                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        await self._emit({"type": "decision", "vehicle": v.to_dict(), "stats": self.stats()})
        return v.to_dict()

    async def request_full_report(self, vid: str, port: str | None = None) -> dict:
        """Start (or join) the detailed report for one vehicle. Streams over the
        event bus as `full_token` then `full_report`; always live inference,
        including during a replay, and labelled as such."""
        v = self._vehicle(vid)
        if v.full_report and v.full_report.get("status") == "done":
            return v.to_dict()
        if vid not in self._report_tasks or self._report_tasks[vid].done():
            v.full_report = {"status": "queued"}
            self._report_tasks[vid] = asyncio.create_task(self._full_report(v, port or self.run.port))
        await self._emit({"type": "full_report", "vehicle": v.to_dict()})
        return v.to_dict()

    async def _full_report(self, v: Vehicle, port: str) -> None:
        async def on_token(chunk: str) -> None:
            await self._emit({"type": "full_token", "vid": v.vid, "text": chunk})
        try:
            async with self._model:
                v.full_report = {"status": "running"}
                await self._emit({"type": "full_report", "vehicle": v.to_dict()})
                text, seconds, usage = await self.infer_report(v.view_paths(self.samples_dir), port, on_token)
            parsed = parse_vlm_output(text)
            severity, explicit = classify_severity_checked(parsed)
            v.full_report = {
                "status": "done", "live": True, "seconds": round(seconds, 2),
                "severity": severity, "severity_explicit": explicit,
                "disposition": resolve_disposition(parsed, severity)["key"],
                "vehicle_type": parsed["vehicle_type"], "view_angle": parsed["view_angle"],
                "affected_panels": parsed["affected_panels"], "observations": parsed["damage_observations"],
                "damage_views": parse_damage_views(parsed, v.angles),
                "photo_findings": parse_photo_findings(text, v.angles),
                "affected_area_percent": parse_area_percent(parsed),
                "summary": parsed["inspector_summary"], "raw": text,
                "differs_from_triage": severity != (v.triage or {}).get("severity"),
            }
        except asyncio.CancelledError:
            raise
        except Exception as e:
            v.full_report = {"status": "error", "error": str(e)}
        await self._emit({"type": "full_report", "vehicle": v.to_dict()})

    # -- report ------------------------------------------------------------
    def csv(self) -> str:
        r = self.run
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["vessel", r.vessel if r else "", "port", PORTS[r.port]["name"] if r else "",
                    "mode", ("REPLAY - not live" if r and r.mode == "replay" else "live") if r else "",
                    "run", r.run_id if r else ""])
        w.writerow(["#", "vin (SIMULATED)", "brand", "vehicle", "consignment", "views", "ai_severity",
                    "ai_disposition", "damage_seen_by", "lane", "lane_reason", "ai_seconds", "human_decision",
                    "agrees_with_ai", "inspector_note", "decided_at", "detailed_report_severity", "photo_credits"])
        for v in (r.vehicles if r else []):
            t, d, f = v.triage or {}, v.decision or {}, v.full_report or {}
            credits = "; ".join(f"{w['angle']}: {w.get('author', '')} ({w.get('license', '')}) {w.get('source', '')}"
                                for w in v.sample["views"])
            w.writerow([v.idx, v.vin, v.sample["brand"], v.sample["vehicle"], v.sample.get("consignment", ""),
                        " ".join(v.angles), t.get("severity", ""), t.get("disposition", ""),
                        " ".join(t.get("damage_views") or []), LANES.get(v.lane, "not inspected"), v.reason or "",
                        t.get("seconds", ""), d.get("action", ""),
                        "" if not d else ("yes" if d.get("agrees_with_ai") else "no"), d.get("note", ""),
                        d.get("at", ""), f.get("severity", "") if f.get("status") == "done" else "", credits])
        return out.getvalue()
