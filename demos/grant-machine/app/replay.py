"""Record/replay event streams. Recording is just EventBus(log_path=...)
writing events.jsonl as a run proceeds; this module copies a finished run's
log into data/replays/ and replays one with (capped) scaled timing."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

MAX_GAP_S = 3.0


def save_replay(run_events_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_events_path, dest_path)


def _parse_ts(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()


def iter_replay_events(path: Path, speed: float = 1.0) -> Iterator[tuple[float, dict]]:
    """Yields (seconds_to_sleep_before_this_event, event_dict), gaps capped at
    MAX_GAP_S / speed so idle waits don't stall a live demo fallback."""
    prev_ts: float | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            ts = _parse_ts(event["ts"])
            gap = 0.0 if prev_ts is None else min(MAX_GAP_S, (ts - prev_ts) / max(speed, 0.01))
            prev_ts = ts
            yield gap, event


def list_replays(replays_dir: Path) -> list[dict[str, object]]:
    if not replays_dir.is_dir():
        return []
    out = []
    for p in sorted(replays_dir.glob("*.jsonl")):
        out.append({"name": p.stem, "bytes": p.stat().st_size})
    return out
