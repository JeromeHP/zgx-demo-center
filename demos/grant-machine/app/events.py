"""Event envelope + per-run in-memory event bus (Spec G). Phase 7 wraps this
with SSE; the orchestrator (Phase 6) only needs emit/persist."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class Event:
    run_id: str
    seq: int
    ts: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class EventBus:
    """Per-run event log with subscriber fanout. Optionally persists to
    runtime/runs/<run_id>/events.jsonl."""

    def __init__(self, run_id: str, log_path: Path | None = None):
        self.run_id = run_id
        self._seq = 0
        self._events: list[Event] = []
        self._subscribers: list[Callable[[Event], None]] = []
        self._log_path = log_path
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, type_: str, data: dict[str, Any] | None = None) -> Event:
        self._seq += 1
        event = Event(run_id=self.run_id, seq=self._seq, ts=_now_iso(), type=type_, data=data or {})
        self._events.append(event)
        if self._log_path:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
        for sub in list(self._subscribers):
            sub(event)
        return event

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        self._subscribers.append(callback)

    def events_since(self, seq: int) -> list[Event]:
        return [e for e in self._events if e.seq > seq]

    @property
    def events(self) -> list[Event]:
        return list(self._events)
