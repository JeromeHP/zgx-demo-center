from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.replay import MAX_GAP_S, iter_replay_events, list_replays, save_replay


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def _event(seq: int, ts: str, type_: str = "step_started") -> dict:
    return {"run_id": "r1", "seq": seq, "ts": ts, "type": type_, "data": {}}


def test_iter_replay_events_first_event_has_no_gap(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_events(
        path,
        [
            _event(1, "2026-09-16T00:00:00.000000Z"),
            _event(2, "2026-09-16T00:00:02.000000Z"),  # under MAX_GAP_S so the cap doesn't mask the assertion
        ],
    )
    events = list(iter_replay_events(path, speed=1.0))
    assert events[0][0] == 0.0
    assert events[1][0] == pytest.approx(2.0, abs=0.01)


def test_iter_replay_events_speed_scales_gap(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_events(
        path,
        [
            _event(1, "2026-09-16T00:00:00.000000Z"),
            _event(2, "2026-09-16T00:00:02.000000Z"),
        ],
    )
    events = list(iter_replay_events(path, speed=2.0))
    assert events[1][0] == pytest.approx(1.0, abs=0.01)


def test_iter_replay_events_gap_capped_at_max(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_events(
        path,
        [
            _event(1, "2026-09-16T00:00:00.000000Z"),
            _event(2, "2026-09-16T00:05:00.000000Z"),  # 5-minute idle gap
        ],
    )
    events = list(iter_replay_events(path, speed=1.0))
    assert events[1][0] == MAX_GAP_S


def test_iter_replay_events_skips_blank_lines(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps(_event(1, "2026-09-16T00:00:00.000000Z")) + "\n\n" + json.dumps(_event(2, "2026-09-16T00:00:01.000000Z")) + "\n",
        encoding="utf-8",
    )
    events = list(iter_replay_events(path, speed=1.0))
    assert len(events) == 2


def test_iter_replay_events_preserves_event_payload(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_events(path, [_event(1, "2026-09-16T00:00:00.000000Z", type_="run_started")])
    _, event = next(iter_replay_events(path, speed=1.0))
    assert event["type"] == "run_started"
    assert event["seq"] == 1


def test_save_replay_copies_file(tmp_path):
    src = tmp_path / "src" / "events.jsonl"
    src.parent.mkdir()
    _write_events(src, [_event(1, "2026-09-16T00:00:00.000000Z")])
    dest = tmp_path / "replays" / "my-replay.jsonl"

    save_replay(src, dest)

    assert dest.is_file()
    assert dest.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_list_replays_empty_dir_returns_empty_list(tmp_path):
    assert list_replays(tmp_path / "does-not-exist") == []


def test_list_replays_lists_jsonl_files_sorted(tmp_path):
    (tmp_path / "b.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "not-a-replay.txt").write_text("x", encoding="utf-8")

    result = list_replays(tmp_path)

    assert [r["name"] for r in result] == ["a", "b"]
    assert all("bytes" in r for r in result)
