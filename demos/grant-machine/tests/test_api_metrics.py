"""/api/metrics — the shared Demo Center chrome's token feed.

Every test here exercises the endpoint WITH runs present. The first version of
this endpoint called `run.event_bus.events()` on what is a property, not a
method, and raised TypeError on every request during a run. It was not caught
because the only check made was against an idle server, where there is no
active run, the loop never executes and the bug is unreachable. A test that
cannot reach the code it is meant to cover is not a test.
"""

from __future__ import annotations

import app.main as main_module
from app.events import EventBus


class _FakeRun:
    def __init__(self, run_id: str, metrics: dict | None):
        self.run_id = run_id
        self.event_bus = EventBus(run_id)
        self.event_bus.emit("run_started", {})
        if metrics is not None:
            self.event_bus.emit("metrics", metrics)
        self.event_bus.emit("tool_call", {})   # metrics must be found when not last


def _install(monkeypatch, runs, active_id=None):
    monkeypatch.setattr(main_module.run_manager, "all_runs", lambda: runs)
    monkeypatch.setattr(main_module.run_manager, "get_active_id", lambda: active_id)


def test_totals_a_single_active_run(monkeypatch):
    run = _FakeRun("r1", {"prompt_tokens": 1000, "completion_tokens": 250,
                          "llm_calls": 4, "decode_tps_recent": 51.2})
    _install(monkeypatch, [run], active_id="r1")
    m = main_module.api_metrics()
    assert m["total_tokens"] == 1250
    assert m["requests"] == 4
    assert m["tps"] == 51.2


def test_the_total_survives_the_run_finishing(monkeypatch):
    """The bug this endpoint was rewritten for: reading only the ACTIVE run
    meant the counter climbed all through a run and dropped to zero the moment
    it completed -- exactly when a presenter turns round to point at it."""
    run = _FakeRun("r1", {"prompt_tokens": 599358, "completion_tokens": 14447, "llm_calls": 41})
    _install(monkeypatch, [run], active_id=None)     # finished: no active run
    m = main_module.api_metrics()
    assert m["total_tokens"] == 613805, "the session total reset when the run ended"
    assert m["tps"] is None, "a finished run's last rate is history, not status"


def test_totals_accumulate_across_several_runs(monkeypatch):
    runs = [_FakeRun("r1", {"prompt_tokens": 100, "completion_tokens": 10, "llm_calls": 1}),
            _FakeRun("r2", {"prompt_tokens": 200, "completion_tokens": 20, "llm_calls": 2})]
    _install(monkeypatch, runs, active_id=None)
    m = main_module.api_metrics()
    assert m["total_tokens"] == 330
    assert m["requests"] == 3


def test_a_run_with_no_metrics_event_is_skipped(monkeypatch):
    runs = [_FakeRun("r1", None),
            _FakeRun("r2", {"prompt_tokens": 5, "completion_tokens": 5, "llm_calls": 1})]
    _install(monkeypatch, runs, active_id=None)
    assert main_module.api_metrics()["total_tokens"] == 10


def test_estimated_is_sticky_across_runs(monkeypatch):
    runs = [_FakeRun("r1", {"prompt_tokens": 10, "completion_tokens": 1, "tokens_estimated": True}),
            _FakeRun("r2", {"prompt_tokens": 10, "completion_tokens": 1})]
    _install(monkeypatch, runs, active_id=None)
    assert main_module.api_metrics()["estimated"] is True


def test_no_runs_at_all_reports_zeroes_not_an_error(monkeypatch):
    _install(monkeypatch, [], active_id=None)
    m = main_module.api_metrics()
    assert m["total_tokens"] == 0 and m["tps"] is None


def test_payload_matches_what_the_chrome_reads():
    """demos/ui/zgx-ui.js hides the panel unless total_tokens is a number."""
    for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                "requests", "tps", "estimated"):
        assert key in main_module.api_metrics()


def test_grant_machines_own_token_meter_is_gone():
    """Two token counters used to sit directly above one another showing
    different numbers -- the chrome's session total and this run's."""
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    body = html.split("<!-- zgx-ui:begin")[0]
    assert 'id="meter-tokens"' not in body
    assert 'id="meter-elapsed"' in body, "Agent meter should stay; nothing else has it"
    assert 'id="meter-cost"' in body, "Cloud equiv. should stay"
