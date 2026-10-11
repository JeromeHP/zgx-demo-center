"""Shared token accounting (demos/zgx_metrics.py).

Written against the two server shapes the suite actually talks to, and against
the failure this replaces: pipeline-pulse displayed a client-side chars/4
estimate as "tokens processed locally" while Ollama returned the real count in
a chunk three lines away that the code threw away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import zgx_metrics as zm  # noqa: E402

OLLAMA = {"prompt_eval_count": 2384, "eval_count": 139,
          "eval_duration": 1_800_000_000, "prompt_eval_duration": 300_000_000,
          "response": "..."}
OPENAI = {"usage": {"prompt_tokens": 2384, "completion_tokens": 139, "total_tokens": 2523}}


def test_ollama_native_shape():
    u = zm.normalize(OLLAMA)
    assert (u["prompt"], u["completion"], u["total"]) == (2384, 139, 2523)
    assert u["estimated"] is False


def test_openai_shape():
    u = zm.normalize(OPENAI, wall_s=2.0)
    assert (u["prompt"], u["completion"], u["total"]) == (2384, 139, 2523)
    assert u["estimated"] is False


def test_ollamas_own_duration_is_preferred_over_wall_clock():
    """eval_count / eval_duration is a TRUE decode rate. grant-machine divides
    by wall time for the whole call including prompt processing -- a
    simplification its own PROGRESS.md records at line 334. When the server
    hands over the better number, use it."""
    u = zm.normalize(OLLAMA, wall_s=60.0)          # a wildly wrong wall time
    assert u["tps"] == pytest.approx(139 / 1.8, abs=0.2)
    assert u["tps"] > 70, "fell back to wall-clock when a real duration was available"


def test_wall_clock_is_used_only_when_there_is_no_duration():
    u = zm.normalize(OPENAI, wall_s=2.0)
    assert u["tps"] == pytest.approx(69.5, abs=0.1)


def test_missing_counts_are_estimated_and_say_so():
    u = zm.normalize({}, fallback_text="x" * 350)
    assert u["estimated"] is True
    assert u["completion"] == 100


def test_tps_is_none_not_zero_when_it_cannot_be_computed():
    """0 tok/s is a reading an audience believes; absence is not."""
    assert zm.normalize({"prompt_eval_count": 5, "eval_count": 0})["tps"] is None
    assert zm.normalize(OPENAI)["tps"] is None      # no duration, no wall time


def test_normalize_never_raises_on_junk():
    for junk in (None, {}, {"usage": None}, {"eval_count": "nonsense"},
                 {"usage": {"prompt_tokens": None, "completion_tokens": None}}):
        assert isinstance(zm.normalize(junk), dict)


def test_ledger_accumulates_across_calls_and_shapes():
    led = zm.TokenLedger()
    led.record(OLLAMA)
    led.record(OPENAI, wall_s=1.0)
    snap = led.snapshot()
    assert snap["total_tokens"] == 2523 * 2
    assert snap["requests"] == 2
    assert snap["estimated"] is False


def test_estimated_is_sticky_across_the_session():
    """One estimated call taints the session total, so the UI's leading "≈"
    stays on rather than implying the whole figure was measured."""
    led = zm.TokenLedger()
    led.record(OLLAMA)
    led.record({}, fallback_text="guessed")
    assert led.snapshot()["estimated"] is True


def test_a_zero_token_call_does_not_wipe_the_last_known_rate():
    led = zm.TokenLedger()
    led.record(OLLAMA)
    good = led.snapshot()["tps"]
    led.record({"prompt_eval_count": 3, "eval_count": 0})
    assert led.snapshot()["tps"] == good


def test_snapshot_matches_the_shape_the_chrome_expects():
    """demos/ui/zgx-ui.js hides the panel unless total_tokens is a number."""
    snap = zm.TokenLedger().snapshot()
    assert isinstance(snap["total_tokens"], int)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "requests", "tps", "estimated"):
        assert key in snap
