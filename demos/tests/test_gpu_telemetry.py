"""Vendor-neutral GPU telemetry (demos/zgx_gpu_telemetry.py).

The AMD branch has no AMD hardware behind it yet, so it is tested against
captured/representative rocm-smi JSON. That is a real limitation and is called
out in the module: rocm-smi's key names have moved between ROCm releases, so
these fixtures MUST be re-cut from a real gfx1151 box via
scripts/capture_hw_fixture.sh. Until then these tests prove the parser handles
the shape, not that the shape is right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zgx_gpu_telemetry as tel  # noqa: E402

NVIDIA_CSV = "1275, 41.85, 44, 37\n"
NVIDIA_NA = "[N/A], [N/A], 35, [N/A]\n"   # what a GB10 returns for several fields

ROCM_JSON = json.dumps({
    "card0": {
        "Temperature (Sensor edge) (C)": "48.0",
        "Average Graphics Package Power (W)": "35.0",
        "GPU use (%)": "62",
        "sclk clock speed:": "(1100Mhz)",
        "VRAM Total Memory (B)": "137438953472",
        "VRAM Total Used Memory (B)": "21474836480",
    }
})


def test_nvidia_sample_is_parsed(monkeypatch):
    monkeypatch.setattr(tel, "_run", lambda cmd, timeout=4: NVIDIA_CSV)
    d = tel.read()
    assert d["vendor"] == "NVIDIA" and d["available"] is True
    assert (d["clock_mhz"], d["power_w"], d["temp_c"], d["util_pct"]) == (1275, 41.85, 44, 37)


def test_unreadable_fields_are_none_not_zero(monkeypatch):
    """A GB10 reports '[N/A]' for several fields. 0 W and 0% are values an
    audience will believe; None lets the UI render '--' and stay honest."""
    monkeypatch.setattr(tel, "_run", lambda cmd, timeout=4: NVIDIA_NA)
    d = tel.read()
    assert d["temp_c"] == 35
    assert d["clock_mhz"] is None and d["power_w"] is None and d["util_pct"] is None


def test_amd_sample_is_parsed(monkeypatch):
    def fake_run(cmd, timeout=4):
        return ROCM_JSON if cmd[0] == "rocm-smi" else None
    monkeypatch.setattr(tel, "_run", fake_run)
    d = tel.read()
    assert d["vendor"] == "AMD" and d["source"] == "rocm-smi"
    assert d["temp_c"] == 48
    assert d["power_w"] == 35.0
    assert d["util_pct"] == 62
    assert d["clock_mhz"] == 1100          # parsed out of "(1100Mhz)"
    assert d["mem_total_gb"] == 128.0
    assert d["mem_used_gb"] == 20.0


def test_nvidia_is_preferred_when_both_tools_answer(monkeypatch):
    """Both toolkits can be installed on one box; the NVIDIA probe runs first
    and must win rather than producing a mixed reading."""
    monkeypatch.setattr(tel, "_run",
                        lambda cmd, timeout=4: NVIDIA_CSV if cmd[0] == "nvidia-smi" else ROCM_JSON)
    assert tel.read()["vendor"] == "NVIDIA"


def test_no_gpu_tool_at_all_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setattr(tel, "_run", lambda cmd, timeout=4: None)
    d = tel.read()
    assert d["available"] is False
    assert d["vendor"] == "unknown"
    assert all(d[k] is None for k in
               ("clock_mhz", "power_w", "temp_c", "util_pct", "mem_used_gb", "mem_total_gb"))


def test_read_never_raises_even_on_garbage(monkeypatch):
    for junk in ("not csv at all", "{{{ broken json", "", "1,2"):
        monkeypatch.setattr(tel, "_run", lambda cmd, timeout=4, j=junk: j)
        assert tel.read()["available"] in (True, False)   # returned a dict, did not raise


def test_poller_recovers_instead_of_giving_up(monkeypatch):
    """ai-vision's poller returned permanently on the first FileNotFoundError,
    so a driver blip or a late-starting tool disabled telemetry for the whole
    session. This one re-probes every tick."""
    import zgx_gpu_telemetry
    calls = {"n": 0}

    def flaky(handle=None, include_system=False):
        calls["n"] += 1
        return dict(tel.EMPTY) if calls["n"] == 1 else {**tel.EMPTY, "available": True}

    monkeypatch.setattr(zgx_gpu_telemetry, "read", flaky)
    p = tel.GpuPoller(interval_s=0.01)
    p.start()
    deadline = __import__("time").monotonic() + 2
    while __import__("time").monotonic() < deadline:
        if p.get()["available"]:
            break
        __import__("time").sleep(0.01)
    assert p.get()["available"] is True, "poller never recovered after a failed first probe"


def test_poller_survives_a_read_that_raises(monkeypatch):
    """An unhandled exception in the poll loop kills the thread and freezes
    every meter at its last value, with nothing saying the feed is dead."""
    import zgx_gpu_telemetry
    state = {"boom": True}

    def sometimes(handle=None, include_system=False):
        if state["boom"]:
            raise RuntimeError("driver wedged")
        return {**tel.EMPTY, "available": True}

    monkeypatch.setattr(zgx_gpu_telemetry, "read", sometimes)
    p = tel.GpuPoller(interval_s=0.01)
    p.start()
    import time as _t
    _t.sleep(0.1)
    assert p.get()["available"] is False      # survived, reporting nothing
    state["boom"] = False
    deadline = _t.monotonic() + 2
    while _t.monotonic() < deadline and not p.get()["available"]:
        _t.sleep(0.01)
    assert p.get()["available"] is True, "poller thread died on the exception"


def test_read_carries_cpu_and_ram_when_asked(monkeypatch):
    """CPU/RAM used to exist only in pipeline-pulse, so every other demo's
    hardware panel could show a GPU and nothing else."""
    monkeypatch.setattr(tel, "_run", lambda cmd, timeout=4: NVIDIA_CSV)
    plain = tel.read()
    assert "cpu_pct" not in plain
    full = tel.read(include_system=True)
    assert {"cpu_pct", "ram_used_gb", "ram_pct"} <= set(full)
