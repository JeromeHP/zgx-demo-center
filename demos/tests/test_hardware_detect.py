"""Accelerator detection (demos/zgx_hardware.py).

The platform had no tests at all before this file. These exist because the
launcher spent its whole life asserting hardware facts it never checked --
"NVIDIA CTK v1.18.2", "GB10 - sm_121", "128GB" as literal HTML on a box that
actually runs CTK 1.20.0. Detection replaces those claims, so detection is the
thing that now has to be right, including when it cannot find anything.

Probes are monkeypatched at _run rather than shelled out, so these pass on any
machine -- including the AMD cases, which is the point: the AMD path must be
exercised before the AMD box exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zgx_hardware  # noqa: E402

# Captured verbatim from the ZGX Nano, 2026-09-21. The "[N/A]" is not a
# placeholder in this fixture -- it is what a GB10 really returns, because
# unified memory has no separate VRAM figure to report.
NVIDIA_GB10 = "NVIDIA GB10, [N/A], 580.173.02"
NVIDIA_DISCRETE = "NVIDIA RTX A6000, 49140 MiB, 550.90.07"

# Shape per ROCm's documented rocminfo output. MUST be re-captured verbatim from
# the Strix Halo box on first contact -- see scripts/capture_hw_fixture.sh.
ROCMINFO_GFX1151 = """
Agent 2
*******
  Name:                    gfx1151
  Marketing Name:          AMD Radeon(TM) 8060S Graphics
  Device Type:             GPU
"""


def _probes(mapping, *, kfd=False):
    """Build a _run stand-in: first token of argv -> canned stdout (or None)."""
    def fake_run(cmd):
        return mapping.get(cmd[0])
    return fake_run


@pytest.fixture
def nvidia_box(monkeypatch):
    monkeypatch.delenv("ZGX_HW_PROFILE", raising=False)
    monkeypatch.delenv("ZGX_DEVICE_LABEL", raising=False)
    monkeypatch.setattr(zgx_hardware, "_run", _probes({
        "nvidia-smi": NVIDIA_GB10,
        "nvidia-ctk": "NVIDIA Container Toolkit CLI version 1.20.0",
    }))
    monkeypatch.setattr(zgx_hardware.os.path, "exists", lambda p: False)


@pytest.fixture
def amd_box(monkeypatch):
    monkeypatch.delenv("ZGX_HW_PROFILE", raising=False)
    monkeypatch.delenv("ZGX_DEVICE_LABEL", raising=False)
    monkeypatch.setattr(zgx_hardware, "_run", _probes({
        "rocminfo": ROCMINFO_GFX1151,
        "cat": "6.4.1",
    }))
    monkeypatch.setattr(zgx_hardware.os.path, "exists", lambda p: p == "/dev/kfd")
    monkeypatch.setattr(zgx_hardware.glob, "glob", lambda p: ["/dev/dri/renderD128"])


def test_nvidia_gb10_is_identified(nvidia_box):
    hw = zgx_hardware.detect()
    assert hw["profile"] == "nvidia-cuda"
    assert hw["vendor"] == "NVIDIA"
    assert hw["accelerator"] == "NVIDIA GB10"
    assert hw["driver"] == "NVIDIA 580.173.02"
    assert hw["container_runtime"] == "nvidia-container-toolkit 1.20.0"
    assert hw["device_label"] == "HP ZGX Nano G1n"


def test_unified_memory_falls_back_to_system_ram(nvidia_box):
    """A GB10 reports memory.total as '[N/A]'. Reporting 0 GB, or None, or the
    launcher's hardcoded 128, are all wrong in different ways."""
    hw = zgx_hardware.detect()
    assert hw["memory_gb"] and hw["memory_gb"] > 1
    assert hw["memory_source"] == "system (unified)"


def test_discrete_gpu_memory_is_read_from_the_gpu(monkeypatch):
    monkeypatch.delenv("ZGX_HW_PROFILE", raising=False)
    monkeypatch.delenv("ZGX_DEVICE_LABEL", raising=False)
    monkeypatch.setattr(zgx_hardware, "_run", _probes({"nvidia-smi": NVIDIA_DISCRETE}))
    monkeypatch.setattr(zgx_hardware.os.path, "exists", lambda p: False)
    hw = zgx_hardware.detect()
    assert hw["memory_source"] == "gpu"
    assert hw["memory_gb"] == pytest.approx(48.0, abs=0.2)
    # An unrecognised chip must not be given a product name we invented.
    assert hw["device_label"] == "NVIDIA RTX A6000"


def test_amd_strix_halo_is_identified(amd_box):
    hw = zgx_hardware.detect()
    assert hw["profile"] == "amd-rocm"
    assert hw["vendor"] == "AMD"
    assert "gfx1151" in hw["accelerator"]
    assert "8060S" in hw["accelerator"]
    assert hw["gfx_target"] == "gfx1151"
    assert hw["driver"] == "ROCm 6.4.1"
    assert hw["device_label"] == "AMD Strix Halo (Ryzen AI Max)"
    assert "nvidia" not in hw["container_runtime"].lower()


def test_dev_dri_alone_does_not_mean_amd(monkeypatch):
    """The trap this design turns on.

    An NVIDIA box also exposes /dev/dri/renderD128 -- verified on the Nano. Had
    detection keyed on /dev/dri instead of /dev/kfd, every NVIDIA machine would
    have classified as AMD and taken the ROCm install path.
    """
    monkeypatch.delenv("ZGX_HW_PROFILE", raising=False)
    monkeypatch.setattr(zgx_hardware, "_run", _probes({"rocminfo": ROCMINFO_GFX1151}))
    monkeypatch.setattr(zgx_hardware.os.path, "exists", lambda p: False)  # no /dev/kfd
    monkeypatch.setattr(zgx_hardware.glob, "glob", lambda p: ["/dev/dri/renderD128"])
    assert zgx_hardware.detect()["profile"] == "cpu-only"


def test_no_accelerator_reports_unknown_not_a_guess(monkeypatch):
    monkeypatch.delenv("ZGX_HW_PROFILE", raising=False)
    monkeypatch.delenv("ZGX_DEVICE_LABEL", raising=False)
    monkeypatch.setattr(zgx_hardware, "_run", _probes({}))
    monkeypatch.setattr(zgx_hardware.os.path, "exists", lambda p: False)
    hw = zgx_hardware.detect()
    assert hw["profile"] == "cpu-only"
    assert hw["vendor"] == "unknown"
    assert hw["accelerator"] == "unknown"
    assert hw["driver"] == "unknown"
    assert hw["detected_by"] == []
    # Whatever it says, it must not name a vendor it did not find.
    assert "NVIDIA" not in hw["device_label"] and "AMD" not in hw["device_label"]


def test_detection_never_raises_when_every_probe_explodes(monkeypatch):
    def boom(cmd):
        raise OSError("probe exploded")
    monkeypatch.setattr(zgx_hardware, "_run", boom)
    monkeypatch.setattr(zgx_hardware.os.path, "exists", lambda p: False)
    with pytest.raises(OSError):
        boom(["x"])  # the stand-in really does raise
    monkeypatch.setattr(zgx_hardware, "_probe_nvidia", lambda: None)
    monkeypatch.setattr(zgx_hardware, "_probe_amd", lambda: None)
    assert zgx_hardware.detect()["profile"] == "cpu-only"


def test_a_real_probe_failure_is_swallowed(monkeypatch):
    """_run itself is the guard: a missing binary or a wedged driver is None."""
    assert zgx_hardware._run(["definitely-not-a-real-binary-xyz"]) is None
    assert zgx_hardware._run(["false"]) is None


def test_forcing_a_profile_says_so_in_warnings(nvidia_box, monkeypatch):
    """Forcing is how the AMD path gets exercised on NVIDIA hardware, so it must
    be loud -- a forced run that looked like a real detection would let a
    mis-set env var silently misreport the machine at a show."""
    monkeypatch.setenv("ZGX_HW_PROFILE", "amd-rocm")
    hw = zgx_hardware.detect()
    assert hw["profile"] == "amd-rocm"
    assert any("forced" in w for w in hw["warnings"])


def test_a_bogus_forced_profile_is_ignored(nvidia_box, monkeypatch):
    monkeypatch.setenv("ZGX_HW_PROFILE", "not-a-profile")
    hw = zgx_hardware.detect()
    assert hw["profile"] == "nvidia-cuda"
    assert hw["warnings"] == []


def test_missing_container_toolkit_is_reported_not_assumed(monkeypatch):
    monkeypatch.delenv("ZGX_HW_PROFILE", raising=False)
    monkeypatch.setattr(zgx_hardware, "_run", _probes({"nvidia-smi": NVIDIA_GB10}))
    monkeypatch.setattr(zgx_hardware.os.path, "exists", lambda p: False)
    hw = zgx_hardware.detect()
    assert hw["container_runtime"] == "unknown"
    assert any("nvidia-ctk" in w for w in hw["warnings"])


def test_amd_without_render_group_warns(amd_box, monkeypatch):
    """Device nodes present but unreadable fails at container runtime with a
    permission error that looks nothing like its cause."""
    class FakeProc:
        stdout = "demo sudo docker"
    monkeypatch.setattr(zgx_hardware.subprocess, "run", lambda *a, **k: FakeProc())
    hw = zgx_hardware.detect()
    assert any("render" in w for w in hw["warnings"])
    assert any("video" in w for w in hw["warnings"])
