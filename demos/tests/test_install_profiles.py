"""Hardware-profile resolution in the demo registry.

install_plans.py stays pure data; the merge lives in demo_install_server. These
tests pin the two things that matter: the NVIDIA path must be byte-for-byte
what it was, and an AMD box must get the ONNX/numpy steps instead of
tensorrt-cu13 + pycuda -- which simply do not exist for ROCm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_install_server as server  # noqa: E402
import install_plans  # noqa: E402
import zgx_hardware  # noqa: E402

PLANS = install_plans.INSTALL_PLANS


@pytest.fixture
def as_profile(monkeypatch):
    def _set(profile):
        monkeypatch.setattr(zgx_hardware, "detect", lambda: {"profile": profile})
    return _set


def test_nvidia_plans_are_completely_unchanged(as_profile):
    """The regression control: this is the box the demos ship on today."""
    as_profile("nvidia-cuda")
    for demo_id, plan in PLANS.items():
        assert server.resolve_plan(plan) == plan, f"{demo_id} changed on the NVIDIA path"
        assert server.plan_is_supported(plan), f"{demo_id} became unsupported on NVIDIA"


def test_amd_swaps_the_tensorrt_installs(as_profile):
    as_profile("amd-rocm")
    for demo_id in ("vision", "spectrum"):
        steps = " ".join(s["cmd"] for s in server.resolve_plan(PLANS[demo_id])["steps"])
        assert "tensorrt" not in steps, f"{demo_id} still installs TensorRT on ROCm"
        assert "pycuda" not in steps, f"{demo_id} still installs pycuda on ROCm"
        assert "pypi.nvidia.com" not in steps


def test_amd_vision_installs_an_onnx_runtime(as_profile):
    as_profile("amd-rocm")
    steps = " ".join(s["cmd"] for s in server.resolve_plan(PLANS["vision"])["steps"])
    assert "onnxruntime" in steps, "the AMD vision path needs an ONNX Runtime to detect with"


def test_the_robot_demo_is_refused_on_amd_with_a_reason(as_profile):
    """Robotic-medical is the one demo left out of AMD scope: nvcr.io vLLM,
    the CUDA-only awq_marlin kernel, a whisper.cpp build pinned to
    CMAKE_CUDA_ARCHITECTURES=121, and a physical Reachy Mini. Attempting it on
    AMD would pull images that cannot run, and a mid-install failure buries the
    demos that did work -- so it is skipped with a reason instead."""
    as_profile("amd-rocm")
    assert not server.plan_is_supported(PLANS["medical"])
    assert PLANS["medical"]["requires"]["profiles"] == ["nvidia-cuda"]


def test_maritime_has_a_real_amd_path_not_a_refusal(as_profile):
    """Its default stack is nvcr.io vLLM + an FP8 checkpoint, neither of which
    works on gfx1151 -- but backend/main.py only speaks OpenAI-compatible
    /chat/completions, so on AMD it points at host Ollama instead."""
    as_profile("amd-rocm")
    assert server.plan_is_supported(PLANS["maritime"])
    steps = " ".join(s["cmd"] for s in server.resolve_plan(PLANS["maritime"])["steps"])
    assert "docker-compose.amd.yml" in steps
    assert "FP8" not in steps, "the FP8 checkpoint cannot be served on ROCm"
    assert "ollama pull" in steps


def test_ollama_backed_demos_run_on_every_profile(as_profile):
    """These talk to host Ollama over HTTP and touch no GPU API, which is the
    whole reason they are free to port."""
    for profile in ("nvidia-cuda", "amd-rocm", "cpu-only"):
        as_profile(profile)
        for demo_id in ("grant", "pii", "pharmacy", "ballistics", "vehicle"):
            assert server.plan_is_supported(PLANS[demo_id]), f"{demo_id} blocked on {profile}"


def test_a_demo_with_no_requires_runs_anywhere(as_profile):
    as_profile("cpu-only")
    assert server.plan_is_supported({"name": "x", "steps": []})


def test_install_order_is_preserved_by_filtering():
    """vision pip-installs the TensorRT wheel that spectrum and the Phase 4b
    engine build depend on, so filtering must never reorder the list."""
    import install
    ids = install.WEB_INSTALL_IDS
    assert ids.index("vision") < ids.index("spectrum")

    class Flags:
        skip_demos = {"pii"}
    runnable, _ = install._installable_demos(Flags())
    assert runnable == [d for d in ids if d in runnable], "filtering reordered the install list"
