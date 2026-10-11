"""The AMD variants of the two demos that needed real porting.

Every assertion here corresponds to a bug that reached a real AMD box and was
found by running the thing, not by reading it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import demo_install_server as server  # noqa: E402
import install_plans  # noqa: E402
import zgx_hardware  # noqa: E402

NAVY = ROOT / "Fed-Navy-containerized-docker"


@pytest.fixture
def amd(monkeypatch):
    monkeypatch.setattr(zgx_hardware, "detect", lambda: {"profile": "amd-rocm"})


# ── maritime ────────────────────────────────────────────────────────────────

class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that understands Compose's own tags.

    `!reset` is Compose syntax, not YAML, and plain safe_load raises on it. The
    tag is load-bearing here -- it is the only thing that actually removes the
    base file's nvidia device reservation -- so the test has to parse it rather
    than the file being rewritten to keep the test happy.
    """


_ComposeLoader.add_constructor(
    "!reset", lambda loader, node: loader.construct_sequence(node)
    if isinstance(node, yaml.SequenceNode) else None)


def _merged_navy():
    base = yaml.load((NAVY / "docker-compose.yml").read_text(encoding="utf-8"), Loader=_ComposeLoader)
    amd = yaml.load((NAVY / "docker-compose.amd.yml").read_text(encoding="utf-8"), Loader=_ComposeLoader)
    return base, amd


def test_amd_overlay_targets_the_base_service_name():
    """The overlay first declared `maritime-demo`; the base service is
    `navy-demo`. Compose would have silently created a SECOND service and gone
    on running the NVIDIA one."""
    base, amd = _merged_navy()
    assert set(amd["services"]) <= set(base["services"])


def test_amd_maritime_uses_host_networking():
    """It talks to host Ollama. The base file maps ports instead -- correct for
    the NVIDIA stack, where vLLM runs INSIDE the container and VLLM_BASE_URL
    resolves to the container's own localhost. Keeping that while repointing
    the URL at host Ollama gives you the container's localhost, where nothing
    listens: the entrypoint hangs in its reachability probe and every request
    afterwards fails. Reproduced exactly that before fixing it.
    """
    _, amd = _merged_navy()
    svc = amd["services"]["navy-demo"]
    assert svc.get("network_mode") == "host", \
        "the AMD stack must share the host network to reach Ollama on 11434"


def test_amd_maritime_clears_the_nvidia_device_reservation():
    """Compose MERGES lists rather than replacing them, so `devices: []` left
    `driver: nvidia` in place and the stack still failed with 'could not select
    device driver'. Only !reset removes it."""
    raw = (NAVY / "docker-compose.amd.yml").read_text(encoding="utf-8")
    assert "!reset" in raw
    assert "devices: !reset" in raw


def test_maritime_health_probes_an_endpoint_both_backends_serve():
    """It probed {base}/health, which only vLLM serves. Against Ollama that is
    a 404, so a server that was up and answering was reported 'degraded'."""
    src = (NAVY / "backend" / "main.py").read_text(encoding="utf-8")
    assert "/models" in src.split("async def health_check")[1][:900]
    assert "/health\"" not in src.split("async def health_check")[1][:900]


def test_maritime_health_does_not_hardcode_engine_or_model():
    """Two literals reported vLLM and the FP8 checkpoint regardless of what was
    actually serving."""
    body = (NAVY / "backend" / "main.py").read_text(encoding="utf-8").split("async def health_check")[1][:1400]
    assert '"model": "Qwen3-VL-8B-Instruct"' not in body
    assert '"inference_engine": "vLLM",' not in body


# ── ai-vision ───────────────────────────────────────────────────────────────

def test_amd_vision_readiness_does_not_require_the_trt_engine(amd):
    """install.py Phase 4b deliberately never builds yolov8s.trt on ROCm, so
    requiring it pinned the card at 'Needs Setup' no matter how many times it
    was installed -- which is precisely what happened."""
    plan = server.resolve_plan(install_plans.INSTALL_PLANS["vision"])
    files = plan["readiness_check"]["files"]
    assert not any(f.endswith(".trt") for f in files), files
    assert any(f.endswith(".onnx") for f in files), "AMD vision runs the ONNX; readiness must check for it"


def test_status_endpoint_resolves_profile_overrides():
    """/status did not call resolve_plan, so readiness stayed vendor-blind even
    once the override existed."""
    src = (ROOT / "demo_install_server.py").read_text(encoding="utf-8")
    status_body = src.split('@app.get("/status/{demo_id}")')[1][:700]
    assert "resolve_plan(plan)" in status_body


def test_amd_vision_install_cannot_skip_opencv(amd):
    """Written as `A && B || C`, a failure of A (opencv+numpy) still ran C,
    leaving onnxruntime installed but no cv2 -- and vision_demo.py dying on
    `import cv2` with the pip error long scrolled away."""
    steps = server.resolve_plan(install_plans.INSTALL_PLANS["vision"])["steps"]
    combined = [s for s in steps if "opencv" in s["cmd"] and "onnxruntime" in s["cmd"]]
    assert not combined, "opencv and onnxruntime must install in separate steps"
    assert any("opencv" in s["cmd"] for s in steps)
    assert any("onnxruntime" in s["cmd"] for s in steps)


# ── maritime: the launch gate that actually broke it ────────────────────────

def _start_sh() -> str:
    return (NAVY / "start.sh").read_text(encoding="utf-8")


def test_the_fp8_model_gate_is_scoped_to_the_cuda_stack():
    """THE bug. The gate is a CUDA-stack requirement but ran unconditionally,
    and the AMD plan deliberately never downloads that checkpoint. So on every
    AMD box it tripped, `set -e` exited 1, compose was never invoked, and
    /demo_log reported "(no container output ...)". Reproduced locally by
    moving models/ aside before fixing."""
    src = _start_sh()
    gate = src.split("Qwen3-VL-8B-Instruct-FP8")[0]
    assert 'if [ "$STACK" = "cuda" ]' in gate, \
        "the FP8 model gate is not scoped to the CUDA stack"


def test_profile_is_decided_before_the_model_gate():
    """Ordering is the fix: the gate cannot be conditional on a stack that has
    not been selected yet."""
    src = _start_sh()
    assert src.index("STACK=") < src.index("Qwen3-VL-8B-Instruct-FP8")


def test_unknown_profile_fails_loudly_instead_of_defaulting_to_cuda():
    """An empty profile used to fall through to the NVIDIA stack in silence,
    which on AMD surfaces as an unrelated-looking device-driver error."""
    src = _start_sh()
    assert "Could not determine this machine's accelerator profile" in src
    assert 'PROFILE" != "nvidia-cuda"' not in src, \
        "the old not-nvidia predicate is back; it disagrees with install_plans.py"


def test_start_predicate_matches_the_install_plan_profile_keys():
    """start.sh selected the overlay for anything != nvidia-cuda while the plan
    overrode only amd-rocm, so a cpu-only box built the NGC image and ran the
    overlay."""
    src = _start_sh()
    for profile in ("nvidia-cuda", "amd-rocm", "cpu-only"):
        assert f'"{profile}"' in src, f"start.sh does not handle {profile} explicitly"


def test_previous_container_logs_are_saved_before_removal():
    """`docker rm -f` was the first thing the script did, destroying the only
    record of the previous failure -- which is why a second attempt reported
    nothing at all."""
    src = _start_sh()
    rm_at = src.index("docker rm -f maritime-surveillance-demo")
    assert "docker logs maritime-surveillance-demo" in src[:rm_at], \
        "the container is removed before its log is captured"


def test_the_two_stacks_use_distinct_image_tags():
    """Without an explicit image:, both stacks resolve to the same auto tag and
    `up` (launch_cmd passes no --build) reuses whichever image is there."""
    base = yaml.load((NAVY / "docker-compose.yml").read_text(encoding="utf-8"), Loader=_ComposeLoader)
    amd = yaml.load((NAVY / "docker-compose.amd.yml").read_text(encoding="utf-8"), Loader=_ComposeLoader)
    b = base["services"]["navy-demo"].get("image")
    a = amd["services"]["navy-demo"].get("image")
    assert b and a and b != a, f"stacks share an image tag: base={b} amd={a}"


def test_healthcheck_identifies_this_service_not_just_the_port():
    """Under network_mode: host, port 8000 is shared with pharmacy and pii, so
    a bare `curl -f /api/health` can pass against THEIR endpoint."""
    base = yaml.load((NAVY / "docker-compose.yml").read_text(encoding="utf-8"), Loader=_ComposeLoader)
    test = " ".join(base["services"]["navy-demo"]["healthcheck"]["test"])
    assert "maritime-surveillance" in test, "healthcheck does not verify it reached THIS service"
    assert "service" in (NAVY / "backend" / "main.py").read_text(encoding="utf-8")


def test_maritime_has_explicit_readiness_per_profile(amd):
    """The /status fallback checks `any(models_dir.iterdir())`, and the compose
    bind mount makes Docker create an empty models/ -- pinning the card at
    "Needs Setup" forever on AMD."""
    import copy
    nvidia = install_plans.INSTALL_PLANS["maritime"]["readiness_check"]
    assert any("Qwen3-VL" in f for f in nvidia["files"]), "CUDA readiness should want the checkpoint"
    resolved = server.resolve_plan(install_plans.INSTALL_PLANS["maritime"])["readiness_check"]
    assert not any("Qwen3-VL" in f for f in resolved["files"]), \
        "AMD readiness must not require a checkpoint that path never downloads"
    assert resolved["ollama_model"] == "qwen2.5vl:7b"


def test_an_empty_ollama_model_means_no_requirement():
    """check_ollama_model("") returns True for any loaded model via
    startswith(""), and False when Ollama is merely down -- so a demo that
    needs no Ollama would report Needs Setup if an unrelated service stopped."""
    import asyncio

    import zgx_demo_lib
    res = asyncio.run(zgx_demo_lib.check_files_and_model(["/etc/hostname"], "", None))
    assert res["model_ok"] is True
    assert not any(k.startswith("ollama_") for k in res["checks"]), \
        "an empty model name still produced an Ollama check"


# ── status flapping ─────────────────────────────────────────────────────────

def test_an_unreachable_ollama_does_not_uninstall_a_demo():
    """Cards flashed "Needs Setup" and reverted a second later.

    check_ollama_model returned False for three different situations -- model
    absent, Ollama slow past a 2s timeout, Ollama down -- and /status computes
    installed as files_ok AND model_ok. So one slow /api/tags, which is routine
    while Ollama cold-loads a 42GB model or is saturated mid-inference, flipped
    a working demo's card to uninstalled until the next poll flipped it back.
    """
    import asyncio

    import httpx

    import zgx_demo_lib

    async def scenario():
        # A real answer first, so the cache holds something definitive.
        await zgx_demo_lib.check_files_and_model(["/etc/hostname"], "qwen2.5vl:7b", None)

        class Stalled:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, *a, **k): raise httpx.ConnectTimeout("busy")

        real = httpx.AsyncClient
        httpx.AsyncClient = Stalled
        try:
            cached = await zgx_demo_lib.check_files_and_model(["/etc/hostname"], "qwen2.5vl:7b", None)
            cold = await zgx_demo_lib.check_files_and_model(["/etc/hostname"], "never-seen:1b", None)
        finally:
            httpx.AsyncClient = real
        return cached, cold

    cached, cold = asyncio.run(scenario())
    assert cached["installed"] is True, "a stalled Ollama uninstalled a demo we had already verified"
    assert cold["installed"] is True, "no evidence either way must not mean 'not installed'"
    assert cold["model_state"] == "unknown"


def test_a_genuinely_absent_model_is_still_reported():
    """The fix must not blind the check -- absent is still absent."""
    import asyncio

    import zgx_demo_lib
    res = asyncio.run(zgx_demo_lib.check_files_and_model(
        ["/etc/hostname"], "definitely-not-a-real-model:999b", None))
    assert res["model_state"] == "absent"
    assert res["installed"] is False


def test_the_launcher_does_not_repaint_on_an_unknown_check():
    launcher = (ROOT / "zgx-demo-launcher.html").read_text(encoding="utf-8")
    assert "model_state === 'unknown'" in launcher, \
        "the launcher still repaints cards on an inconclusive status check"
