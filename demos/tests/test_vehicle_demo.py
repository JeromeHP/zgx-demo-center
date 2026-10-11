"""Vehicle Inspection: the upstream assumptions that would break this suite stay gone.

Vendored from a repo written for a different box. Each test pins one thing that
was true upstream and wrong here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "automotive-vehicle-inspection"
sys.path.insert(0, str(ROOT))

import install_plans  # noqa: E402

PLAN = install_plans.INSTALL_PLANS["vehicle"]
COMPOSE = yaml.safe_load((DEMO / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICE = next(iter(COMPOSE["services"].values()))


def test_runs_on_host_networking_with_no_published_ports():
    """Docker-published ports bypass ufw, so lockdown-tailnet.sh could not cover
    them; host networking also means the app's own bind IS the host port."""
    assert SERVICE.get("network_mode") == "host"
    assert "ports" not in SERVICE


def test_binds_its_registered_port_and_nothing_upstream_used():
    env = dict(e.split("=", 1) for e in SERVICE["environment"])
    assert int(env["PORT"]) == PLAN["port"] == 8097
    for path in ("Dockerfile", "docker-compose.yml", "start.sh", "backend/main.py"):
        text = (DEMO / path).read_text(encoding="utf-8")
        # 8095 is Grant Machine; 8090 (upstream's in-container vLLM) is Medical.
        assert "8095" not in text and "8090" not in text, path


def test_no_gpu_reservation_and_no_vllm():
    assert "deploy" not in SERVICE
    assert "nvcr.io" not in (DEMO / "Dockerfile").read_text(encoding="utf-8")


def test_one_model_name_everywhere():
    """Compose, the backend default, readiness and the install steps must agree,
    or the card reads "installed" for a model the demo never calls.

    The served name is `vehicle-inspector`: mistral-small3.2 re-registered with
    an 8k context (Modelfile.vehicle), because the default 131k context held
    36.9 GB resident on the GB10 against 15.8 GB. It must stay a non-thinking
    model -- thinking ate the whole token budget on Ollama and left every report
    empty (README, "Model")."""
    env = dict(e.split("=", 1) for e in SERVICE["environment"])
    model = env["OLLAMA_MODEL"]
    assert model == "vehicle-inspector"
    assert "Mistral Small 3.2" in env["MODEL_LABEL"]
    modelfile = (DEMO / "Modelfile.vehicle").read_text(encoding="utf-8")
    assert "FROM mistral-small3.2" in modelfile and "num_ctx 8192" in modelfile
    assert PLAN["readiness_check"]["ollama_model"] == model
    cmds = [s["cmd"] for s in PLAN["steps"]]
    assert "ollama pull mistral-small3.2" in cmds
    assert any(f"ollama create {model} -f Modelfile.vehicle" in c for c in cmds)


def test_healthcheck_identifies_this_service_not_just_the_port():
    test = " ".join(SERVICE["healthcheck"]["test"])
    assert "grep -q vehicle-inspection" in test
    assert '"service": "vehicle-inspection"' in (DEMO / "backend/main.py").read_text(encoding="utf-8")


def test_start_script_never_prompts():
    """The launcher runs it through an SSE stream; a prompt hangs the launch."""
    text = (DEMO / "start.sh").read_text(encoding="utf-8")
    assert "read -p" not in text and "read -r" not in text


def test_restart_policy_does_not_hide_failures():
    assert SERVICE.get("restart") == "no"
