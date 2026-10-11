"""An install step must install what the demo actually needs.

Grant Machine installed three packages on the reasoning that "the env already
carries 13 of its 14 pyproject deps". That was true of the ZGX Nano, whose
shared zgx env had been populated by its factory image and the other demos --
and false of every fresh machine, because install.py Phase 2 builds that env
from demos/requirements.txt, which is four packages.

On a clean box the demo therefore installed "successfully" and died at launch
with ModuleNotFoundError: No module named 'sse_starlette'. These tests pin the
general rule, not just that one import.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import install_plans  # noqa: E402

PLANS = install_plans.INSTALL_PLANS


def _pip_installed_names(plan) -> set[str]:
    """Distribution names any pip step in this plan installs."""
    names: set[str] = set()
    for step in plan["steps"]:
        for quoted in re.findall(r"'([^']+)'", step["cmd"]):
            if "pip install" not in step["cmd"]:
                continue
            names.add(re.split(r"[><=\[]", quoted)[0].strip().lower().replace("_", "-"))
        # Unquoted pins such as numpy==2.4.6
        if "pip install" in step["cmd"]:
            for tok in re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9._-]*)(?:==|>=)", step["cmd"]):
                names.add(tok.lower().replace("_", "-"))
    return names


def _shared_env_packages() -> set[str]:
    """What install.py Phase 2 puts in the zgx env — the ONLY safe baseline."""
    out = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.add(re.split(r"[><=\[]", line)[0].strip().lower().replace("_", "-"))
    return out


def test_grant_machine_installs_every_runtime_dependency():
    """The regression: a fresh box has only requirements.txt in the env."""
    pyproject = tomllib.loads((ROOT / "grant-machine" / "pyproject.toml").read_text(encoding="utf-8"))
    required = {
        re.split(r"[><=\[]", d)[0].strip().lower().replace("_", "-")
        for d in pyproject["project"]["dependencies"]
    }
    covered = _pip_installed_names(PLANS["grant"]) | _shared_env_packages()
    missing = sorted(required - covered)
    assert not missing, (
        "grant-machine would hit ModuleNotFoundError on a clean box for: "
        f"{missing}. The install step must not assume packages the shared env "
        "(demos/requirements.txt) does not provide."
    )


def test_sse_starlette_specifically_is_installed():
    """Named because this is the import that actually failed in the field."""
    assert "sse-starlette" in _pip_installed_names(PLANS["grant"])


def test_the_verify_step_checks_what_the_app_imports():
    """A verify step that skips a module cannot catch its absence."""
    verify = next(s for s in PLANS["grant"]["steps"] if "Verifying Grant Machine imports" in s["label"])
    for module in ("fastapi", "uvicorn", "httpx", "openai", "pydantic", "pypdf",
                   "rank_bm25", "sse_starlette", "markdownify", "bs4",
                   "jsonschema", "psutil", "yaml", "reportlab"):
        assert module in verify["cmd"], f"verify step does not import {module}"


def test_optional_steps_are_marked_not_inferred():
    """Exit codes are now honoured, so a genuinely best-effort step has to say
    so explicitly rather than relying on the installer ignoring failures."""
    searxng = next(s for s in PLANS["grant"]["steps"] if "SearXNG" in s["label"])
    assert searxng.get("optional") is True
    required = [s for s in PLANS["grant"]["steps"] if not s.get("optional")]
    assert len(required) == 3


@pytest.mark.parametrize("demo_id", sorted(PLANS))
def test_every_demo_declares_a_verify_or_has_no_pip_step(demo_id):
    """If a demo pip-installs into the shared env, something must confirm the
    imports resolve -- otherwise the failure surfaces at launch, far from its
    cause, which is exactly what happened here."""
    plan = PLANS[demo_id]
    installs = any("pip install" in s["cmd"] for s in plan["steps"])
    if not installs:
        pytest.skip(f"{demo_id} installs no Python packages")
    verifies = any("import" in s["cmd"] and "-c" in s["cmd"] for s in plan["steps"])
    assert verifies, f"{demo_id} pip-installs but never verifies the imports resolve"
