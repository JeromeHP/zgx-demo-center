"""The port map's hand-maintained copies must agree.

KNOWN-ISSUES.md, "The port map is maintained by hand in three places": the
launcher holds PORT_CONFLICTS / PORT_PATHS / HEALTH_CHECK_PATHS near the top and
a second port+path per card in the DEMOS array, and install_plans.py holds a
third. Changing a port in one place silently breaks either readiness polling or
the URL the Launch button opens. Nothing enforced agreement; this does, for
every demo, until one map is served from the install API.

It also pins the allocation rules RUNBOOK.md states: one port per demo except
the declared :8000 group, and no demo on an infrastructure port.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import install_plans  # noqa: E402

PLANS = install_plans.INSTALL_PLANS
HTML = (ROOT / "zgx-demo-launcher.html").read_text(encoding="utf-8")

# maritime / pharmacy / pii share :8000 by design (RUNBOOK.md); the launcher
# stops one before starting another. No other sharing is allowed.
SHARED_8000 = {"maritime", "pharmacy", "pii"}
# Launcher UI, install API, SearXNG (grant's side service), Ollama.
INFRA_PORTS = {8085, 8086, 8096, 11434}


def _js_object(name: str) -> str:
    m = re.search(r"const " + name + r"\s*=\s*\{(.*?)\n\};", HTML, re.S)
    assert m, f"{name} not found in zgx-demo-launcher.html"
    return m.group(1)


def _port_conflicts() -> dict[str, list[int]]:
    body = _js_object("PORT_CONFLICTS")
    return {k: [int(p) for p in re.findall(r"\d+", v)]
            for k, v in re.findall(r"(\w+):\s*\[([^\]]*)\]", body)}


def _string_map(name: str) -> dict[str, str]:
    body = re.sub(r"//[^\n]*", "", _js_object(name))
    return dict(re.findall(r"'?(\w+)'?\s*:\s*'([^']*)'", body))


def _demos_entries() -> dict[str, tuple[int, str]]:
    keys = re.search(r"const DEMO_KEY_BY_ID=\[(.*?)\];", HTML).group(1)
    keys = re.findall(r"'(\w+)'", keys)
    out = {}
    for idx, port, path in re.findall(r"\{id:(\d+),[^\n]*?port:(\d+),path:'([^']*)'", HTML):
        out[keys[int(idx)]] = (int(port), path)
    return out


CONFLICTS = _port_conflicts()
PORT_PATHS = _string_map("PORT_PATHS")
HEALTH_PATHS = _string_map("HEALTH_CHECK_PATHS")
DEMOS = _demos_entries()


def test_every_plan_is_registered_in_every_launcher_map():
    for demo_id in PLANS:
        for name, table in (("PORT_CONFLICTS", CONFLICTS), ("PORT_PATHS", PORT_PATHS),
                            ("HEALTH_CHECK_PATHS", HEALTH_PATHS), ("DEMOS", DEMOS)):
            assert demo_id in table, f"{demo_id} is in install_plans.py but missing from launcher {name}"


@pytest.mark.parametrize("demo_id", sorted(PLANS))
def test_launcher_primary_port_matches_the_plan(demo_id):
    plan = PLANS[demo_id]
    assert CONFLICTS[demo_id][0] == plan["port"], "PORT_CONFLICTS[0] is the port the Launch button opens"
    assert DEMOS[demo_id][0] == plan["port"], "DEMOS[] card/modal port disagrees with install_plans.py"


@pytest.mark.parametrize("demo_id", sorted(PLANS))
def test_secondary_ports_are_declared_in_the_plan(demo_id):
    """A port the launcher polls for collisions must be one teardown frees."""
    extra = set(PLANS[demo_id].get("extra_ports") or [])
    assert set(CONFLICTS[demo_id][1:]) <= extra


@pytest.mark.parametrize("demo_id", sorted(PLANS))
def test_launcher_paths_match_the_plan(demo_id):
    plan = PLANS[demo_id]
    expected = plan.get("port_path", "/")
    assert PORT_PATHS[demo_id] == expected
    assert DEMOS[demo_id][1] == expected


@pytest.mark.parametrize("demo_id", sorted(PLANS))
def test_health_url_is_on_a_port_this_demo_owns(demo_id):
    """/health_check takes host:port from health_url and only the path from the
    launcher, so both halves must describe the same endpoint."""
    plan = PLANS[demo_id]
    url = urlparse(plan["health_url"])
    assert url.hostname in ("localhost", "127.0.0.1"), "readiness probes loopback; services bind 0.0.0.0"
    assert url.port in {plan["port"], *(plan.get("extra_ports") or [])}
    assert url.path == HEALTH_PATHS[demo_id]


def test_no_two_demos_share_a_port_outside_the_declared_8000_group():
    owners: dict[int, set[str]] = {}
    for demo_id, plan in PLANS.items():
        for port in {plan["port"], *(plan.get("extra_ports") or [])}:
            owners.setdefault(port, set()).add(demo_id)
    for port, ids in owners.items():
        if len(ids) > 1:
            assert port == 8000 and ids <= SHARED_8000, f"port {port} claimed by {sorted(ids)}"


def test_no_demo_sits_on_an_infrastructure_port():
    for demo_id, plan in PLANS.items():
        clash = {plan["port"], *(plan.get("extra_ports") or [])} & INFRA_PORTS
        assert not clash, f"{demo_id} uses reserved port(s) {sorted(clash)}"


def test_runbook_port_map_lists_every_primary_port():
    runbook = (ROOT.parent / "RUNBOOK.md").read_text(encoding="utf-8")
    table = runbook.split("## Port map", 1)[1].split("\n## ", 1)[0]
    for demo_id, plan in PLANS.items():
        assert re.search(rf"\|\s*{demo_id}\s*\|\s*{plan['port']}\b", table), \
            f"RUNBOOK.md port map has no row '{demo_id} | {plan['port']}'"
