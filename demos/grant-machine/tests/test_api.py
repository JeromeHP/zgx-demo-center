from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import app.main as main_module
from app.settings import Settings
from tests.fake_llm_server import ScriptedResponse, ScriptedToolCall


@pytest.fixture
def app_env(tmp_path, make_settings, fake_llm_app, fake_llm_server, monkeypatch):
    """Points app.main at a throwaway REPO_ROOT (with a minimal vault + seed)
    and the fake LLM server, so the real repo's data/ and runtime/ are never
    touched by tests."""
    repo_root = tmp_path
    (repo_root / "data" / "vault").mkdir(parents=True)
    (repo_root / "data" / "seeds").mkdir(parents=True)
    (repo_root / "data" / "solicitation").mkdir(parents=True)
    (repo_root / "data" / "replays").mkdir(parents=True)
    (repo_root / "runtime").mkdir(parents=True)

    (repo_root / "data" / "vault" / "prior-report.md").write_text(
        "---\ntitle: Prior Report\n---\n# Prior Report\n\nThe project produced 84 completers against a target of 120.\n",
        encoding="utf-8",
    )
    seed = {
        "institution": {"name": "Harrow Creek Community College"},
        "demo_context": {"today": "2026-09-16", "deadline": "2026-10-01"},
    }
    (repo_root / "data" / "seeds" / "harrow_creek.yaml").write_text(yaml.dump(seed), encoding="utf-8")

    # Empty the "quick" scope's sections so these API-level tests only need to
    # script get_requirements/check_budget/finish; full agent behavior
    # (drafting, review, citations) is exercised in test_orchestrator.py.
    settings = make_settings(
        overrides={
            "llm": {"base_url": f"{fake_llm_server.base_url}/v1"},
            "agent": {"scopes": {"quick": {"sections": []}}},
        }
    )

    monkeypatch.setattr(main_module, "REPO_ROOT", repo_root)
    main_module.app.dependency_overrides[main_module.get_settings_dep] = lambda: settings

    client = TestClient(main_module.app)
    client.post("/api/reset")
    yield client, repo_root, settings, fake_llm_app

    client.post("/api/reset")
    main_module.app.dependency_overrides.clear()


def _tc(name: str, args: dict, call_id: str = "call_1") -> ScriptedToolCall:
    return ScriptedToolCall(id=call_id, name=name, arguments=json.dumps(args))


def test_health_ok(app_env):
    client, *_ = app_env
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm"] == "ok"  # fake server responds to GET /v1/models


def test_config_public_has_no_secrets(app_env):
    client, _, settings, _ = app_env
    resp = client.get("/api/config/public")
    assert resp.status_code == 200
    body = resp.json()
    assert settings.llm.api_key not in json.dumps(body)
    assert "api_key" not in body
    assert body["device_label"]
    assert set(body["scopes"]) >= {"quick", "full"}
    assert body["pricing_configured"] is False
    assert body["baselines_configured"] is False


def test_run_lifecycle_completes(app_env):
    client, repo_root, settings, fake_llm_app = app_env
    q = fake_llm_app.state.queue
    q.append(ScriptedResponse(content="", tool_calls=[_tc("get_requirements", {})]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("check_budget", {"track": "track_2"})]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("finish", {"summary": "Done."})]))

    resp = client.post("/api/runs", json={"mode": "story", "scope": "quick", "show_thinking": True})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    deadline = time.monotonic() + 10
    state = None
    while time.monotonic() < deadline:
        state = client.get(f"/api/runs/{run_id}/state").json()
        if state["status"] != "running":
            break
        time.sleep(0.05)

    assert state is not None
    assert state["status"] == "completed"

    packet = client.get(f"/api/runs/{run_id}/packet.md")
    assert packet.status_code == 200
    assert "DRAFT" in packet.text


def test_second_concurrent_run_returns_409(app_env, monkeypatch):
    client, *_rest = app_env
    # Block the background run deterministically (rather than racing on how
    # fast a real orchestrator loop would finish) so it's guaranteed to still
    # be "active" when the second POST arrives.
    release = threading.Event()

    def blocking_orchestrator(ctx, llm, scope, sections, max_steps, info, is_cancelled=lambda: False):
        release.wait(timeout=5)
        ctx.state.status = "cancelled"
        return ctx.state

    monkeypatch.setattr(main_module, "run_orchestrator", blocking_orchestrator)

    resp1 = client.post("/api/runs", json={"mode": "story", "scope": "quick"})
    assert resp1.status_code == 200
    active_run_id = resp1.json()["run_id"]

    resp2 = client.post("/api/runs", json={"mode": "story", "scope": "quick"})
    assert resp2.status_code == 409
    assert resp2.json()["detail"]["active_run_id"] == active_run_id

    release.set()


def test_sse_stream_receives_run_started_and_run_finished(app_env):
    client, *_rest, fake_llm_app = app_env
    q = fake_llm_app.state.queue
    q.append(ScriptedResponse(content="", tool_calls=[_tc("get_requirements", {})]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("check_budget", {"track": "track_2"})]))
    q.append(ScriptedResponse(content="", tool_calls=[_tc("finish", {"summary": "Done."})]))

    run_id = client.post("/api/runs", json={"mode": "story", "scope": "quick"}).json()["run_id"]

    seen_types: list[str] = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        event_type = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line == "" and event_type:
                seen_types.append(event_type)
                event_type = None
                if seen_types[-1] == "run_finished":
                    break

    assert seen_types[0] == "run_started"
    assert seen_types[-1] == "run_finished"


def test_cancel_unknown_run_404(app_env):
    client, *_ = app_env
    resp = client.post("/api/runs/does-not-exist/cancel")
    assert resp.status_code == 404


def test_upload_rejects_bad_extension(app_env):
    client, *_ = app_env
    resp = client.post("/api/explore/upload", files={"file": ("bad.exe", b"data", "application/octet-stream")})
    assert resp.status_code == 400


def test_upload_rejects_oversized_file(app_env):
    client, repo_root, settings, _ = app_env
    big = b"x" * (settings.uploads.max_bytes + 1)
    resp = client.post("/api/explore/upload", files={"file": ("doc.md", big, "text/markdown")})
    assert resp.status_code == 400


def test_upload_accepts_valid_file_and_reset_wipes_it(app_env):
    client, repo_root, settings, _ = app_env
    resp = client.post("/api/explore/upload", files={"file": ("doc.md", b"# Hello\n\nSome content.\n", "text/markdown")})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    session_dir = repo_root / "runtime" / "sessions" / session_id
    assert session_dir.is_dir()
    assert any(session_dir.iterdir())

    reset_resp = client.post("/api/reset")
    assert reset_resp.status_code == 200
    assert not (repo_root / "runtime" / "sessions").exists()


# -- scenario leakage into the prompt -------------------------------------------------
# A live Ashgrove run filed three fabricated "missing partner letter" issues against an
# academic medical centre, because the quick-scope guidance named Harrow Creek's
# industrial employers as string constants. The agent went looking for a fluid-systems
# company in a hospital's vault, found nothing, and wrote that down as a finding.


def test_quick_guidance_names_this_scenarios_partners():
    from app.agent.prompts import quick_scope_guidance

    text = quick_scope_guidance(["Wrenfield Regional Hospital", "Calloway Valley Health"])
    assert "Wrenfield Regional Hospital" in text
    assert "Calloway Valley Health" in text
    assert "Kessler" not in text


def test_quick_guidance_is_generic_when_no_partners_are_known():
    from app.agent.prompts import quick_scope_guidance

    text = quick_scope_guidance([])
    assert "Kessler" not in text and "Northgate" not in text
    assert "search the vault" in text


def test_no_scenario_inherits_another_scenarios_partner_names():
    """Every shipped scenario names its own partners, and nobody else's."""
    import app.main as main

    seen = {}
    for sid in main._scenario_registry():
        seen[sid] = set(main._partner_names(main._scenario_or_default(sid)))
        assert seen[sid], f"{sid} declares no partners"
    for sid, names in seen.items():
        for other, other_names in seen.items():
            if other != sid:
                assert not (names & other_names), f"{sid} and {other} share partner names"


# -- a stale UI must not be possible --------------------------------------------------
# The Demo Center loads `/?v=<timestamp>`, so index.html was always fresh while app.js
# and styles.css -- bare relative paths, no cache headers -- came from the browser cache
# and were never revalidated. The operator's browser ran a build from before the
# scenario picker existed: the markup had the dropdown, the code that fills it did not.
# From the server the two builds were indistinguishable.


def test_the_ui_is_served_uncached():
    from fastapi.testclient import TestClient

    import app.main as main

    with TestClient(main.app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "no-store" in page.headers.get("cache-control", "")
        asset = client.get("/app.js")
        assert asset.status_code == 200
        assert "no-store" in asset.headers.get("cache-control", ""), \
            "a cached app.js is a bug the server cannot see and the operator cannot diagnose"


def test_asset_urls_carry_the_build_id():
    from fastapi.testclient import TestClient

    import app.main as main

    with TestClient(main.app) as client:
        html = client.get("/").text
        build = client.get("/api/build").json()["build"]
        assert f"app.js?v={build}" in html
        assert f"styles.css?v={build}" in html


def test_the_build_id_changes_when_the_ui_changes(tmp_path, monkeypatch):
    import app.main as main

    before = main._asset_build_id()
    target = main._web_dir / "app.js"
    original = target.stat()
    try:
        target.touch()
        assert main._asset_build_id() != before, \
            "an edited UI must produce a new build id, or the stamp cannot bust anything"
    finally:
        os.utime(target, (original.st_atime, original.st_mtime))
