"""Draft mode and the institution document views.

Draft replaced Story and Explore, which were one drafting pipeline differing only in
where the solicitation came from. It resolves to one of those two before anything
else sees it, and it has to honour the presenter's choice of sources -- a file
uploaded earlier must not override "this institution's" once switched back.

The document views read every scenario's vault. The read path used to look only in
data/vault/, so a citation in any run for the other five institutions opened
"Document not found".
"""

from __future__ import annotations

import json as _json
import re
import time
from collections import Counter
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.checks.budget import BudgetRules, check_budget


# -- Draft resolves to story or explore ---------------------------------------------


@pytest.fixture
def draft_env(tmp_path, monkeypatch):
    repo_root = tmp_path
    for d in ("data/vault", "data/seeds", "data/solicitation", "runtime"):
        (repo_root / d).mkdir(parents=True)
    (repo_root / "data" / "vault" / "inst-profile.md").write_text("# Profile\n\nShipped vault.\n", encoding="utf-8")
    (repo_root / "data" / "seeds" / "harrow_creek.yaml").write_text(
        "institution:\n  name: Harrow Creek Community College\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "REPO_ROOT", repo_root)

    seen: dict = {}

    def capture(ctx, llm, scope, sections, max_steps, info, is_cancelled=lambda: False):
        seen["mode"] = ctx.state.mode
        seen["vault_dir"] = ctx.vault_dir
        ctx.state.status = "completed"
        return ctx.state

    monkeypatch.setattr(main_module, "run_orchestrator", capture)
    # Requirement extraction needs a model; these tests only care which branch ran.
    monkeypatch.setattr(main_module, "_intake_requirements",
                        lambda *a, **k: seen.setdefault("intake", True) and (None, []))

    client = TestClient(main_module.app)
    client.post("/api/reset")
    yield client, repo_root, seen
    client.post("/api/reset")


def _run(client, seen, **body):
    seen.clear()
    resp = client.post("/api/runs", json={"mode": "draft", "scope": "quick", **body})
    assert resp.status_code == 200, resp.text
    # The run is a background thread; the captured orchestrator returns at once.
    deadline = time.monotonic() + 5
    while "mode" not in seen and time.monotonic() < deadline:
        time.sleep(0.02)
    # And the run has to be marked finished before the next one may start.
    while main_module.run_manager.get_active_id() and time.monotonic() < deadline:
        time.sleep(0.02)
    return seen


def _upload(client, kind, name, text, session_id=None):
    qs = f"kind={kind}" + (f"&session_id={session_id}" if session_id else "")
    resp = client.post(f"/api/explore/upload?{qs}", files={"file": (name, text.encode(), "text/markdown")})
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def test_draft_with_nothing_uploaded_runs_as_story_on_the_scenario_vault(draft_env):
    client, repo_root, seen = draft_env
    _run(client, seen)
    assert seen["mode"] == "story"
    assert seen["vault_dir"] == repo_root / "data" / "vault"
    assert "intake" not in seen


def test_draft_with_an_uploaded_solicitation_runs_as_explore(draft_env):
    client, _, seen = draft_env
    sid = _upload(client, "solicitation", "call.md", "# Call\n\nApplicants must do X.\n")
    _run(client, seen, session_id=sid, use_uploaded_solicitation=True)
    assert seen["mode"] == "explore"
    assert seen.get("intake")


def test_draft_keeps_the_scenario_solicitation_when_told_to(draft_env):
    client, _, seen = draft_env
    sid = _upload(client, "solicitation", "call.md", "# Call\n\nApplicants must do X.\n")
    _run(client, seen, session_id=sid, use_uploaded_solicitation=False)
    assert seen["mode"] == "story"
    assert "intake" not in seen


def test_draft_evidence_follows_the_documents_choice(draft_env):
    client, repo_root, seen = draft_env
    sid = _upload(client, "documents", "ours.md", "# Our profile\n\nOur own evidence.\n")

    _run(client, seen, session_id=sid, use_uploaded_documents=True)
    assert seen["mode"] == "story"
    assert seen["vault_dir"] == repo_root / "runtime" / "sessions" / sid / "documents"

    _run(client, seen, session_id=sid, use_uploaded_documents=False)
    assert seen["vault_dir"] == repo_root / "data" / "vault"


# -- the document views, against the real shipped scenarios -----------------------


@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_every_scenario_lists_its_documents_and_solicitation(client):
    scenarios = client.get("/api/scenarios").json()["scenarios"]
    assert len(scenarios) > 1
    for sc in scenarios:
        body = client.get(f"/api/scenarios/{sc['id']}/documents").json()
        assert len(body["documents"]) == sc["documents"], sc["id"]
        assert body["solicitation"], f"{sc['id']} has no readable solicitation"
        assert body["groups"] and all(d["group"] in body["groups"] for d in body["documents"])
        assert all(d["title"] for d in body["documents"])
        # Planted issues cite documents that exist in this vault.
        ids = {d["doc_id"] for d in body["documents"]}
        for issue in body["planted_issues"]:
            assert set(issue["evidence"]) <= ids, (sc["id"], issue["id"])


def test_unknown_scenario_is_404(client):
    assert client.get("/api/scenarios/nowhere/documents").status_code == 404


def test_a_citation_from_another_institution_resolves(client):
    """Was 404: the read path only looked in Harrow Creek's vault."""
    resp = client.get("/api/vault/aumc-bio-achterberg")
    assert resp.status_code == 200
    assert resp.json()["text"]


@pytest.mark.parametrize("scenario", ["harrow-creek", "ashgrove"])
def test_the_solicitation_is_readable(client, scenario):
    resp = client.get(f"/api/vault/solicitation?scenario={scenario}")
    assert resp.status_code == 200
    assert len(resp.json()["text"]) > 200


def test_doc_ids_cannot_walk_out_of_the_vault(client):
    assert client.get("/api/vault/..%2F..%2Fconfig%2Fdefault").status_code == 404
    assert client.get("/api/vault/inst-profile-2025?session_id=..%2F..%2F..").status_code == 200


def test_every_scenario_has_an_overview_of_ten_sentences_or_fewer(client):
    for sc in client.get("/api/scenarios").json()["scenarios"]:
        summary = client.get(f"/api/scenarios/{sc['id']}/documents").json()["scenario"]["summary"]
        assert summary, f"{sc['id']} has no scenario overview"
        # "Dr." and initials are not sentence ends.
        sentences = [s for s in re.split(r"(?<!Dr)(?<![A-Z])[.!?]\s+", summary) if s.strip()]
        assert len(sentences) <= 10, f"{sc['id']} overview is {len(sentences)} sentences"


def test_the_overview_is_never_part_of_a_corpus():
    """It is an introduction for the audience; the agent must not retrieve or cite it."""
    path = main_module.REPO_ROOT / "data" / "scenario_summaries.yaml"
    assert path.is_file()
    for sc in main_module._scenario_registry().values():
        assert Path(sc["vault"]).resolve() not in path.resolve().parents


# -- the scenarios themselves -----------------------------------------------------
# Every scenario once shared the same four planted issues, the same $1,038,450 budget
# and the same synthetic $1M call. These keep them apart.

_GENERATED = ["ashgrove", "meridian", "sablecreek", "tierra-sandoval", "vantcross"]


def _registry():
    return main_module._scenario_registry()


def _planted(sc) -> list[dict]:
    return _json.loads((Path(sc["vault"]) / "_planted_issues.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("scenario_id", _GENERATED)
def test_each_generated_scenario_applies_to_a_real_call_with_curated_rules(scenario_id):
    sc = _registry()[scenario_id]
    assert sc["solicitation_is_real"] and sc["solicitation_id"], scenario_id
    rules = (main_module._load_requirements_doc(sc) or {}).get("budget_rules") or {}
    assert rules, f"{scenario_id}: requirements.json has no budget_rules"
    for key, rule in rules.items():
        assert rule.get("quote") and rule.get("source", "").startswith("https://"), f"{scenario_id}.{key} is unsourced"


def test_no_planted_issue_type_is_shared_by_more_than_two_scenarios():
    counts = Counter(code for sc in _registry().values() for code in {i["code"] for i in _planted(sc)})
    assert [c for c, n in counts.items() if n > 2] == [], counts


@pytest.mark.parametrize("scenario_id", _GENERATED)
def test_the_budget_check_finds_exactly_the_planted_budget_issues(scenario_id):
    sc = _registry()[scenario_id]
    rules = (main_module._load_requirements_doc(sc) or {}).get("budget_rules") or {}
    value = lambda k: rules[k].get("value") if isinstance(rules.get(k), dict) else rules.get(k)  # noqa: E731
    result = check_budget(Path(sc["vault"]), cap_usd=value("cap_usd"), track_years=value("max_years") or 5,
                          rules=BudgetRules.from_dict(rules))
    planted = {i["code"] for i in _planted(sc) if i["detection"] == "deterministic"}
    assert {i.code for i in result.issues} == planted


def test_no_generated_scenario_carries_the_old_shared_budget():
    for scenario_id in _GENERATED:
        sc = _registry()[scenario_id]
        total = check_budget(Path(sc["vault"]), cap_usd=None).total_usd  # parses the CSV properly
        assert total != 1_038_450, scenario_id


# -- Find Funding must not write into a shipped vault ------------------------------


def test_fetched_web_pages_never_land_in_a_scenario_vault(tmp_path, monkeypatch):
    """Find Funding on a built-in institution saved fetched pages into data/scenarios/,
    where they became that institution's documents for every later run."""
    repo = tmp_path
    for d in ("data/vault", "data/seeds", "data/scenarios/foo/solicitation", "runtime"):
        (repo / d).mkdir(parents=True)
    (repo / "data/seeds/harrow_creek.yaml").write_text("institution:\n  name: Harrow Creek\n", encoding="utf-8")
    (repo / "data/seeds/foo.yaml").write_text("institution:\n  name: Foo Institute\n", encoding="utf-8")
    (repo / "data/scenarios/foo/profile.md").write_text("# Foo\n\nShipped.\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "REPO_ROOT", repo)
    monkeypatch.setattr(main_module, "_build_egress_guard", lambda *a, **k: object())
    seen: dict = {}

    def capture(ctx, *a, **k):
        seen["web_dir"] = Path(ctx.web._session_dir)
        ctx.state.status = "completed"
        return ctx.state

    monkeypatch.setattr(main_module, "run_orchestrator", capture)
    client = TestClient(main_module.app)

    def run(**body):
        seen.clear()
        assert client.post("/api/runs", json={"mode": "discover", "scope": "quick", **body}).status_code == 200
        deadline = time.monotonic() + 5
        while ("web_dir" not in seen or main_module.run_manager.get_active_id()) and time.monotonic() < deadline:
            time.sleep(0.02)
        return seen["web_dir"]

    web_dir = run(scenario="foo")
    assert repo / "data" not in web_dir.parents, f"web pages would be written into {web_dir}"
    assert (repo / "runtime" / "runs") in web_dir.parents

    # With the institution's own uploads, pages stay beside them.
    sid = client.post("/api/explore/upload?kind=documents",
                      files={"file": ("ours.md", b"# Ours\n\nUploaded.\n", "text/markdown")}).json()["session_id"]
    assert run(scenario="foo", session_id=sid) == repo / "runtime" / "sessions" / sid / "documents"
    client.post("/api/reset")
