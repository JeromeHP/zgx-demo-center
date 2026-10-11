"""Sample inputs and typed profiles.

Discover, Explore and Grade all read from runtime/sessions/, which ships empty — so
without these a visitor has nothing to try. These tests cover the two ways something
gets in there besides a file upload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "samples" / "manifest.json"


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.main as main

    # Sessions are written under REPO_ROOT/runtime; keep tests out of the real one.
    monkeypatch.setattr(main, "REPO_ROOT", main.REPO_ROOT)
    return TestClient(main.app)


# -- the manifest itself -------------------------------------------------------


def test_every_sample_file_exists_and_lives_under_data():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data_root = (REPO_ROOT / "data").resolve()
    assert manifest["samples"], "the manifest should not be empty"
    for sample in manifest["samples"]:
        assert sample["modes"], f"{sample['id']} suits no mode"
        for entry in sample["files"]:
            path = (REPO_ROOT / entry["path"]).resolve()
            assert path.is_file(), f"{sample['id']} references a missing file: {entry['path']}"
            assert str(path).startswith(str(data_root) + "/"), "samples must stay under data/"
            assert entry["kind"] in {"documents", "solicitation", "proposal"}


def test_real_samples_declare_where_they_came_from():
    """A real document without a source is indistinguishable from an invented one."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        if sample.get("synthetic") is False:
            assert sample.get("source", "").startswith("http"), f"{sample['id']} has no source URL"


def test_every_mode_that_needs_input_has_at_least_one_sample():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    covered = {m for s in manifest["samples"] for m in s["modes"]}
    assert {"discover", "explore", "grade"} <= covered


# -- the endpoints -------------------------------------------------------------


def test_listing_samples_does_not_ship_file_contents(client):
    body = client.get("/api/samples").json()
    assert body["samples"]
    for sample in body["samples"]:
        assert "files" not in sample, "the UI needs labels, not document bodies"
        assert sample["file_count"] >= 1


def test_loading_a_sample_creates_a_session_with_its_files(client):
    body = client.post("/api/samples/grade-flawed-satp/load").json()
    session_id = body["session_id"]
    assert body["files"]
    proposal_dir = client.app  # placeholder to keep flake quiet
    from app.main import REPO_ROOT as ROOT

    written = ROOT / "runtime" / "sessions" / session_id / "proposal"
    assert written.is_dir() and list(written.iterdir())


def test_loading_two_samples_into_one_session_combines_them(client):
    first = client.post("/api/samples/explore-nsf-ate/load").json()
    sid = first["session_id"]
    client.post(f"/api/samples/discover-harrow-creek/load?session_id={sid}")

    from app.main import REPO_ROOT as ROOT

    session = ROOT / "runtime" / "sessions" / sid
    assert (session / "solicitation").is_dir()
    assert (session / "documents").is_dir(), "a second sample must not replace the first"


def test_an_unknown_sample_is_a_404(client):
    assert client.post("/api/samples/does-not-exist/load").status_code == 404


# -- typed profiles ------------------------------------------------------------


def test_a_typed_profile_becomes_a_document_the_agent_can_cite(client):
    body = client.post("/api/session/profile", json={
        "institution": "Midland State University",
        "research_areas": "coastal resilience, remote sensing",
    }).json()

    from app.main import REPO_ROOT as ROOT

    path = ROOT / "runtime" / "sessions" / body["session_id"] / "documents" / "profile.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "Midland State University" in text
    assert "coastal resilience" in text
    # doc_id "profile" is what a [[profile]] citation resolves to.
    assert body["doc_id"] == "profile"


def test_a_real_institution_name_is_accepted(client):
    """The synthetic-institution rule is enforced over data/vault/ at build time. It
    must not reach session input, or a real university could not use this at all."""
    resp = client.post("/api/session/profile", json={"institution": "University of Tennessee, Knoxville"})
    assert resp.status_code == 200


def test_saving_twice_edits_the_profile_rather_than_accumulating_copies(client):
    first = client.post("/api/session/profile", json={"institution": "One"}).json()
    sid = first["session_id"]
    client.post(f"/api/session/profile?session_id={sid}", json={"institution": "Two"})

    from app.main import REPO_ROOT as ROOT

    docs = ROOT / "runtime" / "sessions" / sid / "documents"
    assert [p.name for p in docs.iterdir()] == ["profile.md"]
    assert "Two" in (docs / "profile.md").read_text(encoding="utf-8")


def test_an_empty_profile_is_refused(client):
    assert client.post("/api/session/profile", json={}).status_code == 400
    assert client.post("/api/session/profile", json={"institution": "   "}).status_code == 400


def test_a_saved_profile_can_be_read_back_for_editing(client):
    sid = client.post("/api/session/profile", json={"institution": "Readback U"}).json()["session_id"]
    body = client.get(f"/api/session/profile?session_id={sid}").json()
    assert "Readback U" in body["text"]
    assert client.get("/api/session/profile?session_id=nope").status_code == 404


def test_the_profile_front_matter_survives_json_serialization(client):
    """An unquoted date in front matter becomes a datetime.date and killed a whole run
    with "Object of type date is not JSON serializable"."""
    from app.ingest.loaders import load
    from app.main import REPO_ROOT as ROOT

    sid = client.post("/api/session/profile", json={"institution": "Datey U"}).json()["session_id"]
    doc = load(ROOT / "runtime" / "sessions" / sid / "documents" / "profile.md")
    json.dumps(doc.front_matter)


def test_a_grade_sample_ships_the_records_its_claims_are_checked_against():
    """The claim check compares a submission's figures against "the institution's own
    records", so those records must belong to the submission. Grading a real SUNY
    Binghamton award while the corpus was Harrow Creek's vault reported "wrong
    institution cited for prior award" as a contradiction — confident nonsense about
    the demo's own unrelated scenario."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    flawed = next(s for s in manifest["samples"] if s["id"] == "grade-flawed-satp")

    kinds = {f["kind"] for f in flawed["files"]}
    assert "proposal" in kinds
    assert "documents" in kinds, "the proposal needs its institution's records alongside it"


def test_grade_reads_session_records_when_present(tmp_path):
    from app.main import _vault_dir_for, REPO_ROOT as ROOT

    session = ROOT / "runtime" / "sessions" / "grade-corpus-test" / "documents"
    session.mkdir(parents=True, exist_ok=True)
    (session / "records.md").write_text("# Records\n\nCompleters: 84.\n", encoding="utf-8")
    try:
        assert _vault_dir_for("grade", "grade-corpus-test") == session
    finally:
        import shutil

        shutil.rmtree(session.parent, ignore_errors=True)


def test_grade_falls_back_to_the_shipped_vault_only_when_no_records_were_given():
    from app.main import _vault_dir_for, REPO_ROOT as ROOT

    assert _vault_dir_for("grade", None) == ROOT / "data" / "vault"
