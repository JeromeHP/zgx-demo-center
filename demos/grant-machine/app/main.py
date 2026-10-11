"""FastAPI app: routes under /api, static web UI at / (Phase 7, Spec G).

Run with: uv run python -m app.main
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent.orchestrator import run_orchestrator
from app.agent.state import DraftVersion, RunState
from app.agent.tools import ToolContext
from app.events import Event, EventBus
from app.ingest.chunker import chunk_document
from app.ingest.loaders import LoaderError, load
from app.ingest.requirements import ExtractionProgress, extract_requirements, propose_sections
from app.discovery import GrantsGovClient, WebResearch
from app.egress import EgressGuard, SessionCorpus, Vocabulary
from app.llm_client import LLMClient
from app.packet import render_packet_pdf
from app.replay import iter_replay_events, list_replays, save_replay
from app.retrieval.bm25_index import BM25Index
from app.settings import REPO_ROOT, Settings, get_settings
from app import netstatus

app = FastAPI(title="Grant Machine")

_VAULT_EXTENSIONS = (".md", ".html", ".htm", ".pdf", ".csv")


# -- dependencies -------------------------------------------------------------


def get_settings_dep() -> Settings:
    return get_settings()


def get_llm_client(settings: Settings = Depends(get_settings_dep)) -> LLMClient:
    return LLMClient(settings)


# -- run manager (module-level: enforces "one active run at a time") ---------


class RunConflict(Exception):
    def __init__(self, run_id: str):
        self.run_id = run_id


@dataclass
class ActiveRun:
    run_id: str
    state: RunState
    event_bus: EventBus
    thread: threading.Thread | None = None
    cancel_holder: dict[str, bool] = field(default_factory=lambda: {"cancelled": False})


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ActiveRun | None = None
        self._all: dict[str, ActiveRun] = {}

    def start(self, run: ActiveRun) -> None:
        with self._lock:
            if self._active is not None:
                raise RunConflict(self._active.run_id)
            self._active = run
            self._all[run.run_id] = run

    def get(self, run_id: str) -> ActiveRun | None:
        return self._all.get(run_id)

    def all_runs(self) -> list["ActiveRun"]:
        """Every run this process has started, finished or not.

        Used by /api/metrics to total tokens across the whole session: the
        shared chrome's counter is meant to answer "how many tokens has this
        machine processed", which must not reset to zero the moment a run ends.
        """
        with self._lock:
            return list(self._all.values())

    def get_active_id(self) -> str | None:
        with self._lock:
            return self._active.run_id if self._active else None

    def mark_finished(self, run_id: str) -> None:
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                self._active = None

    def cancel(self, run_id: str) -> bool:
        run = self._all.get(run_id)
        if run is None:
            return False
        run.cancel_holder["cancelled"] = True
        return True

    def reset(self) -> None:
        with self._lock:
            if self._active is not None:
                self._active.cancel_holder["cancelled"] = True
            self._active = None
            self._all = {}


run_manager = RunManager()


# -- helpers ------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _load_seed() -> dict[str, Any]:
    path = REPO_ROOT / "data" / "seeds" / "harrow_creek.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_requirements_doc(scenario: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The requirements for a scenario. Defaults to the original, curated NSF 24-584 set."""
    path = (scenario or {}).get("requirements") or (REPO_ROOT / "data" / "solicitation" / "requirements.json")
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _requirements_list(scenario: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    doc = _load_requirements_doc(scenario)
    return doc.get("requirements", []) if doc else []


def _budget_rules(scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    """The curated money rules of a scenario's solicitation, if it declares any."""
    doc = _load_requirements_doc(scenario)
    return (doc or {}).get("budget_rules") or {}


def _tracks(scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = _load_requirements_doc(scenario)
    if doc and doc.get("tracks"):
        return doc["tracks"]
    return {"track_2": {"max_total_usd": 1_000_000, "max_years": 3}}


def _load_seed_for(scenario: dict[str, Any] | None) -> dict[str, Any]:
    path = (scenario or {}).get("seed")
    if path and Path(path).is_file():
        try:
            return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            pass
    return _load_seed()


def _solicitation_info(scenario: dict[str, Any] | None = None) -> dict[str, str]:
    seed = _load_seed_for(scenario)
    doc = _load_requirements_doc(scenario)
    ctx = seed.get("demo_context") or {}
    return {
        "today": ctx.get("today", ""),
        "deadline": doc.get("demo_deadline") if doc and doc.get("demo_deadline") else ctx.get("deadline", ""),
        "solicitation_id": (doc or {}).get("solicitation_id", "NSF 24-584"),
        "program": (doc or {}).get("program", "Advanced Technological Education (ATE)"),
        "track_name": (seed.get("new_project") or {}).get("track_name", "Track 2 Projects"),
    }


def _partner_names(scenario: dict[str, Any] | None = None) -> list[str]:
    """The partners whose support letters this scenario's vault holds.

    Seeds call them `employers` (a technician-education college) or `collaborators` (a
    research institution); both mean the same thing here -- the organisations whose
    letters a quick run is told to go and read.
    """
    seed = _load_seed_for(scenario)
    entries = seed.get("employers") or seed.get("collaborators") or []
    names = []
    for e in entries:
        name = e.get("name") if isinstance(e, dict) else e
        if name:
            names.append(str(name))
    return names


def _institution_name(scenario: dict[str, Any] | None = None) -> str:
    seed = _load_seed_for(scenario)
    return (seed.get("institution") or {}).get("name", "the institution")


def _vault_dir_for(mode: str, session_id: str | None, scenario: dict[str, Any] | None = None) -> Path:
    """The corpus the agent searches.

    Discover swaps in the institution's uploaded documents. Explore does NOT by
    default: its upload is the *solicitation*, and the institution's evidence remains
    the shipped vault. Explore used to swap the vault instead, which is why uploading
    a solicitation never changed the requirements the agent worked to.

    Grade matters most here. The claim check compares a submission's figures against
    "the institution's own records", so those records have to *belong to the
    submission*. Grading a real SUNY Binghamton award while the corpus was Harrow
    Creek's vault produced confident nonsense -- it reported "wrong institution cited
    for prior award" as a contradiction, when the only thing contradicted was the
    demo's own unrelated scenario. Session documents when supplied; otherwise the
    claim check finds nothing and honestly says so.
    """
    # Both modes use the session's `documents/` as the corpus when it has any.
    # Explore's upload is the solicitation, but a session may carry both -- and
    # pairing somebody's own call for proposals with the shipped Harrow Creek vault
    # produces exactly what you would expect: a first live NIH run scored 1.0-2.5
    # because a community college's advanced-manufacturing evidence has nothing to
    # say about hearing research. Honest, and useless. Upload both and it works.
    # Story is included for Draft runs, which resolve to "story" when the solicitation
    # is the scenario's own but may still bring their own evidence.
    if mode in ("discover", "explore", "grade", "story") and session_id:
        docs = REPO_ROOT / "runtime" / "sessions" / session_id / "documents"
        if docs.is_dir() and any(p.is_file() for p in docs.iterdir()):
            return docs
    # Otherwise the selected scenario's own vault, which for the default scenario is
    # the original data/vault.
    return (scenario or {}).get("vault") or (REPO_ROOT / "data" / "vault")


def _submission_upload(session_id: str | None) -> Path | None:
    """The proposal being graded, if this session carries one."""
    return _session_file(session_id, "proposal")


def _session_file(session_id: str | None, kind: str) -> Path | None:
    if not session_id:
        return None
    directory = REPO_ROOT / "runtime" / "sessions" / session_id / kind
    if not directory.is_dir():
        return None
    files = sorted(p for p in directory.iterdir() if p.is_file())
    return files[0] if files else None


def _solicitation_upload(session_id: str | None) -> Path | None:
    """The uploaded solicitation for this session, if there is one."""
    if not session_id:
        return None
    return _session_file(session_id, "solicitation")


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _seed_submission(session_id: str | None, state: RunState, bus: EventBus) -> list[dict[str, str]]:
    """Load an uploaded proposal and seed its sections as v1 drafts.

    Grading reuses `review_section` unchanged, which reads whatever is in
    `state.drafts` -- so a submitted document only has to be put there. Sections come
    from the document's own headings; a flat document with none becomes one section
    rather than nothing, since a PDF export frequently loses heading structure.
    """
    path = _submission_upload(session_id)
    if path is None:
        return []
    try:
        doc = load(path)
    except LoaderError as exc:
        bus.emit("submission_failed", {"filename": path.name, "detail": str(exc)})
        return []

    seeded: list[dict[str, str]] = []
    used: set[str] = set()
    for section in doc.sections:
        text = (section.text or "").strip()
        if len(text) < 120:
            continue  # a heading with no substance under it is not a section to score
        slug = _SLUG_RE.sub("_", section.heading.lower()).strip("_")[:48] or "section"
        while slug in used:
            slug = f"{slug}_2"
        used.add(slug)
        state.drafts[slug] = [DraftVersion(version=1, text=text, citations=[], word_count=len(text.split()))]
        seeded.append({"id": slug, "title": section.heading, "word_target": ""})

    if not seeded and doc.text.strip():
        state.drafts["full_submission"] = [
            DraftVersion(version=1, text=doc.text, citations=[], word_count=len(doc.text.split()))
        ]
        seeded = [{"id": "full_submission", "title": doc.title or path.stem, "word_target": ""}]

    bus.emit("submission_loaded", {
        "filename": path.name,
        "title": doc.title,
        "sections": [{"id": s["id"], "title": s["title"],
                      "words": state.drafts[s["id"]][0].word_count} for s in seeded],
    })
    return seeded


def _intake_requirements(
    session_id: str | None,
    llm: LLMClient,
    settings: Settings,
    bus: EventBus,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Extract requirements from this session's uploaded solicitation, if any.

    Returns (requirements_doc, sections). Both are None/empty when nothing was
    uploaded, in which case the run falls back to the shipped NSF 24-584 set.
    """
    path = _solicitation_upload(session_id)
    if path is None:
        return None, []

    try:
        doc = load(path)
    except LoaderError as exc:
        bus.emit("intake_failed", {"filename": path.name, "detail": str(exc)})
        return None, []

    bus.emit("intake_started", {"filename": path.name, "title": doc.title, "chars": len(doc.text)})

    def on_progress(p: ExtractionProgress) -> None:
        bus.emit("intake_progress", {"index": p.index, "total": p.total,
                                     "heading": p.heading, "candidates": p.candidates})

    # Mechanical extract-and-quote work: thinking off, measured ~18s/section with it
    # off versus 2-6 minutes per section with it on, for marginal accuracy gain --
    # and every candidate is validated against the source text afterwards anyway.
    extractor = settings.llm.profiles.get("extractor")
    if extractor is not None:
        extractor.enable_thinking = False

    try:
        requirements = extract_requirements(
            doc.text,
            llm,
            model=settings.llm.model,
            solicitation_id=doc.title or path.stem,
            source_url="",
            tracks={},
            on_progress=on_progress,
            cache_dir=REPO_ROOT / "runtime" / "cache" / "requirements",
        )
        sections = propose_sections(llm, requirements)
    except Exception as exc:  # noqa: BLE001
        # A failure here must degrade to the shipped requirements, not end the run:
        # the operator has already waited minutes, and an intake problem should cost
        # them the upload, not the whole demo.
        bus.emit("intake_failed", {"filename": path.name, "detail": f"extraction failed: {exc}"})
        return None, []

    bus.emit("intake_complete", {
        "solicitation_id": requirements.get("solicitation_id"),
        "requirements": len(requirements.get("requirements", [])),
        "sections": sections,
        "extraction": requirements.get("extraction", {}),
    })
    return requirements, sections


def _build_egress_guard(settings: Settings, session_dir: Path, bus: EventBus, state: RunState) -> EgressGuard:
    """Construct the boundary for a discovery run.

    Every record is mirrored into both the event stream (so the UI's "what left this
    box" panel is live) and RunState (so it survives into the packet and the saved
    run). Nothing reaches the network without appearing in both.
    """
    vocabulary = Vocabulary.load(REPO_ROOT / settings.egress.vocabulary_dir)
    corpus = SessionCorpus.from_dir(session_dir, ngram_len=settings.egress.ngram_len)

    def on_record(record) -> None:
        payload = record.as_dict()
        state.egress_log.append(payload)
        bus.emit("egress", payload)

    return EgressGuard(
        vocabulary,
        corpus,
        max_requests=settings.egress.max_requests_per_run,
        timeout_s=settings.egress.timeout_s,
        max_response_bytes=settings.egress.max_response_bytes,
        max_term_words=settings.egress.max_term_words,
        # Operator-configured, never model-chosen. See EgressGuard._is_trusted.
        trusted_endpoints=[settings.egress.searxng_url] if settings.egress.searxng_url else [],
        on_record=on_record,
    )


def _build_index(vault_dir: Path) -> BM25Index:
    index = BM25Index()
    if vault_dir.is_dir():
        docs = [load(p) for p in vault_dir.glob("*") if p.suffix.lower() in _VAULT_EXTENSIONS]
        index.build(docs)
    return index


def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _render_packet(state: RunState) -> str:
    lines = [
        "# Grant Machine Packet",
        "",
        "**DRAFT -- AI-generated for human review**",
        "",
        "## Summary",
        "",
        state.summary or "(no summary yet)",
        "",
        "## Compliance Matrix",
        "",
        "| Requirement | Status | Evidence | Note |",
        "|---|---|---|---|",
    ]
    for rid, entry in state.compliance.items():
        lines.append(f"| {rid} | {entry.status} | {', '.join(entry.evidence)} | {entry.note} |")

    lines += ["", "## Issues", ""]
    for issue in state.issues:
        lines.append(f"- **[{issue.severity}] {issue.title}** ({issue.source}): {issue.detail}")

    lines += ["", "## Drafts", ""]
    for section_id, versions in state.drafts.items():
        latest = versions[-1]
        lines += [f"### {section_id} (v{latest.version})", "", latest.text, ""]
        review = (state.reviews.get(section_id) or [None])[-1]
        if review:
            lines += [f"_Reviewer score: {review.mean_score:.1f}/5, passed={review.passed}_", ""]

    return "\n".join(lines)


# -- health / config / network -------------------------------------------------


_health_cache: dict[str, Any] = {"ts": 0.0, "llm_ok": False}


def _check_llm_reachable(settings: Settings) -> bool:
    try:
        resp = httpx.get(f"{settings.llm.base_url}/models", timeout=2.0)
        return resp.status_code < 500
    except httpx.HTTPError:
        return False


@app.get("/api/health")
async def health(settings: Settings = Depends(get_settings_dep)) -> dict[str, str]:
    now = time.monotonic()
    if now - _health_cache["ts"] > 5:
        _health_cache["llm_ok"] = await asyncio.to_thread(_check_llm_reachable, settings)
        _health_cache["ts"] = now
    return {
        "status": "ok",
        "llm": "ok" if _health_cache["llm_ok"] else "unreachable",
        "model": settings.llm.model,
        "version": _git_short_sha(),
    }


@app.get("/api/metrics")
def api_metrics() -> dict[str, Any]:
    """Session tokens for the shared Demo Center chrome (demos/ui/README.md).

    Totals the latest `metrics` event from EVERY run this process has seen, not
    just the active one. The first version read only the active run's bus, so
    the counter climbed during a run and dropped back to zero the instant it
    finished -- which is precisely when a presenter turns round to point at it.
    A session counter that resets when the work completes is worse than none.

    Still no second tally: this app instruments tokens properly already (it is
    what the shared ledger was modelled on), so this reads the numbers it
    publishes rather than counting them again.
    """
    prompt = completion = calls = 0
    estimated = False
    tps = None
    active_id = run_manager.get_active_id()

    for run in run_manager.all_runs():
        latest = None
        for ev in reversed(run.event_bus.events):   # a property, not a method
            if ev.type == "metrics":
                latest = ev.data
                break
        if not latest:
            continue
        prompt += latest.get("prompt_tokens", 0) or 0
        completion += latest.get("completion_tokens", 0) or 0
        calls += latest.get("llm_calls", 0) or 0
        estimated = estimated or bool(latest.get("tokens_estimated"))
        # Rate is a live figure, so only the run currently producing tokens has
        # a meaningful one; a finished run's last rate is history, not status.
        if run.run_id == active_id:
            tps = latest.get("decode_tps_recent")

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "requests": calls,
        "tps": tps,
        "estimated": estimated,
    }


@app.get("/api/config/public")
def config_public(settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    baselines = settings.demo.human_baseline_minutes
    pricing = settings.demo.cloud_pricing
    return {
        "device_label": settings.demo.device_label,
        "model": settings.llm.model,
        "scopes": list(settings.agent.scopes.keys()),
        "pricing_configured": pricing.input_per_million_usd is not None and pricing.output_per_million_usd is not None,
        "baselines_configured": any(
            v is not None for v in (baselines.compliance_matrix, baselines.budget_review, baselines.section_draft)
        ),
        "disclaimer": "Fictional institution. All people, organizations, figures, and documents are synthetic.",
    }


_network_cache: dict[str, Any] = {"status": None, "ts": 0.0}
_network_poller_stop = threading.Event()
_NETWORK_CACHE_STALE_S = 10.0  # fall back to a fresh synchronous probe if the poller hasn't run recently
_NETWORK_EMIT_INTERVAL_S = 10.0  # emit into the active run's stream at least this often, per Spec I


def _network_poller_loop() -> None:
    settings = get_settings()
    last_status: dict[str, Any] | None = None
    last_emit = 0.0
    while not _network_poller_stop.is_set():
        status = netstatus.get_status(settings)
        _network_cache["status"] = status
        _network_cache["ts"] = time.monotonic()

        now = time.monotonic()
        if status != last_status or now - last_emit >= _NETWORK_EMIT_INTERVAL_S:
            active_id = run_manager.get_active_id()
            if active_id:
                run = run_manager.get(active_id)
                if run:
                    run.event_bus.emit("network_status", status)
            last_status = status
            last_emit = now

        _network_poller_stop.wait(settings.demo.connectivity_probe.interval_s)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    thread = threading.Thread(target=_network_poller_loop, daemon=True)
    thread.start()
    yield
    _network_poller_stop.set()


app.router.lifespan_context = _lifespan


@app.get("/api/network")
async def network_status(settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    cached = _network_cache["status"]
    if cached is not None and time.monotonic() - _network_cache["ts"] < _NETWORK_CACHE_STALE_S:
        return cached
    return await asyncio.to_thread(netstatus.get_status, settings)


# -- run lifecycle --------------------------------------------------------------


class StartRunRequest(BaseModel):
    # "draft" is what the UI sends. It resolves to "story" (the scenario's own
    # solicitation) or "explore" (an uploaded one) before anything else sees it, so
    # the packet, state and replay code only ever meet the two values they know.
    # "story" and "explore" stay accepted: saved replays and API callers use them.
    mode: Literal["draft", "story", "explore", "discover", "grade"]
    # Which institution the run is for. Defaults to the original scenario.
    scenario: str | None = None
    scope: Literal["quick", "full"]
    show_thinking: bool = True
    session_id: str | None = None
    # Draft only. Which of the session's uploads this run uses. None means "use it if
    # the session has one"; False keeps the scenario's own even when an upload exists,
    # so switching the UI back to "this institution's" is not overridden by a file
    # uploaded earlier in the same session.
    use_uploaded_solicitation: bool | None = None
    use_uploaded_documents: bool | None = None


@app.post("/api/runs")
def start_run(
    body: StartRunRequest, settings: Settings = Depends(get_settings_dep), llm: LLMClient = Depends(get_llm_client)
) -> dict[str, str]:
    scope_cfg = settings.agent.scopes.get(body.scope)
    if scope_cfg is None:
        raise HTTPException(400, f"unknown scope: {body.scope}")

    run_id = uuid.uuid4().hex[:12]
    scenario = _scenario_or_default(body.scenario)
    corpus_session = body.session_id
    if body.mode == "draft":
        uploaded_solicitation = body.use_uploaded_solicitation is not False and _solicitation_upload(body.session_id)
        body.mode = "explore" if uploaded_solicitation else "story"
        if body.use_uploaded_documents is False:
            corpus_session = None
    vault_dir = _vault_dir_for(body.mode, corpus_session, scenario)
    runtime_dir = REPO_ROOT / "runtime"
    log_path = runtime_dir / "runs" / run_id / "events.jsonl"

    state = RunState(run_id=run_id, mode=body.mode, scope=body.scope,
                     scenario=scenario["id"], requirements=_requirements_list(scenario))
    bus = EventBus(run_id=run_id, log_path=log_path)

    playbook = {"discover": "discovery", "grade": "grade"}.get(body.mode, "proposal")
    grants_client = web_client = None
    if playbook == "discovery":
        if not settings.egress.enabled:
            raise HTTPException(400, "discovery needs egress.enabled; it is disabled in this config")
        # The guard's corpus is built from the session's uploads, so the institution's
        # own words are blocked from leaving before the first request is ever made.
        guard = _build_egress_guard(settings, vault_dir, bus, state)
        grants_client = GrantsGovClient(guard)
        # Fetched pages are saved so a citation to one opens in the reader. Never into a
        # shipped scenario vault: "not data/vault" used to stand for "a session's uploads",
        # but it is also true of every built-in institution but Harrow Creek, so Find
        # Funding runs wrote web pages into data/scenarios/ -- where they became that
        # institution's documents for every later run. Session uploads keep them beside
        # the uploads; everything else gets the run's own folder under runtime/.
        sessions_root = (REPO_ROOT / "runtime" / "sessions").resolve()
        in_session = str(Path(vault_dir).resolve()).startswith(str(sessions_root) + "/")
        web_client = WebResearch(
            guard,
            session_dir=vault_dir if in_session else REPO_ROOT / "runtime" / "runs" / run_id / "web",
            searxng_url=settings.egress.searxng_url,
            max_page_chars=settings.egress.max_page_chars,
        )

    ctx = ToolContext(
        settings=settings,
        llm=llm,
        vault_dir=vault_dir,
        index=_build_index(vault_dir),
        state=state,
        event_bus=bus,
        runtime_dir=runtime_dir,
        institution_name=_institution_name(scenario),
        partner_names=_partner_names(scenario),
        tracks=_tracks(scenario),
        budget_rules=_budget_rules(scenario),
        playbook=playbook,
        grants=grants_client,
        web=web_client,
    )

    active = ActiveRun(run_id=run_id, state=state, event_bus=bus)

    def target() -> None:
        bus.emit(
            "run_started",
            {
                "mode": body.mode,
                "scope": body.scope,
                "model": settings.llm.model,
                "base_url_host_class": netstatus.classify_host(httpx.URL(settings.llm.base_url).host or ""),
                "device_label": settings.demo.device_label,
                "started_at": _now_iso(),
            },
        )
        try:
            sections = list(scope_cfg.sections)
            solicitation_info = _solicitation_info(scenario)

            # A generated scenario carries its own solicitation, and that solicitation
            # asks for its own sections. Without this, Story drafted the NSF ATE names
            # from config/default.yaml whatever the scenario was -- an academic medical
            # centre applying to a critical-care programme wrote "Results from Prior
            # NSF Support". Harrow Creek has no `sections` key and keeps the configured
            # names, which are the right ones for the real NSF 24-584 it applies to.
            if body.mode == "story":
                scenario_sections = (_load_requirements_doc(scenario) or {}).get("sections") or []
                if scenario_sections:
                    chosen = scenario_sections[: len(scope_cfg.sections)]
                    sections = [sec["id"] for sec in chosen]
                    ctx.section_titles = {sec["id"]: sec["title"] for sec in chosen}
                    ctx.section_word_targets = {
                        sec["id"]: sec["word_target"] for sec in chosen if sec.get("word_target")
                    }

            # Intake: if this session carried a solicitation, the run works to ITS
            # requirements and ITS sections, not the shipped NSF 24-584 ones. Done in
            # the thread rather than in the request because extraction takes minutes,
            # and it streams progress into this run's own event feed.
            if body.mode == "grade":
                seeded = _seed_submission(body.session_id, state, bus)
                # Grading against an uploaded solicitation too, when one was supplied.
                extracted, _ = _intake_requirements(body.session_id, llm, settings, bus)
                if extracted is not None:
                    state.requirements = extracted.get("requirements", [])
                    ctx.uploaded_solicitation = True
                    ctx.budget_rules = {}  # the scenario's call no longer applies
                    ctx.reviewer_persona = "reviewer for this funding programme"
                    ctx.merit_criteria = []
                    solicitation_info = {
                        **solicitation_info,
                        "solicitation_id": extracted.get("solicitation_id", ""),
                        "program": extracted.get("program", ""),
                    }
                sections = [sec["id"] for sec in seeded]
                ctx.section_titles = {sec["id"]: sec["title"] for sec in seeded}

            if body.mode == "explore":
                extracted, proposed = _intake_requirements(body.session_id, llm, settings, bus)
                if extracted is not None:
                    state.requirements = extracted.get("requirements", [])
                    ctx.uploaded_solicitation = True
                    ctx.budget_rules = {}  # the scenario's call no longer applies
                    ctx.tracks = extracted.get("tracks") or ctx.tracks
                    sections = [sec["id"] for sec in proposed]
                    ctx.section_titles = {sec["id"]: sec["title"] for sec in proposed}
                    ctx.section_word_targets = {sec["id"]: sec["word_target"] for sec in proposed}
                    # Not an NSF panel, and not NSF's merit criteria. The reviewer
                    # scores against what THIS solicitation asks for.
                    ctx.reviewer_persona = "reviewer for this funding programme"
                    ctx.merit_criteria = []
                    solicitation_info = {
                        **solicitation_info,
                        "solicitation_id": extracted.get("solicitation_id", ""),
                        "program": extracted.get("program", ""),
                        "track_name": "",
                        "deadline": extracted.get("demo_deadline", "") or "(not stated in the uploaded document)",
                    }

            run_orchestrator(
                ctx,
                llm,
                body.scope,
                sections,
                settings.agent.max_steps,
                solicitation_info,
                is_cancelled=lambda: active.cancel_holder["cancelled"],
            )
        except Exception as exc:  # noqa: BLE001 -- must not crash the background thread silently
            state.status = "error"
            state.save(runtime_dir)
            bus.emit("error", {"message": str(exc), "recoverable": False})
        finally:
            run_manager.mark_finished(run_id)

    active.thread = threading.Thread(target=target, daemon=True)

    try:
        run_manager.start(active)
    except RunConflict as exc:
        raise HTTPException(409, detail={"active_run_id": exc.run_id})

    active.thread.start()
    return {"run_id": run_id}


async def _sse_event_generator(bus: EventBus, last_event_id: int):
    import queue

    q: queue.Queue[Event] = queue.Queue()
    bus.subscribe(q.put)
    max_seq_sent = last_event_id
    for e in bus.events_since(last_event_id):
        max_seq_sent = e.seq
        yield {"event": e.type, "id": str(e.seq), "data": json.dumps(e.data)}
        if e.type == "run_finished":
            return

    while True:
        try:
            e = await asyncio.to_thread(q.get, True, 0.2)
        except Exception:  # noqa: BLE001 -- queue.Empty via timeout
            continue
        if e.seq <= max_seq_sent:
            continue
        max_seq_sent = e.seq
        yield {"event": e.type, "id": str(e.seq), "data": json.dumps(e.data)}
        if e.type == "run_finished":
            break


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, request: Request) -> EventSourceResponse:
    run = run_manager.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    last_event_id = int(request.headers.get("last-event-id") or request.query_params.get("last_event_id") or 0)
    return EventSourceResponse(_sse_event_generator(run.event_bus, last_event_id))


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, bool]:
    if not run_manager.cancel(run_id):
        raise HTTPException(404, "run not found")
    return {"ok": True}


@app.get("/api/runs/{run_id}/state")
def get_run_state(run_id: str) -> dict[str, Any]:
    """Same disk fallback as the packet endpoints.

    Without it a finished run's packet.pdf kept working while its /state 404'd, because
    `run_manager` is in-memory and Reset clears it. Two endpoints reading the same run
    should not disagree about whether it exists.
    """
    return _load_run_state(run_id).to_dict()


def _load_run_state(run_id: str) -> RunState:
    """The run if it is still in memory, else rehydrate it from disk.

    Runs only live in memory for the life of the process, and Reset clears them, so
    without this a finished run became a 404 while its state.json sat right there.
    """
    run = run_manager.get(run_id)
    if run is not None:
        return run.state
    path = REPO_ROOT / "runtime" / "runs" / run_id / "state.json"
    if not path.is_file():
        raise HTTPException(404, "run not found")
    try:
        return RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(500, f"stored run state is unreadable: {exc}") from exc


@app.get("/api/runs/{run_id}/packet.md")
def get_packet(run_id: str) -> PlainTextResponse:
    """Markdown source. Kept for machine consumption and diffing; humans want the PDF."""
    return PlainTextResponse(_render_packet(_load_run_state(run_id)), media_type="text/markdown")


@app.get("/api/runs/{run_id}/packet.pdf")
def get_packet_pdf(run_id: str, settings: Settings = Depends(get_settings_dep)) -> Response:
    """The hand-off artifact: a paginated PDF a grants director can file or email.

    This exists because serving the packet as text/markdown meant the browser rendered
    a wall of unstyled plain text -- the content was right, the artifact was unusable.
    Content-Disposition is `attachment` so it downloads as a named file rather than
    opening as yet another tab of text.
    """
    state = _load_run_state(run_id)
    # The packet's header names the solicitation, so use the one this run was for.
    info = _solicitation_info(_scenario_or_default(getattr(state, "scenario", None)))
    solicitation = " \u2014 ".join(p for p in (info.get("solicitation_id"), info.get("program")) if p)
    pdf = render_packet_pdf(
        state,
        institution=_institution_name(_scenario_or_default(getattr(state, "scenario", None))),
        solicitation=solicitation,
        device_label=settings.demo.device_label,
        model=settings.llm.model,
    )
    filename = f"grant-machine-packet-{run_id}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class SaveReplayRequest(BaseModel):
    name: str


@app.post("/api/runs/{run_id}/save-replay")
def save_replay_endpoint(run_id: str, body: SaveReplayRequest) -> dict[str, str]:
    run = run_manager.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    src = REPO_ROOT / "runtime" / "runs" / run_id / "events.jsonl"
    if not src.is_file():
        raise HTTPException(400, "no recorded events for this run")
    dest = REPO_ROOT / "data" / "replays" / f"{body.name}.jsonl"
    save_replay(src, dest)
    return {"path": str(dest.relative_to(REPO_ROOT))}


# -- explore uploads / reset ----------------------------------------------------


@app.post("/api/explore/upload")
async def explore_upload(
    file: UploadFile,
    session_id: str | None = None,
    kind: Literal["solicitation", "documents", "proposal"] = "documents",
    settings: Settings = Depends(get_settings_dep),
) -> dict[str, Any]:
    """Accept one document into a session corpus.

    `session_id` is optional and additive: Discover needs several documents (a
    strategic plan, CVs, prior awards) to reach the agent as ONE corpus, and before
    this each upload minted a fresh session so only the last file was ever visible.
    Pass the id returned by the first upload to add to it.

    `kind` says what the file IS, which the two modes need to distinguish:
      - "solicitation" (Explore/Intake) -- the call for proposals to extract
        requirements from. The institution's evidence stays the shipped vault.
      - "documents" (Discover) -- the institution's own material, which becomes the
        searchable corpus.
      - "proposal" (Grade) -- an already-written proposal to assess.
    They land in separate subdirectories so a session can hold both without one being
    mistaken for the other.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.uploads.allowed_extensions:
        raise HTTPException(400, f"unsupported file type: {ext!r}")

    contents = await file.read()
    if len(contents) > settings.uploads.max_bytes:
        raise HTTPException(400, f"file too large: {len(contents)} bytes exceeds {settings.uploads.max_bytes}")

    sessions_root = (REPO_ROOT / "runtime" / "sessions").resolve()
    if session_id:
        # Path-traversal guard: this id comes from the client, and it selects a
        # directory we are about to write into.
        session_dir = (sessions_root / session_id).resolve()
        if not str(session_dir).startswith(str(sessions_root) + "/") or not session_dir.is_dir():
            raise HTTPException(400, "unknown session_id")
    else:
        session_id = uuid.uuid4().hex[:12]
        session_dir = sessions_root / session_id
    target_dir = session_dir / kind
    target_dir.mkdir(parents=True, exist_ok=True)

    # Never trust the client's filename as a path.
    safe_name = Path(file.filename or f"upload{ext}").name or f"upload{ext}"
    dest = target_dir / safe_name
    dest.write_bytes(contents)

    files = sorted(p.name for p in target_dir.iterdir() if p.is_file())
    return {"session_id": session_id, "kind": kind, "filename": dest.name,
            "bytes": len(contents), "files": files}


# -- scenarios -------------------------------------------------------------------


def _scenario_registry() -> dict[str, dict[str, Any]]:
    """Every institution Story mode can run as.

    Story used to be hardwired to `data/vault/` and NSF 24-584's curated requirements,
    so the other scenarios existed but could not be selected — they reached the UI only
    as Discover/Explore/Grade samples. A scenario needs BOTH halves: the institution's
    vault, and something for it to be applying to.
    """
    scenarios: dict[str, dict[str, Any]] = {
        "harrow-creek": {
            "id": "harrow-creek",
            "label": "Harrow Creek Community College",
            "type": "Public two-year college",
            "vault": REPO_ROOT / "data" / "vault",
            "requirements": REPO_ROOT / "data" / "solicitation" / "requirements.json",
            "seed": REPO_ROOT / "data" / "seeds" / "harrow_creek.yaml",
            "solicitation_is_real": True,
            "solicitation_id": "NSF 24-584",
            "solicitation_agency": "NSF",
            "description": "The original scenario: a rural technician college applying to the real NSF ATE solicitation.",
        }
    }
    seeds_dir = REPO_ROOT / "data" / "seeds"
    for seed_path in sorted(seeds_dir.glob("*.yaml")):
        if seed_path.stem == "harrow_creek":
            continue
        vault = REPO_ROOT / "data" / "scenarios" / seed_path.stem
        requirements = vault / "solicitation" / "requirements.json"
        if not vault.is_dir():
            continue
        try:
            seed = yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            continue
        inst = seed.get("institution") or {}
        # A seed says what it applies to. Absent means the older synthetic call.
        sol = seed.get("solicitation") or {}
        sid = seed_path.stem.replace("_", "-")
        scenarios[sid] = {
            "id": sid,
            "label": inst.get("name", sid),
            "type": inst.get("type", ""),
            "vault": vault,
            "requirements": requirements,
            "seed": seed_path,
            "solicitation_is_real": bool(sol.get("real")),
            "solicitation_id": sol.get("id", ""),
            "solicitation_agency": sol.get("agency", ""),
            "description": (seed.get("new_project") or {}).get("title", ""),
        }
    return scenarios


DEFAULT_SCENARIO = "harrow-creek"


@app.get("/api/scenarios")
def get_scenarios() -> dict[str, Any]:
    out = []
    for sc in _scenario_registry().values():
        docs = len([p for p in sc["vault"].glob("*") if p.suffix.lower() in _VAULT_EXTENSIONS])
        out.append({
            "id": sc["id"], "label": sc["label"], "type": sc["type"],
            "description": sc["description"], "documents": docs,
            "solicitation_is_real": sc["solicitation_is_real"],
            "solicitation_id": sc.get("solicitation_id", ""),
            "solicitation_agency": sc.get("solicitation_agency", ""),
            "ready": sc["requirements"].is_file(),
        })
    return {"scenarios": out, "default": DEFAULT_SCENARIO}


def _scenario_or_default(scenario_id: str | None) -> dict[str, Any]:
    registry = _scenario_registry()
    return registry.get(scenario_id or DEFAULT_SCENARIO) or registry[DEFAULT_SCENARIO]


# -- typed institutional profile -------------------------------------------------


class ProfileRequest(BaseModel):
    """What a visitor types instead of producing a document first.

    Real institution names are expected and fine here. `scripts/validate_vault.py`
    enforces the synthetic-institution rule over `data/vault/` at build time only --
    it never sees session data, so nothing about typing a real university's name
    trips it. This text is also treated as the institution's own material by the
    egress guard, meaning it can never leave the device as a search term.
    """

    institution: str = ""
    institution_type: str = ""
    unit: str = ""
    research_areas: str = ""
    investigators: str = ""
    prior_awards: str = ""
    facilities: str = ""
    capacity: str = ""
    constraints: str = ""


_PROFILE_FIELDS = [
    ("institution", "Institution"),
    ("institution_type", "Institution type"),
    ("unit", "Department, centre or unit"),
    ("research_areas", "Research and programme areas"),
    ("investigators", "Lead investigators"),
    ("prior_awards", "Prior awards and outcomes"),
    ("facilities", "Facilities and instrumentation"),
    ("capacity", "Capacity and staffing"),
    ("constraints", "Constraints and eligibility notes"),
]


def _render_profile(body: ProfileRequest) -> str:
    title = body.institution.strip() or "Institution profile"
    lines = [
        "---",
        f"title: {title} — profile",
        "source: entered in the Grant Machine UI",
        f"date: '{_now_iso()[:10]}'",
        "---",
        "",
        f"# {title}",
        "",
    ]
    for field, label in _PROFILE_FIELDS:
        value = (getattr(body, field) or "").strip()
        if not value:
            continue
        lines += [f"## {label}", "", value, ""]
    return "\n".join(lines)


@app.post("/api/session/profile")
def save_profile(body: ProfileRequest, session_id: str | None = None) -> dict[str, Any]:
    """Write a typed profile into the session as a citable Markdown document.

    Deliberately a *document* rather than run configuration: put it where uploads go
    and BM25 indexes it, the agent cites it as [[profile]] like any other source,
    verify_citations and check_claims apply to claims resting on it, and the egress
    guard's SessionCorpus picks it up so typed research descriptions are contained
    exactly as uploaded ones are. Run configuration would get none of that.
    """
    if not any((getattr(body, field) or "").strip() for field, _ in _PROFILE_FIELDS):
        raise HTTPException(400, "the profile is empty — fill in at least one field")

    sessions_root = (REPO_ROOT / "runtime" / "sessions").resolve()
    if session_id:
        session_dir = (sessions_root / session_id).resolve()
        if not str(session_dir).startswith(str(sessions_root) + "/"):
            raise HTTPException(400, "unknown session_id")
    else:
        session_id = uuid.uuid4().hex[:12]
        session_dir = sessions_root / session_id

    target_dir = session_dir / "documents"
    target_dir.mkdir(parents=True, exist_ok=True)
    text = _render_profile(body)
    # Fixed name, so re-saving edits the profile rather than accumulating copies.
    (target_dir / "profile.md").write_text(text, encoding="utf-8")

    return {"session_id": session_id, "doc_id": "profile", "chars": len(text)}


@app.get("/api/session/profile")
def get_profile(session_id: str) -> dict[str, Any]:
    """Read back a saved profile so the form can be re-opened and edited."""
    path = REPO_ROOT / "runtime" / "sessions" / session_id / "documents" / "profile.md"
    if not path.is_file():
        raise HTTPException(404, "no profile saved for this session")
    return {"session_id": session_id, "text": path.read_text(encoding="utf-8")}


# -- sample inputs ---------------------------------------------------------------

_SAMPLES_PATH = REPO_ROOT / "data" / "samples" / "manifest.json"


def _load_samples() -> list[dict[str, Any]]:
    if not _SAMPLES_PATH.is_file():
        return []
    try:
        return json.loads(_SAMPLES_PATH.read_text(encoding="utf-8")).get("samples") or []
    except (json.JSONDecodeError, OSError):
        return []


@app.get("/api/samples")
def get_samples() -> dict[str, Any]:
    """Sample inputs the UI can offer per mode.

    Discover, Explore and Grade all read from `runtime/sessions/`, which ships empty --
    so without these a visitor has nothing to try and no idea what a good input looks
    like. File contents are not returned; the UI only needs the labels.
    """
    return {
        "samples": [
            {k: v for k, v in sample.items() if k != "files"}
            | {"file_count": len(sample.get("files") or []),
               "kinds": sorted({f.get("kind", "documents") for f in sample.get("files") or []})}
            for sample in _load_samples()
        ]
    }


@app.post("/api/samples/{sample_id}/load")
def load_sample(sample_id: str, session_id: str | None = None) -> dict[str, Any]:
    """Copy a sample's files into a session, exactly as uploading them would."""
    sample = next((s for s in _load_samples() if s.get("id") == sample_id), None)
    if sample is None:
        raise HTTPException(404, f"unknown sample: {sample_id}")

    sessions_root = (REPO_ROOT / "runtime" / "sessions").resolve()
    if session_id:
        session_dir = (sessions_root / session_id).resolve()
        if not str(session_dir).startswith(str(sessions_root) + "/"):
            raise HTTPException(400, "unknown session_id")
    else:
        session_id = uuid.uuid4().hex[:12]
        session_dir = sessions_root / session_id

    loaded: list[str] = []
    kinds: set[str] = set()
    for entry in sample.get("files") or []:
        # The manifest is a repo file, not user input, but it addresses the filesystem
        # -- so it is still resolved and confined to data/ rather than trusted.
        source = (REPO_ROOT / entry["path"]).resolve()
        data_root = (REPO_ROOT / "data").resolve()
        if not str(source).startswith(str(data_root) + "/") or not source.is_file():
            raise HTTPException(500, f"sample {sample_id} references a file outside data/: {entry['path']}")
        kind = entry.get("kind", "documents")
        kinds.add(kind)
        target_dir = session_dir / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target_dir / source.name)
        loaded.append(source.name)

    # The UI has to say which corpus the next run will search, and only a sample
    # carrying `documents` displaces the scenario's vault -- a solicitation does not.
    return {"session_id": session_id, "sample_id": sample_id, "files": loaded,
            "kinds": sorted(kinds)}


@app.post("/api/reset")
def reset() -> dict[str, bool]:
    run_manager.reset()
    sessions_dir = REPO_ROOT / "runtime" / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    return {"ok": True}


# -- vault (read-only; backs the UI's citation/evidence source drawer) --------


def _scenario_solicitation_file(scenario: dict[str, Any]) -> Path | None:
    """The human-readable solicitation a scenario applies to.

    Harrow Creek applies to the real NSF 24-584, kept at data/solicitation/. Generated
    scenarios carry theirs beside their requirements as `<name>-solicitation.md`.
    """
    if scenario["id"] == DEFAULT_SCENARIO:
        path = REPO_ROOT / "data" / "solicitation" / "nsf24-584.md"
        return path if path.is_file() else None
    folder = Path(scenario["requirements"]).parent
    found = sorted(folder.glob("*-solicitation.md")) if folder.is_dir() else []
    return found[0] if found else None


def _session_documents_dir(session_id: str | None) -> Path | None:
    if not session_id:
        return None
    sessions_root = (REPO_ROOT / "runtime" / "sessions").resolve()
    directory = (sessions_root / session_id / "documents").resolve()
    if not str(directory).startswith(str(sessions_root) + "/"):
        return None
    return directory if directory.is_dir() else None


def _find_document(doc_id: str, scenario_id: str | None, session_id: str | None) -> Path | None:
    """Where a cited doc_id lives.

    This used to look only in data/vault/, Harrow Creek's vault, so every citation in
    a run for any other institution opened "Document not found". Searched in order:
    the session's uploads (they are the corpus when present), the named scenario, then
    every scenario -- a replay or an old citation may not say which one it came from.
    """
    registry = _scenario_registry()
    dirs = [_session_documents_dir(session_id)]
    if scenario_id in registry:
        dirs.append(registry[scenario_id]["vault"])
    dirs += [sc["vault"] for sc in registry.values()]
    for directory in dirs:
        if directory is None:
            continue
        for ext in _VAULT_EXTENSIONS:
            candidate = Path(directory) / f"{doc_id}{ext}"
            if candidate.is_file():
                return candidate
    # A web page a Find Funding run fetched lives in that run's folder; newest first.
    if doc_id.startswith("web-"):
        pages = sorted((REPO_ROOT / "runtime" / "runs").glob(f"*/web/{doc_id}.md"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if pages:
            return pages[0]
    return None


@app.get("/api/vault/{doc_id}")
def get_vault_document(
    doc_id: str, chunk_id: str | None = None, scenario: str | None = None, session_id: str | None = None
) -> dict[str, Any]:
    """Not in Spec G's endpoint table, but Spec H requires citation chips to
    open a source drawer with the cited excerpt, which needs some read path.

    `solicitation` with a `scenario` returns what that institution is applying to."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", doc_id):
        raise HTTPException(404, "document not found")
    if doc_id == "solicitation" and scenario:
        registry = _scenario_registry()
        path = _scenario_solicitation_file(registry[scenario]) if scenario in registry else None
    else:
        path = _find_document(doc_id, scenario, session_id)
    if path is None:
        raise HTTPException(404, "document not found")
    try:
        doc = load(path, doc_id=doc_id)
    except LoaderError as exc:
        raise HTTPException(500, str(exc)) from exc

    text = doc.text
    if chunk_id:
        chunks = {c.chunk_id: c for c in chunk_document(doc)}
        if chunk_id not in chunks:
            raise HTTPException(404, "chunk not found")
        text = chunks[chunk_id].text

    return {"doc_id": doc_id, "title": doc.title, "chunk_id": chunk_id, "text": text}


# The brief groups documents by what they are for. Vault file names follow one
# convention across every scenario (see scripts/generate_vault.py), so the name is
# enough; first match wins.
_DOC_GROUPS = [
    ("People", ("bio-",)),
    ("Letters of support", ("letter-",)),
    ("Budget & rates", ("budget", "rate-agreement", "cost-share", "subaward")),
    ("Prior work", ("prior-", "declined-")),
    ("Institution", ("profile", "strategic", "research-strategy", "facilities", "capacity",
                     "infrastructure", "core-facilities", "census", "labor-market", "advisory")),
]
_DOC_GROUP_ORDER = [g for g, _ in _DOC_GROUPS] + ["Plans & compliance"]


def _doc_group(doc_id: str) -> str:
    name = doc_id.lower()
    for group, needles in _DOC_GROUPS:
        if any(n in name for n in needles):
            return group
    return "Plans & compliance"


@app.get("/api/scenarios/{scenario_id}/documents")
def get_scenario_documents(scenario_id: str) -> dict[str, Any]:
    """Everything an institution's runs work from, for the document panel and brief.

    Planted issues are included: the UI keeps them behind a presenter toggle so an
    audience watches the agent find them, but this is a local demo and there is
    nothing to protect by withholding them from the API.
    """
    registry = _scenario_registry()
    sc = registry.get(scenario_id)
    if sc is None:
        raise HTTPException(404, f"unknown scenario: {scenario_id}")
    vault = Path(sc["vault"])

    manifest: dict[str, dict[str, Any]] = {}
    try:
        for entry in json.loads((vault / "manifest.json").read_text(encoding="utf-8")).get("documents", []):
            manifest[entry["doc_id"]] = entry
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    documents = []
    for path in sorted(vault.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _VAULT_EXTENSIONS or path.name.startswith("_"):
            continue
        doc_id = path.stem
        entry = manifest.get(doc_id) or {}
        title = entry.get("title")
        if not title:
            try:
                title = load(path, doc_id=doc_id).title
            except LoaderError:
                title = doc_id
        documents.append({
            "doc_id": doc_id,
            "title": title,
            "doc_type": entry.get("doc_type") or path.suffix.lstrip(".").lower(),
            "date": entry.get("date", ""),
            "group": _doc_group(doc_id),
        })
    documents.sort(key=lambda d: (_DOC_GROUP_ORDER.index(d["group"]), d["title"].lower()))

    solicitation = None
    sol_path = _scenario_solicitation_file(sc)
    if sol_path is not None:
        req = _load_requirements_doc(sc) or {}
        try:
            sol_title = load(sol_path, doc_id="solicitation").title
        except LoaderError:
            sol_title = ""
        solicitation = {
            "title": sol_title or req.get("program") or "Solicitation",
            "solicitation_id": req.get("solicitation_id", ""),
            "program": req.get("program", ""),
            "is_real": sc["solicitation_is_real"],
        }

    summary = ""
    try:
        summaries = yaml.safe_load((REPO_ROOT / "data" / "scenario_summaries.yaml").read_text(encoding="utf-8")) or {}
        summary = str(summaries.get(scenario_id) or "").strip()
    except (OSError, yaml.YAMLError):
        pass

    try:
        planted = json.loads((vault / "_planted_issues.json").read_text(encoding="utf-8"))
        if not isinstance(planted, list):
            planted = []
    except (OSError, json.JSONDecodeError):
        planted = []

    return {
        "scenario": {"id": sc["id"], "label": sc["label"], "type": sc["type"],
                     "description": sc["description"], "solicitation_is_real": sc["solicitation_is_real"],
                     "summary": summary},
        "documents": documents,
        "groups": [g for g in _DOC_GROUP_ORDER if any(d["group"] == g for d in documents)],
        "solicitation": solicitation,
        "planted_issues": planted,
    }


# -- replays --------------------------------------------------------------------


@app.get("/api/replays")
def get_replays() -> dict[str, Any]:
    return {"replays": list_replays(REPO_ROOT / "data" / "replays")}


class PlayReplayRequest(BaseModel):
    speed: Literal[1, 2, 4] = 1


@app.post("/api/replays/{name}/play")
def play_replay(name: str, body: PlayReplayRequest) -> dict[str, str]:
    replay_path = REPO_ROOT / "data" / "replays" / f"{name}.jsonl"
    if not replay_path.is_file():
        raise HTTPException(404, "replay not found")

    run_id = f"replay-{uuid.uuid4().hex[:8]}"
    bus = EventBus(run_id=run_id)
    state = RunState(run_id=run_id, mode="replay", scope="replay")
    active = ActiveRun(run_id=run_id, state=state, event_bus=bus)

    def target() -> None:
        for gap, event in iter_replay_events(replay_path, speed=body.speed):
            if gap > 0:
                time.sleep(gap)
            data = dict(event["data"])
            if event["type"] == "run_started":
                data["mode"] = "replay"
            bus.emit(event["type"], data)
        state.status = "completed"
        run_manager.mark_finished(run_id)

    active.thread = threading.Thread(target=target, daemon=True)
    try:
        run_manager.start(active)
    except RunConflict as exc:
        raise HTTPException(409, detail={"active_run_id": exc.run_id})
    active.thread.start()
    return {"run_id": run_id}


# -- static UI (mounted last so it never shadows an /api route) ----------------

_web_dir = REPO_ROOT / "web"


def _asset_build_id() -> str:
    """A token that changes whenever the UI files change.

    The git sha is not enough on its own: during development the files change
    without a commit, and a browser holding a stale app.js is exactly the failure
    this exists to prevent.
    """
    stamp = 0.0
    if _web_dir.is_dir():
        for path in sorted(_web_dir.rglob("*")):
            if path.is_file():
                stamp = max(stamp, path.stat().st_mtime)
    return f"{_git_short_sha()}-{int(stamp)}"


class _NoStoreStatic(StaticFiles):
    """Serve the UI with caching off.

    A demo box had the Demo Center loading `/?v=<timestamp>`, so index.html was always
    fresh -- while `app.js` and `styles.css`, referenced as bare relative paths with no
    cache headers at all, were served from the browser's cache and never revalidated.
    The operator's browser was running a build from before the scenario picker existed:
    the markup had the dropdown, the code that fills it did not. Two separate "you
    didn't fix it" reports came out of that, and from the server the two builds are
    indistinguishable.

    These are a handful of small files over loopback or a LAN. There is nothing to gain
    by caching them and a whole class of phantom bug to lose.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Any:
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["pragma"] = "no-cache"
        return response


@app.get("/", include_in_schema=False)
def index() -> Response:
    """index.html with the asset URLs version-stamped.

    no-store above fixes every future load; this also defeats anything holding a copy
    from before it -- the browser cannot reuse a cached `app.js` when the page asks for
    `app.js?v=<build>`.
    """
    path = _web_dir / "index.html"
    if not path.is_file():
        raise HTTPException(404, "UI not installed")
    build = _asset_build_id()
    html = path.read_text(encoding="utf-8")
    html = html.replace('href="styles.css"', f'href="styles.css?v={build}"')
    html = html.replace('src="app.js"', f'src="app.js?v={build}"')
    return Response(
        html,
        media_type="text/html",
        headers={"cache-control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/build", include_in_schema=False)
def build_info() -> dict[str, str]:
    """What the server is serving, so "which build is that browser running?" has an
    answer without guessing."""
    return {"build": _asset_build_id(), "version": _git_short_sha()}


if _web_dir.is_dir():
    app.mount("/", _NoStoreStatic(directory=str(_web_dir), html=True), name="web")


def main() -> None:
    settings = get_settings()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
