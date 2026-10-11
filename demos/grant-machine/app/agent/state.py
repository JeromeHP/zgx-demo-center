"""Run state: compliance matrix, drafts, scores, issues (Phase 6.1).

Persisted as JSON under runtime/runs/<run_id>/ at each step; drafts also
saved as drafts/<section_id>_v<n>.md for human inspection/download.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ComplianceEntry:
    requirement_id: str
    status: str = "missing"  # met | partial | missing | needs_human | not_applicable
    evidence: list[str] = field(default_factory=list)
    note: str = ""
    updated_step: int = 0


@dataclass
class DraftVersion:
    version: int
    text: str
    citations: list[str] = field(default_factory=list)
    word_count: int = 0


@dataclass
class ReviewResult:
    version: int
    criteria: list[dict[str, Any]] = field(default_factory=list)
    mean_score: float = 0.0
    passed: bool = False
    must_fix: list[str] = field(default_factory=list)


@dataclass
class CitationCheckRecord:
    version: int
    passed: bool
    missing_docs: list[str] = field(default_factory=list)
    unsupported_claims: list[dict[str, Any]] = field(default_factory=list)
    uncited_numbers: list[dict[str, Any]] = field(default_factory=list)
    malformed_citations: list[str] = field(default_factory=list)
    citation_count: int = 0


@dataclass
class Issue:
    id: str
    source: str  # "check" | "agent"
    severity: str
    title: str
    detail: str
    evidence: list[str] = field(default_factory=list)
    code: str | None = None
    recommended_action: str | None = None


@dataclass
class ProposedOpportunity:
    """A funding opportunity the agent is recommending the institution consider.

    `opportunity_id` and `number` are not free text: the finish gate checks them
    against what the catalogue actually returned this run, so the agent cannot
    recommend a grant it never found.
    """

    opportunity_id: str
    number: str
    title: str
    agency: str = ""
    close_date: str = ""
    url: str = ""
    fit_rationale: str = ""
    citations: list[str] = field(default_factory=list)
    confidence: str = "medium"  # high | medium | low
    # Which of the institution's own projects, priorities or needs this would serve, in
    # the agent's words -- the "why us" a research office reads first.
    aligned_with: list[str] = field(default_factory=list)
    announcement_url: str = ""
    award_ceiling: str = ""
    cost_sharing: bool | None = None
    concerns: list[str] = field(default_factory=list)


@dataclass
class RunState:
    run_id: str
    mode: str = "story"  # "story" | "explore" | "discover" | "grade" | "replay"
    # Which institution this run was for, so a packet rendered after the fact names the
    # right solicitation rather than the default one.
    scenario: str = ""
    scope: str = "quick"
    requirements: list[dict[str, Any]] = field(default_factory=list)
    compliance: dict[str, ComplianceEntry] = field(default_factory=dict)
    drafts: dict[str, list[DraftVersion]] = field(default_factory=dict)
    reviews: dict[str, list[ReviewResult]] = field(default_factory=dict)
    citation_checks: dict[str, list[CitationCheckRecord]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    tool_calls_made: dict[str, int] = field(default_factory=dict)
    # -- discovery (v2 Phase 1) --
    # Opportunities seen this run, keyed by grants.gov id. Populated by searches and
    # detail lookups; this is the set the finish gate validates proposals against.
    opportunities_seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    proposals: list[ProposedOpportunity] = field(default_factory=list)
    # Counted by the search tool itself rather than read off tool_calls_made, which
    # only the orchestrator maintains -- the finish gate must hold however the tool
    # is driven.
    searches_run: int = 0
    egress_log: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    status: str = "running"  # running | completed | incomplete | cancelled | error
    summary: str = ""

    def record_tool_call(self, name: str) -> None:
        self.tool_calls_made[name] = self.tool_calls_made.get(name, 0) + 1

    def latest_draft(self, section_id: str) -> DraftVersion | None:
        versions = self.drafts.get(section_id) or []
        return versions[-1] if versions else None

    def latest_review(self, section_id: str) -> ReviewResult | None:
        reviews = self.reviews.get(section_id) or []
        return reviews[-1] if reviews else None

    def latest_citation_check(self, section_id: str) -> CitationCheckRecord | None:
        checks = self.citation_checks.get(section_id) or []
        return checks[-1] if checks else None

    def add_issue(self, issue: Issue) -> None:
        self.issues.append(issue)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "scenario": self.scenario,
            "scope": self.scope,
            "requirements": self.requirements,
            "compliance": {k: asdict(v) for k, v in self.compliance.items()},
            "drafts": {k: [asdict(v) for v in vs] for k, vs in self.drafts.items()},
            "reviews": {k: [asdict(v) for v in vs] for k, vs in self.reviews.items()},
            "citation_checks": {k: [asdict(v) for v in vs] for k, vs in self.citation_checks.items()},
            "issues": [asdict(i) for i in self.issues],
            "tool_calls_made": self.tool_calls_made,
            "opportunities_seen": self.opportunities_seen,
            "proposals": [asdict(p) for p in self.proposals],
            "searches_run": self.searches_run,
            "egress_log": self.egress_log,
            "steps": self.steps,
            "status": self.status,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        """Rehydrate a run persisted by `save()`.

        Runs live in memory for the life of the process, so before this existed a
        restart made every finished run's packet a 404 even though its state.json was
        sitting on disk. Unknown keys are ignored so an older state.json still loads.
        """
        state = cls(
            run_id=data.get("run_id", ""),
            mode=data.get("mode", "story"),
            scenario=data.get("scenario", ""),
            scope=data.get("scope", "quick"),
            requirements=data.get("requirements") or [],
        )
        state.compliance = {
            k: ComplianceEntry(**v) for k, v in (data.get("compliance") or {}).items()
        }
        state.drafts = {
            k: [DraftVersion(**v) for v in vs] for k, vs in (data.get("drafts") or {}).items()
        }
        state.reviews = {
            k: [ReviewResult(**v) for v in vs] for k, vs in (data.get("reviews") or {}).items()
        }
        state.citation_checks = {
            k: [CitationCheckRecord(**v) for v in vs]
            for k, vs in (data.get("citation_checks") or {}).items()
        }
        state.issues = [Issue(**i) for i in (data.get("issues") or [])]
        state.tool_calls_made = data.get("tool_calls_made") or {}
        state.opportunities_seen = data.get("opportunities_seen") or {}
        state.proposals = [ProposedOpportunity(**p) for p in (data.get("proposals") or [])]
        state.searches_run = data.get("searches_run", 0)
        state.egress_log = data.get("egress_log") or []
        state.steps = data.get("steps", 0)
        state.status = data.get("status", "running")
        state.summary = data.get("summary", "")
        return state

    def save(self, runtime_dir: Path) -> None:
        run_dir = runtime_dir / "runs" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "state.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_draft_file(self, runtime_dir: Path, section_id: str, version: int, text: str) -> None:
        drafts_dir = runtime_dir / "runs" / self.run_id / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        (drafts_dir / f"{section_id}_v{version}.md").write_text(text, encoding="utf-8")
