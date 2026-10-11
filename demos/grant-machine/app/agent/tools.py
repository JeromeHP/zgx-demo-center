"""Tool schemas + implementations (Spec E). Schemas are strict
(additionalProperties: false) and validated server-side with jsonschema
before execution; invalid args never crash the orchestrator loop."""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import jsonschema

from app.agent.prompts import (
    SECTION_TITLES,
    SECTION_WORD_TARGETS,
    drafter_prompt,
    reviewer_prompt,
)
from app.agent.state import (
    CitationCheckRecord,
    ComplianceEntry,
    DraftVersion,
    Issue,
    ProposedOpportunity,
    ReviewResult,
    RunState,
)
from app.checks.budget import BudgetRules, check_budget as run_budget_check
from app.checks.citations import verify_citations as run_citation_check
from app.checks.claims import check_claims as run_claim_check
from app.ingest.chunker import chunk_document
from app.ingest.loaders import load
from app.llm_client import LLMClient, Usage

SECTION_IDS = list(SECTION_TITLES.keys())

# Shared by the single- and batch- forms of update_compliance.
_COMPLIANCE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requirement_id": {"type": "string"},
        "status": {"type": "string", "enum": ["met", "partial", "missing", "needs_human", "not_applicable"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "note": {"type": "string", "maxLength": 400},
    },
    "required": ["requirement_id", "status"],
}

# The institution profile lives in the normal drafts store under this id, so
# verify_citations and the drafts UI apply to it with no special-casing.
_PROFILE_SECTION = "institution_profile"



def _check_section(ctx: "ToolContext", section_id: str, *, allow_profile: bool = False) -> dict[str, Any] | None:
    """Validate a section id against THIS run's sections. None means it is fine.

    These used to be a JSON-schema enum built from the three hardcoded NSF ATE
    section names. That broke twice: once when the institution profile became a
    citable draft, and again when Intake began deriving sections from an uploaded
    solicitation -- in both cases the model asked for exactly the right thing and was
    told it "is not one of [...]", with no way to discover what it should have said.
    Validating here instead means the error can name the sections that actually exist.
    """
    allowed = list(ctx.scope_sections)
    if allow_profile:
        allowed.append(_PROFILE_SECTION)
    if section_id in allowed:
        return None
    return {
        "error": "unknown_section",
        "detail": f"{section_id!r} is not a section in this run. Sections in scope: {allowed}",
    }
_VAULT_EXTENSIONS = (".md", ".html", ".htm", ".pdf", ".csv")
# Chunk ids are already `doc_id#cNNN`, so a model shown both a doc_id and a chunk_id
# writes [[doc#doc#c000]] often enough to matter. The two-segment pattern did not
# match that at all, which meant the citation was invisible -- and a draft whose
# every citation was malformed passed verification with nothing to verify.
_CITATION_RE = re.compile(r"\[\[([A-Za-z0-9_\-]+)(?:#([A-Za-z0-9_\-]+))?(?:#([A-Za-z0-9_\-]+))?\]\]")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_requirements",
            "description": "List solicitation requirements, optionally filtered by category.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "project_description",
                            "merit_review",
                            "program_specific",
                            "formatting",
                            "budget",
                            "supplementary_documents",
                            "eligibility",
                        ],
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": "Search the institutional document vault with BM25.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "minLength": 3, "maxLength": 200},
                    "k": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read a vault document, or one chunk of it.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "doc_id": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 6000},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_budget",
            "description": "Run the deterministic budget check against the draft budget.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"track": {"type": "string", "enum": ["track_2"]}},
                "required": ["track"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_section",
            "description": "Draft a proposal section from vault evidence, with inline citations.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "section_id": {"type": "string"},  # validated against the run's sections, not a fixed enum
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"doc_id": {"type": "string"}, "chunk_id": {"type": "string"}},
                            "required": ["doc_id"],
                        },
                    },
                    "guidance": {"type": "string", "maxLength": 1500},
                },
                "required": ["section_id", "evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_section",
            "description": "Score the latest draft of a section against linked requirements and NSF merit review criteria.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"section_id": {"type": "string"}},
                "required": ["section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_citations",
            "description": (
                "Verify every citation and number claim in the latest draft of a section, "
                "or of the institution profile."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                # The profile is a citable draft like any other, and the discovery
                # finish gate requires it to be verified -- so it has to be nameable
                # here. Leaving it out of the enum meant the agent asked for exactly
                # the right check, got "'institution_profile' is not one of [...]",
                # and then looped on finish twelve times because the gate it could
                # not satisfy was the gate it was being held to.
                "properties": {"section_id": {"type": "string"}},
                "required": ["section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_compliance",
            "description": (
                "Update the compliance matrix. Pass `items` to rule on several requirements "
                "in one call, which is far cheaper than one call each."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item": _COMPLIANCE_ITEM_SCHEMA,
                    "items": {"type": "array", "items": _COMPLIANCE_ITEM_SCHEMA, "maxItems": 60},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_issue",
            "description": "Flag a problem a human grant writer would need to catch.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "issue": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                            "title": {"type": "string", "maxLength": 120},
                            "detail": {"type": "string", "maxLength": 800},
                            "evidence": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}},
                            "recommended_action": {"type": "string", "maxLength": 400},
                        },
                        "required": ["severity", "title", "detail", "evidence"],
                    }
                },
                "required": ["issue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_claims",
            "description": (
                "Cross-check the factual claims in a section of the submission against the "
                "institution's documentation. Reports each numeric claim as supported, "
                "contradicted, or unverifiable. Contradictions are what a panel catches."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"section_id": {"type": "string"}},
                "required": ["section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_profile",
            "description": (
                "Write the institution's research/capability profile from the uploaded documents. "
                "Cite every claim with [[doc_id]]. This is what the funding search is based on, so "
                "it must describe real, documented strengths -- not aspirations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The profile in Markdown, with [[doc_id]] citations."},
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Generic, published field names to search funding catalogues with "
                            "(e.g. 'robotics', 'workforce development'). These leave the device, so "
                            "they must be public field language -- never project names, people, or "
                            "phrases taken from the documents."
                        ),
                    },
                },
                "required": ["text", "search_terms"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_grants",
            "description": (
                "Search the federal funding catalogue (grants.gov) by field terms. Terms are "
                "validated against a published vocabulary before anything leaves the device; "
                "rejected terms come back explained so you can choose different ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "rows": {"type": "integer", "minimum": 1, "maximum": 50},
                    "statuses": {"type": "string", "description": "e.g. 'posted|forecasted'"},
                },
                "required": ["terms"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_opportunity",
            "description": "Full detail for one opportunity returned by search_grants, by its numeric id.",
            "parameters": {
                "type": "object",
                "properties": {"opportunity_id": {"type": "string"}},
                "required": ["opportunity_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Retrieve a public web page (e.g. a program page linked from an opportunity) and "
                "register it as a citable document. Returns a doc_id you can then cite."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "General web search, when a provider is configured. Same term validation as "
                "search_grants. Use for agency priorities and programme context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["terms"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_opportunity",
            "description": (
                "Recommend an opportunity the institution should consider. Only opportunities "
                "actually returned by search_grants or get_opportunity this run can be proposed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunity_id": {"type": "string"},
                    "fit_rationale": {"type": "string", "description": "Why it fits, citing [[doc_id]] evidence."},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "concerns": {"type": "array", "items": {"type": "string"},
                                 "description": "Eligibility or capacity problems a human must check."},
                    "aligned_with": {"type": "array", "items": {"type": "string"},
                                     "description": ("The institution's own projects, priorities or needs this "
                                                     "opportunity would serve, each a short phrase from its "
                                                     "documents, e.g. 'coastal flood forecasting programme'.")},
                },
                "required": ["opportunity_id", "fit_rationale", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the run with a short summary for the grants director.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"summary": {"type": "string", "maxLength": 1500}},
                "required": ["summary"],
            },
        },
    },
]


_SCHEMA_BY_NAME = {t["function"]["name"]: t["function"]["parameters"] for t in TOOL_SCHEMAS}


@dataclass
class ToolContext:
    settings: Any
    llm: LLMClient
    vault_dir: Path
    index: Any  # app.retrieval.bm25_index.BM25Index
    state: RunState
    event_bus: Any  # app.events.EventBus
    runtime_dir: Path
    institution_name: str = "the institution"
    tracks: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"track_2": {"max_total_usd": 1_000_000, "max_years": 3}}
    )
    scope_sections: list[str] = field(default_factory=list)
    # The scenario solicitation's money rules (requirements.json "budget_rules"): cap,
    # duration, salary cap, subaward MTDC limit... Empty means the historical defaults.
    budget_rules: dict[str, Any] = field(default_factory=dict)
    # -- discovery (v2 Phase 1). None in a proposal run; the discovery tools refuse
    # cleanly rather than crash if they are somehow reached without them.
    # Section titles/word targets for THIS run. Empty means the shipped NSF ATE
    # defaults; Intake fills them in from the uploaded solicitation.
    # Who the reviewer is told they are. Solicitation-derived under Intake.
    reviewer_persona: str = "NSF ATE panel reviewer"
    merit_criteria: list[str] = field(default_factory=lambda: ["Intellectual Merit", "Broader Impacts"])
    section_titles: dict[str, str] = field(default_factory=dict)
    section_word_targets: dict[str, str] = field(default_factory=dict)
    # True when the requirements came from a solicitation uploaded this session, so
    # nothing about the shipped scenarios (partner letters, merit criteria) applies.
    uploaded_solicitation: bool = False
    # The partners whose support letters this scenario's vault holds, named in the
    # quick-scope guidance so the agent checks THIS institution's letters.
    partner_names: list[str] = field(default_factory=list)
    playbook: str = "proposal"  # "proposal" | "discovery"
    grants: Any = None          # app.discovery.GrantsGovClient
    web: Any = None             # app.discovery.WebResearch
    total_usage: Usage = field(default_factory=lambda: Usage(0, 0, estimated=False))
    llm_call_count: int = 0
    last_decode_tps: float | None = None

    def add_usage(self, usage: Usage) -> None:
        self.total_usage.prompt_tokens += usage.prompt_tokens
        self.total_usage.completion_tokens += usage.completion_tokens
        self.total_usage.estimated = self.total_usage.estimated or usage.estimated

    def note_llm_call(self, timings: Any) -> None:
        self.llm_call_count += 1
        if getattr(timings, "decode_tps", None):
            self.last_decode_tps = timings.decode_tps

    def known_doc_ids(self) -> set[str]:
        return {p.stem for p in self.vault_dir.glob("*") if p.suffix.lower() in _VAULT_EXTENSIONS}

    def find_doc_path(self, doc_id: str) -> Path | None:
        for ext in _VAULT_EXTENSIONS:
            candidate = self.vault_dir / f"{doc_id}{ext}"
            if candidate.is_file():
                return candidate
        return None


# -- tool implementations ----------------------------------------------------


def tool_get_requirements(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    category = args.get("category")
    reqs = ctx.state.requirements
    if not ctx.state.compliance and ctx.state.requirements:
        for r in ctx.state.requirements:
            ctx.state.compliance[r["id"]] = ComplianceEntry(requirement_id=r["id"], status="missing", updated_step=ctx.state.steps)
        ctx.event_bus.emit(
            "compliance_update",
            {"requirement_id": "*", "status": "initialized", "evidence": [], "note": f"{len(ctx.state.requirements)} requirements loaded"},
        )
    if category:
        reqs = [r for r in reqs if r.get("category") == category]
    return {
        "requirements": [
            {"id": r["id"], "category": r.get("category"), "text": r.get("text"), "section_link": r.get("section_link")}
            for r in reqs
        ]
    }


def tool_search_vault(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    hits = ctx.index.search(args["query"], k=args.get("k", 5))
    return {"hits": [asdict(h) for h in hits]}


def tool_read_document(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    doc_id = args["doc_id"]
    chunk_id = args.get("chunk_id")
    max_chars = args.get("max_chars", 4000)

    path = ctx.find_doc_path(doc_id)
    if path is None:
        close = difflib.get_close_matches(doc_id, sorted(ctx.known_doc_ids()), n=3)
        return {"error": "not_found", "doc_id": doc_id, "closest_matches": close}

    doc = load(path, doc_id=doc_id)
    if chunk_id:
        chunks = {c.chunk_id: c for c in chunk_document(doc)}
        if chunk_id not in chunks:
            return {"error": "chunk_not_found", "doc_id": doc_id, "chunk_id": chunk_id, "valid_chunk_ids": sorted(chunks)}
        text = chunks[chunk_id].text
    else:
        text = doc.text

    truncated = len(text) > max_chars
    return {
        "doc_id": doc_id,
        "title": doc.title,
        "front_matter": doc.front_matter,
        "text": text[:max_chars],
        "truncated": truncated,
    }


def _rule(ctx: ToolContext, key: str) -> Any:
    raw = ctx.budget_rules.get(key)
    return raw.get("value") if isinstance(raw, dict) else raw


def _track_cap(ctx: ToolContext, track: str) -> int | None:
    # A scenario's curated rules win; "cap_usd": null means the call has no ceiling.
    if "cap_usd" in ctx.budget_rules:
        cap = _rule(ctx, "cap_usd")
        return int(cap) if cap is not None else None
    return int(ctx.tracks.get(track, {}).get("max_total_usd", 1_000_000))


def _track_years(ctx: ToolContext, track: str) -> int:
    if _rule(ctx, "max_years") is not None:
        return int(_rule(ctx, "max_years"))
    return int(ctx.tracks.get(track, {}).get("max_years", 3))


def tool_check_budget(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    track = args["track"]
    result = run_budget_check(
        ctx.vault_dir,
        cap_usd=_track_cap(ctx, track),
        track_years=_track_years(ctx, track),
        equipment_threshold_usd=ctx.settings.budget.equipment_threshold_usd,
        rules=BudgetRules.from_dict(ctx.budget_rules) if ctx.budget_rules else None,
    )
    for issue in result.issues:
        state_issue = Issue(
            id=f"check-{issue.code}-{len(ctx.state.issues)}",
            source="check",
            severity=issue.severity,
            title=issue.code,
            detail=issue.detail,
            evidence=issue.evidence,
            code=issue.code,
        )
        ctx.state.add_issue(state_issue)
        ctx.event_bus.emit("issue_flagged", {"issue": asdict(state_issue)})
    return {
        "total_usd": result.total_usd,
        "cap_usd": result.cap_usd,
        "over_by_usd": result.over_by_usd,
        "fringe_rate_budget": result.fringe_rate_budget,
        "fringe_rate_agreement": result.fringe_rate_agreement,
        "total_if_agreement_fringe_usd": result.total_if_agreement_fringe_usd,
        "issues": [asdict(i) for i in result.issues],
    }


def tool_draft_section(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    section_id = args["section_id"]
    bad = _check_section(ctx, section_id)
    if bad:
        return bad
    guidance = args.get("guidance", "")

    existing_versions = ctx.state.drafts.get(section_id, [])
    max_revisions = ctx.settings.agent.max_revisions_per_section
    if len(existing_versions) >= max_revisions + 1:
        # Guardrail 3 (Section 6.3): after the cap, review_section marks the
        # section needs_human -- but that alone doesn't stop the model from
        # calling draft_section again, which was letting runs draft the same
        # section 5-7 times in a row (one e2e rep hit 60 steps / 1.16M prompt
        # tokens / 29 minutes wall time this way). Refuse outright instead.
        return {
            "error": "revision_cap_reached",
            "detail": (
                f"{section_id} has already used all {max_revisions} allowed revisions "
                f"(version {len(existing_versions)} exists) and has been marked needs_human. "
                "Do not draft this section again; move on to other work."
            ),
        }

    evidence_blocks: list[str] = []
    for ref in args["evidence"]:
        result = tool_read_document(ctx, {"doc_id": ref["doc_id"], "chunk_id": ref.get("chunk_id"), "max_chars": 4000})
        if "error" in result:
            continue
        label = f"[[{ref['doc_id']}#{ref['chunk_id']}]]" if ref.get("chunk_id") else f"[[{ref['doc_id']}]]"
        evidence_blocks.append(f"{label} {result['title']}\n{result['text']}")

    linked_reqs = [r for r in ctx.state.requirements if r.get("section_link") == section_id]
    linked_text = "\n".join(f"- {r['id']}: {r['text']}" for r in linked_reqs) or "(none linked)"

    prior_versions = ctx.state.drafts.get(section_id, [])
    revision_guidance = ""
    if prior_versions:
        review = ctx.state.latest_review(section_id)
        check = ctx.state.latest_citation_check(section_id)
        lines = []
        if review and review.must_fix:
            lines.append("Revise to address:")
            lines.extend(f"- {m}" for m in review.must_fix)
        if check and check.unsupported_claims:
            lines.append("Remove or fix these unsupported claims:")
            lines.extend(f"- {c.get('number')}: \"{c.get('sentence')}\"" for c in check.unsupported_claims)
        revision_guidance = "\n".join(lines)

    prompt = drafter_prompt(
        institution_name=ctx.institution_name,
        section_title=ctx.section_titles.get(section_id) or SECTION_TITLES.get(section_id, section_id.replace("_", " ").title()),
        linked_requirements=linked_text,
        evidence_blocks="\n\n".join(evidence_blocks) or "(no evidence provided)",
        word_target=ctx.section_word_targets.get(section_id) or SECTION_WORD_TARGETS.get(section_id, "400-700 words"),
        revision_guidance=revision_guidance,
    )

    def on_content(delta: str) -> None:
        ctx.event_bus.emit("draft_delta", {"section_id": section_id, "delta": delta})

    version = len(prior_versions) + 1
    ctx.event_bus.emit("draft_started", {"section_id": section_id, "version": version})
    result = ctx.llm.chat(
        "drafter",
        [{"role": "system", "content": prompt}, {"role": "user", "content": guidance or "Draft the section now."}],
        stream=True,
        on_content=on_content,
    )
    ctx.add_usage(result.usage)
    ctx.note_llm_call(result.timings)

    text = result.content.strip()
    citations = sorted({m.group(0) for m in _CITATION_RE.finditer(text)})
    word_count = len(text.split())
    draft = DraftVersion(version=version, text=text, citations=citations, word_count=word_count)
    ctx.state.drafts.setdefault(section_id, []).append(draft)
    ctx.state.save_draft_file(ctx.runtime_dir, section_id, version, text)
    ctx.event_bus.emit(
        "draft_completed", {"section_id": section_id, "version": version, "word_count": word_count, "citations": citations}
    )
    return {"section_id": section_id, "version": version, "word_count": word_count, "citations": citations}


def tool_review_section(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    import json

    section_id = args["section_id"]
    bad = _check_section(ctx, section_id)
    if bad:
        return bad
    draft = ctx.state.latest_draft(section_id)
    if draft is None:
        return {"error": "no_draft", "detail": f"No draft exists yet for {section_id}; call draft_section first."}

    linked_reqs = [r for r in ctx.state.requirements if r.get("section_link") == section_id]
    criteria_lines = [f"- {r['id']}: {r['text']}" for r in linked_reqs]
    # Intellectual Merit / Broader Impacts are NSF's review criteria specifically.
    # Appending them to a review of somebody else's solicitation would score a draft
    # against criteria that solicitation never asked for.
    if ctx.merit_criteria:
        criteria_lines += [f"- {c}" for c in ctx.merit_criteria]

    # An example instance (not a JSON Schema) -- models reliably echo an
    # abstract schema's "type"/"properties" wrapper back verbatim instead of
    # producing a compliant instance; a concrete example doesn't have that trap.
    example = {
        "criteria": [
            {"id": "<requirement id or criterion name>", "score": 4, "rationale": "<one sentence>", "gaps": ["<gap>"]}
        ],
        "mean_score": 4.0,
        "must_fix": ["<smallest change needed>"],
    }
    prompt = reviewer_prompt(
        reviewer_persona=ctx.reviewer_persona,
        criteria_list="\n".join(criteria_lines),
        threshold=ctx.settings.agent.review_pass_threshold,
        json_example=json.dumps(example),
    )
    result = ctx.llm.chat("reviewer", [{"role": "system", "content": prompt}, {"role": "user", "content": draft.text}], json_mode=True)
    ctx.add_usage(result.usage)
    ctx.note_llm_call(result.timings)

    try:
        parsed = json.loads(result.content)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    if "criteria" not in parsed and isinstance(parsed.get("properties"), dict):
        # Defensive unwrap: some models echo a schema-shaped wrapper
        # ({"type": "object", "properties": {...}}) despite the prompt asking
        # for an instance; the prompt wording change above should make this
        # rare, but this keeps a stray occurrence from silently zeroing a review.
        parsed = parsed["properties"]

    criteria = parsed.get("criteria") or []
    mean_score = parsed.get("mean_score")
    if mean_score is None and criteria:
        mean_score = sum(c.get("score", 0) for c in criteria) / len(criteria)
    mean_score = float(mean_score or 0.0)
    must_fix = parsed.get("must_fix") or []
    passed = mean_score >= ctx.settings.agent.review_pass_threshold and not must_fix

    review = ReviewResult(version=draft.version, criteria=criteria, mean_score=mean_score, passed=passed, must_fix=must_fix)
    ctx.state.reviews.setdefault(section_id, []).append(review)

    revision_count = len(ctx.state.drafts.get(section_id, [])) - 1
    if not passed and revision_count >= ctx.settings.agent.max_revisions_per_section:
        for r in ctx.state.requirements:
            if r.get("section_link") == section_id and r["id"] in ctx.state.compliance:
                entry = ctx.state.compliance[r["id"]]
                entry.status = "needs_human"
                entry.note = "Revision cap reached; must-fix: " + "; ".join(must_fix)
                entry.updated_step = ctx.state.steps
                ctx.event_bus.emit(
                    "compliance_update",
                    {"requirement_id": r["id"], "status": "needs_human", "evidence": entry.evidence, "note": entry.note},
                )

    ctx.event_bus.emit(
        "section_scored",
        {"section_id": section_id, "version": draft.version, "mean_score": mean_score, "passed": passed, "criteria": criteria, "must_fix": must_fix},
    )
    return {"version": draft.version, "criteria": criteria, "mean_score": mean_score, "passed": passed, "must_fix": must_fix}


def tool_verify_citations(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    section_id = args["section_id"]
    bad = _check_section(ctx, section_id, allow_profile=True)
    if bad:
        return bad
    draft = ctx.state.latest_draft(section_id)
    if draft is None:
        return {"error": "no_draft", "detail": f"No draft exists yet for {section_id}; call draft_section first."}

    result = run_citation_check(draft.text, ctx.vault_dir, known_doc_ids=ctx.known_doc_ids())
    record = CitationCheckRecord(
        version=draft.version,
        passed=result.passed,
        missing_docs=result.missing_docs,
        unsupported_claims=[asdict(u) for u in result.unsupported_claims],
        uncited_numbers=[asdict(u) for u in result.uncited_numbers],
        malformed_citations=result.malformed_citations,
        citation_count=result.citation_count,
    )
    ctx.state.citation_checks.setdefault(section_id, []).append(record)
    ctx.event_bus.emit(
        "citations_checked",
        {
            "section_id": section_id,
            "version": draft.version,
            "passed": result.passed,
            "unsupported_claims": record.unsupported_claims,
            "uncited_numbers": record.uncited_numbers,
            "malformed_citations": record.malformed_citations,
            "citation_count": record.citation_count,
        },
    )
    out: dict[str, Any] = {
        "version": draft.version,
        "passed": result.passed,
        "missing_docs": result.missing_docs,
        "unsupported_claims": record.unsupported_claims,
        "uncited_numbers": record.uncited_numbers,
        "citation_count": result.citation_count,
    }
    # Say what is wrong in words. A bare `passed: false` on a draft whose citations all
    # failed to parse reads as "the evidence is bad" and sends the model rewriting
    # correct prose, when the fix is the bracket syntax.
    if result.malformed_citations:
        out["malformed_citations"] = result.malformed_citations
        out["detail"] = (
            f"{len(result.malformed_citations)} double-bracket marker(s) are not citations "
            "this check can read: " + "; ".join(result.malformed_citations[:5]) + ". "
            "[[...]] means a citation and nothing else: [[doc_id]] or [[doc_id#chunk_id]]. "
            "If one of these is a placeholder for something you could not find, delete it and "
            "say in plain prose what is missing, or flag_issue it -- a draft cannot go to a "
            "funder with an unfilled placeholder in it."
        )
    elif result.citation_count == 0:
        out["detail"] = (
            "This draft contains no citations at all, so there is nothing to verify. Every "
            "factual statement must carry a [[doc_id]] marker naming the vault document it "
            "rests on."
        )
    return out


def tool_update_compliance(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """One requirement, or a batch.

    Grading rules on every requirement in the solicitation -- 54 of them for NSF
    24-584 -- and doing that one call per requirement spent 39 of a 60-step budget on
    bookkeeping, leaving the run no room to finish. The batch form collapses that to a
    couple of steps. Each item still goes through the same guardrail individually.
    """
    items = args.get("items")
    if items is None:
        single = args.get("item")
        if single is None:
            return {"error": "invalid_arguments", "detail": "pass either `item` or `items`"}
        items = [single]

    results = [_update_one_compliance(ctx, i) for i in items]
    if len(results) == 1:
        return results[0]
    rejected = [r for r in results if r.get("error")]
    return {
        "updated": [r["requirement_id"] for r in results if not r.get("error")],
        "rejected": rejected,
        "detail": f"{len(results) - len(rejected)} updated, {len(rejected)} rejected" if rejected else None,
    }


def _update_one_compliance(ctx: ToolContext, item: dict[str, Any]) -> dict[str, Any]:
    requirement_id = item["requirement_id"]
    status = item["status"]

    # The matrix is a claim about THIS solicitation's requirements, so a row that
    # matches no requirement corrupts it. A live run filed rulings under
    # "BUDGET_OVER_CAP" and "FRINGE_RATE_MISMATCH" -- issue codes, not requirement
    # ids -- and setdefault created them silently, giving a 54-requirement
    # solicitation 56 rows, two of them with no requirement text behind them.
    known = {r["id"] for r in ctx.state.requirements}
    if known and requirement_id not in known:
        return {
            "requirement_id": requirement_id,
            "error": "unknown_requirement",
            "detail": (
                f"{requirement_id!r} is not a requirement in this solicitation. Use an id from "
                f"get_requirements (e.g. {sorted(known)[:3]}). To record a problem that is not "
                "tied to a requirement, use flag_issue."
            ),
        }

    if status == "met":
        if ctx.playbook == "grade":
            # A submitted proposal carries no [[doc_id]] markers -- it was written by
            # a human for a funder, not by this agent from a vault -- so the citation
            # gate cannot apply and would simply block every requirement. The grading
            # equivalent is that "met" must point at the part of the submission that
            # meets it, so a reviewer can check the call rather than take it on trust.
            evidence = item.get("evidence") or []
            if not evidence:
                return {
                    "requirement_id": requirement_id,
                    "error": "guardrail_rejected",
                    "detail": (
                        f"Cannot mark {requirement_id} met with no evidence. Give the section id(s) "
                        "of the submission that satisfy it, so the call can be checked."
                    ),
                }
            unknown = [e for e in evidence if e not in ctx.state.drafts]
            if unknown and not any(e in ctx.state.drafts for e in evidence):
                return {
                    "requirement_id": requirement_id,
                    "error": "guardrail_rejected",
                    "detail": (
                        f"Cannot mark {requirement_id} met: none of {unknown} is a section of the "
                        f"submission. Sections present: {sorted(ctx.state.drafts)}"
                    ),
                }
        else:
            req = next((r for r in ctx.state.requirements if r["id"] == requirement_id), None)
            section_id = req.get("section_link") if req else None
            if section_id:
                check = ctx.state.latest_citation_check(section_id)
                if check is None or not check.passed or check.unsupported_claims:
                    return {
                        "requirement_id": requirement_id,
                        "error": "guardrail_rejected",
                        "detail": (
                            f"Cannot mark {requirement_id} met: {section_id} has not passed verify_citations "
                            "with zero unsupported claims."
                        ),
                    }

    entry = ctx.state.compliance.setdefault(requirement_id, ComplianceEntry(requirement_id=requirement_id))
    entry.status = status
    entry.evidence = item.get("evidence", [])
    entry.note = item.get("note", "")
    entry.updated_step = ctx.state.steps
    ctx.event_bus.emit("compliance_update", {"requirement_id": requirement_id, "status": status, "evidence": entry.evidence, "note": entry.note})
    return {"requirement_id": requirement_id, "status": status}


def tool_flag_issue(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    issue_args = args["issue"]
    unknown = [e for e in issue_args["evidence"] if e not in ctx.known_doc_ids()]
    if unknown:
        return {"error": "invalid_evidence", "detail": f"Unknown evidence doc_id(s): {unknown}"}
    issue = Issue(
        id=f"agent-{len(ctx.state.issues)}",
        source="agent",
        severity=issue_args["severity"],
        title=issue_args["title"],
        detail=issue_args["detail"],
        evidence=issue_args["evidence"],
        recommended_action=issue_args.get("recommended_action"),
    )
    ctx.state.add_issue(issue)
    ctx.event_bus.emit("issue_flagged", {"issue": asdict(issue)})
    return {"issue_id": issue.id}


def tool_check_claims(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """The grading analogue of verify_citations.

    A submitted proposal carries no [[doc_id]] markers, so citation verification
    cannot run -- which left the submission's facts entirely unexamined. A test
    proposal claimed 120 completers where the institution's own data recorded 84, and
    the grader praised the section for "exceeding graduation targets".
    """
    section_id = args["section_id"]
    bad = _check_section(ctx, section_id)
    if bad:
        return bad
    draft = ctx.state.latest_draft(section_id)
    if draft is None:
        return {"error": "no_section", "detail": f"no submission text loaded for {section_id}"}

    checks = run_claim_check(draft.text, ctx.index, ctx.llm)
    contradicted = [c for c in checks if c.verdict == "contradicted"]

    record = CitationCheckRecord(
        version=draft.version,
        passed=not contradicted,
        unsupported_claims=[c.as_dict() for c in checks if c.verdict != "supported"],
    )
    ctx.state.citation_checks.setdefault(section_id, []).append(record)

    for c in contradicted:
        issue = Issue(
            id=f"claim-{len(ctx.state.issues) + 1}",
            source="check",
            severity="high",
            title=f"Claim contradicted by the institution's own records ({section_id})",
            detail=f"{c.claim} — {c.detail}",
            evidence=c.evidence,
            code="CLAIM_CONTRADICTED",
            recommended_action="Correct the figure, or drop the claim.",
        )
        ctx.state.add_issue(issue)
        ctx.event_bus.emit("issue_flagged", asdict(issue))

    ctx.event_bus.emit("claims_checked", {
        "section_id": section_id,
        "checked": len(checks),
        "contradicted": len(contradicted),
        "unverifiable": sum(1 for c in checks if c.verdict == "unverifiable"),
        "claims": [c.as_dict() for c in checks],
    })
    return {
        "section_id": section_id,
        "checked": len(checks),
        "contradicted": [c.as_dict() for c in contradicted],
        "unverifiable": [c.as_dict() for c in checks if c.verdict == "unverifiable"],
    }


# -- discovery tools (v2 Phase 1) ---------------------------------------------


def tool_draft_profile(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Record the institution profile as a citable draft.

    Stored under the section id "institution_profile" in the normal drafts store, so
    verify_citations applies to it unchanged -- the profile is held to the same
    evidence standard as a proposal section, which matters because everything the
    search does downstream is derived from it.
    """
    text = (args.get("text") or "").strip()
    if not text:
        return {"error": "empty_profile", "detail": "the profile text is empty"}

    citations = sorted({m.group(0) for m in _CITATION_RE.finditer(text)})
    version = len(ctx.state.drafts.get(_PROFILE_SECTION, [])) + 1
    draft = DraftVersion(version=version, text=text, citations=citations, word_count=len(text.split()))
    ctx.state.drafts.setdefault(_PROFILE_SECTION, []).append(draft)
    ctx.state.save_draft_file(ctx.runtime_dir, _PROFILE_SECTION, version, text)

    # Validate the proposed search terms now, so the operator sees what will and will
    # not be allowed to leave *before* any request is made.
    raw_terms = [t for t in (args.get("search_terms") or []) if isinstance(t, str)]
    decisions = []
    if ctx.grants is not None:
        decisions = [d.as_dict() for d in ctx.grants._guard.validator.validate_all(raw_terms)]

    ctx.event_bus.emit(
        "profile_drafted",
        {"version": version, "text": text, "citations": citations,
         "word_count": draft.word_count, "search_terms": decisions},
    )
    return {
        "version": version,
        "word_count": draft.word_count,
        "citations": citations,
        "search_terms": decisions,
        "note": "Call verify_citations on section_id 'institution_profile' before proposing anything.",
    }


def _remember_opportunities(ctx: ToolContext, opportunities: list[dict[str, Any]]) -> None:
    for opp in opportunities:
        oid = str(opp.get("opportunity_id") or "")
        if oid:
            # Merge, not replace: a later search hit is a thin synopsis and must not wipe
            # the detail (announcement link, ceiling, cost sharing) get_opportunity fetched.
            prior = ctx.state.opportunities_seen.get(oid) or {}
            ctx.state.opportunities_seen[oid] = {**prior, **{k: v for k, v in opp.items() if v not in ("", None, [])}}


def tool_search_grants(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.grants is None:
        return {"error": "discovery_unavailable", "detail": "no funding catalogue client in this run"}

    ctx.state.searches_run += 1
    result = ctx.grants.search(
        [t for t in args.get("terms", []) if isinstance(t, str)],
        rows=args.get("rows", 15),
        statuses=args.get("statuses") or "posted|forecasted",
    )
    _remember_opportunities(ctx, result.get("opportunities") or [])
    ctx.event_bus.emit(
        "grants_searched",
        {"terms_sent": result.get("terms_sent", []), "rejected_terms": result.get("rejected_terms", []),
         "hit_count": result.get("hit_count", 0), "opportunities": result.get("opportunities", [])[:25],
         "error": result.get("error")},
    )
    return result


def tool_get_opportunity(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.grants is None:
        return {"error": "discovery_unavailable", "detail": "no funding catalogue client in this run"}
    result = ctx.grants.fetch_detail(args["opportunity_id"])
    opp = result.get("opportunity")
    if opp:
        _remember_opportunities(ctx, [opp])
        ctx.event_bus.emit("opportunity_detail", {"opportunity": opp})
    return result


def tool_fetch_url(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.web is None:
        return {"error": "discovery_unavailable", "detail": "no web client in this run"}
    result = ctx.web.fetch(args["url"])
    if not result.get("error"):
        ctx.event_bus.emit("page_fetched", {k: result.get(k) for k in ("doc_id", "url", "title", "chars", "cached")})
    return result


def tool_search_web(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.web is None:
        return {"error": "discovery_unavailable", "detail": "no web client in this run"}
    result = ctx.web.search([t for t in args.get("terms", []) if isinstance(t, str)], limit=args.get("limit", 8))
    ctx.event_bus.emit(
        "web_searched",
        {"query_sent": result.get("query_sent"), "rejected_terms": result.get("rejected_terms", []),
         "results": result.get("results", []), "error": result.get("error")},
    )
    return result


def tool_propose_opportunity(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Recommend an opportunity -- but only one the catalogue actually returned.

    This is the discovery equivalent of the citation gate: a recommendation the
    institution might act on must correspond to a real, retrievable opportunity, so
    the id is checked against what came back over the wire this run rather than
    trusted from the model.
    """
    oid = str(args.get("opportunity_id") or "").strip()
    known = ctx.state.opportunities_seen.get(oid)
    if known is None:
        return {
            "error": "unknown_opportunity",
            "detail": (
                f"{oid!r} was not returned by search_grants or get_opportunity in this run. "
                "You can only propose opportunities the catalogue actually returned. "
                f"Available ids: {sorted(ctx.state.opportunities_seen)[:20]}"
            ),
        }

    profile = ctx.state.latest_draft(_PROFILE_SECTION)
    if profile is None:
        return {"error": "no_profile", "detail": "call draft_profile before proposing opportunities"}

    rationale = (args.get("fit_rationale") or "").strip()
    citations = sorted({m.group(0) for m in _CITATION_RE.finditer(rationale)})
    if not citations:
        return {
            "error": "uncited_rationale",
            "detail": ("the fit rationale must cite the evidence it rests on with [[doc_id]], "
                       "the same as any other claim in this system"),
        }

    proposal = ProposedOpportunity(
        opportunity_id=oid,
        number=str(known.get("number") or ""),
        title=str(known.get("title") or ""),
        agency=str(known.get("agency") or ""),
        close_date=str(known.get("close_date") or ""),
        url=str(known.get("url") or ""),
        fit_rationale=rationale,
        citations=citations,
        confidence=str(args.get("confidence") or "medium"),
        concerns=[c for c in (args.get("concerns") or []) if isinstance(c, str)],
        aligned_with=[a.strip() for a in (args.get("aligned_with") or []) if isinstance(a, str) and a.strip()],
        announcement_url=str(known.get("announcement_url") or ""),
        award_ceiling=str(known.get("award_ceiling") or ""),
        cost_sharing=known.get("cost_sharing") if isinstance(known.get("cost_sharing"), bool) else None,
    )
    ctx.state.proposals = [p for p in ctx.state.proposals if p.opportunity_id != oid]
    ctx.state.proposals.append(proposal)
    ctx.event_bus.emit("opportunity_proposed", asdict(proposal))
    return {"accepted": True, "opportunity_id": oid, "number": proposal.number, "citations": citations}


def _finish_discovery(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Discovery's finish gate, enforced in code the same way the proposal one is."""
    missing: list[str] = []

    profile = ctx.state.latest_draft(_PROFILE_SECTION)
    if profile is None:
        missing.append("draft_profile has not been called")
    else:
        check = ctx.state.latest_citation_check(_PROFILE_SECTION)
        if check is None or check.version != profile.version:
            missing.append("verify_citations has not been run on the latest institution_profile")
        elif check.unsupported_claims:
            missing.append(
                f"institution_profile still has {len(check.unsupported_claims)} unsupported claim(s); "
                "revise it with draft_profile"
            )

    if ctx.state.searches_run == 0:
        missing.append("search_grants has not been called")

    # Deliberately NOT requiring at least one proposal. An earlier version did, and it
    # was wrong in a way worth recording: the prompt tells the agent a short list of
    # genuinely suitable opportunities beats a padded one, and then the gate would have
    # refused to let it finish having honestly found nothing -- pressuring it to pad.
    # The integrity guarantee here is that a proposal must correspond to a real
    # retrieved opportunity (see tool_propose_opportunity), not that some minimum
    # number of them exists.

    if missing:
        return {"accepted": False, "missing": missing}

    ctx.state.summary = args["summary"]
    return {"accepted": True, "missing": [], "proposals": len(ctx.state.proposals)}


def _finish_grade(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Grading's finish gate: every section actually scored, every requirement ruled on."""
    missing: list[str] = []

    if not ctx.state.drafts:
        missing.append("no submission sections were loaded")
    for section_id in ctx.state.drafts:
        if ctx.state.latest_review(section_id) is None:
            missing.append(f"{section_id}: review_section has not been run")
        if ctx.state.latest_citation_check(section_id) is None:
            missing.append(f"{section_id}: check_claims has not been run")

    if ctx.state.tool_calls_made.get("get_requirements", 0) == 0:
        missing.append("get_requirements has not been called")

    # A scorecard that silently skipped requirements would be worse than none: the
    # reader cannot tell "compliant" from "not looked at".
    unruled = [r["id"] for r in ctx.state.requirements if r["id"] not in ctx.state.compliance]
    if unruled:
        missing.append(
            f"{len(unruled)} requirement(s) have no compliance ruling, e.g. {unruled[:5]}. "
            "Every requirement needs a status, including 'not_applicable'."
        )

    if missing:
        return {"accepted": False, "missing": missing}

    ctx.state.summary = args["summary"]
    return {"accepted": True, "missing": [], "sections_scored": len(ctx.state.reviews)}


def tool_finish(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.playbook == "discovery":
        return _finish_discovery(ctx, args)
    if ctx.playbook == "grade":
        return _finish_grade(ctx, args)

    missing: list[str] = []
    if ctx.state.tool_calls_made.get("get_requirements", 0) == 0:
        missing.append("get_requirements has not been called")
    if ctx.state.tool_calls_made.get("check_budget", 0) == 0:
        missing.append("check_budget has not been called")

    for section_id in ctx.scope_sections:
        draft = ctx.state.latest_draft(section_id)
        if draft is None:
            missing.append(f"{section_id}: draft_section has not been called")
            continue
        review = ctx.state.latest_review(section_id)
        if review is None or review.version != draft.version:
            missing.append(f"{section_id}: review_section has not been run on the latest version")
        check = ctx.state.latest_citation_check(section_id)
        if check is None or check.version != draft.version:
            missing.append(f"{section_id}: verify_citations has not been run on the latest version")

    if missing:
        return {"accepted": False, "missing": missing}

    ctx.state.summary = args["summary"]
    return {"accepted": True, "missing": []}


TOOL_IMPLS: dict[str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]] = {
    "get_requirements": tool_get_requirements,
    "search_vault": tool_search_vault,
    "read_document": tool_read_document,
    "check_budget": tool_check_budget,
    "draft_section": tool_draft_section,
    "review_section": tool_review_section,
    "verify_citations": tool_verify_citations,
    "update_compliance": tool_update_compliance,
    "flag_issue": tool_flag_issue,
    "check_claims": tool_check_claims,
    "draft_profile": tool_draft_profile,
    "search_grants": tool_search_grants,
    "get_opportunity": tool_get_opportunity,
    "fetch_url": tool_fetch_url,
    "search_web": tool_search_web,
    "propose_opportunity": tool_propose_opportunity,
    "finish": tool_finish,
}

# Which tools each playbook exposes. The model is only offered the tools its run can
# actually use: handing a discovery run draft_section/check_budget invites it to wander
# into a proposal it has no solicitation for.
PROPOSAL_TOOLS = (
    "get_requirements", "search_vault", "read_document", "check_budget", "draft_section",
    "review_section", "verify_citations", "update_compliance", "flag_issue", "finish",
)
DISCOVERY_TOOLS = (
    "search_vault", "read_document", "draft_profile", "verify_citations", "search_grants",
    "get_opportunity", "fetch_url", "search_web", "propose_opportunity", "flag_issue", "finish",
)
# Grading reads a submission; it never writes one. draft_section is deliberately out
# of reach -- an agent asked to assess work should not be able to quietly rewrite it
# and then score its own version. verify_citations is out too: a submitted proposal
# has no [[doc_id]] markers to verify.
GRADE_TOOLS = (
    "get_requirements", "search_vault", "read_document", "check_budget",
    "review_section", "check_claims", "update_compliance", "flag_issue", "finish",
)
_PLAYBOOK_TOOLS = {"proposal": PROPOSAL_TOOLS, "discovery": DISCOVERY_TOOLS, "grade": GRADE_TOOLS}


def schemas_for(playbook: str) -> list[dict[str, Any]]:
    names = set(_PLAYBOOK_TOOLS.get(playbook, PROPOSAL_TOOLS))
    return [sch for sch in TOOL_SCHEMAS if sch["function"]["name"] in names]


def execute_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    schema = _SCHEMA_BY_NAME.get(name)
    if schema is None:
        return {"error": "unknown_tool", "detail": f"no such tool: {name}"}
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as exc:
        return {"error": "invalid_arguments", "detail": exc.message}
    try:
        return TOOL_IMPLS[name](ctx, args)
    except Exception as exc:  # noqa: BLE001 -- a tool bug must not crash the orchestrator loop
        return {"error": "tool_exception", "detail": str(exc)}
