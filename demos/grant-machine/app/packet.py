"""Render a finished run as a PDF packet a grants director can actually file.

The packet used to be served as `text/markdown`, which every browser renders as a
wall of unstyled plain text -- technically the content, practically unusable as the
hand-off artifact it is meant to be. This produces a real paginated document:
compliance matrix as a table, issues ranked by severity, each drafted section with
its reviewer score and citation-check result.

Two things this must not do, both inherited from the rest of the app's discipline:

  - It must never present AI output as finished work. Every page carries the DRAFT
    provenance, and the reviewer scores and unsupported-claim counts travel with the
    drafts rather than being quietly dropped.
  - It must never trust model text. Draft bodies are escaped for XML *before* the
    small Markdown subset is applied, the same escape-then-transform order the web
    UI uses (see docs/ARCHITECTURE.md "Security"); ReportLab's Paragraph parses a
    mini-HTML dialect, so unescaped model output could otherwise break the document
    or inject markup.

ReportLab is BSD-licensed and ships a pure-Python wheel, so it adds no build step on
aarch64 and clears scripts/license_audit.sh.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .agent.state import RunState

# Muted, print-friendly palette. Status colours have to survive greyscale printing,
# so they differ in value as well as hue.
_INK = colors.HexColor("#1a1a1a")
_MUTED = colors.HexColor("#5b5b5b")
_RULE = colors.HexColor("#c9c9c9")
_ACCENT_HEX = "#2f6d5a"
_ACCENT = colors.HexColor(_ACCENT_HEX)
_DRAFT_RED = colors.HexColor("#9b2226")

_STATUS_COLORS = {
    "met": colors.HexColor("#2d6a4f"),
    "partial": colors.HexColor("#b07d10"),
    "missing": colors.HexColor("#9b2226"),
    "needs_human": colors.HexColor("#6d3eb0"),
    "not_applicable": _MUTED,
}
_SEVERITY_ORDER = {"high": 0, "critical": 0, "medium": 1, "moderate": 1, "low": 2}

# The packet is a different artifact per mode, and saying "working packet" over an
# assessment of somebody else's proposal would misdescribe it.
_PROVENANCE = {
    "default": (
        "<b>This document was produced by an AI agent running entirely on the device named "
        "above.</b> Every section below is a draft. Reviewer scores, unsupported-claim counts "
        "and flagged issues are reproduced with the drafts so nothing reads as more finished "
        "than it is. Verify every figure and citation against the source documents before any "
        "part of this is submitted."
    ),
    "discover": (
        "<b>This search was run by an AI agent entirely on the device named above.</b> The "
        "institution's documents were not transmitted; the section below lists every term that "
        "was. These are candidates for a human to assess, not a shortlist to act on — confirm "
        "eligibility and deadlines against each funder's own announcement before committing."
    ),
    "grade": (
        "<b>This assessment was produced by an AI agent running entirely on the device named "
        "above.</b> It is an opinion formed from the submitted text and the institution's own "
        "records, not a funding decision and not a substitute for the programme officer. Where "
        "it reports a claim as contradicted, check the underlying record yourself before acting."
    ),
}

_SUBTITLES = {
    "discover": "Funding opportunities found — for human review, not a decision",
    "grade": "Assessment of a submitted proposal — for the author, not a verdict",
}


# -- text handling -------------------------------------------------------------


def _escape(text: str) -> str:
    """XML-escape first. Everything else happens after this, never before."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.S)
_CODE_RE = re.compile(r"`([^`]+)`")
_CITE_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _inline(text: str) -> str:
    """Escaped text -> ReportLab's mini-HTML. Order matters: escape, then transform."""
    out = _escape(text)
    out = _CODE_RE.sub(r'<font face="Courier" size="9">\1</font>', out)
    out = _BOLD_RE.sub(r"<b>\1</b>", out)
    out = _ITALIC_RE.sub(r"<i>\1</i>", out)
    # Citations are the load-bearing detail of this demo -- make them visible in the
    # printed artifact rather than letting them read as stray brackets.
    out = _CITE_RE.sub(rf'<font color="{_ACCENT_HEX}" size="8">[\1]</font>', out)
    return out


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# -- page furniture ------------------------------------------------------------


def _draw_furniture(canvas, doc) -> None:
    """Header rule and footer on every page, including the DRAFT provenance."""
    canvas.saveState()
    width, height = LETTER

    canvas.setStrokeColor(_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(0.75 * inch, height - 0.62 * inch, width - 0.75 * inch, height - 0.62 * inch)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_DRAFT_RED)
    canvas.drawString(0.75 * inch, height - 0.52 * inch, "DRAFT — AI-GENERATED, FOR HUMAN REVIEW")
    canvas.setFillColor(_MUTED)
    canvas.drawRightString(width - 0.75 * inch, height - 0.52 * inch, doc.packet_subtitle)

    canvas.line(0.75 * inch, 0.62 * inch, width - 0.75 * inch, 0.62 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(0.75 * inch, 0.45 * inch, doc.packet_footer)
    canvas.drawRightString(width - 0.75 * inch, 0.45 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, leading=24,
                                textColor=_INK, spaceAfter=2, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10.5,
                                   textColor=_MUTED, spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=17,
                             textColor=_ACCENT, spaceBefore=16, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, leading=14,
                             textColor=_INK, spaceBefore=11, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=10.5, leading=13,
                             textColor=_INK, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13.5,
                               textColor=_INK, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=9.5, leading=13.5,
                                 textColor=_INK, leftIndent=14, bulletIndent=4, spaceAfter=3),
        "meta": ParagraphStyle("meta", parent=base["Normal"], fontSize=8.5, leading=11,
                               textColor=_MUTED),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8, leading=10.5,
                               textColor=_INK),
        "note": ParagraphStyle("note", parent=base["Normal"], fontSize=8.5, leading=11.5,
                               textColor=_MUTED, spaceAfter=8),
    }


# -- markdown subset -> flowables ----------------------------------------------


def _markdown_flowables(text: str, st: dict[str, ParagraphStyle]) -> list:
    """Headings, bullets and paragraphs. Deliberately small: the drafts the model
    writes use exactly this much Markdown, and a fuller parser would be more surface
    for untrusted text to misbehave in."""
    flow: list = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            flow.append(Paragraph(_inline(" ".join(buffer)), st["body"]))
            buffer.clear()

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            style = st["h2"] if level <= 2 else st["h3"]
            flow.append(Paragraph(_inline(line.lstrip("#").strip()), style))
            continue
        if re.match(r"^\s*[-*+]\s+", line):
            flush()
            flow.append(Paragraph(_inline(re.sub(r"^\s*[-*+]\s+", "", line)), st["bullet"], bulletText="•"))
            continue
        if re.match(r"^\s*\d+[.)]\s+", line):
            flush()
            flow.append(Paragraph(_inline(line.strip()), st["bullet"]))
            continue
        buffer.append(line.strip())

    flush()
    return flow


# -- sections ------------------------------------------------------------------


def _meta_table(rows: list[tuple[str, str]], st: dict[str, ParagraphStyle]) -> Table:
    data = [[Paragraph(f"<b>{_escape(k)}</b>", st["cell"]), Paragraph(_escape(v), st["cell"])] for k, v in rows]
    t = Table(data, colWidths=[1.35 * inch, 5.4 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _compliance_table(state: RunState, st: dict[str, ParagraphStyle]) -> list:
    if not state.compliance:
        return [Paragraph("No compliance entries were recorded in this run.", st["note"])]

    req_text = {r["id"]: r.get("text", "") for r in state.requirements}
    header = ["Req", "Requirement", "Status", "Evidence", "Note"]
    data = [[Paragraph(f"<b>{h}</b>", st["cell"]) for h in header]]
    styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f0")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, _RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e4e4e4")),
    ]

    for i, (rid, entry) in enumerate(sorted(state.compliance.items()), start=1):
        colour = _STATUS_COLORS.get(entry.status, _MUTED)
        data.append([
            Paragraph(_escape(rid), st["cell"]),
            Paragraph(_escape(_truncate(req_text.get(rid, ""), 180)), st["cell"]),
            # Underscores give the wrapper no break opportunity, so "needs_human" in a
            # narrow column split as "needs_hum / an". Spaces read better in prose anyway.
            Paragraph(f'<b><font color="#{colour.hexval()[2:]}">{_escape(entry.status.replace("_", " "))}</font></b>', st["cell"]),
            Paragraph(_escape(_truncate(", ".join(entry.evidence), 90)) or "—", st["cell"]),
            Paragraph(_escape(_truncate(entry.note, 140)) or "—", st["cell"]),
        ])
        if i % 2 == 0:
            styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafafa")))

    t = Table(data, colWidths=[0.55 * inch, 2.35 * inch, 0.95 * inch, 1.25 * inch, 1.65 * inch],
              repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle(styles))
    return [t]


def _issues_section(state: RunState, st: dict[str, ParagraphStyle]) -> list:
    if not state.issues:
        return [Paragraph("No issues were flagged in this run.", st["note"])]

    flow: list = []
    ordered = sorted(state.issues, key=lambda i: _SEVERITY_ORDER.get((i.severity or "").lower(), 3))
    for issue in ordered:
        sev = (issue.severity or "unknown").upper()
        colour = _STATUS_COLORS["missing"] if _SEVERITY_ORDER.get(issue.severity or "", 3) == 0 else _STATUS_COLORS["partial"]
        head = f'<b><font color="#{colour.hexval()[2:]}">[{_escape(sev)}]</font> {_escape(issue.title or issue.code or "Issue")}</b>'
        parts = [Paragraph(head, st["h3"])]
        if issue.detail:
            parts.append(Paragraph(_inline(issue.detail), st["body"]))
        bits = []
        if issue.code:
            bits.append(f"code: {issue.code}")
        if issue.source:
            bits.append(f"source: {issue.source}")
        if issue.evidence:
            bits.append("evidence: " + _truncate(", ".join(issue.evidence), 120))
        if issue.recommended_action:
            parts.append(Paragraph(f"<b>Recommended action:</b> {_inline(issue.recommended_action)}", st["body"]))
        if bits:
            parts.append(Paragraph(_escape(" · ".join(bits)), st["meta"]))
        parts.append(Spacer(1, 5))
        flow.append(KeepTogether(parts))
    return flow


def _opportunities_section(state: RunState, st: dict[str, ParagraphStyle]) -> list:
    """Discover's shortlist, ordered by confidence."""
    if not state.proposals:
        return []
    rank = {"high": 0, "medium": 1, "low": 2}
    flow: list = [Paragraph("Recommended opportunities", st["h1"])]
    for opp in sorted(state.proposals, key=lambda o: rank.get(o.confidence, 3)):
        colour = {"high": _STATUS_COLORS["met"], "medium": _STATUS_COLORS["partial"]}.get(opp.confidence, _MUTED)
        parts = [
            Paragraph(
                f'<b><font color="#{colour.hexval()[2:]}">[{_escape(opp.confidence.upper())}]</font> '
                f'{_escape(opp.number or opp.opportunity_id)}</b> — {_escape(opp.title)}',
                st["h3"],
            ),
            Paragraph(_escape(" · ".join(
                x for x in (opp.agency, f"closes {opp.close_date}" if opp.close_date else "") if x)), st["meta"]),
        ]
        if opp.fit_rationale:
            parts.append(Paragraph(_inline(opp.fit_rationale), st["body"]))
        if opp.concerns:
            parts.append(Paragraph("<b>A human must check</b>", st["h3"]))
            parts += [Paragraph(_inline(c), st["bullet"], bulletText="•") for c in opp.concerns]
        if opp.url:
            parts.append(Paragraph(_escape(opp.url), st["meta"]))
        parts.append(Spacer(1, 6))
        flow.append(KeepTogether(parts))
    return flow


def _claims_section(state: RunState, st: dict[str, ParagraphStyle]) -> list:
    """Grade's claim cross-check. Contradictions first -- they are the finding."""
    contradicted: list[tuple[str, dict]] = []
    unverifiable: list[tuple[str, dict]] = []
    for section_id, records in (state.citation_checks or {}).items():
        for claim in (records[-1].unsupported_claims or []):
            if not isinstance(claim, dict) or "verdict" not in claim:
                continue
            (contradicted if claim.get("verdict") == "contradicted" else unverifiable).append((section_id, claim))

    if not contradicted and not unverifiable:
        return []

    flow: list = [Paragraph("Claims checked against your own records", st["h1"])]
    if contradicted:
        flow.append(Paragraph(
            f'<b><font color="#{_DRAFT_RED.hexval()[2:]}">{len(contradicted)} claim(s) contradicted '
            "by the institution's own documentation.</font></b> These will not survive a panel.",
            st["body"]))
        for section_id, claim in contradicted:
            flow.append(KeepTogether([
                Paragraph(f"<b>{_escape(section_id)}</b>", st["h3"]),
                Paragraph(_escape(_truncate(claim.get("claim", ""), 400)), st["body"]),
                Paragraph(_escape(_truncate(claim.get("detail", ""), 300)), st["meta"]),
                Spacer(1, 5),
            ]))
    if unverifiable:
        flow.append(Paragraph(
            f"{len(unverifiable)} further claim(s) could not be checked against available "
            "documentation — not wrong, but unevidenced. A reviewer may ask.", st["note"]))
    return flow


def _egress_section(state: RunState, st: dict[str, ParagraphStyle]) -> list:
    """Everything that left the device, for the reader who will be asked about it.

    Deliberately lists only what was SENT. Reprinting a refused term would put the
    institution's own language into a document meant to be shared -- which is the
    exact thing the boundary stopped.
    """
    if not state.egress_log:
        return []
    sent = [r for r in state.egress_log if r.get("allowed")]
    refused = [r for r in state.egress_log if not r.get("allowed")]
    terms = sorted({t["term"] for r in sent for t in (r.get("terms") or []) if t.get("allowed")})

    flow: list = [
        Paragraph("What left this device", st["h1"]),
        Paragraph(
            f"{len(sent)} request(s) were made and {len(refused)} refused. Your documents were not "
            "sent anywhere. The complete list of terms transmitted:", st["body"]),
        Paragraph(_escape(", ".join(terms)) or "none", st["body"]),
    ]
    if refused:
        reasons = sorted({str(r.get("reason", "")) for r in refused})
        flow.append(Paragraph(
            _escape(f"Refusals: {', '.join(reasons)} — the boundary working, not an error."), st["meta"]))
    hosts = sorted({(r.get("url") or "").split("/")[2] for r in sent if (r.get("url") or "").count("/") > 2})
    if hosts:
        flow.append(Paragraph(_escape("Contacted: " + ", ".join(hosts)), st["meta"]))
    return flow


def _drafts_section(state: RunState, st: dict[str, ParagraphStyle]) -> list:
    if not state.drafts:
        return [Paragraph("No sections were drafted in this run.", st["note"])]

    flow: list = []
    for section_id, versions in state.drafts.items():
        if not versions:
            continue
        latest = versions[-1]
        title = section_id.replace("_", " ").title()
        flow.append(Paragraph(f"{_escape(title)} <font size='9' color='#{_MUTED.hexval()[2:]}'>(v{latest.version})</font>", st["h2"]))

        review = state.latest_review(section_id)
        check = state.latest_citation_check(section_id)
        status_bits = []
        if review:
            verdict = "passed" if review.passed else "did not pass"
            status_bits.append(f"Reviewer score {review.mean_score:.1f}/5 — {verdict}")
        if check:
            n = len(check.unsupported_claims)
            status_bits.append("citations verified" if check.passed and not n
                               else f"{n} unsupported claim(s)")
        if latest.word_count:
            status_bits.append(f"{latest.word_count} words")
        if status_bits:
            flow.append(Paragraph(_escape(" · ".join(status_bits)), st["meta"]))
            flow.append(Spacer(1, 5))

        flow.extend(_markdown_flowables(latest.text, st))

        if review and review.must_fix:
            items = [Paragraph("<b>Reviewer must-fix</b>", st["h3"])]
            items += [Paragraph(_inline(m), st["bullet"], bulletText="•") for m in review.must_fix]
            flow.append(KeepTogether(items))
        flow.append(Spacer(1, 10))
    return flow


# -- entry point ---------------------------------------------------------------


def render_packet_pdf(
    state: RunState,
    *,
    institution: str = "",
    solicitation: str = "",
    device_label: str = "",
    model: str = "",
    generated_at: datetime | None = None,
) -> bytes:
    """Render `state` to PDF bytes."""
    st = _styles()
    # Discovery is not working to a solicitation -- it is looking for one. This has to
    # happen before the document is built, because the value also becomes the running
    # page header, where it asserted a connection to NSF 24-584 on every page of a run
    # that had nothing to do with it.
    if state.mode == "discover":
        solicitation = ""
    ts = generated_at or datetime.now(timezone.utc)
    stamp = ts.strftime("%Y-%m-%d %H:%M UTC")

    buf = BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.85 * inch, bottomMargin=0.8 * inch,
        title="Grant Machine Packet (DRAFT)",
        author="Grant Machine — on-device AI",
        subject=solicitation or "Grant proposal working packet",
    )
    doc.packet_subtitle = solicitation or "Grant Machine"
    doc.packet_footer = f"Generated on-device {stamp}" + (f" · {device_label}" if device_label else "")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_draw_furniture)])

    flow: list = [
        Paragraph("Grant Machine Packet", st["title"]),
        Paragraph(_SUBTITLES.get(state.mode, "Working packet for human review — not a submission"), st["subtitle"]),
    ]

    meta: list[tuple[str, str]] = []
    if institution:
        meta.append(("Institution", institution))
    if solicitation:
        meta.append(("Solicitation", solicitation))
    meta += [
        ("Run", f"{state.run_id} · {state.mode}/{state.scope} · {state.status}"),
        ("Steps", str(state.steps)),
        ("Generated", stamp),
    ]
    if model:
        meta.append(("Model", f"{model} (local)"))
    if device_label:
        meta.append(("Device", device_label))
    flow.append(_meta_table(meta, st))
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(_PROVENANCE.get(state.mode, _PROVENANCE["default"]), st["note"]))

    flow.append(Paragraph("Summary", st["h1"]))
    flow.extend(_markdown_flowables(state.summary or "_(no summary was recorded for this run)_", st))

    # A discovery run has no requirements and did not draft anything, so "No issues
    # were flagged" and "No compliance entries were recorded" are not findings there --
    # they are noise in front of the content. Print a section when it has something to
    # say, or when its emptiness is itself a finding for that kind of run.
    assessing = state.mode != "discover"
    if state.issues or assessing:
        flow.append(Paragraph("Issues requiring attention", st["h1"]))
        flow.extend(_issues_section(state, st))

    if state.compliance or (assessing and state.requirements):
        flow.append(Paragraph("Compliance matrix", st["h1"]))
        flow.extend(_compliance_table(state, st))

    flow.extend(_opportunities_section(state, st))
    flow.extend(_claims_section(state, st))
    flow.extend(_egress_section(state, st))

    if state.drafts:
        flow.append(PageBreak())
        # A grading run did not draft these -- it was handed them.
        heading = "Submitted sections, as assessed" if state.mode == "grade" else "Drafted sections"
        flow.append(Paragraph(heading, st["h1"]))
        flow.extend(_drafts_section(state, st))

    doc.build(flow)
    return buf.getvalue()
