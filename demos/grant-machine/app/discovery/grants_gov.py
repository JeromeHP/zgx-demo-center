"""grants.gov client (Phase 1: Discover).

The federal funding catalogue, via the public `search2` / `fetchOpportunity`
endpoints -- no API key, no account, U.S. Government work.

This module builds requests and parses responses. It does not decide what may be
sent: the `EgressGuard` passed in owns that, and the keyword terms handed to it are
validated against the published vocabulary before anything leaves the device. Note
that the search body is assembled *here* from already-validated terms rather than
taking a caller-supplied query string -- there is no code path that puts arbitrary
text into an outbound field.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from ..egress import EgressBlocked, EgressGuard

SEARCH_URL = "https://api.grants.gov/v1/api/search2"
DETAIL_URL = "https://api.grants.gov/v1/api/fetchOpportunity"

# Statuses worth showing a research office. "closed" is excluded by default -- a
# closed opportunity is not something anyone can apply to.
DEFAULT_STATUSES = "posted|forecasted"


@dataclass
class Opportunity:
    """One funding opportunity as returned by grants.gov."""

    opportunity_id: str
    number: str
    title: str
    agency: str = ""
    agency_code: str = ""
    status: str = ""
    close_date: str = ""
    open_date: str = ""
    doc_type: str = ""
    cfda: list[str] = field(default_factory=list)
    # Filled in by fetch_detail(), not by search.
    description: str = ""
    award_ceiling: str = ""
    award_floor: str = ""
    cost_sharing: bool | None = None
    applicant_types: list[str] = field(default_factory=list)
    url: str = ""
    # The agency's own full announcement, when the catalogue links one. The grants.gov
    # record is a synopsis; the announcement is what a human actually has to read.
    announcement_url: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "number": self.number,
            "title": self.title,
            "agency": self.agency,
            "agency_code": self.agency_code,
            "status": self.status,
            "close_date": self.close_date,
            "open_date": self.open_date,
            "doc_type": self.doc_type,
            "cfda": self.cfda,
            "description": self.description,
            "award_ceiling": self.award_ceiling,
            "award_floor": self.award_floor,
            "cost_sharing": self.cost_sharing,
            "applicant_types": self.applicant_types,
            "url": self.url,
            "announcement_url": self.announcement_url,
        }


def _opportunity_url(opportunity_id: str) -> str:
    return f"https://grants.gov/search-results-detail/{opportunity_id}"


def _https(value: Any) -> str:
    """An agency link from the catalogue, upgraded to https; blank if it is not a URL."""
    url = str(value or "").strip()
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url if url.startswith("https://") else ""


_NIH_NUMBER = re.compile(r"^(PA|PAR|RFA|OTA|NOT)-[A-Z]{0,2}-?\d{2}-\d{3}$")


def _nih_guide_url(number: str) -> str:
    """The NIH Guide search for an announcement grants.gov gave no link to.

    Not a guessed file path: older announcements sit at /grants/guide/pa-files/<n>.html,
    but 2026-and-later ones do not (PAR-26-040 there is a 404), and a dead link in a
    report handed to a research office is worse than none. The search always resolves.
    """
    n = number.strip().upper()
    if not _NIH_NUMBER.match(n):
        return ""
    return f"https://grants.nih.gov/funding/searchguide/index.html#/?query={n}"


def _clean(value: Any) -> str:
    """Catalogue text arrives HTML-escaped ("CHIPS Incentives Program &ndash; ...")."""
    return html.unescape(str(value or "")).strip()


def _parse_hit(hit: dict[str, Any]) -> Opportunity:
    cfda = hit.get("cfdaList") or []
    return Opportunity(
        opportunity_id=str(hit.get("id") or ""),
        number=_clean(hit.get("number")),
        title=_clean(hit.get("title")),
        agency=_clean(hit.get("agency")),
        agency_code=_clean(hit.get("agencyCode")),
        status=_clean(hit.get("oppStatus")),
        close_date=_clean(hit.get("closeDate")),
        open_date=_clean(hit.get("openDate")),
        doc_type=_clean(hit.get("docType")),
        cfda=[str(c) for c in cfda] if isinstance(cfda, list) else [],
        url=_opportunity_url(str(hit.get("id") or "")),
    )


class GrantsGovClient:
    def __init__(self, guard: EgressGuard) -> None:
        self._guard = guard

    def search(
        self,
        terms: list[str],
        *,
        rows: int = 15,
        statuses: str = DEFAULT_STATUSES,
        agencies: str = "",
        funding_categories: str = "",
    ) -> dict[str, Any]:
        """Search by validated keyword terms.

        Returns {"hit_count", "opportunities", "terms_sent"} or, when the guard
        refuses, {"error", "detail", "rejected_terms"} -- a refusal is a normal
        result the model must be able to read and react to, not an exception that
        kills the run.
        """
        decisions = self._guard.validator.validate_all(terms)
        allowed = [d.term for d in decisions if d.allowed]
        rejected = [{"term": d.term, "reason": d.reason} for d in decisions if not d.allowed]

        if not allowed:
            return {
                "error": "no_permitted_terms",
                "detail": (
                    "None of these terms may leave the device. Search terms must be generic, "
                    "published field names (see data/vocabulary/), not language taken from the "
                    "institution's own documents."
                ),
                "rejected_terms": rejected,
                "hit_count": 0,
                "opportunities": [],
            }

        body: dict[str, Any] = {
            "rows": max(1, min(int(rows), 50)),
            "keyword": " ".join(allowed),
            "oppStatuses": statuses,
        }
        if agencies:
            body["agencies"] = agencies
        if funding_categories:
            body["fundingCategories"] = funding_categories

        try:
            resp = self._guard.request("POST", SEARCH_URL, terms=allowed, json_body=body)
        except EgressBlocked as exc:
            return {"error": "egress_blocked", "detail": str(exc), "rejected_terms": rejected,
                    "hit_count": 0, "opportunities": []}

        payload = resp.json()
        if payload.get("errorcode") not in (0, None):
            return {"error": "api_error", "detail": str(payload.get("msg")), "hit_count": 0, "opportunities": []}

        data = payload.get("data") or {}
        hits = data.get("oppHits") or []
        return {
            "hit_count": int(data.get("hitCount") or 0),
            "terms_sent": allowed,
            "rejected_terms": rejected,
            "opportunities": [_parse_hit(h).as_dict() for h in hits],
        }

    def fetch_detail(self, opportunity_id: str) -> dict[str, Any]:
        """Full detail for one opportunity.

        The id is a grants.gov identifier the catalogue itself returned, not user
        text, so it carries no institutional content -- but it still goes through
        the guard so it lands in the audit log like everything else.
        """
        oid = str(opportunity_id).strip()
        if not oid.isdigit():
            return {"error": "invalid_opportunity_id",
                    "detail": f"expected a numeric grants.gov id, got {opportunity_id!r}"}

        try:
            resp = self._guard.request("POST", DETAIL_URL, terms=[], json_body={"opportunityId": oid})
        except EgressBlocked as exc:
            return {"error": "egress_blocked", "detail": str(exc)}

        payload = resp.json()
        if payload.get("errorcode") not in (0, None):
            return {"error": "api_error", "detail": str(payload.get("msg"))}

        data = payload.get("data") or {}
        syn = data.get("synopsis") or {}
        cfdas = data.get("cfdas") or []

        opp = Opportunity(
            opportunity_id=oid,
            number=_clean(data.get("opportunityNumber")),
            title=_clean(data.get("opportunityTitle")),
            # synopsis.agencyName is sometimes the contact person ("Joseph Perez\nProgram
            # Analyst" for a USDA-NIFA record); agencyDetails carries the agency itself.
            agency=_clean((data.get("agencyDetails") or {}).get("agencyName") or syn.get("agencyName")),
            agency_code=_clean(syn.get("agencyCode")),
            close_date=str(syn.get("responseDate") or syn.get("responseDateDesc") or ""),
            open_date=str(syn.get("postingDateStr") or syn.get("postingDate") or ""),
            doc_type=str(data.get("docType") or ""),
            cfda=[str(c.get("cfdaNumber")) for c in cfdas if isinstance(c, dict) and c.get("cfdaNumber")],
            description=_clean(syn.get("synopsisDesc")),
            award_ceiling=_clean(syn.get("awardCeiling")),
            award_floor=_clean(syn.get("awardFloor")),
            cost_sharing=syn.get("costSharing") if isinstance(syn.get("costSharing"), bool) else None,
            applicant_types=[
                str(a.get("description")) for a in (syn.get("applicantTypes") or [])
                if isinstance(a, dict) and a.get("description")
            ],
            url=_opportunity_url(oid),
            announcement_url=_https(syn.get("fundingDescLinkUrl")) or _nih_guide_url(_clean(data.get("opportunityNumber"))),
        )
        return {"opportunity": opp.as_dict()}
