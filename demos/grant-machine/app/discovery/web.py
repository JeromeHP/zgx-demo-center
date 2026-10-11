"""Open-web research (Phase 1: Discover).

Two capabilities, both through `EgressGuard`:

  - `fetch(url)`   -- retrieve a public page and convert it to Markdown.
  - `search(terms)`-- general web search via a self-hosted SearXNG instance, when one
                      is configured. There is deliberately no built-in commercial
                      search provider: an API key on a demo box, and queries routed
                      through a third party, would both undercut the point.

**Fetched pages become citable documents.** Anything retrieved is written into the
session corpus with a `doc_id`, so the agent cites a web page exactly as it cites a
vault document and `verify_citations` checks it the same way. Without that, open-web
content would be the one source in this app that the citation gate could not see --
which would quietly undermine every other guarantee the demo makes.

The corpus is also extended with the fetched text, so material pulled off the web
cannot then be echoed back out in a later search term.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify

from ..egress import EgressBlocked, EgressGuard

# Stripped before conversion: script/style are not content, and nav/footer chrome
# adds noise the model then has to pay attention tokens for.
_STRIP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "form", "svg")
_WS_RE = re.compile(r"\n{3,}")


@dataclass
class FetchedPage:
    doc_id: str
    url: str
    title: str
    text: str
    bytes_in: int

    def as_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "url": self.url, "title": self.title,
                "chars": len(self.text), "bytes_in": self.bytes_in}


def _doc_id_for(url: str) -> str:
    """Stable, readable id: host plus a short hash so repeat fetches collapse."""
    host = (urlparse(url).hostname or "web").replace("www.", "").replace(".", "-")
    return f"web-{host}-{hashlib.sha256(url.encode()).hexdigest()[:8]}"


def html_to_markdown(html: str) -> tuple[str, str]:
    """(title, markdown). Same bs4 + markdownify path scripts/fetch_solicitation.py
    already uses, so there is one conversion behaviour in the project, not two."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    body = soup.body or soup
    md = markdownify(str(body), heading_style="ATX")
    return title, _WS_RE.sub("\n\n", md).strip()


class WebResearch:
    def __init__(
        self,
        guard: EgressGuard,
        session_dir: Path | None = None,
        *,
        searxng_url: str = "",
        max_page_chars: int = 40_000,
    ) -> None:
        self._guard = guard
        self._session_dir = Path(session_dir) if session_dir else None
        self._searxng_url = (searxng_url or "").rstrip("/")
        self._max_page_chars = max_page_chars
        self.fetched: dict[str, FetchedPage] = {}

    # -- search ---------------------------------------------------------------

    @property
    def search_available(self) -> bool:
        return bool(self._searxng_url)

    def search(self, terms: list[str], *, limit: int = 8) -> dict[str, Any]:
        if not self.search_available:
            return {
                "error": "web_search_unavailable",
                "detail": (
                    "No web search provider is configured. Opportunity search via "
                    "search_grants still works; set egress.searxng_url to a self-hosted "
                    "SearXNG instance to enable general web search."
                ),
                "results": [],
            }

        decisions = self._guard.validator.validate_all(terms)
        allowed = [d.term for d in decisions if d.allowed]
        rejected = [{"term": d.term, "reason": d.reason} for d in decisions if not d.allowed]
        if not allowed:
            return {"error": "no_permitted_terms",
                    "detail": "None of these terms may leave the device.",
                    "rejected_terms": rejected, "results": []}

        query = " ".join(allowed)
        try:
            resp = self._guard.request(
                "GET", f"{self._searxng_url}/search", terms=allowed,
                params={"q": query, "format": "json"},
                headers={"Accept": "application/json"},
            )
        except EgressBlocked as exc:
            return {"error": "egress_blocked", "detail": str(exc), "results": []}

        try:
            payload = resp.json()
        except ValueError:
            return {"error": "bad_search_response",
                    "detail": "SearXNG did not return JSON; is the json format enabled in its settings.yml?",
                    "results": []}

        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in (payload.get("results") or [])[:limit]
        ]
        return {"query_sent": query, "rejected_terms": rejected, "results": results}

    # -- fetch ----------------------------------------------------------------

    def fetch(self, url: str) -> dict[str, Any]:
        """Retrieve a public page, convert it, and register it as a citable document."""
        doc_id = _doc_id_for(url)
        if doc_id in self.fetched:
            page = self.fetched[doc_id]
            return {"cached": True, **page.as_dict(), "text": page.text[:4000]}

        # No `terms`: a URL chosen from search results is public, not user data. The
        # guard still scans the URL's query string, so a URL carrying institutional
        # text in a parameter is refused.
        try:
            resp = self._guard.request("GET", url, terms=[], headers={"Accept": "text/html,text/plain,*/*"})
        except EgressBlocked as exc:
            return {"error": "egress_blocked", "detail": str(exc)}

        content_type = (resp.headers.get("content-type") or "").lower()
        raw = resp.text or ""
        if "html" in content_type:
            title, text = html_to_markdown(raw)
        else:
            title, text = "", raw.strip()

        if not text:
            return {"error": "empty_page", "detail": f"{url} returned no readable text"}

        truncated = len(text) > self._max_page_chars
        text = text[: self._max_page_chars]

        page = FetchedPage(doc_id=doc_id, url=url, title=title or url,
                           text=text, bytes_in=len(resp.content or b""))
        self.fetched[doc_id] = page
        self._persist(page)

        return {"cached": False, "truncated": truncated, **page.as_dict(), "text": text[:4000]}

    def _persist(self, page: FetchedPage) -> None:
        """Write into the session corpus so the page is citable and verifiable.

        Also folded into the egress guard's corpus: once text is on this device it is
        treated as the institution's working material, so it can never be echoed back
        out as a search term.
        """
        self._guard.corpus.add_text(page.text)
        if self._session_dir is None:
            return
        self._session_dir.mkdir(parents=True, exist_ok=True)
        front = f"---\ntitle: {page.title}\nsource_url: {page.url}\n---\n\n"
        (self._session_dir / f"{page.doc_id}.md").write_text(front + page.text, encoding="utf-8")
