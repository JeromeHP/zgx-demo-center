from __future__ import annotations

from app.ingest.loaders import Document, Section
from app.retrieval.bm25_index import BM25Index, tokenize


def _doc(doc_id: str, title: str, heading: str, text: str) -> Document:
    return Document(
        doc_id=doc_id, title=title, doc_type="md", text=text, sections=[Section(heading=heading, level=1, text=text)]
    )


def test_tokenize_lowercases_strips_stopwords_keeps_numbers():
    tokens = tokenize("The Budget Total is 1000000 and it exceeds the cap")
    assert "the" not in tokens
    assert "budget" in tokens
    assert "1000000" in tokens


def test_search_returns_expected_doc_in_top_3():
    index = BM25Index()
    docs = [
        _doc(
            "prior-annual-report-2025",
            "Final Outcomes Report",
            "Completers",
            "The project produced 84 completers against a target of 120 completers over the grant period.",
        ),
        _doc(
            "facilities-equipment",
            "Facilities and Equipment",
            "Lab",
            "The automation lab includes robotic arms and PLC training stations for hands-on instruction.",
        ),
        _doc(
            "advisory-board-2026",
            "Industry Advisory Board",
            "Minutes",
            "The advisory board discussed regional employer needs and automation skills gaps in August 2026.",
        ),
        _doc(
            "declined-dol-2023",
            "Declined Proposal",
            "Reviewer Comments",
            "Reviewers noted weak employer engagement and insufficient letters of support in the 2023 submission.",
        ),
    ]
    index.build(docs)

    hits = index.search("prior outcomes completers target", k=3)
    assert hits
    assert hits[0].doc_id == "prior-annual-report-2025"


def test_search_snippet_centers_on_match():
    # BM25 idf is degenerate with a single-document corpus (rank_bm25 can
    # legitimately score every term <= 0), so include filler documents.
    index = BM25Index()
    long_text = "Padding text before. " * 30 + "The keyword appears here in context. " + "Padding text after. " * 30
    index.build(
        [
            _doc("doc1", "Doc One", "Body", long_text),
            _doc("doc2", "Doc Two", "Body", "Completely unrelated content about lab equipment and travel budgets."),
            _doc("doc3", "Doc Three", "Body", "Advisory board minutes discussing employer partnerships and outcomes."),
        ]
    )

    hits = index.search("keyword appears", k=1)
    assert hits
    snippet_lower = hits[0].snippet.lower()
    assert "keyword" in snippet_lower or "appears" in snippet_lower
    assert len(hits[0].snippet) <= 402  # 400 chars + up to 2 ellipsis markers


def test_search_all_stopwords_returns_no_hits():
    index = BM25Index()
    index.build([_doc("doc1", "Doc", "H", "some content here")])
    assert index.search("the of and", k=5) == []


def test_search_empty_index_returns_no_hits():
    index = BM25Index()
    index.build([])
    assert index.search("anything", k=5) == []


def test_small_corpora_still_return_hits(tmp_path):
    """BM25's IDF goes to zero -- negative, in rank_bm25 -- for a term present in
    every document. On the two-document corpora Discover works with that made every
    query score <= 0, and filtering those out left the agent with an apparently
    empty vault while the uploads sat right there.
    """
    from app.ingest.loaders import load
    from app.retrieval.bm25_index import BM25Index

    (tmp_path / "plan.md").write_text(
        "# Strategic Plan\n\nWe run robotics and advanced manufacturing technician programs.\n",
        encoding="utf-8",
    )
    (tmp_path / "capability.md").write_text(
        "# Capability\n\nOur faculty work in mechatronics and industrial automation.\n",
        encoding="utf-8",
    )

    index = BM25Index()
    index.build([load(p) for p in sorted(tmp_path.glob("*.md"))])

    # Every query term appears in both chunks, so raw BM25 scores are non-positive.
    assert max(index._bm25.get_scores(["robotics"])) <= 0

    hits = index.search("robotics advanced manufacturing", k=5)
    assert hits, "a two-document corpus must still be searchable"
    assert hits[0].doc_id == "plan"

    # A query with no overlap at all must still return nothing -- the fallback
    # relaxes the score floor, it does not invent relevance.
    assert index.search("volcanology seismology", k=5) == []
