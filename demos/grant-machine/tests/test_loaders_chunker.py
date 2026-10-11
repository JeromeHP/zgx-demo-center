from __future__ import annotations

import pytest
from pypdf import PdfWriter

from app.ingest.chunker import chunk_document
from app.ingest.loaders import LoaderError, load


def test_load_markdown_with_front_matter(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(
        "---\ntitle: Test Document\ndoc_type: md\n---\n"
        "# Overview\n\nSome overview text with enough words to matter.\n\n"
        "## Details\n\nMore details go here about the project.\n",
        encoding="utf-8",
    )
    doc = load(path, doc_id="test-doc")
    assert doc.doc_id == "test-doc"
    assert doc.title == "Test Document"
    assert doc.doc_type == "md"
    assert doc.front_matter["doc_type"] == "md"
    assert [s.heading for s in doc.sections] == ["Overview", "Details"]


def test_load_markdown_without_front_matter_uses_first_heading(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# My Title\n\nBody text.\n", encoding="utf-8")
    doc = load(path, doc_id="doc2")
    assert doc.title == "My Title"
    assert doc.front_matter == {}


def test_load_html_strips_nav_and_footer(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Page Title</title></head><body>"
        "<nav>Site nav</nav>"
        "<main><h1>Main Heading</h1><p>Body content.</p></main>"
        "<footer>Footer text</footer>"
        "</body></html>",
        encoding="utf-8",
    )
    doc = load(path, doc_id="page")
    assert doc.title == "Page Title"
    assert "Site nav" not in doc.text
    assert "Footer text" not in doc.text
    assert "Body content" in doc.text


def test_load_csv_front_matter_and_rows(tmp_path):
    path = tmp_path / "budget.csv"
    path.write_text(
        "# doc_id: budget-test\n# title: Test Budget\n# synthetic: true\n"
        "year,category,amount_usd\n1,equipment,5000\n2,travel,1000\n",
        encoding="utf-8",
    )
    doc = load(path, doc_id="budget-test")
    assert doc.doc_type == "csv"
    assert doc.front_matter["title"] == "Test Budget"
    assert doc.front_matter["synthetic"] is True
    assert doc.csv_rows == [
        {"year": "1", "category": "equipment", "amount_usd": "5000"},
        {"year": "2", "category": "travel", "amount_usd": "1000"},
    ]
    assert "| year | category | amount_usd |" in doc.text


def test_load_pdf_page_sections(tmp_path):
    path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)

    doc = load(path, doc_id="pdf-doc")
    assert doc.doc_type == "pdf"
    assert len(doc.sections) == 3
    assert doc.sections[0].heading == "Page 1"


def test_load_pdf_over_page_limit_rejected(tmp_path):
    path = tmp_path / "big.pdf"
    writer = PdfWriter()
    for _ in range(201):
        writer.add_blank_page(width=50, height=50)
    with path.open("wb") as f:
        writer.write(f)

    with pytest.raises(LoaderError):
        load(path, doc_id="big-doc")


def test_load_rejects_oversized_file(tmp_path, monkeypatch):
    path = tmp_path / "huge.md"
    path.write_text("# Title\n\nbody\n", encoding="utf-8")
    monkeypatch.setattr("app.ingest.loaders.MAX_BYTES", 5)
    with pytest.raises(LoaderError):
        load(path, doc_id="huge")


def test_load_unsupported_extension(tmp_path):
    path = tmp_path / "file.docx"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(LoaderError):
        load(path)


def test_chunk_document_word_targets_and_overlap(tmp_path):
    path = tmp_path / "long.md"
    words = " ".join(f"word{i}" for i in range(1000))
    path.write_text(f"# Section\n\n{words}\n", encoding="utf-8")
    doc = load(path, doc_id="long-doc")

    chunks = chunk_document(doc, target_words=425, overlap_words=50)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"long-doc#c{i:03d}"
        assert c.doc_id == "long-doc"
        assert len(c.text.split()) <= 425

    tail = chunks[0].text.split()[-10:]
    head = chunks[1].text.split()[:60]
    assert any(w in head for w in tail)


def test_chunk_document_empty_returns_no_chunks(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")
    doc = load(path, doc_id="empty-doc")
    assert chunk_document(doc) == []


def test_unquoted_front_matter_dates_do_not_break_json_serialization(tmp_path):
    """YAML turns an unquoted `date: 2026-09-19` into a datetime.date, which travelled
    as far as the tool result and then killed a whole run with "Object of type date is
    not JSON serializable". A visitor's document must not be able to end a run by being
    ordinary Markdown."""
    import json

    from app.ingest.loaders import load

    path = tmp_path / "typed.md"
    path.write_text(
        "---\ntitle: A profile\ndate: 2026-09-19\nreviewed: 2025-01-02\ncount: 3\n---\n\n# Body\n\nText.\n",
        encoding="utf-8",
    )
    doc = load(path)
    assert doc.front_matter["date"] == "2026-09-19"
    assert doc.front_matter["count"] == 3, "real scalars must survive untouched"
    json.dumps(doc.front_matter)  # must not raise
