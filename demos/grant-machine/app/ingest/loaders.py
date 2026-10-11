"""Load .md/.html/.pdf/.csv vault documents into a normalized Document shape
with heading-aware sections (Phase 5.1)."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify
from pypdf import PdfReader

MAX_BYTES = 20_000_000
MAX_PDF_PAGES = 200

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class LoaderError(Exception):
    pass


@dataclass
class Section:
    heading: str
    level: int
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str  # "md" | "html" | "pdf" | "csv"
    text: str
    sections: list[Section] = field(default_factory=list)
    front_matter: dict[str, Any] = field(default_factory=dict)
    csv_rows: list[dict[str, str]] | None = None
    csv_fieldnames: list[str] | None = None


def load(path: str | Path, doc_id: str | None = None) -> Document:
    path = Path(path)
    if not path.is_file():
        raise LoaderError(f"not a file: {path}")
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise LoaderError(f"{path}: {size} bytes exceeds the {MAX_BYTES}-byte limit")

    doc_id = doc_id or path.stem
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _load_markdown(path, doc_id)
    if suffix in (".html", ".htm"):
        return _load_html(path, doc_id)
    if suffix == ".pdf":
        return _load_pdf(path, doc_id)
    if suffix == ".csv":
        return _load_csv(path, doc_id)
    raise LoaderError(f"{path}: unsupported file type {suffix!r}")


def _json_safe(value: Any) -> Any:
    """Front matter, reduced to types that survive json.dumps."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _split_yaml_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    if raw.startswith("---\n") or raw.startswith("---\r\n"):
        end = raw.find("\n---", 4)
        if end != -1:
            fm_text = raw[4:end]
            body = raw[end + 4 :].lstrip("\n")
            try:
                fm = yaml.safe_load(fm_text) or {}
            except yaml.YAMLError:
                fm = {}
            # Coerce to JSON-safe scalars. YAML turns an unquoted `date: 2026-09-19`
            # into a datetime.date, which travels as far as the tool result and then
            # fails the whole run with "Object of type date is not JSON serializable".
            # The vault's own documents quote their dates; an uploaded or typed one
            # cannot be relied on to, and a visitor's document must not be able to end
            # a run by being ordinary Markdown.
            return (_json_safe(fm) if isinstance(fm, dict) else {}), body
    return {}, raw


def _sections_from_markdown(body: str) -> list[Section]:
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [Section(heading="", level=0, text=stripped)] if stripped else []
    sections: list[Section] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append(Section(heading=heading, level=level, text=body[start:end].strip()))
    return sections


def _load_markdown(path: Path, doc_id: str) -> Document:
    raw = path.read_text(encoding="utf-8")
    front_matter, body = _split_yaml_front_matter(raw)
    sections = _sections_from_markdown(body)
    title = front_matter.get("title") or (sections[0].heading if sections and sections[0].heading else doc_id)
    return Document(doc_id=doc_id, title=str(title), doc_type="md", text=body, sections=sections, front_matter=front_matter)


def _load_html(path: Path, doc_id: str) -> Document:
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    for tag_name in ("nav", "header", "footer", "script", "style"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    main = soup.find("main") or soup.body or soup
    md_text = markdownify(str(main), heading_style="ATX")
    md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip() + "\n"
    sections = _sections_from_markdown(md_text)
    if not title:
        title = sections[0].heading if sections and sections[0].heading else doc_id
    return Document(doc_id=doc_id, title=title, doc_type="html", text=md_text, sections=sections)


def _load_pdf(path: Path, doc_id: str) -> Document:
    reader = PdfReader(str(path))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise LoaderError(f"{path}: {len(reader.pages)} pages exceeds the {MAX_PDF_PAGES}-page limit")

    sections: list[Section] = []
    texts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        sections.append(Section(heading=f"Page {i}", level=1, text=page_text))
        texts.append(page_text)

    full_text = "\n\n".join(texts)
    title = doc_id
    for s in sections:
        if s.text:
            title = s.text.splitlines()[0][:120]
            break
    return Document(doc_id=doc_id, title=title, doc_type="pdf", text=full_text, sections=sections)


def _split_hash_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines(keepends=True)
    front_matter: dict[str, Any] = {}
    idx = 0
    while idx < len(lines) and lines[idx].startswith("#"):
        content = lines[idx].lstrip("#").strip()
        if ":" in content:
            key, _, value = content.partition(":")
            front_matter[key.strip()] = _coerce_scalar(value.strip())
        idx += 1
    return front_matter, "".join(lines[idx:])


def _coerce_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def _rows_to_markdown_table(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    if not fieldnames:
        return ""
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join("---" for _ in fieldnames) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fieldnames) + " |")
    return "\n".join(lines)


def _load_csv(path: Path, doc_id: str) -> Document:
    raw = path.read_text(encoding="utf-8")
    front_matter, csv_text = _split_hash_front_matter(raw)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
    md_table = _rows_to_markdown_table(fieldnames, rows)
    title = str(front_matter.get("title") or doc_id)
    sections = [Section(heading=title, level=1, text=md_table)] if md_table else []
    return Document(
        doc_id=doc_id,
        title=title,
        doc_type="csv",
        text=md_table,
        sections=sections,
        front_matter=front_matter,
        csv_rows=rows,
        csv_fieldnames=fieldnames,
    )
