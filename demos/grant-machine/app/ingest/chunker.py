"""Heading-aware chunking with stable chunk ids (Phase 5.1)."""

from __future__ import annotations

from dataclasses import dataclass

from app.ingest.loaders import Document

TARGET_WORDS = 425  # within the 350-500 word target range
OVERLAP_WORDS = 50


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    heading: str
    text: str


def chunk_document(doc: Document, target_words: int = TARGET_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[Chunk]:
    words: list[str] = []
    word_heading: list[str] = []

    if doc.sections:
        for section in doc.sections:
            heading = section.heading or doc.title
            for w in section.text.split():
                words.append(w)
                word_heading.append(heading)
    else:
        for w in doc.text.split():
            words.append(w)
            word_heading.append(doc.title)

    if not words:
        return []

    step = max(1, target_words - overlap_words)
    chunks: list[Chunk] = []
    n = 0
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        text = " ".join(words[start:end])
        heading = word_heading[start]
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}#c{n:03d}", doc_id=doc.doc_id, heading=heading, text=text))
        n += 1
        if end >= len(words):
            break
        start += step
    return chunks
