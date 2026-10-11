"""BM25 retrieval over chunked vault documents (Phase 5.2). No embedding model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingest.chunker import Chunk, chunk_document
from app.ingest.loaders import Document, load

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small built-in English stopword list; numbers are never stopwords.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of to in on for with as at by from
    is are was were be been being this that these those it its it's their his her
    they them we us you your our i me my not no do does did doing have has had
    can could should would will shall may might must about into over under again
    further here there all any both each few more most other some such only own
    same so than too very s t just don now
    """.split()
)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


@dataclass
class Hit:
    doc_id: str
    title: str
    chunk_id: str
    heading: str
    score: float
    snippet: str
    # The full chunk, not just the excerpt. check_claims hands evidence to a model to
    # judge, and a 400-character window centred on the first query token cut off
    # immediately before the value being checked -- so the judge answered
    # "unverifiable" while looking at a table whose relevant cell had been trimmed.
    text: str = ""


def _snippet(text: str, query_tokens: list[str], width: int = 400) -> str:
    lower = text.lower()
    pos = -1
    for t in query_tokens:
        idx = lower.find(t)
        if idx != -1 and (pos == -1 or idx < pos):
            pos = idx
    if pos == -1:
        pos = 0
    start = max(0, pos - width // 2)
    end = min(len(text), start + width)
    start = max(0, end - width)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._chunk_tokens: list[set[str]] = []
        self._titles: dict[str, str] = {}
        self._bm25: BM25Okapi | None = None

    def build(self, documents: list[Document]) -> None:
        self._chunks = []
        self._titles = {}
        corpus_tokens: list[list[str]] = []
        for doc in documents:
            self._titles[doc.doc_id] = doc.title
            for chunk in chunk_document(doc):
                self._chunks.append(chunk)
                corpus_tokens.append(tokenize(chunk.text))
        # Kept so search() can check real token overlap without re-tokenizing.
        self._chunk_tokens = [set(toks) for toks in corpus_tokens]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if self._bm25 is None or not self._chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)
        query_set = set(query_tokens)
        hits: list[Hit] = []
        for i in ranked:
            # BM25's IDF term goes to zero -- and in rank_bm25's formulation,
            # negative -- for a word that occurs in EVERY document. On a large
            # corpus that never matters. On the two- or three-document corpora
            # Discover works with (an institution uploads a strategic plan and a
            # capability statement) it means every query scores <= 0, and dropping
            # those left the agent staring at an apparently empty vault while the
            # documents sat right there. Fall back to requiring genuine token
            # overlap instead, so ranking still works and nothing irrelevant is
            # invented.
            if scores[i] <= 0 and not (query_set & self._chunk_tokens[i]):
                continue
            chunk = self._chunks[i]
            hits.append(
                Hit(
                    doc_id=chunk.doc_id,
                    title=self._titles.get(chunk.doc_id, chunk.doc_id),
                    chunk_id=chunk.chunk_id,
                    heading=chunk.heading,
                    score=float(scores[i]),
                    snippet=_snippet(chunk.text, query_tokens),
                    text=chunk.text,
                )
            )
            if len(hits) >= k:
                break
        return hits


def build_index_from_paths(paths: list[Path]) -> BM25Index:
    documents = [load(p) for p in paths]
    index = BM25Index()
    index.build(documents)
    return index
