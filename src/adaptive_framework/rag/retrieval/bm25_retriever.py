"""BM25Retriever -- sparse lexical retrieval using BM25Okapi (Phase 4.8).

Implements deterministic lexical retrieval over Chunk objects using the standard
Okapi BM25 ranking function with Robertson-Spärck Jones non-negative IDF.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.interfaces.i_sparse_retriever import ISparseRetriever
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)

# Default BM25 parameters
DEFAULT_K1: float = 1.5
DEFAULT_B: float = 0.75


def tokenize_clinical_text(text: str) -> list[str]:
    """Deterministic tokenizer for English clinical and biomedical text.

    Preserves alphanumeric tokens, dosages (e.g. 500mg), numbers, and medical
    acronyms (e.g. t2dm, cad) while standardizing case and stripping non-alphanumeric
    punctuation. Does NOT aggressively remove clinical stopwords.

    Args:
        text: Input string.

    Returns:
        List of lowercase token strings.
    """
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(ISparseRetriever):
    """Sparse retriever implementing Okapi BM25 over Chunk objects.

    Attributes:
        k1: Term frequency saturation parameter (default: 1.5).
        b: Document length normalization parameter (default: 0.75).
    """

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """Initialize BM25Retriever.

        Args:
            k1: Term frequency saturation parameter. Must be >= 0.
            b: Document length normalization parameter. Must be between 0.0 and 1.0.
        """
        if k1 < 0.0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be between 0.0 and 1.0, got {b}")

        self.k1 = k1
        self.b = b

        self._chunks: list[Chunk] = []
        self._doc_lengths: list[int] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._term_freqs: list[Counter[str]] = []
        self._idf_cache: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._doc_count: int = 0

        self._query_count: int = 0
        self._total_retrieval_time_ms: float = 0.0

    @property
    def doc_count(self) -> int:
        """Total number of indexed chunks."""
        return self._doc_count

    @property
    def avgdl(self) -> float:
        """Average document length in tokens."""
        return self._avgdl

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks into the BM25 inverted index.

        Args:
            chunks: Sequence of Chunk objects.
        """
        self._chunks = list(chunks)
        self._doc_count = len(self._chunks)
        self._doc_lengths = []
        self._term_freqs = []
        self._doc_freqs = defaultdict(int)
        self._idf_cache.clear()

        if self._doc_count == 0:
            self._avgdl = 0.0
            return

        total_length = 0
        for chunk in self._chunks:
            tokens = tokenize_clinical_text(chunk.text)
            length = len(tokens)
            self._doc_lengths.append(length)
            total_length += length

            tf = Counter(tokens)
            self._term_freqs.append(tf)

            for term in tf:
                self._doc_freqs[term] += 1

        self._avgdl = total_length / self._doc_count if self._doc_count > 0 else 0.0

        # Precalculate IDF for all observed terms
        for term, df in self._doc_freqs.items():
            self._idf_cache[term] = self._compute_idf(df)

    def _compute_idf(self, df: int) -> float:
        """Compute non-negative Robertson-Spärck Jones IDF with +1 smoothing."""
        return math.log(1.0 + (self._doc_count - df + 0.5) / (df + 0.5))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve top-K chunks for the given query using BM25 scoring.

        Args:
            query: Query string. Must not be empty.
            top_k: Number of results to return. Must be >= 1.
            min_score: Minimum BM25 score required.

        Returns:
            List of RetrievalResult instances sorted in descending score order.
        """
        if not query or not query.strip():
            raise ValueError("Query string must not be empty or whitespace-only.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        t0 = time.perf_counter()
        query_tokens = tokenize_clinical_text(query)

        if not query_tokens or self._doc_count == 0:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._query_count += 1
            self._total_retrieval_time_ms += elapsed
            return []

        # Find matching documents and accumulate BM25 scores
        doc_scores: dict[int, float] = defaultdict(float)
        matched_terms = [t for t in query_tokens if t in self._doc_freqs]

        if not matched_terms:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._query_count += 1
            self._total_retrieval_time_ms += elapsed
            return []

        for term in matched_terms:
            idf = self._idf_cache.get(term, 0.0)
            for idx, tf_dict in enumerate(self._term_freqs):
                tf = tf_dict.get(term, 0)
                if tf > 0:
                    doc_len = self._doc_lengths[idx]
                    denominator = tf + self.k1 * (
                        1.0 - self.b + self.b * (doc_len / self._avgdl if self._avgdl > 0 else 1.0)
                    )
                    score_contrib = idf * (tf * (self.k1 + 1.0)) / denominator
                    doc_scores[idx] += score_contrib

        # Filter by min_score and sort descending
        scored_candidates = [
            (idx, score)
            for idx, score in doc_scores.items()
            if score >= min_score
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(scored_candidates[:top_k], start=1):
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_file=chunk.source_file,
                    page_numbers=chunk.page_numbers,
                    text=chunk.text,
                    section_heading=chunk.section_heading,
                    document_type=chunk.document_type,
                    chunk_index=chunk.chunk_index,
                    score=float(score),
                    rank=rank,
                )
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        self._query_count += 1
        self._total_retrieval_time_ms += elapsed
        return results

    def get_metrics(self) -> dict[str, Any]:
        """Return runtime metrics."""
        avg_ms = (
            self._total_retrieval_time_ms / self._query_count
            if self._query_count > 0
            else 0.0
        )
        return {
            "total_queries": self._query_count,
            "total_retrieval_time_ms": round(self._total_retrieval_time_ms, 3),
            "avg_retrieval_time_ms": round(avg_ms, 3),
            "doc_count": self._doc_count,
            "avgdl": round(self._avgdl, 2),
            "vocab_size": len(self._doc_freqs),
        }

    def save(self, path: str | Path) -> None:
        """Persist BM25 index and chunk metadata to a JSON file.

        Args:
            path: Destination file path (e.g. outputs/rag/index/bm25_index.json).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self._doc_count,
            "avgdl": self._avgdl,
            "doc_lengths": self._doc_lengths,
            "doc_freqs": dict(self._doc_freqs),
            "term_freqs": [dict(tf) for tf in self._term_freqs],
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "source_file": c.source_file,
                    "page_numbers": list(c.page_numbers),
                    "text": c.text,
                    "section_heading": c.section_heading,
                    "document_type": c.document_type,
                    "chunk_index": c.chunk_index,
                    "character_count": c.character_count,
                    "word_count": c.word_count,
                    "start_char_in_page": c.start_char_in_page,
                    "end_char_in_page": c.end_char_in_page,
                }
                for c in self._chunks
            ],
        }
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load BM25 index and chunk metadata from a persisted JSON file.

        Args:
            path: Source file path.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"BM25 index file not found at {src}")

        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.k1 = float(data.get("k1", DEFAULT_K1))
        self.b = float(data.get("b", DEFAULT_B))
        self._doc_count = int(data.get("doc_count", 0))
        self._avgdl = float(data.get("avgdl", 0.0))
        self._doc_lengths = list(data.get("doc_lengths", []))
        self._doc_freqs = defaultdict(int, data.get("doc_freqs", {}))
        self._term_freqs = [Counter(tf) for tf in data.get("term_freqs", [])]

        self._chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                source_file=c["source_file"],
                page_numbers=tuple(c.get("page_numbers", [1])),
                text=c["text"],
                section_heading=c.get("section_heading"),
                document_type=c.get("document_type"),
                chunk_index=c.get("chunk_index", 0),
                character_count=c.get("character_count", len(c["text"])),
                word_count=c.get("word_count", len(c["text"].split())),
                start_char_in_page=c.get("start_char_in_page", 0),
                end_char_in_page=c.get("end_char_in_page", len(c["text"])),
            )
            for c in data.get("chunks", [])
        ]

        self._idf_cache.clear()
        for term, df in self._doc_freqs.items():
            self._idf_cache[term] = self._compute_idf(df)
