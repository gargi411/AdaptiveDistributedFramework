"""RetrievalResult — flat, immutable result object for Phase 4.4.

A RetrievalResult is the public-facing output of RetrievalEngine.retrieve().
It flattens the Chunk metadata into a single dataclass so that downstream
consumers (Phase 4.4 Milestone 2: context builder) do not need to import
or understand Chunk internals.

Relationship to FAISSManager.SearchResult:
    FAISSManager.search() returns SearchResult objects that carry the raw
    Chunk object and the internal FAISS rank.  RetrievalResult wraps that
    information with a flatter structure that is safe for serialisation and
    does not expose FAISS implementation details.

Fields sourced from Chunk:
    chunk_id       -> Chunk.chunk_id
    document_id    -> Chunk.document_id
    source_file    -> Chunk.source_file
    page_numbers   -> Chunk.page_numbers  (tuple of 1-indexed page ints)
    text           -> Chunk.text
    section_heading -> Chunk.section_heading (None if not detected)
    document_type  -> Chunk.document_type  (None if not available)
    chunk_index    -> Chunk.chunk_index

Fields sourced from FAISSManager.SearchResult:
    score   Cosine similarity in [-1, 1]; higher is more similar.
            For L2-normalized IndexFlatIP, range is effectively [0, 1].
    rank    1-indexed rank within the result set (1 = most similar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    """Flat, immutable record for one chunk retrieved by RetrievalEngine.

    Attributes:
        chunk_id: Deterministic 16-char hex identifier of the source Chunk.
        document_id: Identifier of the parent document.
        source_file: Absolute path to the source PDF.
        page_numbers: Tuple of 1-indexed page numbers the chunk spans.
        text: The chunk text content.
        section_heading: Active section heading at extraction time, or None.
        document_type: Document kind label (e.g. 'research_paper'), or None.
        chunk_index: 0-indexed position of this chunk within its document.
        score: Cosine similarity score.  Range: [-1, 1]; higher = more similar.
            For IndexFlatIP with L2-normalized vectors, effective range is [0, 1].
        rank: 1-indexed rank within the current query's result set.
            rank=1 is the most similar result.

    Example:
        >>> result = RetrievalResult(
        ...     chunk_id="a1b2c3d4e5f60718",
        ...     document_id="doc-001",
        ...     source_file="/data/paper.pdf",
        ...     page_numbers=(3,),
        ...     text="The patient presented with elevated troponin.",
        ...     section_heading="Clinical Findings",
        ...     document_type="clinical_note",
        ...     chunk_index=2,
        ...     score=0.87,
        ...     rank=1,
        ... )
        >>> result.rank
        1
        >>> result.score
        0.87
    """

    chunk_id: str
    document_id: str
    source_file: str
    page_numbers: tuple[int, ...]
    text: str
    section_heading: str | None
    document_type: str | None
    chunk_index: int
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary.

        Returns:
            Dictionary with all result fields.  page_numbers is a list
            so the result is directly JSON-serialisable.
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "page_numbers": list(self.page_numbers),
            "text": self.text,
            "section_heading": self.section_heading,
            "document_type": self.document_type,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "rank": self.rank,
        }

    def __repr__(self) -> str:
        return (
            f"RetrievalResult("
            f"rank={self.rank}, "
            f"score={self.score:.4f}, "
            f"chunk_id='{self.chunk_id}', "
            f"doc='{self.document_id}')"
        )
