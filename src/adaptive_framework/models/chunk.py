"""Immutable Chunk model — the atomic output of the SemanticChunker.

Every Chunk is fully traceable back to its source document and source pages.
Provenance is mandatory because the doctor-facing RAG system must be able to
cite evidence for every answer it produces.

Architecture position:
    UnifiedDocument
        -> SemanticChunker (rag/chunker.py)
        -> list[Chunk]            <-- this module
        -> embedding model        (Phase 4.3 — not implemented here)
        -> vector index           (Phase 4.3 — not implemented here)

Design rules:
    - Frozen dataclass; matches the existing pattern used by Page and
      UnifiedDocument.
    - chunk_id is a deterministic SHA-256 digest so that identical pipeline
      runs produce byte-identical IDs.  This enables idempotent upserts into
      any downstream vector store without generating random UUIDs on each run.
    - No mutable defaults; all collection fields use immutable tuples.
    - to_dict() follows the same convention as every other model in the
      project so that serialisation is straightforward.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Derive a deterministic chunk identifier.

    The identifier is a 16-character hex prefix of the SHA-256 digest of
    ``<document_id>:<chunk_index>``.  This is short enough to be readable in
    logs while being collision-resistant for any realistic document corpus.

    Args:
        document_id: Parent document identifier.
        chunk_index: 0-indexed position of the chunk within the document.

    Returns:
        16-character lowercase hex string.
    """
    raw = f"{document_id}:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    """Immutable, provenance-aware text chunk produced by SemanticChunker.

    A Chunk carries everything the embedding and retrieval layers need:

    - The text content itself.
    - Full traceability back to the source document and the exact pages it
      spans (required for RAG citation).
    - The section heading that was active when the chunk was extracted, so
      that retrieved chunks can be presented with structural context.
    - Pre-computed character and word counts to avoid re-scanning text
      downstream.

    Attributes:
        chunk_id: Deterministic 16-character hex identifier derived from
            (document_id, chunk_index) via SHA-256.  Identical runs produce
            identical IDs.
        document_id: Identifier of the parent UnifiedDocument.
        source_file: Absolute path to the source PDF, copied from
            UnifiedDocument.file_path.
        chunk_index: 0-indexed sequential position of this chunk within the
            document.  Monotonically increasing; no gaps.
        page_numbers: Tuple of 1-indexed page numbers that contributed text
            to this chunk.  A chunk that spans a page boundary will list both
            pages.  Single-page chunks list exactly one page.
        text: The chunk content.  Never empty; the chunker guarantees at
            least one non-whitespace character.
        section_heading: The most recent heading-level LayoutElement text
            seen before this chunk was emitted, or None if no heading was
            detected.  Populated from Page.layout_elements where
            element_type == 'heading'.
        document_type: A human-readable label for the document kind (e.g.
            'research_paper', 'clinical_note').  Populated from
            DocumentLayout or document metadata where available; None
            otherwise.
        character_count: len(text).  Pre-computed at construction time.
        word_count: Approximate word count (len(text.split())).
        start_char_in_page: Character offset in the source page's plain text
            string where this chunk begins.  Meaningful only when the chunk
            covers a single page; set to 0 for multi-page chunks.
        end_char_in_page: Character offset (exclusive) in the source page's
            plain text string where this chunk ends.  Set to
            character_count for multi-page chunks.

    Example:
        >>> c = Chunk(
        ...     chunk_id="a1b2c3d4e5f60718",
        ...     document_id="doc-001",
        ...     source_file="/data/paper.pdf",
        ...     chunk_index=0,
        ...     page_numbers=(1,),
        ...     text="Introduction to biomedical NLP.",
        ...     section_heading="Introduction",
        ...     document_type="research_paper",
        ...     character_count=31,
        ...     word_count=5,
        ...     start_char_in_page=0,
        ...     end_char_in_page=31,
        ... )
        >>> c.chunk_id
        'a1b2c3d4e5f60718'
        >>> c.page_numbers
        (1,)
    """

    chunk_id: str
    document_id: str
    source_file: str
    chunk_index: int
    page_numbers: tuple[int, ...]
    text: str
    section_heading: str | None
    document_type: str | None
    character_count: int
    word_count: int
    start_char_in_page: int
    end_char_in_page: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary.

        Returns:
            Dictionary containing all chunk fields.  Collection fields are
            converted to lists so the result is JSON-serialisable without any
            further transformation.
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "page_numbers": list(self.page_numbers),
            "text": self.text,
            "section_heading": self.section_heading,
            "document_type": self.document_type,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "start_char_in_page": self.start_char_in_page,
            "end_char_in_page": self.end_char_in_page,
        }

    def __repr__(self) -> str:
        heading_part = (
            f", heading='{self.section_heading}'" if self.section_heading else ""
        )
        return (
            f"Chunk(id='{self.chunk_id}', doc='{self.document_id}', "
            f"index={self.chunk_index}, pages={self.page_numbers}, "
            f"chars={self.character_count}{heading_part})"
        )
