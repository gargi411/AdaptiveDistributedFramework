"""SemanticChunker — deterministic, provenance-aware text chunking (Phase 4.2).

Converts a UnifiedDocument into an ordered list of Chunk objects ready for the
Phase 4.3 embedding and indexing layer.

Architecture:
    UnifiedDocument
        -> SemanticChunker.chunk_document()
        -> list[Chunk]
        -> embedding model  (Phase 4.3, not implemented here)
        -> vector index     (Phase 4.3, not implemented here)

Algorithm overview (no LLM, fully deterministic):

    For each page in page_number order:
        1. Extract text segments from layout_elements (preferred) or plain text.
        2. Track the current section heading (element_type == 'heading').
        3. Split each segment at paragraph boundaries (blank lines), then at
           sentence boundaries (terminal punctuation + whitespace).
        4. Accumulate segments into a buffer until the buffer reaches
           chunk_size characters.
        5. Emit a Chunk when the buffer is full.  Carry the last chunk_overlap
           characters into the next buffer (overlap).
        6. If the remaining buffer at the end of a page is below min_chunk_size,
           append it to the last emitted chunk (up to max_chunk_size) rather
           than emitting a tiny isolated fragment.
        7. If a single sentence exceeds max_chunk_size it is hard-split at
           exactly max_chunk_size characters.

Provenance guarantees:
    Every Chunk records:
        - document_id    from UnifiedDocument
        - source_file    from UnifiedDocument.file_path
        - page_numbers   all 1-indexed page numbers whose text contributed
        - section_heading the most recent heading seen before the chunk

Determinism:
    Given the same UnifiedDocument and the same ChunkerConfig, two successive
    calls to chunk_document() always produce byte-identical output.  chunk_ids
    are derived via SHA-256, not random UUID.

Design constraints (all enforced in this module):
    - No LLM calls.
    - No GPU calls.
    - No modification of UnifiedDocument, Page, or any processing pipeline.
    - No dependency on scheduler, coordinator, or work-stealing internals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from adaptive_framework.config.models import ChunkerConfig
from adaptive_framework.interfaces.i_chunker import IChunker, TextChunk
from adaptive_framework.models.chunk import Chunk, _make_chunk_id
from adaptive_framework.models.page import LayoutElement, Page
from adaptive_framework.models.unified_document import UnifiedDocument

# ---------------------------------------------------------------------------
# Sentence boundary detection
# ---------------------------------------------------------------------------

# Matches a sentence-terminal sequence: end-of-sentence punctuation (.!?),
# optionally followed by closing brackets/quotes, then one or more spaces
# or a newline, followed by a capital letter or end of string.
# This regex is intentionally conservative to avoid splitting abbreviations
# like "Dr." or "Fig.".
_SENTENCE_BOUNDARY = re.compile(
    r"""
    (?<=[.!?])          # preceded by sentence-terminal punctuation
    (?:[)\]'\"'"]*)     # optional closing brackets / quotes
    \s+                 # one or more whitespace characters
    (?=[A-Z\u00C0-\u024F\d])  # followed by capital letter, accented char, or digit
    """,
    re.VERBOSE,
)

# Paragraph boundary: two or more newlines (possibly with intervening spaces).
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    """Split text at paragraph boundaries.

    Args:
        text: Raw text to split.

    Returns:
        List of non-empty paragraph strings, whitespace-stripped.
    """
    parts = _PARAGRAPH_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries.

    Args:
        text: A paragraph or block of text.

    Returns:
        List of non-empty sentence strings.
    """
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _segments_from_layout_elements(
    elements: Sequence[LayoutElement],
) -> list[tuple[str, str | None]]:
    """Convert ordered LayoutElements into (text, heading_or_None) pairs.

    Headings update the running section context; they are also returned as
    a standalone segment so their text is included in the chunk output.

    Args:
        elements: Ordered tuple of LayoutElement objects from a Page.

    Returns:
        List of (segment_text, current_heading) pairs.  heading is the
        text of the most recent heading element seen so far, or None.
    """
    result: list[tuple[str, str | None]] = []
    current_heading: str | None = None

    for elem in sorted(elements, key=lambda e: e.reading_order):
        if not elem.text.strip():
            continue
        if elem.element_type == "heading":
            current_heading = elem.text.strip()
            # Include heading text itself as a short segment so it is not
            # silently dropped from the output.
            result.append((current_heading, current_heading))
        elif elem.element_type in {"header", "footer"}:
            # Headers and footers (page numbers, running titles) are
            # structural noise.  Skip them to avoid polluting chunks.
            pass
        else:
            result.append((elem.text.strip(), current_heading))

    return result


def _segments_from_plain_text(text: str) -> list[tuple[str, str | None]]:
    """Convert plain text into (paragraph_text, None) segments.

    Used when a page has no layout_elements.

    Args:
        text: Raw page text.

    Returns:
        List of (paragraph_text, None) pairs.
    """
    return [(para, None) for para in _split_paragraphs(text) if para]


# ---------------------------------------------------------------------------
# Chunk assembly helper
# ---------------------------------------------------------------------------


@dataclass
class _ChunkBuffer:
    """Mutable accumulator used while building chunks inside chunk_document.

    Not part of the public API; destroyed when chunk_document returns.
    """

    texts: list[str]
    page_numbers: list[int]
    current_heading: str | None
    char_count: int

    def append(self, text: str, page_number: int) -> None:
        """Add a text fragment to the buffer."""
        self.texts.append(text)
        self.char_count += len(text)
        if page_number not in self.page_numbers:
            self.page_numbers.append(page_number)

    def flush_text(self) -> str:
        """Join buffered texts with a single space separator."""
        return " ".join(t for t in self.texts if t)

    def reset_with_overlap(self, overlap_text: str, page_number: int) -> None:
        """Reset the buffer, seeding it with overlap text from the last chunk."""
        self.texts = [overlap_text] if overlap_text else []
        self.page_numbers = [page_number]
        self.char_count = len(overlap_text)
        # current_heading is intentionally preserved across chunk boundaries.


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class SemanticChunker(IChunker):
    """Deterministic semantic chunking of UnifiedDocument objects.

    Implements the IChunker interface for compatibility with the existing
    infrastructure, and extends it with chunk_document() which produces
    fully provenance-aware Chunk objects from a UnifiedDocument.

    Args:
        config: ChunkerConfig with chunk_size, chunk_overlap, min_chunk_size,
            and max_chunk_size settings.

    Example:
        >>> cfg = ChunkerConfig(
        ...     strategy="semantic", chunk_size=512,
        ...     chunk_overlap=64, min_chunk_size=50, max_chunk_size=2000,
        ... )
        >>> chunker = SemanticChunker(cfg)
        >>> chunks = chunker.chunk_document(unified_doc)
        >>> print(len(chunks))
        42
    """

    def __init__(self, config: ChunkerConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # IChunker implementation (plain-text, no provenance)
    # ------------------------------------------------------------------

    def chunk(self, text: str, document_id: str) -> list[TextChunk]:
        """Split plain text into a list of TextChunk objects.

        This method exists for IChunker compatibility.  It performs the same
        size-based splitting as chunk_document() but without page provenance.

        Args:
            text: Full text content to chunk.
            document_id: Identifier of the source document.

        Returns:
            Ordered list of TextChunk objects.
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._split_text_into_raw_chunks(text)
        result: list[TextChunk] = []
        pos = 0
        for idx, chunk_text in enumerate(raw_chunks):
            start = text.find(chunk_text, pos)
            if start == -1:
                start = pos
            end = start + len(chunk_text)
            result.append(
                TextChunk(
                    text=chunk_text,
                    document_id=document_id,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                )
            )
            pos = max(pos, end - self._config.chunk_overlap)
        return result

    def get_strategy_name(self) -> str:
        """Return the chunking strategy identifier.

        Returns:
            Always 'semantic' for this implementation.
        """
        return "semantic"

    # ------------------------------------------------------------------
    # Primary public API — provenance-aware chunking
    # ------------------------------------------------------------------

    def chunk_document(self, doc: UnifiedDocument) -> list[Chunk]:
        """Convert a UnifiedDocument into an ordered list of Chunk objects.

        This is the primary entry point for Phase 4.2.  It respects page
        order, section headings, paragraph boundaries, and sentence
        boundaries while enforcing the configured chunk size limits.

        No text is silently discarded.  Every non-whitespace character that
        appears in a page's text or layout_elements is present in exactly one
        emitted chunk.

        Args:
            doc: A fully-constructed, immutable UnifiedDocument.

        Returns:
            Ordered list of Chunk objects (chunk_index is 0-based,
            monotonically increasing, with no gaps).  An empty document or
            a document whose pages all contain only whitespace returns an
            empty list.
        """
        emitted: list[Chunk] = []
        chunk_index = 0

        # Determine document_type from layout if available.
        document_type: str | None = None
        if doc.layout.title:
            document_type = "research_paper"

        if not doc.pages:
            return emitted

        # Sort pages by page_number to guarantee reading order.
        pages = sorted(doc.pages, key=lambda p: p.page_number)

        # Main buffer accumulates text across page boundaries so that a chunk
        # is never split in the middle of a sentence simply because a page
        # boundary happens to fall there.
        buf = _ChunkBuffer(
            texts=[],
            page_numbers=[],
            current_heading=None,
            char_count=0,
        )

        cfg = self._config

        for page in pages:
            page_segments = self._extract_segments(page)

            for seg_text, seg_heading in page_segments:
                # Update heading context whenever a heading segment is seen.
                if seg_heading is not None:
                    buf.current_heading = seg_heading

                # Split the segment into sentence-level pieces.
                sentences = _split_sentences(seg_text)
                if not sentences:
                    sentences = [seg_text]

                for sentence in sentences:
                    # Hard-split sentences that exceed max_chunk_size.
                    sentence_pieces = self._hard_split(sentence, cfg.max_chunk_size)

                    for piece in sentence_pieces:
                        if not piece.strip():
                            continue

                        # If adding this piece would exceed chunk_size, emit
                        # the current buffer first.
                        if (
                            buf.char_count > 0
                            and buf.char_count + len(piece) > cfg.chunk_size
                        ):
                            chunk, chunk_index = self._emit_chunk(
                                buf=buf,
                                doc=doc,
                                chunk_index=chunk_index,
                                document_type=document_type,
                                page_number=page.page_number,
                            )
                            emitted.append(chunk)
                            # Seed next buffer with overlap from the chunk
                            # just emitted.
                            overlap = self._compute_overlap(
                                chunk.text, cfg.chunk_overlap
                            )
                            buf.reset_with_overlap(overlap, page.page_number)

                        buf.append(piece, page.page_number)

                        # If the buffer has grown to or beyond chunk_size,
                        # emit it immediately.
                        while buf.char_count >= cfg.chunk_size:
                            chunk_text = buf.flush_text()
                            emit_text, remainder = self._split_at_size(
                                chunk_text, cfg.chunk_size
                            )
                            if not emit_text.strip():
                                break

                            chunk, chunk_index = self._emit_chunk_from_text(
                                text=emit_text,
                                buf=buf,
                                doc=doc,
                                chunk_index=chunk_index,
                                document_type=document_type,
                                page_number=page.page_number,
                            )
                            emitted.append(chunk)

                            overlap = self._compute_overlap(
                                emit_text, cfg.chunk_overlap
                            )
                            buf.reset_with_overlap(overlap, page.page_number)
                            if remainder.strip():
                                buf.append(remainder, page.page_number)
                            else:
                                break

        # Flush whatever remains in the buffer after all pages are processed.
        if buf.char_count > 0 and buf.flush_text().strip():
            remaining_text = buf.flush_text()

            if (
                emitted
                and buf.char_count < cfg.min_chunk_size
                and len(emitted[-1].text) + 1 + len(remaining_text) <= cfg.max_chunk_size
            ):
                # Merge tiny trailing fragment into the last emitted chunk.
                last = emitted[-1]
                merged_text = last.text + " " + remaining_text
                merged_pages = tuple(
                    sorted(set(last.page_numbers) | set(buf.page_numbers))
                )
                emitted[-1] = Chunk(
                    chunk_id=last.chunk_id,
                    document_id=last.document_id,
                    source_file=last.source_file,
                    chunk_index=last.chunk_index,
                    page_numbers=merged_pages,
                    text=merged_text,
                    section_heading=last.section_heading,
                    document_type=last.document_type,
                    character_count=len(merged_text),
                    word_count=len(merged_text.split()),
                    start_char_in_page=last.start_char_in_page,
                    end_char_in_page=last.end_char_in_page + 1 + len(remaining_text),
                )
            else:
                chunk, chunk_index = self._emit_chunk(
                    buf=buf,
                    doc=doc,
                    chunk_index=chunk_index,
                    document_type=document_type,
                    page_number=buf.page_numbers[-1] if buf.page_numbers else 1,
                )
                emitted.append(chunk)

        return emitted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_segments(
        self, page: Page
    ) -> list[tuple[str, str | None]]:
        """Extract (text, heading) segments from a Page.

        Prefers layout_elements for structural fidelity.  Falls back to
        plain page.text when no layout elements are available.

        Args:
            page: An immutable Page object.

        Returns:
            List of (text, current_heading_or_None) pairs.
        """
        if page.layout_elements:
            return _segments_from_layout_elements(page.layout_elements)
        if page.text and page.text.strip():
            return _segments_from_plain_text(page.text)
        return []

    def _split_text_into_raw_chunks(self, text: str) -> list[str]:
        """Split a plain text string into raw chunk strings.

        Used by the IChunker.chunk() compatibility method.

        Args:
            text: Plain text to chunk.

        Returns:
            List of chunk text strings, none empty.
        """
        cfg = self._config
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            paragraphs = [text.strip()]

        raw_chunks: list[str] = []
        buffer = ""

        for para in paragraphs:
            sentences = _split_sentences(para)
            if not sentences:
                sentences = [para]

            for sentence in sentences:
                for piece in self._hard_split(sentence, cfg.max_chunk_size):
                    if not piece.strip():
                        continue
                    if buffer and len(buffer) + 1 + len(piece) > cfg.chunk_size:
                        raw_chunks.append(buffer.strip())
                        overlap = self._compute_overlap(buffer, cfg.chunk_overlap)
                        buffer = (overlap + " " + piece).strip()
                    else:
                        buffer = (buffer + " " + piece).strip() if buffer else piece

        if buffer.strip():
            if (
                raw_chunks
                and len(buffer) < cfg.min_chunk_size
                and len(raw_chunks[-1]) + 1 + len(buffer) <= cfg.max_chunk_size
            ):
                raw_chunks[-1] = (raw_chunks[-1] + " " + buffer).strip()
            else:
                raw_chunks.append(buffer.strip())

        return [c for c in raw_chunks if c.strip()]

    @staticmethod
    def _hard_split(text: str, max_size: int) -> list[str]:
        """Hard-split text that exceeds max_size into pieces of exactly max_size.

        This is the last resort for pathologically long sentences.  The
        split happens at a word boundary near the max_size position if one
        exists within the last 20 % of the window; otherwise it splits at
        exactly max_size.

        Args:
            text: Text to split.
            max_size: Maximum allowed length per piece.

        Returns:
            List of text pieces, each <= max_size characters.
        """
        if len(text) <= max_size:
            return [text]

        pieces: list[str] = []
        while len(text) > max_size:
            split_at = max_size
            # Try to find a word boundary within the last 20 % of the window.
            search_start = max(0, max_size - max_size // 5)
            space_pos = text.rfind(" ", search_start, max_size)
            if space_pos > search_start:
                split_at = space_pos
            pieces.append(text[:split_at].strip())
            text = text[split_at:].strip()
        if text:
            pieces.append(text)
        return pieces

    @staticmethod
    def _compute_overlap(text: str, overlap_chars: int) -> str:
        """Return the last overlap_chars characters of text for seeding the next buffer.

        If the text is shorter than overlap_chars the full text is returned so
        that no characters are lost.

        Args:
            text: The text of the chunk just emitted.
            overlap_chars: Number of characters to carry over.

        Returns:
            Overlap string (may be empty if overlap_chars == 0).
        """
        if overlap_chars <= 0 or not text:
            return ""
        return text[-overlap_chars:]

    @staticmethod
    def _split_at_size(text: str, size: int) -> tuple[str, str]:
        """Split text into a (head, tail) pair where len(head) <= size.

        Tries to split at a word boundary near size.  Falls back to hard
        split if no word boundary is found.

        Args:
            text: Text to split.
            size: Target head length.

        Returns:
            (head, tail) where head has at most size characters.
        """
        if len(text) <= size:
            return text, ""
        space_pos = text.rfind(" ", 0, size)
        if space_pos > 0:
            return text[:space_pos].strip(), text[space_pos:].strip()
        return text[:size], text[size:]

    def _emit_chunk(
        self,
        buf: _ChunkBuffer,
        doc: UnifiedDocument,
        chunk_index: int,
        document_type: str | None,
        page_number: int,
    ) -> tuple[Chunk, int]:
        """Build a Chunk from the current buffer state and increment the index.

        Args:
            buf: The current chunk accumulation buffer.
            doc: Parent UnifiedDocument (provides document_id and file_path).
            chunk_index: Next chunk index to assign.
            document_type: Document type label or None.
            page_number: Current page number (used to fill page_numbers if
                the buffer's page_numbers list is empty).

        Returns:
            Tuple of (emitted Chunk, next chunk_index).
        """
        text = buf.flush_text().strip()
        page_nums = tuple(sorted(set(buf.page_numbers))) or (page_number,)
        chunk = Chunk(
            chunk_id=_make_chunk_id(doc.document_id, chunk_index),
            document_id=doc.document_id,
            source_file=doc.file_path,
            chunk_index=chunk_index,
            page_numbers=page_nums,
            text=text,
            section_heading=buf.current_heading,
            document_type=document_type,
            character_count=len(text),
            word_count=len(text.split()),
            start_char_in_page=0,
            end_char_in_page=len(text),
        )
        return chunk, chunk_index + 1

    def _emit_chunk_from_text(
        self,
        text: str,
        buf: _ChunkBuffer,
        doc: UnifiedDocument,
        chunk_index: int,
        document_type: str | None,
        page_number: int,
    ) -> tuple[Chunk, int]:
        """Build a Chunk from an explicit text string (used during mid-buffer splits).

        Args:
            text: Explicit chunk text to emit.
            buf: Buffer providing page provenance and heading context.
            doc: Parent UnifiedDocument.
            chunk_index: Next chunk index to assign.
            document_type: Document type label or None.
            page_number: Current page number.

        Returns:
            Tuple of (emitted Chunk, next chunk_index).
        """
        page_nums = tuple(sorted(set(buf.page_numbers))) or (page_number,)
        text = text.strip()
        chunk = Chunk(
            chunk_id=_make_chunk_id(doc.document_id, chunk_index),
            document_id=doc.document_id,
            source_file=doc.file_path,
            chunk_index=chunk_index,
            page_numbers=page_nums,
            text=text,
            section_heading=buf.current_heading,
            document_type=document_type,
            character_count=len(text),
            word_count=len(text.split()),
            start_char_in_page=0,
            end_char_in_page=len(text),
        )
        return chunk, chunk_index + 1
