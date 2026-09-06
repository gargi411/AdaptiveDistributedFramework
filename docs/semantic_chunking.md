# Semantic Chunking Engine — Phase 4.2

## Purpose

The Semantic Chunking Engine converts a UnifiedDocument (the immutable aggregate
output of the distributed processing pipeline) into an ordered list of Chunk
objects.  These Chunk objects are the direct input to the Phase 4.3 embedding
and indexing layer.

Chunking is necessary because embedding models have a fixed context window.
A typical biomedical paper contains tens of thousands of characters; a typical
embedding model accepts at most a few hundred tokens at a time.  The chunker
breaks the full document into pieces that fit within that window while
preserving as much semantic cohesion as possible.

---

## Input and Output

Input:  adaptive_framework.models.unified_document.UnifiedDocument

    A frozen dataclass produced by the coordinator after all distributed
    workers have finished.  It contains an ordered tuple of Page objects,
    each carrying extracted text, layout elements, tables, and figures.

Output: list[adaptive_framework.models.chunk.Chunk]

    An ordered list of immutable Chunk objects.  Each Chunk contains the
    text content, full provenance metadata, and pre-computed counts.

    The list is ordered by document reading order (page 1 first, last page
    last; within a page, reading order is determined by LayoutElement
    reading_order or by paragraph position in the plain text).

    The list may be empty if the document contains no extractable text.
    It is never None.

---

## Algorithm

The algorithm is deterministic.  Two identical calls with the same document
and the same configuration always produce byte-identical output.  No LLM is
involved.

Step 1: Sort pages by page_number (ascending).

Step 2: For each page, extract text segments.

    If the page has layout_elements:
        - Sort elements by reading_order.
        - Skip header and footer elements (page numbers, running titles).
        - Treat heading elements as both a section-heading update and a short
          text segment so the heading text itself appears in the output.
        - Treat all other element types as text segments.

    If the page has no layout_elements:
        - Split page.text at paragraph boundaries (blank lines).
        - Each paragraph becomes a segment.

Step 3: For each segment, split at sentence boundaries.

    The sentence splitter uses a regular expression that matches terminal
    punctuation (.!?) followed by whitespace and a capital letter.  It is
    conservative to avoid splitting on abbreviations like "Fig." or "Dr."

Step 4: Accumulate segments into a buffer.

    When the buffer reaches chunk_size characters, emit a Chunk.  Carry
    the last chunk_overlap characters of the emitted chunk into the next
    buffer so that context is not lost at chunk boundaries.

Step 5: Handle edge cases.

    - Single sentences longer than max_chunk_size are hard-split at a word
      boundary near the limit.  No single Chunk ever exceeds max_chunk_size.

    - When a buffer is flushed and the remaining text is shorter than
      min_chunk_size, the remaining text is appended to the previous chunk
      (if the merged result does not exceed max_chunk_size) rather than
      emitting a tiny isolated fragment.

    - Empty pages and pages containing only whitespace produce no chunks.

Step 6: Flush the final buffer after all pages are processed.

---

## Chunk Metadata Fields

Every Chunk contains the following fields:

    chunk_id
        A 16-character lowercase hex string derived from
        SHA-256(document_id + ":" + chunk_index).
        Deterministic: identical pipeline runs produce identical chunk_ids.
        This enables idempotent upserts into any downstream vector store.

    document_id
        Copied from UnifiedDocument.document_id.
        Allows any retrieved chunk to be traced back to its source document.

    source_file
        Copied from UnifiedDocument.file_path.
        The absolute path to the source PDF on the filesystem.

    chunk_index
        0-indexed, monotonically increasing, no gaps.
        Allows reconstruction of document reading order from a set of chunks.

    page_numbers
        A tuple of one or more 1-indexed page numbers.
        A chunk that spans a page boundary lists both pages.
        A single-page chunk lists exactly one page.

    text
        The chunk content.  Never empty.  Never contains only whitespace.

    section_heading
        The text of the most recent heading-type LayoutElement seen before
        this chunk was emitted.  None if no heading has been seen.
        Populated from Page.layout_elements where element_type == 'heading'.

    document_type
        A human-readable label for the document kind.  Currently set to
        "research_paper" when the document has a layout title; None otherwise.
        This field is reserved for future enrichment from document metadata.

    character_count
        len(text).  Pre-computed to avoid repeated scanning downstream.

    word_count
        len(text.split()).  Approximate word count.

    start_char_in_page
        Character offset in the source page's plain text where this chunk
        begins.  Meaningful for single-page chunks; set to 0 for multi-page
        chunks.

    end_char_in_page
        Character offset (exclusive) where this chunk ends in the source
        page's plain text.  Set to character_count for multi-page chunks.

---

## Why Provenance Is Preserved

The Adaptive Distributed Framework is designed for a doctor-facing RAG system.
Every answer generated by the system must cite the specific document and page
from which the supporting evidence was taken.  Without provenance, a retrieved
chunk cannot be attributed to a source, and the system cannot satisfy
clinical or regulatory traceability requirements.

The chunk_id, document_id, source_file, and page_numbers fields together
provide complete traceability:

    Given a chunk_id, you can find:
        - Which document it came from (document_id)
        - Which file on disk contains that document (source_file)
        - Which pages of that document contributed the text (page_numbers)
        - Which section of the document it belongs to (section_heading)

This chain of evidence is maintained from the moment a PDF is read by a worker
through the distributed pipeline and all the way into the vector store.

---

## Configuration

Chunking is configured in configs/rag.yaml under the chunker section.
All values are read by ConfigManager and passed to SemanticChunker as a
ChunkerConfig dataclass.

    chunk_size (default: 512)
        Target chunk length in characters.  When the accumulation buffer
        reaches this length, a chunk is emitted.

    chunk_overlap (default: 64)
        Number of characters carried from the end of one chunk into the
        beginning of the next.  Overlap preserves cross-boundary context.
        Must be >= 0 and strictly less than chunk_size.

    min_chunk_size (default: 50)
        Minimum characters a chunk must contain to be emitted as a standalone
        fragment.  Shorter fragments are merged into the previous chunk.
        Must be >= 1.

    max_chunk_size (default: 2000)
        Hard ceiling on chunk length in characters.  A chunk that would
        exceed this limit is split at the nearest word boundary before the
        limit, or at exactly max_chunk_size if no word boundary is found.
        Must be >= chunk_size.

Example configuration (rag.yaml):

    rag:
      chunker:
        strategy: "semantic"
        chunk_size: 512
        chunk_overlap: 64
        min_chunk_size: 50
        max_chunk_size: 2000

---

## Usage Example

    from adaptive_framework.config.models import ChunkerConfig
    from adaptive_framework.rag import SemanticChunker
    from adaptive_framework.models import UnifiedDocument

    cfg = ChunkerConfig(
        strategy="semantic",
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_size=50,
        max_chunk_size=2000,
    )
    chunker = SemanticChunker(cfg)
    chunks = chunker.chunk_document(unified_doc)

    for chunk in chunks:
        print(chunk.chunk_id, chunk.page_numbers, chunk.section_heading)
        print(chunk.text[:80])
        print()

---

## IChunker Compatibility

SemanticChunker also implements the IChunker abstract interface so it can be
used anywhere the codebase expects an IChunker:

    chunks = chunker.chunk(text="Full text...", document_id="doc-001")

This method performs the same size-based splitting but returns TextChunk
objects (defined in interfaces/i_chunker.py) without page provenance.
Use chunk_document() for full provenance.

---

## What Is Not Implemented in Phase 4.2

The following are explicitly out of scope and will be implemented in
Phase 4.3 and later phases:

    - Embedding model (sentence-transformers or OpenVINO-accelerated)
    - FAISS or any vector index
    - Retrieval pipeline (query embedding, similarity search)
    - LLM-based answer generation
    - GPU inference
    - Any modification to the distributed scheduler or PDF processing pipeline

---

## Files Added or Modified in Phase 4.2

New files:
    src/adaptive_framework/models/chunk.py
    src/adaptive_framework/rag/chunker.py
    tests/unit/test_semantic_chunker.py
    docs/semantic_chunking.md

Modified files:
    src/adaptive_framework/config/models.py     (ChunkerConfig extended)
    src/adaptive_framework/models/__init__.py   (Chunk exported)
    src/adaptive_framework/rag/__init__.py      (SemanticChunker exported)
    configs/rag.yaml                            (min/max_chunk_size added)
    project_progress.md                         (Phase 4.2 recorded)
