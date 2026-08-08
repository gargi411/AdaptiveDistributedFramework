"""IChunker — Abstract text chunking interface (RAG pipeline)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A single chunk of text produced by an IChunker.

    Attributes:
        text: The chunk content.
        document_id: Parent document identifier.
        chunk_index: 0-indexed position within the document.
        start_char: Character offset of the first character in the source text.
        end_char: Character offset (exclusive) of the last character.
    """

    text: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int


class IChunker(ABC):
    """Abstract interface for text chunkers in the RAG pipeline.

    A chunker splits document text into overlapping or non-overlapping
    chunks suitable for embedding. The chunk strategy is configured via
    rag.yaml (fixed_size | sentence | semantic).

    Example:
        >>> chunker: IChunker = FixedSizeChunker(cfg.rag.chunker, logger)
        >>> chunks = chunker.chunk(text="Long document text...", document_id="doc-001")
        >>> print(len(chunks))
        12
    """

    @abstractmethod
    def chunk(self, text: str, document_id: str) -> list[TextChunk]:
        """Split text into a list of TextChunk objects.

        Args:
            text: Full text content of a document (or page range).
            document_id: Identifier of the source document.

        Returns:
            Ordered list of TextChunk objects.

        Raises:
            PipelineError: If chunking fails.
        """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the chunking strategy identifier.

        Returns:
            Strategy name (e.g., 'fixed_size', 'sentence', 'semantic').
        """
