"""Context builder for assembling retrieved chunks into an LLM-ready context block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class EvidenceBlock:
    """A formatted block of evidence derived from a single retrieval result."""

    rank: int
    document_id: str
    source_file: str
    page_numbers: tuple[int, ...]
    chunk_id: str
    score: float
    section_heading: str | None
    text: str

    def to_dict(self) -> dict:
        """Convert evidence block to a dictionary."""
        return {
            "rank": self.rank,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "page_numbers": list(self.page_numbers),
            "chunk_id": self.chunk_id,
            "score": self.score,
            "section_heading": self.section_heading,
            "text": self.text,
        }


@dataclass(frozen=True)
class BuiltContext:
    """Assembled context payload ready for prompt integration."""

    evidence_blocks: tuple[EvidenceBlock, ...]
    total_chars: int
    truncated: bool
    retained_results: tuple[RetrievalResult, ...]

    def to_dict(self) -> dict:
        """Convert built context to a dictionary."""
        return {
            "evidence_blocks": [block.to_dict() for block in self.evidence_blocks],
            "total_chars": self.total_chars,
            "truncated": self.truncated,
            "retained_results": [r.to_dict() for r in self.retained_results],
        }


class ContextBuilder:
    """Filters, deduplicates, bounds, and structures retrieval results for LLM prompts.

    Maintains source provenance (document_id, source_file, page_numbers)
    and enforces chunk/character limits while filtering by minimum score.
    """

    def __init__(
        self,
        max_chunks: int = 5,
        max_context_chars: int = 4000,
        min_score: float = 0.0,
    ) -> None:
        """Initialise ContextBuilder.

        Args:
            max_chunks: Maximum number of chunks to retain.
            max_context_chars: Maximum aggregate characters of chunk text retained.
            min_score: Minimum similarity score required to keep a chunk.
        """
        self.max_chunks = max(1, max_chunks)
        self.max_context_chars = max(100, max_context_chars)
        self.min_score = min_score

    def build(self, results: Sequence[RetrievalResult]) -> BuiltContext:
        """Assemble structured context from retrieval results.

        Filtering order:
        1. Discard results with score < min_score.
        2. Sort/preserve by rank ascending.
        3. Cap by max_chunks.
        4. Accumulate chunks until max_context_chars is reached; mark truncated if exceeded.

        Args:
            results: Sequence of RetrievalResult instances.

        Returns:
            BuiltContext containing structured evidence blocks and retained results.
        """
        if not results:
            return BuiltContext(
                evidence_blocks=(),
                total_chars=0,
                truncated=False,
                retained_results=(),
            )

        # 1. Score filter
        filtered = [r for r in results if r.score >= self.min_score]
        if not filtered:
            return BuiltContext(
                evidence_blocks=(),
                total_chars=0,
                truncated=False,
                retained_results=(),
            )

        # 2. Sort by rank ascending
        sorted_results = sorted(filtered, key=lambda r: r.rank)

        # 3. Cap by max_chunks
        capped = sorted_results[: self.max_chunks]

        # 4. Enforce max_context_chars budget
        accumulated_chars = 0
        retained: list[RetrievalResult] = []
        blocks: list[EvidenceBlock] = []
        truncated = False

        for r in capped:
            chunk_len = len(r.text)
            if accumulated_chars + chunk_len <= self.max_context_chars:
                accumulated_chars += chunk_len
                retained.append(r)
                blocks.append(
                    EvidenceBlock(
                        rank=r.rank,
                        document_id=r.document_id,
                        source_file=r.source_file,
                        page_numbers=r.page_numbers,
                        chunk_id=r.chunk_id,
                        score=r.score,
                        section_heading=r.section_heading,
                        text=r.text,
                    )
                )
            else:
                # Character budget exceeded; include partial chunk if room remains
                remaining_budget = self.max_context_chars - accumulated_chars
                if remaining_budget > 50:
                    truncated_text = r.text[:remaining_budget]
                    accumulated_chars += len(truncated_text)
                    retained.append(r)
                    blocks.append(
                        EvidenceBlock(
                            rank=r.rank,
                            document_id=r.document_id,
                            source_file=r.source_file,
                            page_numbers=r.page_numbers,
                            chunk_id=r.chunk_id,
                            score=r.score,
                            section_heading=r.section_heading,
                            text=truncated_text,
                        )
                    )
                truncated = True
                break

        return BuiltContext(
            evidence_blocks=tuple(blocks),
            total_chars=accumulated_chars,
            truncated=truncated,
            retained_results=tuple(retained),
        )
