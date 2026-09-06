"""RAG answer and provenance data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class SourceReference:
    """Distinct source reference supporting an answer."""

    document_id: str
    source_file: str
    page_numbers: tuple[int, ...]

    def to_dict(self) -> dict:
        """Convert source reference to a serialisable dictionary."""
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "page_numbers": list(self.page_numbers),
        }


@dataclass(frozen=True)
class RAGAnswer:
    """Final answer container combining generated text with complete provenance."""

    query: str
    answer_text: str
    evidence: tuple[RetrievalResult, ...]
    source_references: tuple[SourceReference, ...]
    retrieval_metrics: dict[str, Any]
    provider_name: str
    model_name: str
    retrieval_latency_s: float
    context_latency_s: float
    prompt_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    context_chars: int
    num_retrieved: int
    generation_status: str

    def to_dict(self) -> dict:
        """Convert answer to a serialisable dictionary."""
        return {
            "query": self.query,
            "answer_text": self.answer_text,
            "evidence": [e.to_dict() for e in self.evidence],
            "source_references": [s.to_dict() for s in self.source_references],
            "retrieval_metrics": self.retrieval_metrics,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "retrieval_latency_s": self.retrieval_latency_s,
            "context_latency_s": self.context_latency_s,
            "prompt_latency_s": self.prompt_latency_s,
            "generation_latency_s": self.generation_latency_s,
            "total_latency_s": self.total_latency_s,
            "context_chars": self.context_chars,
            "num_retrieved": self.num_retrieved,
            "generation_status": self.generation_status,
        }
