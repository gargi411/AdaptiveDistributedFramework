"""Clinical RAG data models for doctor-facing integration (Phase 4.11).

Provides structured request, upload, ingestion, evidence, and response representations
for clinical document intelligence and decision support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adaptive_framework.rag.generation.answer import SourceReference
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult
from adaptive_framework.rag.reranking.rerank_result import RerankedRetrievalResult

CLINICAL_DISCLAIMER = (
    "Clinical Document Intelligence Prototype - Decision-support only. "
    "Not an autonomous diagnostic or prescription system. "
    "All findings must be verified against original medical records by a qualified physician."
)


@dataclass(frozen=True)
class ClinicalQueryRequest:
    """Doctor query request with optional patient presentation / symptoms."""

    query: str
    document_id: str | None = None
    clinical_context: str | None = None
    top_k: int = 5
    candidate_top_k: int = 15
    metadata_filters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("Clinical query must not be empty.")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.candidate_top_k < 1:
            raise ValueError(f"candidate_top_k must be >= 1, got {self.candidate_top_k}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "document_id": self.document_id,
            "clinical_context": self.clinical_context,
            "top_k": self.top_k,
            "candidate_top_k": self.candidate_top_k,
            "metadata_filters": self.metadata_filters,
        }


@dataclass(frozen=True)
class ClinicalDocumentUpload:
    """Container for clinical document uploads (PDF, EHR text, or structured record)."""

    filename: str
    file_path: str | None = None
    file_bytes: bytes | None = None
    document_id: str | None = None
    document_type: str = "clinical_discharge_summary"

    def __post_init__(self) -> None:
        if not self.filename or not self.filename.strip():
            raise ValueError("Filename must not be empty.")
        if not self.file_path and not self.file_bytes:
            raise ValueError("Either file_path or file_bytes must be provided.")


@dataclass(frozen=True)
class DocumentIngestionResult:
    """Metadata summary of document ingestion and indexing."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    ingestion_latency_ms: float
    status: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "ingestion_latency_ms": self.ingestion_latency_ms,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class DoctorEvidenceItem:
    """Individual piece of clinical evidence supporting an answer."""

    chunk_id: str
    document_id: str
    source_file: str
    page_numbers: tuple[int, ...]
    section_heading: str
    text: str
    score: float
    rank: int
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    reranker_rank: int | None = None

    @classmethod
    def from_retrieval_result(cls, res: RetrievalResult) -> DoctorEvidenceItem:
        """Construct a DoctorEvidenceItem from RetrievalResult or RerankedRetrievalResult."""
        dense_rank = getattr(res, "dense_rank", None)
        sparse_rank = getattr(res, "sparse_rank", None)
        rrf_score = getattr(res, "rrf_score", None)
        reranker_score = getattr(res, "reranker_score", None)
        reranker_rank = getattr(res, "reranker_rank", None)

        return cls(
            chunk_id=res.chunk_id,
            document_id=res.document_id,
            source_file=res.source_file,
            page_numbers=res.page_numbers,
            section_heading=res.section_heading,
            text=res.text,
            score=res.score,
            rank=res.rank,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            rrf_score=rrf_score,
            reranker_score=reranker_score,
            reranker_rank=reranker_rank,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "page_numbers": list(self.page_numbers),
            "section_heading": self.section_heading,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rrf_score": self.rrf_score,
            "reranker_score": self.reranker_score,
            "reranker_rank": self.reranker_rank,
        }

    @property
    def snippet(self) -> str:
        """Convenience property returning the underlying evidence text snippet."""
        return self.text


@dataclass(frozen=True)
class DoctorClinicalResponse:
    """Structured response container presented to the clinician."""

    query: str
    clinical_context: str | None
    answer_text: str
    evidence: tuple[DoctorEvidenceItem, ...]
    source_references: tuple[SourceReference, ...]
    stage_latencies_ms: dict[str, float]
    generation_status: str
    provider_name: str
    model_name: str
    disclaimer: str = CLINICAL_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "clinical_context": self.clinical_context,
            "answer_text": self.answer_text,
            "evidence": [e.to_dict() for e in self.evidence],
            "source_references": [s.to_dict() for s in self.source_references],
            "stage_latencies_ms": self.stage_latencies_ms,
            "generation_status": self.generation_status,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "disclaimer": self.disclaimer,
        }
