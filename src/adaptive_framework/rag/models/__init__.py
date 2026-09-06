"""Phase 4.3 RAG data models — Embedding.

Exports the Embedding frozen dataclass produced by BGEEmbedder and
consumed by FAISSManager.
"""

from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.models.clinical_models import (
    CLINICAL_DISCLAIMER,
    ClinicalDocumentUpload,
    ClinicalQueryRequest,
    DoctorClinicalResponse,
    DoctorEvidenceItem,
    DocumentIngestionResult,
)

__all__ = [
    "Embedding",
    "CLINICAL_DISCLAIMER",
    "ClinicalDocumentUpload",
    "ClinicalQueryRequest",
    "DoctorClinicalResponse",
    "DoctorEvidenceItem",
    "DocumentIngestionResult",
]
