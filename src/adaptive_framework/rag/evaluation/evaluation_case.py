"""Data model representing a RAG retrieval evaluation ground-truth test case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RAGEvaluationCase:
    """A ground-truth query case for evaluating retrieval quality.

    Attributes:
        case_id: Unique identifier for the evaluation case.
        question: Clinical question / search inquiry.
        relevant_document_ids: Tuple of document_ids that contain relevant evidence.
        relevant_chunk_ids: Tuple of chunk_ids that contain relevant evidence.
        category: Clinical facet (diagnosis, symptoms, investigations, treatment, multi_evidence).
        description: Brief clinical context or annotation.
    """

    case_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...] = ()
    category: str = "general"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation case to serializable dictionary."""
        return {
            "case_id": self.case_id,
            "question": self.question,
            "relevant_document_ids": list(self.relevant_document_ids),
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "category": self.category,
            "description": self.description,
        }
