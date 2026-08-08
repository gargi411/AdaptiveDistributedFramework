"""Processing Cost Estimator Interface — Improvement 3 (Architecture Placeholder).

Defines the interface for future scheduler cost prediction.

Version 2 of the Adaptive Scheduler will use predicted processing cost
to make smarter dispatch decisions BEFORE processing begins:
    - Heavy scanned pages (high image density) → assign to GPU workers
    - Light digital pages → assign to CPU workers
    - Cost prediction feeds the page-count priority queue weight

This interface is defined now so:
    a) Future implementation requires no architectural change
    b) Research paper can describe it as a planned contribution
    c) ProcessingStrategyFactory can already accept cost hints

Status: Interface only. Concrete implementation is Future Work.
        Documented in architecture_v2.0_locked.md §6 Future Work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from adaptive_framework.models.document import PDFMetadata


@dataclass(frozen=True)
class ProcessingCostEstimate:
    """Predicted processing cost for one document.

    Produced by IProcessingCostEstimator before scheduling begins.
    Used by the scheduler to weight task priority.

    Attributes:
        document_id: Document being estimated.
        predicted_seconds_per_page: Estimated processing time per page.
        predicted_total_seconds: Total predicted processing time.
        image_density_factor: Relative weight of image content (1.0 = normal).
        resolution_factor: Relative weight of resolution (1.0 = normal).
        ocr_required_pages: Estimated number of pages requiring OCR.
        confidence: Estimator confidence in this prediction [0.0, 1.0].
        estimator_name: Name of the estimator that produced this record.
    """

    document_id: str
    predicted_seconds_per_page: float
    predicted_total_seconds: float
    image_density_factor: float = 1.0
    resolution_factor: float = 1.0
    ocr_required_pages: int = 0
    confidence: float = 0.5
    estimator_name: str = "unknown"

    @property
    def is_high_cost(self) -> bool:
        """Return True if this document is predicted to be expensive to process."""
        return (
            self.predicted_seconds_per_page > 2.0
            or self.image_density_factor > 1.5
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "document_id": self.document_id,
            "predicted_seconds_per_page": self.predicted_seconds_per_page,
            "predicted_total_seconds": self.predicted_total_seconds,
            "image_density_factor": self.image_density_factor,
            "resolution_factor": self.resolution_factor,
            "ocr_required_pages": self.ocr_required_pages,
            "confidence": self.confidence,
            "estimator_name": self.estimator_name,
            "is_high_cost": self.is_high_cost,
        }

    def __repr__(self) -> str:
        return (
            f"ProcessingCostEstimate(doc='{self.document_id}', "
            f"~{self.predicted_total_seconds:.1f}s, "
            f"confidence={self.confidence:.2f})"
        )


class IProcessingCostEstimator(ABC):
    """Abstract interface for processing cost prediction.

    Implementations predict the computational cost of processing a document
    based on its metadata (page count, size, image density, resolution).

    The Adaptive Scheduler (Version 2) will use this interface to:
        - Pre-sort the priority queue by predicted cost
        - Assign high-cost (scanned, high-resolution) documents to GPU workers
        - Balance load more accurately than page-count alone

    Future implementations to consider:
        - HeuristicCostEstimator: Simple formula based on page count × image density
        - MLCostEstimator: Trained regressor on historical processing times
        - ProfiledCostEstimator: Uses hardware profiling data per node

    Example:
        >>> class SimpleCostEstimator(IProcessingCostEstimator):
        ...     def estimate(self, metadata):
        ...         base = metadata.pages * 0.5
        ...         density_factor = 3.0 if metadata.source_type == "scanned" else 1.0
        ...         return ProcessingCostEstimate(
        ...             document_id=metadata.document_id,
        ...             predicted_seconds_per_page=0.5 * density_factor,
        ...             predicted_total_seconds=base * density_factor,
        ...             confidence=0.6,
        ...         )
    """

    @abstractmethod
    def estimate(self, metadata: PDFMetadata) -> ProcessingCostEstimate:
        """Predict the processing cost for a document.

        Args:
            metadata: PDFMetadata record from the metadata generator.
                      Contains page count, size, source type, resolution.

        Returns:
            ProcessingCostEstimate with predicted times and cost factors.
        """

    @abstractmethod
    def batch_estimate(
        self, metadata_list: list[PDFMetadata]
    ) -> list[ProcessingCostEstimate]:
        """Predict costs for a batch of documents.

        Args:
            metadata_list: List of PDFMetadata records.

        Returns:
            List of ProcessingCostEstimate in the same order.
        """


class NullCostEstimator(IProcessingCostEstimator):
    """No-op cost estimator — returns neutral estimates for all documents.

    Used as the default until a concrete estimator is implemented.
    Allows the framework to call estimate() without null-checks.

    Returns 1.0 second per page as a safe default.
    """

    SECONDS_PER_PAGE_DEFAULT: float = 1.0

    def estimate(self, metadata: PDFMetadata) -> ProcessingCostEstimate:
        """Return a neutral cost estimate.

        Args:
            metadata: PDFMetadata for the document.

        Returns:
            ProcessingCostEstimate with default values.
        """
        return ProcessingCostEstimate(
            document_id=metadata.document_id,
            predicted_seconds_per_page=self.SECONDS_PER_PAGE_DEFAULT,
            predicted_total_seconds=metadata.pages * self.SECONDS_PER_PAGE_DEFAULT,
            confidence=0.0,  # No real prediction made
            estimator_name="null",
        )

    def batch_estimate(
        self, metadata_list: list[PDFMetadata]
    ) -> list[ProcessingCostEstimate]:
        """Return neutral estimates for all documents.

        Args:
            metadata_list: List of PDFMetadata records.

        Returns:
            List of neutral ProcessingCostEstimate records.
        """
        return [self.estimate(m) for m in metadata_list]
