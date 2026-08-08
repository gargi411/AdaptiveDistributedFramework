"""Document data models for the Adaptive Distributed Framework.

Models:
    PDFMetadata: Full metadata record for a PDF document (architecture v2.0 §2.4).
    PageMetadata: Metadata for a single page within a document.
    DocumentResult: Aggregated processing result for a complete document.
    PageResult: Processing result for a single page.

All models are pure dataclasses: no algorithms, no I/O.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.core.constants import (
    MIN_RESOLUTION_DPI,
    SOURCE_TYPE_DIGITAL,
    SOURCE_TYPE_SCANNED,
    VALID_SOURCE_TYPES,
)
from adaptive_framework.core.exceptions import ValidationError


# =============================================================
# PDFMetadata
# =============================================================


@dataclass
class PDFMetadata:
    """Structured metadata record for a PDF document.

    Corresponds exactly to the metadata schema defined in
    architecture v2.0 §2.4 (Metadata Generator).

    Attributes:
        document_id: Unique identifier (UUID4 string). Auto-generated if not provided.
        pages: Total page count. Required for the scheduler (page-count strategy).
        estimated_size_mb: Estimated file size in megabytes. Required scheduling weight.
        file_path: Absolute path to the source PDF file.
        resolution_dpi: Scan resolution in DPI. None if not detectable.
        source_type: 'scanned' or 'digital'. None if not detected.
        language: ISO 639-1 language code (e.g. 'en'). None if not detected.
        processing_timestamp: ISO 8601 timestamp set at extraction time.

    Example:
        >>> meta = PDFMetadata(pages=42, estimated_size_mb=3.7,
        ...                    file_path="/data/raw/paper.pdf")
        >>> meta.pages
        42
        >>> meta.source_type  # None until detected
    """

    pages: int
    estimated_size_mb: float
    file_path: str
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resolution_dpi: int | None = None
    source_type: str | None = None
    language: str | None = None
    processing_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate all PDFMetadata fields.

        Raises:
            ValidationError: If any field has an invalid value.
        """
        if self.pages < 1:
            raise ValidationError(
                "PDFMetadata.pages must be >= 1.",
                field="pages",
                value=self.pages,
            )
        if self.estimated_size_mb < 0.0:
            raise ValidationError(
                "PDFMetadata.estimated_size_mb must be >= 0.",
                field="estimated_size_mb",
                value=self.estimated_size_mb,
            )
        if not self.file_path:
            raise ValidationError(
                "PDFMetadata.file_path must not be empty.",
                field="file_path",
            )
        if self.resolution_dpi is not None and self.resolution_dpi < MIN_RESOLUTION_DPI:
            raise ValidationError(
                f"PDFMetadata.resolution_dpi must be >= {MIN_RESOLUTION_DPI}.",
                field="resolution_dpi",
                value=self.resolution_dpi,
            )
        if self.source_type is not None and self.source_type not in VALID_SOURCE_TYPES:
            raise ValidationError(
                f"PDFMetadata.source_type must be one of {sorted(VALID_SOURCE_TYPES)}.",
                field="source_type",
                value=self.source_type,
            )

    def is_scanned(self) -> bool:
        """Return True if this document is identified as a scanned image PDF.

        Returns:
            True when source_type is 'scanned', False otherwise.
        """
        return self.source_type == SOURCE_TYPE_SCANNED

    def is_digital(self) -> bool:
        """Return True if this document is a native digital PDF.

        Returns:
            True when source_type is 'digital', False otherwise.
        """
        return self.source_type == SOURCE_TYPE_DIGITAL

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (JSON-compatible).

        Returns:
            Dictionary representation matching the metadata schema in §2.4.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"PDFMetadata(document_id='{self.document_id}', "
            f"pages={self.pages}, "
            f"estimated_size_mb={self.estimated_size_mb:.2f}, "
            f"source_type={self.source_type!r}, "
            f"language={self.language!r})"
        )


# =============================================================
# PageMetadata
# =============================================================


@dataclass
class PageMetadata:
    """Metadata for a single page within a PDF document.

    Attributes:
        document_id: Parent document identifier.
        page_number: 1-indexed page number within the document.
        width_pts: Page width in PDF points (1 pt = 1/72 inch).
        height_pts: Page height in PDF points.
        has_text_layer: True if a native text layer exists (digital PDF).
        has_images: True if the page contains embedded images.

    Example:
        >>> pm = PageMetadata(document_id="abc-123", page_number=1,
        ...                   width_pts=595.0, height_pts=842.0,
        ...                   has_text_layer=False, has_images=True)
    """

    document_id: str
    page_number: int
    width_pts: float
    height_pts: float
    has_text_layer: bool
    has_images: bool

    def __post_init__(self) -> None:
        """Validate PageMetadata fields.

        Raises:
            ValidationError: If page_number < 1 or dimensions are non-positive.
        """
        if self.page_number < 1:
            raise ValidationError(
                "PageMetadata.page_number must be >= 1 (1-indexed).",
                field="page_number",
                value=self.page_number,
            )
        if self.width_pts <= 0:
            raise ValidationError(
                "PageMetadata.width_pts must be > 0.",
                field="width_pts",
                value=self.width_pts,
            )
        if self.height_pts <= 0:
            raise ValidationError(
                "PageMetadata.height_pts must be > 0.",
                field="height_pts",
                value=self.height_pts,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this page metadata.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"PageMetadata(document_id='{self.document_id}', "
            f"page_number={self.page_number}, "
            f"has_text_layer={self.has_text_layer})"
        )


# =============================================================
# PageResult
# =============================================================


@dataclass
class PageResult:
    """Processing result for a single page.

    Produced by the Document Processing Engine after a page has been
    processed through OCR, Layout Analysis, Table Extraction, and
    Figure Detection.

    Attributes:
        document_id: Parent document identifier.
        page_number: 1-indexed page number.
        text: Extracted text content from this page.
        worker_id: ID of the worker that processed this page.
        processing_time_seconds: Wall-clock time to process this page.
        success: True if processing completed without errors.
        error_message: Description of the error if success is False.

    Example:
        >>> result = PageResult(document_id="abc-123", page_number=1,
        ...                     text="Introduction...", worker_id="worker_01",
        ...                     processing_time_seconds=0.85, success=True)
    """

    document_id: str
    page_number: int
    worker_id: str
    processing_time_seconds: float
    success: bool
    text: str = ""
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValidationError(
                "PageResult.page_number must be >= 1.",
                field="page_number",
                value=self.page_number,
            )
        if self.processing_time_seconds < 0:
            raise ValidationError(
                "PageResult.processing_time_seconds must be >= 0.",
                field="processing_time_seconds",
                value=self.processing_time_seconds,
            )
        if not self.success and not self.error_message:
            raise ValidationError(
                "PageResult.error_message must be set when success=False.",
                field="error_message",
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this page result.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"PageResult(document_id='{self.document_id}', "
            f"page_number={self.page_number}, "
            f"success={self.success}, "
            f"time={self.processing_time_seconds:.3f}s)"
        )


# =============================================================
# DocumentResult
# =============================================================


@dataclass
class DocumentResult:
    """Aggregated processing result for a complete PDF document.

    Combines results from all pages into a single result record that
    is returned to the result collector after the pipeline completes.

    Attributes:
        document_id: Unique document identifier.
        file_path: Source PDF file path.
        total_pages: Total number of pages in the document.
        processed_pages: Number of pages successfully processed.
        failed_pages: Number of pages that failed.
        page_results: Ordered list of per-page results.
        total_processing_time_seconds: End-to-end wall-clock time.
        success: True if all pages were processed successfully.

    Example:
        >>> doc_result = DocumentResult(
        ...     document_id="abc-123",
        ...     file_path="/data/raw/paper.pdf",
        ...     total_pages=5,
        ...     processed_pages=5,
        ...     failed_pages=0,
        ...     page_results=[],
        ...     total_processing_time_seconds=4.2,
        ...     success=True,
        ... )
    """

    document_id: str
    file_path: str
    total_pages: int
    processed_pages: int
    failed_pages: int
    page_results: list[PageResult]
    total_processing_time_seconds: float
    success: bool

    def __post_init__(self) -> None:
        if self.total_pages < 1:
            raise ValidationError(
                "DocumentResult.total_pages must be >= 1.",
                field="total_pages",
                value=self.total_pages,
            )
        if self.processed_pages < 0:
            raise ValidationError(
                "DocumentResult.processed_pages must be >= 0.",
                field="processed_pages",
                value=self.processed_pages,
            )
        if self.failed_pages < 0:
            raise ValidationError(
                "DocumentResult.failed_pages must be >= 0.",
                field="failed_pages",
                value=self.failed_pages,
            )
        if self.processed_pages + self.failed_pages > self.total_pages:
            raise ValidationError(
                "processed_pages + failed_pages must be <= total_pages.",
            )

    @property
    def success_rate(self) -> float:
        """Compute the fraction of successfully processed pages.

        Returns:
            Success rate in [0.0, 1.0]. Returns 0.0 if total_pages is 0.
        """
        if self.total_pages == 0:
            return 0.0
        return self.processed_pages / self.total_pages

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation including all page results.
        """
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "failed_pages": self.failed_pages,
            "success_rate": self.success_rate,
            "total_processing_time_seconds": self.total_processing_time_seconds,
            "success": self.success,
            "page_results": [pr.to_dict() for pr in self.page_results],
        }

    def __repr__(self) -> str:
        return (
            f"DocumentResult(document_id='{self.document_id}', "
            f"pages={self.total_pages}, "
            f"success={self.success}, "
            f"success_rate={self.success_rate:.1%})"
        )
