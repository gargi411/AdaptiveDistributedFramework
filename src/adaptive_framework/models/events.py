"""Processing event models for the internal event bus — Module 12.

Events are published by workers during document processing and consumed
by subscribers: the engineering dashboard and the BenchmarkLogger.

Design: immutable frozen dataclasses — safe to pass across threads.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """All event types published on the internal event bus.

    Values:
        PAGE_STARTED: Worker began processing a page.
        PAGE_FINISHED: Worker completed processing a page.
        OCR_STARTED: PaddleOCR engine started on a page.
        OCR_FINISHED: PaddleOCR engine completed on a page.
        LAYOUT_COMPLETED: Docling layout analysis finished for a page.
        TABLE_EXTRACTED: A table was successfully extracted.
        FIGURE_EXTRACTED: A figure or image was successfully extracted.
        WORKER_COMPLETED: Worker finished its entire PageWorkUnit batch.
        COORDINATOR_MERGE: Coordinator began merging Page objects.
        MERGE_COMPLETED: Coordinator finished building UnifiedDocument.
        VALIDATION_WARNING: PageValidator raised a non-fatal warning.
        VALIDATION_FAILED: PageValidator found a critical issue.
        HEALTH_CHECK: System health report was generated.
    """

    PAGE_STARTED = "page_started"
    PAGE_FINISHED = "page_finished"
    OCR_STARTED = "ocr_started"
    OCR_FINISHED = "ocr_finished"
    LAYOUT_COMPLETED = "layout_completed"
    TABLE_EXTRACTED = "table_extracted"
    FIGURE_EXTRACTED = "figure_extracted"
    WORKER_COMPLETED = "worker_completed"
    COORDINATOR_MERGE = "coordinator_merge"
    MERGE_COMPLETED = "merge_completed"
    VALIDATION_WARNING = "validation_warning"
    VALIDATION_FAILED = "validation_failed"
    HEALTH_CHECK = "health_check"


@dataclass(frozen=True)
class ProcessingEvent:
    """An immutable event published on the internal event bus.

    Subscribers (dashboard, BenchmarkLogger) receive this object.
    All fields are frozen — no mutation after construction.

    Attributes:
        event_type: Type of this event (see EventType).
        document_id: Document being processed. None for system-level events.
        page_number: Page being processed (1-indexed). None for doc-level events.
        worker_id: Worker that emitted this event. None for coordinator events.
        node_id: Hostname of the emitting node. None if not applicable.
        timestamp: ISO 8601 UTC timestamp of event creation.
        event_id: Unique event identifier.
        wall_time_seconds: Duration associated with this event (e.g. OCR time).
        payload: Optional extra key-value data for this event type.
        message: Human-readable description.

    Example:
        >>> event = ProcessingEvent(
        ...     event_type=EventType.PAGE_STARTED,
        ...     document_id="doc-001",
        ...     page_number=3,
        ...     worker_id="worker-0",
        ...     node_id="laptop-1",
        ...     message="Starting page 3",
        ... )
        >>> event.event_type
        <EventType.PAGE_STARTED: 'page_started'>
    """

    event_type: EventType
    document_id: str | None = None
    page_number: int | None = None
    worker_id: str | None = None
    node_id: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    wall_time_seconds: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary for CSV or JSON logging.

        Returns:
            Dictionary with all fields at the top level.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "wall_time_seconds": self.wall_time_seconds,
            "message": self.message,
            **self.payload,
        }

    def __repr__(self) -> str:
        parts = [f"event_type={self.event_type.value!r}"]
        if self.document_id:
            parts.append(f"document_id='{self.document_id}'")
        if self.page_number is not None:
            parts.append(f"page_number={self.page_number}")
        if self.worker_id:
            parts.append(f"worker_id='{self.worker_id}'")
        return f"ProcessingEvent({', '.join(parts)})"


# ── Convenience factory functions ────────────────────────────────────────────


def page_started_event(
    document_id: str,
    page_number: int,
    worker_id: str,
    node_id: str,
    processing_method: str,
) -> ProcessingEvent:
    """Create a PAGE_STARTED event.

    Args:
        document_id: Document being processed.
        page_number: Page number (1-indexed).
        worker_id: Worker processing this page.
        node_id: Node hostname.
        processing_method: 'direct_text', 'ocr', or 'mixed'.

    Returns:
        Immutable ProcessingEvent of type PAGE_STARTED.
    """
    return ProcessingEvent(
        event_type=EventType.PAGE_STARTED,
        document_id=document_id,
        page_number=page_number,
        worker_id=worker_id,
        node_id=node_id,
        payload={"processing_method": processing_method},
        message=f"Started page {page_number} [{processing_method}]",
    )


def page_finished_event(
    document_id: str,
    page_number: int,
    worker_id: str,
    node_id: str,
    wall_time_seconds: float,
    success: bool,
    char_count: int = 0,
    table_count: int = 0,
    figure_count: int = 0,
) -> ProcessingEvent:
    """Create a PAGE_FINISHED event.

    Args:
        document_id: Document being processed.
        page_number: Page number (1-indexed).
        worker_id: Worker processing this page.
        node_id: Node hostname.
        wall_time_seconds: Total time spent on this page.
        success: Whether processing succeeded.
        char_count: Characters extracted.
        table_count: Tables extracted.
        figure_count: Figures extracted.

    Returns:
        Immutable ProcessingEvent of type PAGE_FINISHED.
    """
    return ProcessingEvent(
        event_type=EventType.PAGE_FINISHED,
        document_id=document_id,
        page_number=page_number,
        worker_id=worker_id,
        node_id=node_id,
        wall_time_seconds=wall_time_seconds,
        payload={
            "success": success,
            "char_count": char_count,
            "table_count": table_count,
            "figure_count": figure_count,
        },
        message=f"Finished page {page_number} in {wall_time_seconds:.3f}s",
    )


def worker_completed_event(
    worker_id: str,
    node_id: str,
    document_id: str,
    pages_processed: int,
    total_wall_time_seconds: float,
) -> ProcessingEvent:
    """Create a WORKER_COMPLETED event.

    Args:
        worker_id: Worker that finished.
        node_id: Node hostname.
        document_id: Document that was processed.
        pages_processed: Number of pages completed.
        total_wall_time_seconds: Total time for all pages.

    Returns:
        Immutable ProcessingEvent of type WORKER_COMPLETED.
    """
    return ProcessingEvent(
        event_type=EventType.WORKER_COMPLETED,
        document_id=document_id,
        worker_id=worker_id,
        node_id=node_id,
        wall_time_seconds=total_wall_time_seconds,
        payload={"pages_processed": pages_processed},
        message=(
            f"Worker {worker_id} completed {pages_processed} pages "
            f"in {total_wall_time_seconds:.3f}s"
        ),
    )


def coordinator_merge_event(
    document_id: str,
    page_count: int,
    merge_time_seconds: float,
) -> ProcessingEvent:
    """Create a MERGE_COMPLETED event.

    Args:
        document_id: Document that was merged.
        page_count: Number of pages merged.
        merge_time_seconds: Time spent in coordinator merge.

    Returns:
        Immutable ProcessingEvent of type MERGE_COMPLETED.
    """
    return ProcessingEvent(
        event_type=EventType.MERGE_COMPLETED,
        document_id=document_id,
        wall_time_seconds=merge_time_seconds,
        payload={"page_count": page_count},
        message=(
            f"Coordinator merged {page_count} pages "
            f"in {merge_time_seconds:.3f}s → UnifiedDocument"
        ),
    )
