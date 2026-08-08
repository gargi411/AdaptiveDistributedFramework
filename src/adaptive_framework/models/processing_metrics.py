"""Per-page processing metrics model — Module 11: Runtime Instrumentation.

Every processing stage emits a PageProcessingMetrics object.
These are consumed by the evaluation framework and benchmark logger.

Architecture §4.2: Scheduler overhead = scheduler_time / total_wall_time.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class StageMetrics:
    """Timing and resource usage for a single processing stage.

    Attributes:
        stage_name: Human-readable stage identifier (e.g. 'ocr', 'layout').
        start_time: Monotonic start time (time.perf_counter() value).
        end_time: Monotonic end time.
        wall_time_seconds: end_time - start_time.
        cpu_percent: CPU utilisation during this stage (0–100+).
        ram_percent: RAM utilisation during this stage (0–100).
        error: Error message if this stage failed. None on success.
    """

    stage_name: str
    start_time: float
    end_time: float
    wall_time_seconds: float
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return True if this stage completed without error."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "stage_name": self.stage_name,
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "success": self.success,
            "error": self.error,
        }


@dataclass(frozen=True)
class PageProcessingMetrics:
    """Complete instrumentation record for one page processed by one worker.

    Produced by ProcessingInstrument after the page is fully processed.
    Consumed by BenchmarkLogger (event bus subscriber) and the evaluation
    framework in Phase 5.

    Attributes:
        document_id: Parent document identifier.
        page_number: 1-indexed page number.
        worker_id: ID of the worker that processed this page.
        node_id: Hostname / node identifier.
        processing_method: 'direct_text', 'ocr', 'mixed', 'skipped', 'failed'.
        total_wall_time_seconds: End-to-end time for this page.
        pdf_load_time_seconds: Time spent loading the page via PyMuPDF.
        text_extraction_time_seconds: Time for direct text extraction (0 if OCR).
        ocr_time_seconds: Time spent in PaddleOCR (0 if digital).
        layout_time_seconds: Time spent in Docling layout analysis.
        table_extraction_time_seconds: Time for table extraction.
        image_extraction_time_seconds: Time for figure/image extraction.
        validation_time_seconds: Time spent in PageValidator.
        page_build_time_seconds: Time to construct the Page object.
        cpu_percent_avg: Average CPU utilisation across all stages.
        ram_percent_avg: Average RAM utilisation across all stages.
        gpu_percent: GPU utilisation during OCR (0 if no GPU).
        ocr_confidence: Average OCR confidence (1.0 = direct extraction).
        char_count: Characters extracted from this page.
        table_count: Tables extracted from this page.
        figure_count: Figures extracted from this page.
        retry_count: Number of retries for this page work unit.
        error_count: Number of non-fatal errors encountered.
        stage_metrics: Ordered tuple of per-stage metrics.
        recorded_at: ISO 8601 UTC timestamp.
        metrics_id: Unique ID for this metrics record.
    """

    document_id: str
    page_number: int
    worker_id: str
    node_id: str
    processing_method: str
    total_wall_time_seconds: float
    pdf_load_time_seconds: float = 0.0
    text_extraction_time_seconds: float = 0.0
    ocr_time_seconds: float = 0.0
    layout_time_seconds: float = 0.0
    table_extraction_time_seconds: float = 0.0
    image_extraction_time_seconds: float = 0.0
    validation_time_seconds: float = 0.0
    page_build_time_seconds: float = 0.0
    cpu_percent_avg: float = 0.0
    ram_percent_avg: float = 0.0
    gpu_percent: float = 0.0
    ocr_confidence: float = 1.0
    char_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    retry_count: int = 0
    error_count: int = 0
    stage_metrics: tuple[StageMetrics, ...] = field(default_factory=tuple)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metrics_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dictionary suitable for CSV export.

        Returns:
            Dictionary with all numeric fields at the top level.
        """
        return {
            "metrics_id": self.metrics_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "processing_method": self.processing_method,
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "pdf_load_time_seconds": self.pdf_load_time_seconds,
            "text_extraction_time_seconds": self.text_extraction_time_seconds,
            "ocr_time_seconds": self.ocr_time_seconds,
            "layout_time_seconds": self.layout_time_seconds,
            "table_extraction_time_seconds": self.table_extraction_time_seconds,
            "image_extraction_time_seconds": self.image_extraction_time_seconds,
            "validation_time_seconds": self.validation_time_seconds,
            "page_build_time_seconds": self.page_build_time_seconds,
            "cpu_percent_avg": self.cpu_percent_avg,
            "ram_percent_avg": self.ram_percent_avg,
            "gpu_percent": self.gpu_percent,
            "ocr_confidence": self.ocr_confidence,
            "char_count": self.char_count,
            "table_count": self.table_count,
            "figure_count": self.figure_count,
            "retry_count": self.retry_count,
            "error_count": self.error_count,
            "recorded_at": self.recorded_at,
        }

    @property
    def scheduler_overhead_seconds(self) -> float:
        """Time NOT spent in document processing (scheduling overhead).

        Returns:
            Difference between total wall time and sum of processing stage times.
            Negative values are clamped to 0 (measurement noise).
        """
        processing_sum = (
            self.pdf_load_time_seconds
            + self.text_extraction_time_seconds
            + self.ocr_time_seconds
            + self.layout_time_seconds
            + self.table_extraction_time_seconds
            + self.image_extraction_time_seconds
            + self.validation_time_seconds
            + self.page_build_time_seconds
        )
        return max(0.0, self.total_wall_time_seconds - processing_sum)

    def __repr__(self) -> str:
        return (
            f"PageProcessingMetrics(doc='{self.document_id}', "
            f"page={self.page_number}, "
            f"method={self.processing_method}, "
            f"wall={self.total_wall_time_seconds:.3f}s)"
        )
