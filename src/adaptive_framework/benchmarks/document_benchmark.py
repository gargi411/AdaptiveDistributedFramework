"""Document Processing Benchmark — Batch 18: Performance Evaluation Support.

Renamed from "Document Benchmark" to "Performance Evaluation Support"
to avoid confusion with Phase 5 evaluation benchmarks.

Measures:
    - PDF load time (PyMuPDF open + rasterise)
    - Text extraction time (direct PyMuPDF)
    - OCR time (PaddleOCR per-page)
    - Layout analysis time (Docling/heuristic)
    - Table extraction time
    - Image/figure extraction time
    - Average page processing time
    - Pages per second throughput
    - Worker throughput
    - Coordinator merge time

Generates a timestamped CSV report for Phase 5 evaluation.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DocumentBenchmarkResult:
    """Benchmark timing results for one document processing run.

    Attributes:
        document_id: Document identifier.
        file_path: Source PDF path.
        total_pages: Total pages in document.
        pages_processed: Pages successfully processed.
        total_wall_time_seconds: End-to-end wall clock time.
        pdf_load_time_seconds: Time to open and rasterise the PDF.
        text_extraction_time_seconds: Direct text extraction (digital pages).
        ocr_time_seconds: OCR time (scanned pages).
        layout_time_seconds: Layout analysis time.
        table_extraction_time_seconds: Table extraction time.
        image_extraction_time_seconds: Image/figure extraction time.
        coordinator_merge_time_seconds: UnifiedDocument build time.
        avg_page_time_seconds: Average per-page processing time.
        pages_per_second: Throughput metric.
        worker_id: Worker that processed this document.
        node_id: Node hostname.
        run_id: Pipeline run identifier.
        recorded_at: ISO 8601 UTC timestamp.
        per_page_times: Per-page wall times (page_number → seconds).
    """

    document_id: str
    file_path: str
    total_pages: int
    pages_processed: int = 0
    total_wall_time_seconds: float = 0.0
    pdf_load_time_seconds: float = 0.0
    text_extraction_time_seconds: float = 0.0
    ocr_time_seconds: float = 0.0
    layout_time_seconds: float = 0.0
    table_extraction_time_seconds: float = 0.0
    image_extraction_time_seconds: float = 0.0
    coordinator_merge_time_seconds: float = 0.0
    avg_page_time_seconds: float = 0.0
    pages_per_second: float = 0.0
    worker_id: str = "benchmark"
    node_id: str = "local"
    run_id: str = "benchmark_run"
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    per_page_times: dict[int, float] = field(default_factory=dict)

    def to_csv_row(self) -> dict[str, Any]:
        """Serialise to a flat dictionary for CSV export.

        Returns:
            Flat dictionary with all numeric and string fields.
        """
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "pages_processed": self.pages_processed,
            "total_wall_time_seconds": round(self.total_wall_time_seconds, 4),
            "pdf_load_time_seconds": round(self.pdf_load_time_seconds, 4),
            "text_extraction_time_seconds": round(self.text_extraction_time_seconds, 4),
            "ocr_time_seconds": round(self.ocr_time_seconds, 4),
            "layout_time_seconds": round(self.layout_time_seconds, 4),
            "table_extraction_time_seconds": round(self.table_extraction_time_seconds, 4),
            "image_extraction_time_seconds": round(self.image_extraction_time_seconds, 4),
            "coordinator_merge_time_seconds": round(self.coordinator_merge_time_seconds, 4),
            "avg_page_time_seconds": round(self.avg_page_time_seconds, 4),
            "pages_per_second": round(self.pages_per_second, 4),
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "run_id": self.run_id,
            "recorded_at": self.recorded_at,
        }

    def __repr__(self) -> str:
        return (
            f"DocumentBenchmarkResult(doc='{self.document_id}', "
            f"pages={self.total_pages}, "
            f"throughput={self.pages_per_second:.2f} p/s, "
            f"wall={self.total_wall_time_seconds:.3f}s)"
        )


# CSV field order for the output file
_CSV_FIELDS = [
    "document_id",
    "file_path",
    "total_pages",
    "pages_processed",
    "total_wall_time_seconds",
    "pdf_load_time_seconds",
    "text_extraction_time_seconds",
    "ocr_time_seconds",
    "layout_time_seconds",
    "table_extraction_time_seconds",
    "image_extraction_time_seconds",
    "coordinator_merge_time_seconds",
    "avg_page_time_seconds",
    "pages_per_second",
    "worker_id",
    "node_id",
    "run_id",
    "recorded_at",
]


class DocumentBenchmark:
    """Runs performance benchmarks on the document processing pipeline.

    Measures all processing stage times and generates a CSV report
    for Phase 5 evaluation (speedup, throughput, scheduler overhead).

    Usage:
        >>> bench = DocumentBenchmark(output_dir="output/benchmarks")
        >>> result = bench.benchmark_document(
        ...     file_path="/data/paper.pdf",
        ...     document_id="doc-001",
        ...     worker=worker_instance,
        ...     builder=builder_instance,
        ... )
        >>> print(f"Throughput: {result.pages_per_second:.2f} p/s")
        Throughput: 3.47 p/s

    Args:
        output_dir: Directory to write CSV reports. Created if not exists.
        run_id: Run identifier for grouping multiple benchmark runs.
    """

    def __init__(
        self,
        output_dir: str = "output/benchmarks",
        run_id: str = "benchmark_run",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._run_id = run_id
        self._results: list[DocumentBenchmarkResult] = []

    def benchmark_document(
        self,
        file_path: str,
        document_id: str,
        worker: Any,  # DocumentProcessingWorker
        builder: Any | None = None,  # UnifiedDocumentBuilder
        start_page: int = 1,
        end_page: int | None = None,
    ) -> DocumentBenchmarkResult:
        """Benchmark processing of a single document.

        Opens the document, processes all pages through the worker,
        and times each stage. Optionally runs the coordinator merge.

        Args:
            file_path: Absolute path to the PDF.
            document_id: Document identifier.
            worker: DocumentProcessingWorker instance.
            builder: UnifiedDocumentBuilder instance (optional).
            start_page: First page to benchmark (1-indexed).
            end_page: Last page to benchmark. None = all pages.

        Returns:
            DocumentBenchmarkResult with all timing data.
        """
        from adaptive_framework.models.scheduling import PageWorkUnit

        path = Path(file_path)
        if not path.exists():
            logger.error("Benchmark: file not found: %s", file_path)
            return DocumentBenchmarkResult(
                document_id=document_id,
                file_path=file_path,
                total_pages=0,
                worker_id=getattr(worker, "_worker_id", "unknown"),
            )

        # Determine page range
        try:
            import fitz  # type: ignore[import]
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
        except Exception:
            total_pages = 1

        if end_page is None:
            end_page = total_pages

        wall_start = time.perf_counter()

        # Create a PageWorkUnit and process it
        work_unit = PageWorkUnit(
            document_id=document_id,
            file_path=file_path,
            start_page=start_page,
            end_page=end_page,
            page_count=end_page - start_page + 1,
        )

        worker_result = worker.process_work_unit(work_unit)
        worker_wall = time.perf_counter() - wall_start

        # Coordinator merge timing
        merge_time = 0.0
        if builder is not None and worker_result.pages:
            t_merge = time.perf_counter()
            builder.build(
                document_id=document_id,
                file_path=file_path,
                pages=worker_result.pages,
                run_id=self._run_id,
            )
            merge_time = time.perf_counter() - t_merge

        total_wall = time.perf_counter() - wall_start
        pages_count = len(worker_result.pages)
        per_page_times = {
            p.page_number: p.processing_time_seconds
            for p in worker_result.pages
        }

        avg_page = (
            sum(per_page_times.values()) / len(per_page_times)
            if per_page_times else 0.0
        )
        pages_per_sec = pages_count / total_wall if total_wall > 0 else 0.0

        result = DocumentBenchmarkResult(
            document_id=document_id,
            file_path=file_path,
            total_pages=total_pages,
            pages_processed=worker_result.pages_succeeded,
            total_wall_time_seconds=total_wall,
            coordinator_merge_time_seconds=merge_time,
            avg_page_time_seconds=avg_page,
            pages_per_second=pages_per_sec,
            worker_id=getattr(worker, "_worker_id", "benchmark"),
            node_id=getattr(worker, "_node_id", "local"),
            run_id=self._run_id,
            per_page_times=per_page_times,
        )

        self._results.append(result)
        return result

    def save_csv(self, filename: str | None = None) -> Path:
        """Save all benchmark results to a CSV file.

        Args:
            filename: Output filename. Auto-generated from timestamp if None.

        Returns:
            Path to the written CSV file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"document_benchmark_{ts}.csv"

        out_path = self._output_dir / filename

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for result in self._results:
                writer.writerow(result.to_csv_row())

        logger.info(
            "Benchmark report saved: %s (%d documents)",
            out_path, len(self._results),
        )
        return out_path

    def summary(self) -> dict[str, Any]:
        """Return aggregate summary across all benchmarked documents.

        Returns:
            Dictionary with aggregate metrics.
        """
        if not self._results:
            return {"document_count": 0}

        total_pages = sum(r.total_pages for r in self._results)
        total_wall = sum(r.total_wall_time_seconds for r in self._results)
        avg_throughput = (
            sum(r.pages_per_second for r in self._results) / len(self._results)
        )

        return {
            "run_id": self._run_id,
            "document_count": len(self._results),
            "total_pages": total_pages,
            "total_wall_time_seconds": round(total_wall, 3),
            "avg_pages_per_second": round(avg_throughput, 4),
            "avg_page_time_seconds": round(
                sum(r.avg_page_time_seconds for r in self._results) / len(self._results),
                4,
            ),
        }
