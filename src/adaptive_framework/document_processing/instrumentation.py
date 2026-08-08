"""Runtime Instrumentation — Module 11: Per-stage timing and resource capture.

Every processing stage wraps its work in ProcessingInstrument.
This captures wall time, CPU%, RAM%, and aggregates into PageProcessingMetrics.

Architecture §4.2:
    Scheduler Overhead (%) = (Scheduler Time / Total Execution Time) × 100
    Target: < 1%
    Measurement: time.perf_counter() around each dispatch loop iteration.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from adaptive_framework.models.processing_metrics import (
    PageProcessingMetrics,
    StageMetrics,
)

logger = logging.getLogger(__name__)

# Try psutil for CPU/RAM measurement
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _cpu_percent() -> float:
    """Return current process CPU utilisation (0.0 if psutil unavailable)."""
    if not _PSUTIL_AVAILABLE:
        return 0.0
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return 0.0


def _ram_percent() -> float:
    """Return current system RAM utilisation (0.0 if psutil unavailable)."""
    if not _PSUTIL_AVAILABLE:
        return 0.0
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


@dataclass
class _StageTiming:
    """Internal mutable timing record for one processing stage."""

    stage_name: str
    start_time: float = field(default_factory=time.perf_counter)
    start_cpu: float = field(default_factory=_cpu_percent)
    start_ram: float = field(default_factory=_ram_percent)
    error: str | None = None

    def finish(self) -> StageMetrics:
        """Finalise timing and return an immutable StageMetrics record.

        Returns:
            StageMetrics with wall time and resource utilisation.
        """
        end_time = time.perf_counter()
        return StageMetrics(
            stage_name=self.stage_name,
            start_time=self.start_time,
            end_time=end_time,
            wall_time_seconds=end_time - self.start_time,
            cpu_percent=(_cpu_percent() + self.start_cpu) / 2.0,
            ram_percent=(_ram_percent() + self.start_ram) / 2.0,
            error=self.error,
        )


class ProcessingInstrument:
    """Accumulates per-stage metrics for one page processing session.

    Usage:
        >>> inst = ProcessingInstrument(
        ...     document_id="doc-001",
        ...     page_number=5,
        ...     worker_id="worker-0",
        ...     node_id="laptop-1",
        ... )
        >>> with inst.stage("pdf_load"):
        ...     page = loader.load(...)
        >>> with inst.stage("ocr"):
        ...     result = engine.ocr(page_array)
        >>> metrics = inst.build_metrics(processing_method="ocr")
        >>> metrics.total_wall_time_seconds
        0.843
    """

    def __init__(
        self,
        document_id: str,
        page_number: int,
        worker_id: str,
        node_id: str,
    ) -> None:
        self._document_id = document_id
        self._page_number = page_number
        self._worker_id = worker_id
        self._node_id = node_id
        self._stage_records: list[StageMetrics] = []
        self._session_start = time.perf_counter()
        self._extra: dict[str, float | int | str] = {}

    @contextmanager
    def stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager that times a single processing stage.

        Args:
            stage_name: Human-readable stage identifier.

        Yields:
            Nothing — measures wall time of the wrapped block.

        Example:
            >>> with instrument.stage("layout"):
            ...     result = analyser.analyse_page(page, page_number, blocks)
        """
        timing = _StageTiming(stage_name=stage_name)
        error: str | None = None
        try:
            yield
        except Exception as exc:
            error = str(exc)
            timing.error = error
            raise
        finally:
            record = timing.finish()
            self._stage_records.append(record)

    def set(self, key: str, value: float | int | str) -> None:
        """Store an extra numeric or string value for this session.

        Args:
            key: Metric key (e.g. 'ocr_confidence', 'char_count').
            value: Metric value.
        """
        self._extra[key] = value

    def build_metrics(
        self,
        processing_method: str,
        retry_count: int = 0,
    ) -> PageProcessingMetrics:
        """Build an immutable PageProcessingMetrics from accumulated stages.

        Args:
            processing_method: 'direct_text', 'ocr', 'mixed', etc.
            retry_count: Number of retries for this page.

        Returns:
            PageProcessingMetrics frozen record.
        """
        total_wall = time.perf_counter() - self._session_start

        # Extract known stage times
        stage_map: dict[str, float] = {
            s.stage_name: s.wall_time_seconds for s in self._stage_records
        }

        cpu_values = [s.cpu_percent for s in self._stage_records]
        ram_values = [s.ram_percent for s in self._stage_records]
        avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
        avg_ram = sum(ram_values) / len(ram_values) if ram_values else 0.0

        error_count = sum(1 for s in self._stage_records if not s.success)

        return PageProcessingMetrics(
            document_id=self._document_id,
            page_number=self._page_number,
            worker_id=self._worker_id,
            node_id=self._node_id,
            processing_method=processing_method,
            total_wall_time_seconds=total_wall,
            pdf_load_time_seconds=stage_map.get("pdf_load", 0.0),
            text_extraction_time_seconds=stage_map.get("text_extraction", 0.0),
            ocr_time_seconds=stage_map.get("ocr", 0.0),
            layout_time_seconds=stage_map.get("layout", 0.0),
            table_extraction_time_seconds=stage_map.get("table_extraction", 0.0),
            image_extraction_time_seconds=stage_map.get("image_extraction", 0.0),
            validation_time_seconds=stage_map.get("validation", 0.0),
            page_build_time_seconds=stage_map.get("page_build", 0.0),
            cpu_percent_avg=round(avg_cpu, 2),
            ram_percent_avg=round(avg_ram, 2),
            gpu_percent=float(self._extra.get("gpu_percent", 0.0)),
            ocr_confidence=float(self._extra.get("ocr_confidence", 1.0)),
            char_count=int(self._extra.get("char_count", 0)),
            table_count=int(self._extra.get("table_count", 0)),
            figure_count=int(self._extra.get("figure_count", 0)),
            retry_count=retry_count,
            error_count=error_count,
            stage_metrics=tuple(self._stage_records),
        )
