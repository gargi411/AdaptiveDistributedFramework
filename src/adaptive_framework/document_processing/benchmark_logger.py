"""Benchmark Logger — Improvement 4: Event Bus → CSV performance data.

Subscribes to the EventBus and converts every processing event into a
performance data record. Written to CSV automatically.

Architecture:
    Event Bus ──► BenchmarkLogger ──► CSV ──► Phase 5 Evaluation

When Phase 5 (evaluation framework) arrives, performance data is already
collected — no manual instrumentation needed.

This is the automated bridge between processing events and the evaluation
pipeline specified in architecture §4.1 and §4.2.
"""

from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_framework.document_processing.event_bus import EventBus
from adaptive_framework.models.events import EventType, ProcessingEvent

logger = logging.getLogger(__name__)

# Fields written to CSV for each event
_CSV_FIELDS = [
    "event_id",
    "event_type",
    "timestamp",
    "document_id",
    "page_number",
    "worker_id",
    "node_id",
    "wall_time_seconds",
    "message",
]


class BenchmarkLogger:
    """Subscribes to EventBus and writes performance data to CSV.

    Every page_started, page_finished, ocr_started, ocr_finished,
    layout_completed, table_extracted, figure_extracted, worker_completed,
    and coordinator_merge event is automatically logged.

    The CSV file can be loaded directly into pandas or Excel for Phase 5
    evaluation (speedup, throughput, scheduler overhead analysis).

    Args:
        output_path: Path to the CSV output file.
        event_bus: EventBus to subscribe to. Uses default bus if None.
        auto_flush: If True, flush the CSV file after every write.

    Usage:
        >>> bus = EventBus()
        >>> logger = BenchmarkLogger("output/benchmark.csv", event_bus=bus)
        >>> logger.start()
        >>> # ... processing happens, events are published ...
        >>> logger.stop()
        >>> logger.summary()
    """

    # Event types to capture for benchmarking
    _CAPTURE_TYPES = {
        EventType.PAGE_STARTED,
        EventType.PAGE_FINISHED,
        EventType.OCR_STARTED,
        EventType.OCR_FINISHED,
        EventType.LAYOUT_COMPLETED,
        EventType.TABLE_EXTRACTED,
        EventType.FIGURE_EXTRACTED,
        EventType.WORKER_COMPLETED,
        EventType.COORDINATOR_MERGE,
        EventType.MERGE_COMPLETED,
    }

    def __init__(
        self,
        output_path: str,
        event_bus: EventBus | None = None,
        auto_flush: bool = True,
    ) -> None:
        self._output_path = Path(output_path)
        self._bus = event_bus
        self._auto_flush = auto_flush
        self._lock = threading.Lock()
        self._file: Any = None
        self._writer: Any = None
        self._event_count: int = 0
        self._started = False

    def start(self) -> None:
        """Open the CSV file and subscribe to the event bus.

        Creates parent directories if they do not exist.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file, fieldnames=_CSV_FIELDS, extrasaction="ignore"
        )
        self._writer.writeheader()
        if self._auto_flush:
            self._file.flush()

        # Subscribe to all benchmark-relevant event types
        if self._bus is not None:
            for event_type in self._CAPTURE_TYPES:
                self._bus.subscribe(event_type, self._handle_event)

        self._started = True
        logger.info("BenchmarkLogger started → %s", self._output_path)

    def stop(self) -> None:
        """Flush and close the CSV file. Unsubscribe from bus."""
        if not self._started:
            return

        if self._bus is not None:
            for event_type in self._CAPTURE_TYPES:
                self._bus.unsubscribe(event_type, self._handle_event)

        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()

        self._started = False
        logger.info(
            "BenchmarkLogger stopped. %d events logged → %s",
            self._event_count, self._output_path,
        )

    def _handle_event(self, event: ProcessingEvent) -> None:
        """Write one event to the CSV file.

        Args:
            event: ProcessingEvent received from the event bus.
        """
        row = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "document_id": event.document_id or "",
            "page_number": event.page_number if event.page_number is not None else "",
            "worker_id": event.worker_id or "",
            "node_id": event.node_id or "",
            "wall_time_seconds": event.wall_time_seconds,
            "message": event.message,
        }

        with self._lock:
            if self._writer is None or self._file is None or self._file.closed:
                return
            self._writer.writerow(row)
            self._event_count += 1
            if self._auto_flush:
                self._file.flush()

    def summary(self) -> dict[str, Any]:
        """Return a summary of all logged events.

        Returns:
            Dictionary with output path, event count, and run metadata.
        """
        return {
            "output_path": str(self._output_path),
            "event_count": self._event_count,
            "started": self._started,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def event_count(self) -> int:
        """Total number of events written to CSV."""
        return self._event_count

    def __enter__(self) -> "BenchmarkLogger":
        """Start the logger as a context manager."""
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        """Stop the logger on context exit."""
        self.stop()
