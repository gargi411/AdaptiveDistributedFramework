"""Benchmark Runner — Module 6 of Phase 2A.

Provides lightweight, zero-dependency benchmarking for each pipeline stage.
Records elapsed time for named operations and writes CSV reports.

Measured operations (Phase 2A):
    - dataset_loading_time_s
    - metadata_generation_time_s
    - partition_time_s
    - queue_creation_time_s
    - scheduler_overhead_s (see architecture §4.2)

Design:
    - ``BenchmarkTimer``   — Context manager for timing named stages.
    - ``BenchmarkReport``  — Accumulates results and writes CSV / text.
    - No charts, no matplotlib (those belong to the evaluation dashboard).

Architecture §4.2 compliance:
    Scheduler overhead is instrumented using ``time.perf_counter()`` around
    the dispatch loop. This module stores the raw measurements; the
    ``compute_overhead_fraction`` function (in time_utils) computes the final
    fraction for the evaluation engine.
"""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


@dataclass
class BenchmarkResult:
    """A single timed benchmark result.

    Attributes:
        stage_name: Human-readable stage name (e.g. 'partition_time').
        elapsed_seconds: Wall-clock time in seconds using perf_counter.
        metadata: Optional additional context (e.g. document_count).
        timestamp_utc: ISO 8601 UTC timestamp when measurement completed.

    Example:
        >>> result = BenchmarkResult(
        ...     stage_name="partition_time_s",
        ...     elapsed_seconds=0.0023,
        ...     metadata={"documents": 150, "workers": 4},
        ... )
    """

    stage_name: str
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary with all result fields.
        """
        return {
            "stage_name": self.stage_name,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "timestamp_utc": self.timestamp_utc,
            **self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(stage='{self.stage_name}', "
            f"elapsed={self.elapsed_seconds:.6f}s)"
        )


class BenchmarkTimer:
    """Context manager for timing a named pipeline stage.

    Usage::

        timer = BenchmarkTimer("partition_time_s")
        with timer:
            partitions, stats = partitioner.partition(dataset, num_workers=4)
        result = timer.result
        print(result.elapsed_seconds)

    Alternatively, use ``BenchmarkReport.time()`` for automatic accumulation.
    """

    def __init__(self, stage_name: str, metadata: dict[str, Any] | None = None) -> None:
        """Initialise the BenchmarkTimer.

        Args:
            stage_name: Name of the pipeline stage being timed.
            metadata: Optional context dictionary stored with the result.
        """
        self.stage_name = stage_name
        self.metadata: dict[str, Any] = metadata or {}
        self._start: float = 0.0
        self.result: BenchmarkResult | None = None

    def __enter__(self) -> "BenchmarkTimer":
        """Start the timer."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the timer and store the result."""
        elapsed = time.perf_counter() - self._start
        self.result = BenchmarkResult(
            stage_name=self.stage_name,
            elapsed_seconds=elapsed,
            metadata=self.metadata,
        )


class BenchmarkReport:
    """Accumulates BenchmarkResult objects and generates reports.

    Attributes:
        _results: Ordered list of benchmark results.
        _run_id: Identifier for this benchmark run.

    Example:
        >>> report = BenchmarkReport(run_id="run_001")
        >>> with report.time("partition_time_s", {"documents": 150}):
        ...     partitions, stats = partitioner.partition(dataset, num_workers=4)
        >>> report.print_summary()
        >>> report.save_csv(Path("reports/benchmark.csv"))
    """

    def __init__(self, run_id: str = "") -> None:
        """Initialise the BenchmarkReport.

        Args:
            run_id: Optional identifier for this benchmark run
                    (e.g. framework run_id from ConfigManager).
        """
        self._results: list[BenchmarkResult] = []
        self._run_id = run_id
        self._session_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Measurement API                                                      #
    # ------------------------------------------------------------------ #

    @contextmanager
    def time(
        self,
        stage_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager that times a named stage and records the result.

        Args:
            stage_name: Stage label.
            metadata: Optional dict stored alongside the timing.

        Yields:
            None.

        Example::

            with report.time("dataset_loading_time_s", {"files": 50}):
                dataset = builder.build()
        """
        timer = BenchmarkTimer(stage_name, metadata)
        with timer:
            yield
        if timer.result:
            self._results.append(timer.result)

    def record(self, result: BenchmarkResult) -> None:
        """Manually add a pre-computed BenchmarkResult.

        Args:
            result: The result to add.
        """
        self._results.append(result)

    def record_manual(
        self,
        stage_name: str,
        elapsed_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a manually timed result.

        Args:
            stage_name: Stage label.
            elapsed_seconds: Pre-measured elapsed time.
            metadata: Optional context.
        """
        self._results.append(
            BenchmarkResult(
                stage_name=stage_name,
                elapsed_seconds=elapsed_seconds,
                metadata=metadata or {},
            )
        )

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def save_csv(
        self,
        output_path: Path,
        append: bool = False,
    ) -> Path:
        """Write benchmark results to a CSV file.

        Args:
            output_path: Destination CSV file path.
            append: If True and file exists, appends rows rather than overwriting.

        Returns:
            Resolved path of the written CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append and output_path.exists() else "w"

        # Collect all unique metadata keys across results
        extra_keys: list[str] = []
        for r in self._results:
            for k in r.metadata:
                if k not in extra_keys:
                    extra_keys.append(k)

        fieldnames = ["run_id", "stage_name", "elapsed_seconds", "timestamp_utc"] + extra_keys

        with output_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
            for result in self._results:
                row = result.to_dict()
                row["run_id"] = self._run_id
                writer.writerow(row)

        return output_path.resolve()

    def print_summary(self) -> None:
        """Print a formatted benchmark summary table to stdout."""
        width = 60
        sep = "─" * width
        print(f"\n┌{sep}┐")
        print(f"│{'BENCHMARK REPORT':^{width}}│")
        if self._run_id:
            print(f"│  Run ID: {self._run_id:<{width - 10}}│")
        print(f"├{sep}┤")
        print(f"│  {'Stage':<35} {'Time (s)':>10}  │")
        print(f"├{sep}┤")
        for result in self._results:
            name = result.stage_name[:35]
            print(f"│  {name:<35} {result.elapsed_seconds:>10.6f}  │")
        print(f"├{sep}┤")
        total = self.total_elapsed_seconds()
        print(f"│  {'TOTAL SESSION'::<35} {total:>10.6f}  │")
        print(f"└{sep}┘\n")

    def get_result(self, stage_name: str) -> BenchmarkResult | None:
        """Retrieve the result for a specific stage.

        Args:
            stage_name: The stage name to look up.

        Returns:
            The most recent BenchmarkResult for this stage, or None.
        """
        matches = [r for r in self._results if r.stage_name == stage_name]
        return matches[-1] if matches else None

    def total_elapsed_seconds(self) -> float:
        """Return the sum of all recorded elapsed times.

        Returns:
            Total seconds across all benchmark stages.
        """
        return sum(r.elapsed_seconds for r in self._results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all results to a dictionary.

        Returns:
            Dictionary with run_id and list of result dicts.
        """
        return {
            "run_id": self._run_id,
            "total_elapsed_seconds": round(self.total_elapsed_seconds(), 6),
            "results": [r.to_dict() for r in self._results],
        }

    @property
    def results(self) -> list[BenchmarkResult]:
        """Read-only access to the accumulated results list.

        Returns:
            List of BenchmarkResult objects.
        """
        return list(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return (
            f"BenchmarkReport(run_id='{self._run_id}', "
            f"stages={len(self._results)}, "
            f"total={self.total_elapsed_seconds():.4f}s)"
        )
