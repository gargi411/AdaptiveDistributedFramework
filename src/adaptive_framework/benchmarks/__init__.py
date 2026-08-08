"""Benchmarks package — Phase 2A lightweight benchmarking.

Public API:
    BenchmarkRunner  — alias for BenchmarkReport (primary entry point).
    BenchmarkReport  — Accumulates BenchmarkResult objects, generates CSV.
    BenchmarkTimer   — Context manager for timing individual stages.
    BenchmarkResult  — Dataclass for a single timed measurement.

Usage::

    from adaptive_framework.benchmarks import BenchmarkReport

    report = BenchmarkReport(run_id="run_001")

    with report.time("dataset_loading_time_s", {"files": 50}):
        dataset = builder.build()

    with report.time("partition_time_s", {"documents": len(dataset), "workers": 4}):
        partitions, stats = partitioner.partition(dataset, num_workers=4)

    report.print_summary()
    report.save_csv(Path("reports/benchmark_run_001.csv"))
"""

from adaptive_framework.benchmarks.benchmark_runner import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTimer,
)

# Convenient alias
BenchmarkRunner = BenchmarkReport

__all__ = [
    "BenchmarkRunner",
    "BenchmarkReport",
    "BenchmarkResult",
    "BenchmarkTimer",
]
