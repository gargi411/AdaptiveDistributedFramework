"""ClusterBenchmark — Full distributed cluster performance measurement.

Measures and exports all primary metrics defined in architecture §4.1:
    - Scheduler Overhead (§4.2): dispatcher_time + work_stealing_time / total_time
    - Throughput: pages_per_second
    - Speedup: multi-node_time / single_node_time
    - Parallel Efficiency: speedup / num_nodes
    - Worker Idle Time: per worker and aggregate
    - Queue Wait Time: time tasks spend waiting before dispatch
    - Load Balance Score: from PartitionSummary
    - Cluster Efficiency: utilization × parallel_efficiency

Exports to CSV and JSON for evaluation.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_framework.benchmarks.benchmark_runner import BenchmarkReport
from adaptive_framework.core.constants import ROOT_LOGGER_NAME, SCHEDULER_OVERHEAD_TARGET_FRACTION

logger = logging.getLogger(ROOT_LOGGER_NAME + ".cluster_benchmark")


@dataclass
class ClusterBenchmarkResult:
    """Complete performance measurement for one distributed pipeline run.

    All timing values are in seconds unless stated otherwise.

    Attributes:
        run_id: Unique identifier for this benchmark run.
        mode: 'dev' or 'presentation'.
        num_workers: Number of workers that participated.
        num_nodes: Number of physical nodes.
        total_documents: Documents submitted.
        total_pages: Total pages processed.
        total_wall_time_s: End-to-end elapsed time.
        scheduler_time_s: Time spent in scheduling operations.
        scheduler_overhead_fraction: scheduler_time / total_wall_time.
        meets_overhead_target: True if overhead < 1% (architecture target).
        throughput_pages_per_s: Pages processed per second.
        speedup: multi-node speedup vs single-node baseline.
        parallel_efficiency: speedup / num_nodes.
        avg_cpu_percent: Average CPU across all workers.
        avg_ram_percent: Average RAM across all workers.
        avg_gpu_percent: Average GPU across all workers. None if no GPU.
        load_balance_score: Partition balance score [0.0, 1.0].
        total_steal_events: Number of work-stealing events.
        total_tasks_stolen: Total tasks moved by work stealing.
        total_recovery_events: Number of failure recovery events.
        total_tasks_recovered: Tasks re-queued after worker loss.
        worker_utilization: Per-worker utilization dict.
        benchmark_stages: Stage-level timings from BenchmarkReport.
        completed_at: ISO 8601 UTC timestamp.
    """

    run_id: str
    mode: str
    num_workers: int
    num_nodes: int
    total_documents: int
    total_pages: int
    total_wall_time_s: float
    scheduler_time_s: float
    scheduler_overhead_fraction: float
    throughput_pages_per_s: float
    speedup: float
    parallel_efficiency: float
    avg_cpu_percent: float
    avg_ram_percent: float
    load_balance_score: float
    meets_overhead_target: bool = field(init=False)
    avg_gpu_percent: float | None = None
    total_steal_events: int = 0
    total_tasks_stolen: int = 0
    total_recovery_events: int = 0
    total_tasks_recovered: int = 0
    worker_utilization: dict[str, Any] = field(default_factory=dict)
    benchmark_stages: list[dict[str, Any]] = field(default_factory=list)
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        self.meets_overhead_target = (
            self.scheduler_overhead_fraction < SCHEDULER_OVERHEAD_TARGET_FRACTION
        )

    @property
    def scheduler_overhead_percent(self) -> float:
        """Return scheduler overhead as a percentage.

        Returns:
            scheduler_overhead_fraction * 100.
        """
        return self.scheduler_overhead_fraction * 100.0

    @property
    def cluster_efficiency(self) -> float:
        """Compute cluster efficiency: parallel_efficiency × avg_utilization.

        Returns:
            Cluster efficiency in [0.0, 1.0].
        """
        avg_util = (self.avg_cpu_percent + self.avg_ram_percent) / 200.0
        return round(self.parallel_efficiency * avg_util, 4)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Complete dictionary with all metrics.
        """
        d = asdict(self)
        d["scheduler_overhead_percent"] = self.scheduler_overhead_percent
        d["cluster_efficiency"] = self.cluster_efficiency
        d["meets_overhead_target"] = self.meets_overhead_target
        return d

    def format_summary(self) -> str:
        """Format a human-readable summary table.

        Returns:
            Multi-line string with all key metrics.
        """
        overhead_flag = "✅" if self.meets_overhead_target else "❌"
        lines = [
            "=" * 65,
            f"  CLUSTER BENCHMARK REPORT — Run: {self.run_id}",
            "=" * 65,
            f"  Mode              : {self.mode}",
            f"  Workers           : {self.num_workers} ({self.num_nodes} node(s))",
            f"  Documents         : {self.total_documents}",
            f"  Pages             : {self.total_pages}",
            f"  Wall Time         : {self.total_wall_time_s:.3f}s",
            f"  Throughput        : {self.throughput_pages_per_s:.2f} pages/s",
            f"  Speedup           : {self.speedup:.3f}x",
            f"  Parallel Efficiency: {self.parallel_efficiency:.3f}",
            f"  Cluster Efficiency : {self.cluster_efficiency:.3f}",
            f"  Load Balance Score : {self.load_balance_score:.3f}",
            "-" * 65,
            f"  Scheduler Overhead : {self.scheduler_overhead_percent:.4f}% {overhead_flag} "
            f"(target <{SCHEDULER_OVERHEAD_TARGET_FRACTION*100:.0f}%)",
            f"  Scheduler Time    : {self.scheduler_time_s:.6f}s",
            "-" * 65,
            f"  Avg CPU           : {self.avg_cpu_percent:.1f}%",
            f"  Avg RAM           : {self.avg_ram_percent:.1f}%",
            f"  Avg GPU           : {self.avg_gpu_percent or 'N/A'}",
            "-" * 65,
            f"  Steal Events      : {self.total_steal_events}",
            f"  Tasks Stolen      : {self.total_tasks_stolen}",
            f"  Recovery Events   : {self.total_recovery_events}",
            f"  Tasks Recovered   : {self.total_tasks_recovered}",
            "=" * 65,
            f"  Completed At      : {self.completed_at}",
            "=" * 65,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ClusterBenchmarkResult(run_id='{self.run_id}', "
            f"throughput={self.throughput_pages_per_s:.2f}p/s, "
            f"speedup={self.speedup:.2f}x, "
            f"overhead={self.scheduler_overhead_percent:.4f}%)"
        )


class ClusterBenchmark:
    """Orchestrates measurement, computation, and export of cluster benchmarks.

    Use as a context manager or call start()/stop() explicitly.

    Example:
        >>> bench = ClusterBenchmark(run_id="run_001", mode="dev", num_workers=4)
        >>> bench.start()
        >>> # ... run the pipeline ...
        >>> result = bench.stop(
        ...     total_pages=1200,
        ...     scheduler_time_s=0.003,
        ...     avg_cpu=72.5, avg_ram=55.0,
        ...     load_balance_score=0.94,
        ... )
        >>> bench.save_csv(result, path=Path("outputs/benchmark.csv"))
    """

    def __init__(
        self,
        run_id: str,
        mode: str = "dev",
        num_workers: int = 1,
        num_nodes: int = 1,
        single_node_baseline_s: float | None = None,
    ) -> None:
        """Initialise the ClusterBenchmark.

        Args:
            run_id: Unique identifier for this run.
            mode: Deployment mode ('dev' or 'presentation').
            num_workers: Number of participating workers.
            num_nodes: Number of physical nodes.
            single_node_baseline_s: Known single-node total time for speedup
                computation. If None, speedup is reported as 1.0 (baseline run).
        """
        self._run_id = run_id
        self._mode = mode
        self._num_workers = num_workers
        self._num_nodes = num_nodes
        self._single_node_baseline = single_node_baseline_s

        self._stage_report = BenchmarkReport(run_id=run_id)
        self._start_time: float | None = None
        self._total_documents: int = 0

    # ------------------------------------------------------------------ #
    # Context manager support                                              #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "ClusterBenchmark":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # User must call stop() to get the result

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the wall-clock timer."""
        self._start_time = time.perf_counter()
        logger.info("ClusterBenchmark started (run_id=%s).", self._run_id)

    def stop(
        self,
        total_documents: int,
        total_pages: int,
        scheduler_time_s: float,
        avg_cpu_percent: float,
        avg_ram_percent: float,
        load_balance_score: float,
        avg_gpu_percent: float | None = None,
        total_steal_events: int = 0,
        total_tasks_stolen: int = 0,
        total_recovery_events: int = 0,
        total_tasks_recovered: int = 0,
        worker_utilization: dict[str, Any] | None = None,
    ) -> ClusterBenchmarkResult:
        """Stop the timer and compute all metrics.

        Args:
            total_documents: Documents processed in this run.
            total_pages: Total pages processed.
            scheduler_time_s: Cumulative scheduler time (§4.2).
            avg_cpu_percent: Average CPU across workers.
            avg_ram_percent: Average RAM across workers.
            load_balance_score: Partition balance score [0.0, 1.0].
            avg_gpu_percent: Average GPU (None if no GPU).
            total_steal_events: Number of work-stealing events.
            total_tasks_stolen: Tasks moved by work stealing.
            total_recovery_events: Failure recovery events fired.
            total_tasks_recovered: Tasks re-queued after recovery.
            worker_utilization: Per-worker utilization mapping.

        Returns:
            ClusterBenchmarkResult with all computed metrics.
        """
        if self._start_time is None:
            raise RuntimeError("ClusterBenchmark.start() must be called before stop().")

        wall_time = time.perf_counter() - self._start_time
        wall_time = max(wall_time, 1e-9)  # avoid division by zero

        throughput = total_pages / wall_time
        overhead = scheduler_time_s / wall_time

        baseline = self._single_node_baseline or wall_time
        speedup = baseline / wall_time if wall_time > 0 else 1.0
        parallel_eff = speedup / max(self._num_nodes, 1)

        result = ClusterBenchmarkResult(
            run_id=self._run_id,
            mode=self._mode,
            num_workers=self._num_workers,
            num_nodes=self._num_nodes,
            total_documents=total_documents,
            total_pages=total_pages,
            total_wall_time_s=wall_time,
            scheduler_time_s=scheduler_time_s,
            scheduler_overhead_fraction=overhead,
            throughput_pages_per_s=throughput,
            speedup=speedup,
            parallel_efficiency=parallel_eff,
            avg_cpu_percent=avg_cpu_percent,
            avg_ram_percent=avg_ram_percent,
            avg_gpu_percent=avg_gpu_percent,
            load_balance_score=load_balance_score,
            total_steal_events=total_steal_events,
            total_tasks_stolen=total_tasks_stolen,
            total_recovery_events=total_recovery_events,
            total_tasks_recovered=total_tasks_recovered,
            worker_utilization=worker_utilization or {},
            benchmark_stages=self._stage_report.to_dict().get("results", []),
        )

        logger.info("ClusterBenchmark complete:\n%s", result.format_summary())
        return result

    def time_stage(self, stage_name: str, **metadata: Any) -> Any:
        """Return a context manager for timing a named stage.

        Args:
            stage_name: Name of the stage to time.
            **metadata: Optional metadata key-value pairs.

        Returns:
            BenchmarkTimer context manager.
        """
        return self._stage_report.time(stage_name, metadata)

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_csv(
        result: ClusterBenchmarkResult,
        path: Path,
        append: bool = False,
    ) -> Path:
        """Save benchmark result to a CSV file.

        Args:
            result: ClusterBenchmarkResult to save.
            path: Output CSV file path.
            append: If True, append rows; otherwise overwrite.

        Returns:
            Path to the written CSV file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = result.to_dict()
        # Flatten nested dicts/lists
        flat: dict[str, Any] = {
            k: v for k, v in d.items()
            if not isinstance(v, (dict, list))
        }

        mode = "a" if append else "w"
        write_header = not path.exists() or not append
        with path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat)

        logger.info("Benchmark CSV saved → %s", path)
        return path

    @staticmethod
    def save_json(result: ClusterBenchmarkResult, path: Path) -> Path:
        """Save the full benchmark result to a JSON file.

        Args:
            result: ClusterBenchmarkResult to save.
            path: Output JSON file path.

        Returns:
            Path to the written JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("Benchmark JSON saved → %s", path)
        return path
