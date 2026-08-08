"""Evaluation result data model for the Adaptive Distributed Framework.

Models:
    EvaluationResult: Complete evaluation output for a pipeline run,
        covering all 6 metrics defined in architecture v2.0 §4.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.core.constants import (
    SCHEDULER_OVERHEAD_TARGET_FRACTION,
    SPEEDUP_BASELINE_NODES,
)
from adaptive_framework.core.exceptions import ValidationError


@dataclass
class EvaluationResult:
    """Complete evaluation result for one pipeline run.

    Records all 6 metrics from architecture v2.0 §4.1:

        1. Speedup         — multi-node vs single-node time ratio
        2. Throughput      — pages per second
        3. CPU Utilization — average CPU % across all nodes
        4. GPU Utilization — average GPU % (None if no GPU)
        5. Energy          — Joules per document batch
        6. Scheduler Overhead — % of total time in scheduler

    Attributes:
        run_id: Unique pipeline run identifier.
        node_count: Number of worker nodes in this run.
        total_documents: Number of documents processed.
        total_pages: Total pages processed.

        speedup: multi-node wall time / baseline_wall_time.
        throughput_pages_per_second: Pages processed per second.
        avg_cpu_percent: Average CPU utilization (%).
        avg_gpu_percent: Average GPU utilization (%). None if no GPU.
        total_energy_joules: Estimated total energy (Joules).
        scheduler_overhead_percent: Scheduler time / total time × 100.

        baseline_nodes: Reference node count for speedup calculation.
        baseline_wall_time_seconds: Single-node reference wall time.
        measured_wall_time_seconds: This run's wall-clock time.
        scheduler_time_seconds: Total time inside the scheduler.

        passes_overhead_target: True if scheduler_overhead_percent < 1%.
        report_path: Path to the generated evaluation report file.
        generated_at: ISO 8601 timestamp when this result was created.

    Example:
        >>> result = EvaluationResult(
        ...     run_id="adf_run_001", node_count=4,
        ...     total_documents=10, total_pages=500,
        ...     speedup=3.8, throughput_pages_per_second=4.17,
        ...     avg_cpu_percent=68.0, total_energy_joules=240.0,
        ...     scheduler_overhead_percent=0.67,
        ...     baseline_nodes=1, baseline_wall_time_seconds=456.0,
        ...     measured_wall_time_seconds=120.0,
        ...     scheduler_time_seconds=0.8)
        >>> result.passes_overhead_target
        True
    """

    run_id: str
    node_count: int
    total_documents: int
    total_pages: int

    # Metric 1: Speedup
    speedup: float

    # Metric 2: Throughput
    throughput_pages_per_second: float

    # Metric 3: CPU Utilization
    avg_cpu_percent: float

    # Metric 5: Energy
    total_energy_joules: float

    # Metric 6: Scheduler Overhead
    scheduler_overhead_percent: float

    # Supporting timing data
    baseline_nodes: int
    baseline_wall_time_seconds: float
    measured_wall_time_seconds: float
    scheduler_time_seconds: float

    # Metric 4: GPU Utilization (optional — None if no GPU)
    avg_gpu_percent: float | None = None

    # Derived
    report_path: str | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate all evaluation result fields.

        Raises:
            ValidationError: If any metric value is invalid.
        """
        if self.node_count < 1:
            raise ValidationError(
                "EvaluationResult.node_count must be >= 1.",
                field="node_count",
                value=self.node_count,
            )
        if self.speedup < 0:
            raise ValidationError(
                "EvaluationResult.speedup must be >= 0.",
                field="speedup",
                value=self.speedup,
            )
        if self.throughput_pages_per_second < 0:
            raise ValidationError(
                "EvaluationResult.throughput_pages_per_second must be >= 0.",
                field="throughput_pages_per_second",
                value=self.throughput_pages_per_second,
            )
        if not (0.0 <= self.avg_cpu_percent <= 100.0):
            raise ValidationError(
                "EvaluationResult.avg_cpu_percent must be in [0.0, 100.0].",
                field="avg_cpu_percent",
                value=self.avg_cpu_percent,
            )
        if self.scheduler_overhead_percent < 0:
            raise ValidationError(
                "EvaluationResult.scheduler_overhead_percent must be >= 0.",
                field="scheduler_overhead_percent",
                value=self.scheduler_overhead_percent,
            )
        if self.baseline_nodes < SPEEDUP_BASELINE_NODES:
            raise ValidationError(
                f"EvaluationResult.baseline_nodes must be >= {SPEEDUP_BASELINE_NODES}.",
                field="baseline_nodes",
                value=self.baseline_nodes,
            )

    @property
    def passes_overhead_target(self) -> bool:
        """Return True if scheduler overhead is within the architecture target.

        The architecture spec §4.2 requires overhead < 1%.

        Returns:
            True when scheduler_overhead_percent < 1.0.
        """
        target_percent = SCHEDULER_OVERHEAD_TARGET_FRACTION * 100.0
        return self.scheduler_overhead_percent < target_percent

    @property
    def energy_per_page_joules(self) -> float:
        """Compute energy consumption per page.

        Returns:
            Joules per page. Returns 0.0 if total_pages is 0.
        """
        if self.total_pages == 0:
            return 0.0
        return self.total_energy_joules / self.total_pages

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary including all computed properties for reporting.
        """
        d = asdict(self)
        d["passes_overhead_target"] = self.passes_overhead_target
        d["energy_per_page_joules"] = self.energy_per_page_joules
        return d

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(run_id='{self.run_id}', "
            f"nodes={self.node_count}, "
            f"speedup={self.speedup:.2f}x, "
            f"throughput={self.throughput_pages_per_second:.2f} p/s, "
            f"sched_overhead={self.scheduler_overhead_percent:.3f}%, "
            f"overhead_ok={self.passes_overhead_target})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EvaluationResult):
            return NotImplemented
        return self.run_id == other.run_id
