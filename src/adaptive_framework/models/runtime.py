"""Runtime monitoring data models for the Adaptive Distributed Framework.

Models:
    ResourceSnapshot: Point-in-time system resource measurement.
    RuntimeMetrics: Aggregated performance metrics for a run.
    WorkerStatus: Status snapshot of a single Ray worker node.
    ClusterStatus: Aggregated status of the entire Ray cluster.
    FrameworkStatus: Top-level framework health indicator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from adaptive_framework.core.constants import UNKNOWN_FLOAT, UNKNOWN_INT
from adaptive_framework.core.exceptions import ValidationError


class WorkerState(str, Enum):
    """Lifecycle state of a worker node.

    Values:
        IDLE: Connected but has no assigned tasks.
        ACTIVE: Processing one or more work units.
        OVERLOADED: Task queue depth exceeds steal_threshold.
        LOST: Heartbeat timeout — Failure Recovery initiated.
        SHUTTING_DOWN: Graceful shutdown in progress.
    """

    IDLE = "idle"
    ACTIVE = "active"
    OVERLOADED = "overloaded"
    LOST = "lost"
    SHUTTING_DOWN = "shutting_down"


class FrameworkState(str, Enum):
    """Top-level framework operational state.

    Values:
        INITIALIZING: Loading configs, setting up logging.
        READY: All components initialized, awaiting jobs.
        RUNNING: At least one pipeline job is active.
        DEGRADED: One or more workers are lost; Failure Recovery active.
        SHUTTING_DOWN: Graceful shutdown in progress.
        STOPPED: Framework has halted.
    """

    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


# =============================================================
# ResourceSnapshot
# =============================================================


@dataclass
class ResourceSnapshot:
    """Point-in-time system resource measurement for a single node.

    Captured periodically during a pipeline run for the evaluation engine.
    All utilization values are in the range [0.0, 100.0] (percent).

    Attributes:
        node_id: Identifier of the node this snapshot was taken from.
        timestamp: ISO 8601 timestamp of the measurement.
        cpu_percent: CPU utilization across all cores (%).
        memory_percent: RAM utilization (%).
        gpu_percent: GPU utilization (%). None if no GPU present.
        gpu_memory_percent: GPU memory utilization (%). None if no GPU.
        disk_read_mb_s: Disk read throughput (MB/s).
        disk_write_mb_s: Disk write throughput (MB/s).
        net_sent_mb_s: Network bytes sent per second (MB/s).
        net_recv_mb_s: Network bytes received per second (MB/s).

    Example:
        >>> snap = ResourceSnapshot(
        ...     node_id="node_01", cpu_percent=72.5, memory_percent=45.0)
        >>> snap.cpu_percent
        72.5
    """

    node_id: str
    cpu_percent: float
    memory_percent: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    gpu_percent: float | None = None
    gpu_memory_percent: float | None = None
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    net_sent_mb_s: float = 0.0
    net_recv_mb_s: float = 0.0

    def __post_init__(self) -> None:
        """Validate resource snapshot values.

        Raises:
            ValidationError: If any utilization value is out of range.
        """
        for attr_name, val in [
            ("cpu_percent", self.cpu_percent),
            ("memory_percent", self.memory_percent),
        ]:
            if not (0.0 <= val <= 100.0):
                raise ValidationError(
                    f"ResourceSnapshot.{attr_name} must be in [0.0, 100.0].",
                    field=attr_name,
                    value=val,
                )
        if self.gpu_percent is not None and not (0.0 <= self.gpu_percent <= 100.0):
            raise ValidationError(
                "ResourceSnapshot.gpu_percent must be in [0.0, 100.0].",
                field="gpu_percent",
                value=self.gpu_percent,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this resource snapshot.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"ResourceSnapshot(node_id='{self.node_id}', "
            f"cpu={self.cpu_percent:.1f}%, "
            f"mem={self.memory_percent:.1f}%)"
        )


# =============================================================
# RuntimeMetrics
# =============================================================


@dataclass
class RuntimeMetrics:
    """Aggregated performance metrics for a complete pipeline run.

    Computed after all documents have been processed. Consumed by the
    Evaluation Engine (Phase 6) to produce the final evaluation report.

    Attributes:
        run_id: Unique identifier for this pipeline run.
        total_documents: Number of documents submitted.
        total_pages: Total pages across all documents.
        total_wall_time_seconds: End-to-end wall-clock time.
        scheduler_time_seconds: Time spent inside the scheduler.
        scheduler_overhead_fraction: scheduler_time / total_wall_time.
        throughput_pages_per_second: Pages processed per wall-clock second.
        speedup: Ratio of single-node time to multi-node time.
        avg_cpu_percent: Average CPU utilization across all nodes.
        avg_gpu_percent: Average GPU utilization across all nodes. None if no GPU.
        total_energy_joules: Estimated total energy consumed (Joules).
        node_count: Number of worker nodes involved.
        resource_snapshots: All resource snapshots collected during the run.

    Example:
        >>> metrics = RuntimeMetrics(
        ...     run_id="adf_run_001", total_documents=10, total_pages=500,
        ...     total_wall_time_seconds=120.0, scheduler_time_seconds=0.8,
        ...     scheduler_overhead_fraction=0.0067,
        ...     throughput_pages_per_second=4.17, speedup=3.8,
        ...     avg_cpu_percent=68.0, total_energy_joules=240.0, node_count=4)
    """

    run_id: str
    total_documents: int
    total_pages: int
    total_wall_time_seconds: float
    scheduler_time_seconds: float
    scheduler_overhead_fraction: float
    throughput_pages_per_second: float
    speedup: float
    avg_cpu_percent: float
    total_energy_joules: float
    node_count: int
    avg_gpu_percent: float | None = None
    resource_snapshots: list[ResourceSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_documents < 0:
            raise ValidationError(
                "RuntimeMetrics.total_documents must be >= 0.",
                field="total_documents",
                value=self.total_documents,
            )
        if self.scheduler_overhead_fraction < 0:
            raise ValidationError(
                "RuntimeMetrics.scheduler_overhead_fraction must be >= 0.",
                field="scheduler_overhead_fraction",
                value=self.scheduler_overhead_fraction,
            )
        if self.node_count < 1:
            raise ValidationError(
                "RuntimeMetrics.node_count must be >= 1.",
                field="node_count",
                value=self.node_count,
            )

    @property
    def scheduler_overhead_percent(self) -> float:
        """Return scheduler overhead as a percentage.

        Returns:
            scheduler_overhead_fraction * 100.
        """
        return self.scheduler_overhead_fraction * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary including scheduler_overhead_percent.
        """
        d = asdict(self)
        d["scheduler_overhead_percent"] = self.scheduler_overhead_percent
        return d

    def __repr__(self) -> str:
        return (
            f"RuntimeMetrics(run_id='{self.run_id}', "
            f"pages={self.total_pages}, "
            f"throughput={self.throughput_pages_per_second:.2f} p/s, "
            f"speedup={self.speedup:.2f}x, "
            f"sched_overhead={self.scheduler_overhead_percent:.3f}%)"
        )


# =============================================================
# WorkerStatus
# =============================================================


@dataclass
class WorkerStatus:
    """Status snapshot of a single Ray worker node.

    Emitted by the Heartbeat Monitor to the Distributed Coordinator.

    Attributes:
        worker_id: Unique worker identifier.
        node_id: Physical node the worker is running on.
        state: Current WorkerState.
        active_work_units: IDs of work units currently being processed.
        completed_work_units: Number of work units completed this session.
        failed_work_units: Number of work units that failed this session.
        last_heartbeat_timestamp: ISO 8601 timestamp of last heartbeat.
        resource_snapshot: Latest resource snapshot for this node.

    Example:
        >>> ws = WorkerStatus(worker_id="worker_01", node_id="node_01",
        ...                   state=WorkerState.ACTIVE)
    """

    worker_id: str
    node_id: str
    state: WorkerState
    active_work_units: list[str] = field(default_factory=list)
    completed_work_units: int = 0
    failed_work_units: int = 0
    last_heartbeat_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resource_snapshot: ResourceSnapshot | None = None

    def __post_init__(self) -> None:
        if self.completed_work_units < 0:
            raise ValidationError(
                "WorkerStatus.completed_work_units must be >= 0.",
                field="completed_work_units",
                value=self.completed_work_units,
            )

    def is_available(self) -> bool:
        """Return True if this worker can accept new tasks.

        Returns:
            True if state is IDLE or ACTIVE (not OVERLOADED, LOST, etc.).
        """
        return self.state in (WorkerState.IDLE, WorkerState.ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation including state value string.
        """
        d = asdict(self)
        d["state"] = self.state.value
        return d

    def __repr__(self) -> str:
        return (
            f"WorkerStatus(worker_id='{self.worker_id}', "
            f"state={self.state.value}, "
            f"active={len(self.active_work_units)} units)"
        )


# =============================================================
# ClusterStatus
# =============================================================


@dataclass
class ClusterStatus:
    """Aggregated status of the entire Ray cluster.

    Published by the Distributed Coordinator at regular intervals.

    Attributes:
        total_workers: Total registered workers.
        active_workers: Workers in ACTIVE or IDLE state.
        lost_workers: Workers in LOST state (Failure Recovery active).
        total_active_work_units: Work units currently in-flight.
        worker_statuses: List of individual worker status snapshots.
        timestamp: ISO 8601 timestamp of this cluster snapshot.

    Example:
        >>> cs = ClusterStatus(total_workers=4, active_workers=3,
        ...                    lost_workers=1, total_active_work_units=12)
    """

    total_workers: int
    active_workers: int
    lost_workers: int
    total_active_work_units: int
    worker_statuses: list[WorkerStatus] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.total_workers < 0:
            raise ValidationError(
                "ClusterStatus.total_workers must be >= 0.",
                field="total_workers",
                value=self.total_workers,
            )
        if self.active_workers + self.lost_workers > self.total_workers:
            raise ValidationError(
                "active_workers + lost_workers cannot exceed total_workers.",
            )

    @property
    def is_degraded(self) -> bool:
        """Return True if any workers are in the LOST state.

        Returns:
            True when lost_workers > 0 (Failure Recovery is active).
        """
        return self.lost_workers > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation including is_degraded flag.
        """
        d = asdict(self)
        d["is_degraded"] = self.is_degraded
        return d

    def __repr__(self) -> str:
        return (
            f"ClusterStatus(workers={self.total_workers}, "
            f"active={self.active_workers}, "
            f"lost={self.lost_workers}, "
            f"degraded={self.is_degraded})"
        )


# =============================================================
# FrameworkStatus
# =============================================================


@dataclass
class FrameworkStatus:
    """Top-level framework health and operational status.

    Aggregates cluster, scheduler, and pipeline state into a single
    status object that can be queried via health-check endpoints.

    Attributes:
        state: Current FrameworkState.
        run_id: Active run identifier. None if no run is active.
        cluster_status: Current cluster status snapshot.
        active_jobs: Number of pipeline jobs currently executing.
        completed_jobs: Total jobs completed since framework start.
        failed_jobs: Total jobs that failed since framework start.
        uptime_seconds: Seconds since the framework was initialized.
        timestamp: ISO 8601 timestamp of this status snapshot.

    Example:
        >>> fs = FrameworkStatus(state=FrameworkState.RUNNING,
        ...                      run_id="adf_run_001",
        ...                      cluster_status=cs,
        ...                      active_jobs=2, completed_jobs=5,
        ...                      failed_jobs=0, uptime_seconds=300.0)
    """

    state: FrameworkState
    cluster_status: ClusterStatus
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    uptime_seconds: float
    run_id: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.active_jobs < 0:
            raise ValidationError(
                "FrameworkStatus.active_jobs must be >= 0.",
                field="active_jobs",
                value=self.active_jobs,
            )
        if self.uptime_seconds < 0:
            raise ValidationError(
                "FrameworkStatus.uptime_seconds must be >= 0.",
                field="uptime_seconds",
                value=self.uptime_seconds,
            )

    @property
    def is_healthy(self) -> bool:
        """Return True if the framework is in a healthy operational state.

        Returns:
            True when state is READY or RUNNING and no workers are lost.
        """
        return (
            self.state in (FrameworkState.READY, FrameworkState.RUNNING)
            and not self.cluster_status.is_degraded
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary including is_healthy and state value string.
        """
        d = asdict(self)
        d["state"] = self.state.value
        d["is_healthy"] = self.is_healthy
        return d

    def __repr__(self) -> str:
        return (
            f"FrameworkStatus(state={self.state.value}, "
            f"healthy={self.is_healthy}, "
            f"active_jobs={self.active_jobs}, "
            f"uptime={self.uptime_seconds:.1f}s)"
        )
