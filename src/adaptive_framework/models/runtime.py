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
    """Point-in-time system resource measurement for a single node or cluster.

    Captured periodically during pipeline runs for monitoring and observability.
    All utilization values are in the range [0.0, 100.0] (percent).
    Optional metrics are None when telemetry is unsupported or unavailable.

    Attributes:
        node_id: Identifier of the node this snapshot was taken from.
        cpu_percent: CPU utilization across all cores (%).
        memory_percent: RAM utilization (%).
        timestamp: ISO 8601 UTC timestamp of the measurement.
        cpu_count: Total logical CPU cores on the host.
        ram_used_mb: Currently used system RAM in megabytes.
        ram_available_mb: Available system RAM in megabytes.
        gpu_available: True if a GPU is detected and available.
        gpu_name: Human-readable GPU device name (e.g. Intel Iris Xe).
        gpu_device: OpenVINO execution device ID (e.g. 'GPU', 'GPU.0').
        gpu_utilization_percent: GPU engine utilization (%). None if unavailable.
        gpu_percent: Alias for gpu_utilization_percent for backward compatibility.
        gpu_memory_used_mb: GPU memory used (MB). None if unavailable.
        gpu_memory_total_mb: GPU total memory (MB). None if unavailable.
        gpu_memory_percent: GPU memory utilization (%). None if unavailable.
        gpu_temperature_c: GPU temperature in Celsius. None if unavailable.
        gpu_power_w: GPU power consumption in Watts. None if unavailable.
        queue_length: In-flight or pending task queue depth.
        active_workers: Number of active worker processes.
        ocr_latency_ms: Recent OCR inference latency in ms if measured.
        ocr_throughput_pages_s: Recent OCR throughput in pages/sec if measured.
        disk_read_mb_s: Disk read throughput (MB/s).
        disk_write_mb_s: Disk write throughput (MB/s).
        net_sent_mb_s: Network bytes sent per second (MB/s).
        net_recv_mb_s: Network bytes received per second (MB/s).
    """

    node_id: str = "node_0"
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    cpu_count: int = 1
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0
    gpu_available: bool = False
    gpu_name: str | None = None
    gpu_device: str | None = None
    gpu_utilization_percent: float | None = None
    gpu_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_memory_percent: float | None = None
    gpu_temperature_c: float | None = None
    gpu_power_w: float | None = None
    queue_length: int = 0
    active_workers: int = 0
    ocr_latency_ms: float | None = None
    ocr_throughput_pages_s: float | None = None
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    net_sent_mb_s: float = 0.0
    net_recv_mb_s: float = 0.0

    def __post_init__(self) -> None:
        """Validate resource snapshot values and synchronize aliases."""
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

        # Synchronize gpu_percent and gpu_utilization_percent aliases
        if self.gpu_utilization_percent is not None and self.gpu_percent is None:
            self.gpu_percent = self.gpu_utilization_percent
        elif self.gpu_percent is not None and self.gpu_utilization_percent is None:
            self.gpu_utilization_percent = self.gpu_percent

        if self.gpu_percent is not None and not (0.0 <= self.gpu_percent <= 100.0):
            raise ValidationError(
                "ResourceSnapshot.gpu_percent must be in [0.0, 100.0].",
                field="gpu_percent",
                value=self.gpu_percent,
            )

        if self.gpu_memory_percent is not None and not (0.0 <= self.gpu_memory_percent <= 100.0):
            raise ValidationError(
                "ResourceSnapshot.gpu_memory_percent must be in [0.0, 100.0].",
                field="gpu_memory_percent",
                value=self.gpu_memory_percent,
            )

    @property
    def ram_percent(self) -> float:
        """Alias for memory_percent."""
        return self.memory_percent

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this resource snapshot.
        """
        d = asdict(self)
        d["ram_percent"] = self.ram_percent
        return d

    def format_summary(self) -> str:
        """Format as a clean, readable ASCII summary block."""
        gpu_util_str = (
            f"{self.gpu_utilization_percent:.1f}%"
            if self.gpu_utilization_percent is not None
            else "UNAVAILABLE"
        )
        gpu_mem_str = (
            f"{self.gpu_memory_used_mb:.1f} / {self.gpu_memory_total_mb:.1f} MB"
            if (self.gpu_memory_used_mb is not None and self.gpu_memory_total_mb is not None)
            else (
                f"Total: {self.gpu_memory_total_mb:.1f} MB"
                if self.gpu_memory_total_mb is not None
                else "UNAVAILABLE"
            )
        )
        temp_str = (
            f"{self.gpu_temperature_c:.1f} C"
            if self.gpu_temperature_c is not None
            else "UNAVAILABLE"
        )
        pwr_str = (
            f"{self.gpu_power_w:.1f} W"
            if self.gpu_power_w is not None
            else "UNAVAILABLE"
        )
        ocr_lat_str = (
            f"{self.ocr_latency_ms:.2f} ms"
            if self.ocr_latency_ms is not None
            else "N/A"
        )
        ocr_thru_str = (
            f"{self.ocr_throughput_pages_s:.2f} p/s"
            if self.ocr_throughput_pages_s is not None
            else "N/A"
        )

        lines = [
            f"Resource Snapshot [{self.timestamp}]",
            "-" * 50,
            f"  Node ID:            {self.node_id}",
            f"  CPU Utilization:    {self.cpu_percent:.1f}% ({self.cpu_count} cores)",
            f"  RAM Utilization:    {self.memory_percent:.1f}% (Used: {self.ram_used_mb:.1f} MB, Avail: {self.ram_available_mb:.1f} MB)",
            f"  GPU Available:      {'YES' if self.gpu_available else 'NO'}",
            f"  GPU Device:         {self.gpu_name or 'None'} ({self.gpu_device or 'None'})",
            f"  GPU Utilization:    {gpu_util_str}",
            f"  GPU Memory:         {gpu_mem_str}",
            f"  GPU Temperature:    {temp_str}",
            f"  GPU Power:          {pwr_str}",
            f"  Queue Length:       {self.queue_length}",
            f"  Active Workers:     {self.active_workers}",
            f"  OCR Latency:        {ocr_lat_str}",
            f"  OCR Throughput:     {ocr_thru_str}",
            "-" * 50,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ResourceSnapshot(node='{self.node_id}', "
            f"cpu={self.cpu_percent:.1f}%, "
            f"ram={self.memory_percent:.1f}%, "
            f"gpu={'avail' if self.gpu_available else 'none'}, "
            f"queue={self.queue_length})"
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
