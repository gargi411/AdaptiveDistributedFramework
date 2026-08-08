"""WorkerRecord — Full persistent record for a registered worker.

Extends WorkerStatus (runtime snapshot) with permanent registration data
such as node hardware info, registration time, and cumulative session totals.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.models.runtime import WorkerState, WorkerStatus


@dataclass
class WorkerRecord:
    """Persistent record for a registered cluster worker.

    Stored in the WorkerRegistry for the lifetime of the worker's connection.
    Combines static hardware identity (NodeInfo) with live runtime state
    (WorkerStatus) and cumulative session statistics.

    Attributes:
        worker_id: Unique identifier for this worker process.
        node_info: Hardware metadata of the host node.
        hostname: Shortcut — same as node_info.hostname.
        ip_address: Shortcut — same as node_info.ip_address.
        cpu_count: Logical CPU count on this worker's host.
        ram_total_gb: Total RAM on this worker's host.
        gpu_count: GPU count on this worker's host.
        registered_at: ISO 8601 UTC timestamp of first registration.
        last_seen_at: ISO 8601 UTC timestamp of last heartbeat.
        state: Current WorkerState.
        current_task_id: ID of the work unit currently assigned. None if idle.
        queue_depth: Number of work units currently in this worker's local queue.
        cpu_percent: Most recent CPU utilization reading (%).
        ram_percent: Most recent RAM utilization reading (%).
        gpu_percent: Most recent GPU utilization reading (%). None if no GPU.
        total_completed: Cumulative work units completed this session.
        total_failed: Cumulative work units failed this session.
        total_stolen_from: Work units stolen away from this worker.
        total_stolen_to: Work units stolen to this worker from peers.
        retry_count: Total retry events for tasks on this worker.

    Example:
        >>> node = NodeInfo.from_current_host()
        >>> rec = WorkerRecord.create(node_info=node)
        >>> print(rec.state)
        WorkerState.IDLE
    """

    worker_id: str
    node_info: NodeInfo
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_seen_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    state: WorkerState = WorkerState.IDLE
    current_task_id: str | None = None
    queue_depth: int = 0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    gpu_percent: float | None = None
    total_completed: int = 0
    total_failed: int = 0
    total_stolen_from: int = 0
    total_stolen_to: int = 0
    retry_count: int = 0

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        node_info: NodeInfo,
        worker_id: str | None = None,
    ) -> "WorkerRecord":
        """Create a new WorkerRecord from a NodeInfo.

        Args:
            node_info: Hardware metadata of the host node.
            worker_id: Optional explicit ID; auto-generated UUID4 if None.

        Returns:
            WorkerRecord in IDLE state.
        """
        return cls(
            worker_id=worker_id or str(uuid.uuid4()),
            node_info=node_info,
        )

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def hostname(self) -> str:
        """Return the worker host's hostname.

        Returns:
            Hostname string from NodeInfo.
        """
        return self.node_info.hostname

    @property
    def ip_address(self) -> str:
        """Return the worker host's IP address.

        Returns:
            IP address string from NodeInfo.
        """
        return self.node_info.ip_address

    @property
    def cpu_count(self) -> int:
        """Return the logical CPU count of this worker's host.

        Returns:
            Logical CPU count from NodeInfo.
        """
        return self.node_info.cpu_count_logical

    @property
    def ram_total_gb(self) -> float:
        """Return total RAM (GB) of this worker's host.

        Returns:
            Total RAM from NodeInfo in GB.
        """
        return self.node_info.ram_total_gb

    @property
    def gpu_count(self) -> int:
        """Return GPU count of this worker's host.

        Returns:
            GPU count from NodeInfo.
        """
        return self.node_info.gpu_count

    @property
    def is_available(self) -> bool:
        """Return True if this worker can accept new tasks.

        Returns:
            True when state is IDLE or ACTIVE.
        """
        return self.state in (WorkerState.IDLE, WorkerState.ACTIVE)

    @property
    def is_lost(self) -> bool:
        """Return True if this worker has been declared lost.

        Returns:
            True when state is LOST.
        """
        return self.state == WorkerState.LOST

    @property
    def utilization_score(self) -> float:
        """Compute a composite utilization score in [0.0, 1.0].

        Weights: 60% CPU + 40% RAM (no GPU bias for fairness).

        Returns:
            Utilization score in [0.0, 1.0].
        """
        return round((self.cpu_percent * 0.6 + self.ram_percent * 0.4) / 100.0, 4)

    # ------------------------------------------------------------------ #
    # Mutators                                                             #
    # ------------------------------------------------------------------ #

    def update_heartbeat(
        self,
        cpu_percent: float,
        ram_percent: float,
        gpu_percent: float | None,
        queue_depth: int,
        current_task_id: str | None,
        state: WorkerState,
    ) -> None:
        """Apply a heartbeat update to this record.

        Args:
            cpu_percent: Current CPU utilization (%).
            ram_percent: Current RAM utilization (%).
            gpu_percent: Current GPU utilization (%). None if no GPU.
            queue_depth: Current local task queue depth.
            current_task_id: Currently active work unit ID or None.
            state: Current WorkerState.
        """
        now = datetime.now(timezone.utc).isoformat()
        self.last_seen_at = now
        self.cpu_percent = cpu_percent
        self.ram_percent = ram_percent
        self.gpu_percent = gpu_percent
        self.queue_depth = queue_depth
        self.current_task_id = current_task_id
        self.state = state

    def record_completion(self) -> None:
        """Increment total_completed and update state to IDLE."""
        self.total_completed += 1
        self.current_task_id = None
        if self.queue_depth == 0:
            self.state = WorkerState.IDLE

    def record_failure(self) -> None:
        """Increment total_failed."""
        self.total_failed += 1

    def mark_lost(self) -> None:
        """Set state to LOST."""
        self.state = WorkerState.LOST

    def mark_recovered(self) -> None:
        """Set state back to IDLE after recovery."""
        self.state = WorkerState.IDLE
        self.current_task_id = None
        self.queue_depth = 0

    def to_status(self) -> WorkerStatus:
        """Convert to a WorkerStatus snapshot (for coordinator broadcast).

        Returns:
            WorkerStatus with fields derived from this record.
        """
        active = [self.current_task_id] if self.current_task_id else []
        return WorkerStatus(
            worker_id=self.worker_id,
            node_id=self.node_info.node_id,
            state=self.state,
            active_work_units=active,
            completed_work_units=self.total_completed,
            failed_work_units=self.total_failed,
            last_heartbeat_timestamp=self.last_seen_at,
        )

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for the dashboard.

        Returns:
            Dictionary with all fields including derived properties.
        """
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "cpu_count": self.cpu_count,
            "ram_total_gb": self.ram_total_gb,
            "gpu_count": self.gpu_count,
            "registered_at": self.registered_at,
            "last_seen_at": self.last_seen_at,
            "state": self.state.value,
            "current_task_id": self.current_task_id,
            "queue_depth": self.queue_depth,
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "gpu_percent": self.gpu_percent,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_stolen_from": self.total_stolen_from,
            "total_stolen_to": self.total_stolen_to,
            "retry_count": self.retry_count,
            "utilization_score": self.utilization_score,
            "is_available": self.is_available,
        }

    def __repr__(self) -> str:
        return (
            f"WorkerRecord(worker_id='{self.worker_id[:8]}', "
            f"host='{self.hostname}', "
            f"state={self.state.value}, "
            f"completed={self.total_completed})"
        )
