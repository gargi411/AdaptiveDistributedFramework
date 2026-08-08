"""WorkerRegistry — Thread-safe registry of all cluster workers.

Provides registration, discovery, update, removal, and statistics for
all worker processes in the framework's Ray cluster.

The registry is the single source of truth for worker identity and state.
It is consumed by:
    - TaskDispatcher (who is available?)
    - HeartbeatMonitor (update last-seen timestamps)
    - FailureRecovery (which workers are LOST?)
    - WorkStealing (who is idle / overloaded?)
    - EngineeringDashboard (display worker panel)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.coordinator.worker_registry.worker_record import WorkerRecord
from adaptive_framework.core.exceptions import FrameworkError
from adaptive_framework.models.runtime import ClusterStatus, WorkerState, WorkerStatus


class WorkerRegistry:
    """Thread-safe registry maintaining all cluster worker records.

    Operations are protected by a single ``threading.RLock`` so that
    Ray actor calls and background monitor threads can read/write safely.

    Attributes:
        _workers: Mapping of worker_id → WorkerRecord.
        _lock: Reentrant lock for thread-safe access.
        _created_at: Registry creation timestamp (perf_counter).

    Example:
        >>> registry = WorkerRegistry()
        >>> node = NodeInfo.from_current_host()
        >>> rec = registry.register(node, worker_id="w_001")
        >>> print(registry.worker_count)
        1
        >>> registry.update_heartbeat("w_001", cpu_percent=45.0, ...)
        >>> registry.remove("w_001")
    """

    def __init__(self) -> None:
        """Initialise an empty WorkerRegistry."""
        self._workers: dict[str, WorkerRecord] = {}
        self._lock: threading.RLock = threading.RLock()
        self._created_at: float = time.monotonic()

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(
        self,
        node_info: NodeInfo,
        worker_id: str | None = None,
    ) -> WorkerRecord:
        """Register a new worker and return its WorkerRecord.

        If a worker with the same worker_id already exists, the existing
        record is returned and its last_seen_at is updated.

        Args:
            node_info: Hardware metadata from the registering node.
            worker_id: Optional explicit ID. Auto-generated UUID4 if None.

        Returns:
            WorkerRecord for the newly registered (or refreshed) worker.
        """
        with self._lock:
            record = WorkerRecord.create(node_info=node_info, worker_id=worker_id)
            existing = self._workers.get(record.worker_id)
            if existing is not None:
                existing.last_seen_at = datetime.now(timezone.utc).isoformat()
                return existing
            self._workers[record.worker_id] = record
            return record

    def remove(self, worker_id: str) -> WorkerRecord | None:
        """Remove a worker from the registry.

        Args:
            worker_id: ID of the worker to remove.

        Returns:
            The removed WorkerRecord, or None if not found.
        """
        with self._lock:
            return self._workers.pop(worker_id, None)

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    def get(self, worker_id: str) -> WorkerRecord | None:
        """Retrieve a worker record by ID.

        Args:
            worker_id: Worker identifier.

        Returns:
            WorkerRecord or None if not found.
        """
        with self._lock:
            return self._workers.get(worker_id)

    def get_all(self) -> list[WorkerRecord]:
        """Return a snapshot of all registered worker records.

        Returns:
            List of WorkerRecord objects (copy, safe for iteration).
        """
        with self._lock:
            return list(self._workers.values())

    def get_available(self) -> list[WorkerRecord]:
        """Return workers that can accept new tasks (IDLE or ACTIVE, not LOST).

        Returns:
            List of available WorkerRecord objects.
        """
        with self._lock:
            return [w for w in self._workers.values() if w.is_available]

    def get_idle(self) -> list[WorkerRecord]:
        """Return workers currently in IDLE state.

        Returns:
            List of IDLE WorkerRecord objects.
        """
        with self._lock:
            return [
                w for w in self._workers.values()
                if w.state == WorkerState.IDLE
            ]

    def get_lost(self) -> list[WorkerRecord]:
        """Return workers that have been declared LOST.

        Returns:
            List of LOST WorkerRecord objects.
        """
        with self._lock:
            return [w for w in self._workers.values() if w.is_lost]

    def get_overloaded(self, steal_threshold: int = 2) -> list[WorkerRecord]:
        """Return workers whose queue depth exceeds the steal threshold.

        Args:
            steal_threshold: Minimum queue depth to consider a worker overloaded.

        Returns:
            List of overloaded WorkerRecord objects sorted by queue_depth descending.
        """
        with self._lock:
            overloaded = [
                w for w in self._workers.values()
                if w.queue_depth >= steal_threshold and w.state != WorkerState.LOST
            ]
            return sorted(overloaded, key=lambda w: w.queue_depth, reverse=True)

    # ------------------------------------------------------------------ #
    # Updates                                                              #
    # ------------------------------------------------------------------ #

    def update_heartbeat(
        self,
        worker_id: str,
        cpu_percent: float,
        ram_percent: float,
        gpu_percent: float | None,
        queue_depth: int,
        current_task_id: str | None,
        state: WorkerState,
    ) -> bool:
        """Apply a heartbeat update to a worker record.

        Args:
            worker_id: Target worker ID.
            cpu_percent: CPU utilization (%).
            ram_percent: RAM utilization (%).
            gpu_percent: GPU utilization (%). None if no GPU.
            queue_depth: Current local task queue depth.
            current_task_id: Active work unit ID or None.
            state: Updated WorkerState.

        Returns:
            True if the update was applied; False if worker_id not found.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            record.update_heartbeat(
                cpu_percent=cpu_percent,
                ram_percent=ram_percent,
                gpu_percent=gpu_percent,
                queue_depth=queue_depth,
                current_task_id=current_task_id,
                state=state,
            )
            return True

    def mark_lost(self, worker_id: str) -> bool:
        """Mark a worker as LOST (heartbeat timeout).

        Args:
            worker_id: Target worker ID.

        Returns:
            True if updated; False if not found.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            record.mark_lost()
            return True

    def mark_recovered(self, worker_id: str) -> bool:
        """Mark a LOST worker as recovered (IDLE).

        Args:
            worker_id: Target worker ID.

        Returns:
            True if updated; False if not found.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            record.mark_recovered()
            return True

    def record_completion(self, worker_id: str) -> bool:
        """Increment a worker's completion counter.

        Args:
            worker_id: Target worker ID.

        Returns:
            True if updated; False if not found.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            record.record_completion()
            return True

    def record_failure(self, worker_id: str) -> bool:
        """Increment a worker's failure counter.

        Args:
            worker_id: Target worker ID.

        Returns:
            True if updated; False if not found.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            record.record_failure()
            return True

    def record_stolen_from(self, worker_id: str, count: int) -> bool:
        """Record that *count* work units were stolen from this worker.

        Args:
            worker_id: Source worker ID.
            count: Number of tasks stolen.

        Returns:
            True if updated; False if not found.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            record.total_stolen_from += count
            return True

    def record_stolen_to(self, worker_id: str, count: int) -> bool:
        """Record that *count* work units were stolen to this worker.

        Args:
            worker_id: Destination worker ID.
            count: Number of tasks received via stealing.

        Returns:
            True if updated; False if not found.
        """
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            record.total_stolen_to += count
            return True

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def worker_count(self) -> int:
        """Return total number of registered workers.

        Returns:
            Count of all registered workers.
        """
        with self._lock:
            return len(self._workers)

    @property
    def active_count(self) -> int:
        """Return number of ACTIVE workers.

        Returns:
            Count of workers in ACTIVE state.
        """
        with self._lock:
            return sum(
                1 for w in self._workers.values()
                if w.state == WorkerState.ACTIVE
            )

    @property
    def idle_count(self) -> int:
        """Return number of IDLE workers.

        Returns:
            Count of workers in IDLE state.
        """
        with self._lock:
            return sum(
                1 for w in self._workers.values()
                if w.state == WorkerState.IDLE
            )

    @property
    def lost_count(self) -> int:
        """Return number of LOST workers.

        Returns:
            Count of workers in LOST state.
        """
        with self._lock:
            return sum(
                1 for w in self._workers.values()
                if w.state == WorkerState.LOST
            )

    @property
    def uptime_seconds(self) -> float:
        """Return registry uptime in seconds.

        Returns:
            Elapsed seconds since registry was created.
        """
        return time.monotonic() - self._created_at

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_cluster_status(self) -> ClusterStatus:
        """Build a ClusterStatus snapshot from the current registry state.

        Returns:
            ClusterStatus with aggregated worker counts and status list.
        """
        with self._lock:
            statuses: list[WorkerStatus] = [
                w.to_status() for w in self._workers.values()
            ]
            return ClusterStatus(
                total_workers=len(self._workers),
                active_workers=sum(
                    1 for w in self._workers.values() if w.is_available
                ),
                lost_workers=self.lost_count,
                total_active_work_units=sum(
                    1 for w in self._workers.values()
                    if w.current_task_id is not None
                ),
                worker_statuses=statuses,
            )

    def get_statistics(self) -> dict[str, Any]:
        """Compute aggregate statistics across all workers.

        Returns:
            Dictionary with total, active, idle, lost, overloaded counts
            and aggregate CPU/RAM utilization.
        """
        with self._lock:
            workers = list(self._workers.values())
            if not workers:
                return {
                    "total_workers": 0,
                    "active_workers": 0,
                    "idle_workers": 0,
                    "lost_workers": 0,
                    "avg_cpu_percent": 0.0,
                    "avg_ram_percent": 0.0,
                    "total_completed": 0,
                    "total_failed": 0,
                    "uptime_seconds": self.uptime_seconds,
                }
            n = len(workers)
            return {
                "total_workers": n,
                "active_workers": sum(1 for w in workers if w.state == WorkerState.ACTIVE),
                "idle_workers": sum(1 for w in workers if w.state == WorkerState.IDLE),
                "lost_workers": sum(1 for w in workers if w.state == WorkerState.LOST),
                "avg_cpu_percent": round(sum(w.cpu_percent for w in workers) / n, 2),
                "avg_ram_percent": round(sum(w.ram_percent for w in workers) / n, 2),
                "total_completed": sum(w.total_completed for w in workers),
                "total_failed": sum(w.total_failed for w in workers),
                "uptime_seconds": self.uptime_seconds,
            }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full registry to a plain dictionary.

        Returns:
            Dictionary mapping worker_id → worker dict.
        """
        with self._lock:
            return {wid: rec.to_dict() for wid, rec in self._workers.items()}

    def __len__(self) -> int:
        return self.worker_count

    def __repr__(self) -> str:
        return (
            f"WorkerRegistry(workers={self.worker_count}, "
            f"active={self.active_count}, "
            f"idle={self.idle_count}, "
            f"lost={self.lost_count})"
        )
