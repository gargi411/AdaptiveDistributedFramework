"""DashboardState — Thread-safe shared state store for the Engineering Dashboard.

The DistributedCoordinator writes state to this store at regular intervals.
The Streamlit dashboard reads from it on each auto-refresh cycle.

This approach avoids importing the full coordinator into the Streamlit process
while keeping the data fresh through a polling-based update model.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DashboardStateStore:
    """Thread-safe shared state store for the Engineering Dashboard.

    Can be used in two ways:
        1. In-process: coordinator writes directly to this object.
        2. File-based: coordinator writes to a JSON file; dashboard reads it.

    The auto-refresh Streamlit UI always reads from this store.

    Attributes:
        _data: The state dictionary.
        _lock: Thread safety lock.
        _file_path: Optional file path for file-based mode.
        _last_updated: Monotonic time of last update.

    Example:
        >>> store = DashboardStateStore()
        >>> store.update(coordinator.get_status_dict())
        >>> data = store.get()
    """

    def __init__(self, file_path: str | Path | None = None) -> None:
        """Initialise the DashboardStateStore.

        Args:
            file_path: Optional path to write/read state JSON.
                None = in-process only.
        """
        self._data: dict[str, Any] = self._empty_state()
        self._lock = threading.RLock()
        self._file_path = Path(file_path) if file_path else None
        self._last_updated: float = 0.0

        # History for sparkline charts
        self._cpu_history: list[float] = []
        self._ram_history: list[float] = []
        self._queue_history: list[int] = []
        self._throughput_history: list[float] = []
        self._history_max: int = 60  # 60 data points = 5 min at 5s interval

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        """Return an empty state skeleton matching the coordinator output.

        Returns:
            Empty state dictionary with all expected keys.
        """
        return {
            "run_id": "—",
            "framework_state": "initializing",
            "uptime_seconds": 0.0,
            "registry": {
                "total_workers": 0,
                "active_workers": 0,
                "idle_workers": 0,
                "lost_workers": 0,
                "avg_cpu_percent": 0.0,
                "avg_ram_percent": 0.0,
                "total_completed": 0,
                "total_failed": 0,
            },
            "dispatcher": {
                "total_dispatched": 0,
                "total_completed": 0,
                "total_failed": 0,
                "currently_active": 0,
                "queue_size": 0,
                "scheduler_time_seconds": 0.0,
            },
            "heartbeat_monitor": {
                "total_heartbeats": 0,
                "total_timeouts": 0,
                "total_reconnects": 0,
                "current_lost_workers": 0,
            },
            "work_stealing": {
                "total_steal_events": 0,
                "total_tasks_stolen": 0,
                "scheduler_time_seconds": 0.0,
            },
            "failure_recovery": {
                "total_workers_lost": 0,
                "total_tasks_recovered": 0,
                "total_tasks_permanently_failed": 0,
            },
            "resource_orchestration": {
                "latest_recommendation": None,
                "running": False,
            },
            "cluster_manager": {
                "mode": "dev",
                "initialized": False,
                "node_count": 0,
                "uptime_seconds": 0.0,
            },
            "queue_size": 0,
            "workers": [],
            "heartbeat_events": [],
            "steal_events": [],
            "recovery_events": [],
            "assignment_history": [],
            "dataset_summary": {
                "total_pdfs": 0,
                "total_pages": 0,
                "digital_pdfs": 0,
                "scanned_pdfs": 0,
                "unknown_pdfs": 0,
                "pending": 0,
                "queued": 0,
                "in_progress": 0,
                "completed": 0,
                "failed": 0,
                "avg_pages": 0.0,
                "avg_size_mb": 0.0,
                "metadata_cached": False,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def update(self, data: dict[str, Any]) -> None:
        """Update the store with new coordinator state.

        Args:
            data: Dictionary from coordinator.get_status_dict().
        """
        now = time.monotonic()
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._data = data
            self._last_updated = now
            # Update sparkline histories
            registry = data.get("registry", {})
            self._cpu_history.append(registry.get("avg_cpu_percent", 0.0))
            self._ram_history.append(registry.get("avg_ram_percent", 0.0))
            self._queue_history.append(data.get("queue_size", 0))
            dispatcher = data.get("dispatcher", {})
            completed = dispatcher.get("total_completed", 0)
            self._throughput_history.append(float(completed))

            # Trim histories
            for lst in (
                self._cpu_history,
                self._ram_history,
                self._queue_history,
                self._throughput_history,
            ):
                while len(lst) > self._history_max:
                    lst.pop(0)

        if self._file_path is not None:
            self._flush_to_file(data)

    def _flush_to_file(self, data: dict[str, Any]) -> None:
        """Write state plus chart histories to JSON file (file-based mode).

        Args:
            data: Current state dictionary.
        """
        try:
            with self._lock:
                payload = dict(data)
                payload["_chart_history"] = {
                    "cpu": list(self._cpu_history),
                    "ram": list(self._ram_history),
                    "queue": list(self._queue_history),
                    "throughput": list(self._throughput_history),
                }
            self._file_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            tmp = self._file_path.with_suffix(".tmp")  # type: ignore[union-attr]
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            tmp.replace(self._file_path)  # type: ignore[arg-type]
        except Exception:
            pass  # Best-effort -- don't crash the coordinator

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    def get(self) -> dict[str, Any]:
        """Return a snapshot of the current state.

        Returns:
            Copy of the current state dictionary.
        """
        with self._lock:
            return dict(self._data)

    def get_cpu_history(self) -> list[float]:
        """Return CPU utilization history for sparkline chart.

        Returns:
            List of CPU % values (oldest first).
        """
        with self._lock:
            return list(self._cpu_history)

    def get_ram_history(self) -> list[float]:
        """Return RAM utilization history.

        Returns:
            List of RAM % values (oldest first).
        """
        with self._lock:
            return list(self._ram_history)

    def get_queue_history(self) -> list[int]:
        """Return queue size history.

        Returns:
            List of queue size values (oldest first).
        """
        with self._lock:
            return list(self._queue_history)

    def get_throughput_history(self) -> list[float]:
        """Return throughput history (completed tasks count).

        Returns:
            List of completed task counts (oldest first).
        """
        with self._lock:
            return list(self._throughput_history)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "DashboardStateStore":
        """Build a store and load initial state from a JSON file.

        Restores chart histories from the ``_chart_history`` key written by
        ``_flush_to_file`` so the Performance Graphs panel shows the full
        processing history of the completed run.

        Args:
            file_path: Path to state JSON file.

        Returns:
            DashboardStateStore pre-loaded with file content.
        """
        store = cls(file_path=file_path)
        path = Path(file_path)
        if path.exists():
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                with store._lock:
                    store._data = data
                    # Restore chart histories from the persisted payload
                    ch = data.get("_chart_history", {})
                    store._cpu_history = [float(x) for x in ch.get("cpu", [])]
                    store._ram_history = [float(x) for x in ch.get("ram", [])]
                    store._queue_history = [int(x) for x in ch.get("queue", [])]
                    store._throughput_history = [float(x) for x in ch.get("throughput", [])]
            except Exception:
                pass
        return store

    @property
    def last_updated_seconds_ago(self) -> float:
        """Return seconds since last update.

        Returns:
            Elapsed seconds since last update, or inf if never updated.
        """
        if self._last_updated == 0.0:
            return float("inf")
        return time.monotonic() - self._last_updated
