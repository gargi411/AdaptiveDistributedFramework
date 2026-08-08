"""DistributedCoordinator — Primary orchestration hub for the Ray cluster.

The DistributedCoordinator integrates all Phase 2B modules:

    ┌─────────────────────────────────────────────────────┐
    │              DistributedCoordinator                 │
    │                                                     │
    │  ClusterManager  ←──── cluster.yaml                │
    │  WorkerRegistry  ←──── register / discover          │
    │  HeartbeatMonitor ←─── periodic worker pings        │
    │  TaskDispatcher  ←──── pop queue → assign worker    │
    │  FailureRecovery ←──── on_timeout callback          │
    │  WorkStealingCoordinator ←─── idle/overload balance │
    │  ResourceOrchestrator ←──── CPU/RAM/GPU monitoring  │
    │  ClusterBenchmark ←──── §4.2 overhead measurement   │
    └─────────────────────────────────────────────────────┘

Usage:
    >>> coord = DistributedCoordinator.from_yaml("configs/cluster.yaml")
    >>> coord.start()
    >>> coord.submit_partitions(partitions)
    >>> coord.run_dispatch_loop(until_empty=True)
    >>> report = coord.stop()
"""

from __future__ import annotations

import logging
import time
import threading
from pathlib import Path
from typing import Any

import yaml

from adaptive_framework.benchmarks.cluster_benchmark import ClusterBenchmark, ClusterBenchmarkResult
from adaptive_framework.coordinator.cluster_manager import ClusterManager
from adaptive_framework.coordinator.failure_recovery import FailureRecoveryEngine
from adaptive_framework.coordinator.heartbeat_monitor import (
    HeartbeatMonitor,
    HeartbeatPayload,
)
from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.coordinator.resource_orchestrator import ResourceOrchestrator
from adaptive_framework.coordinator.task_dispatcher import TaskDispatcher
from adaptive_framework.coordinator.worker_registry import WorkerRegistry
from adaptive_framework.core.constants import ROOT_LOGGER_NAME
from adaptive_framework.core.exceptions import FrameworkError
from adaptive_framework.models.runtime import ClusterStatus, FrameworkState, WorkerState
from adaptive_framework.models.scheduling import Partition, PageWorkUnit
from adaptive_framework.scheduler.priority_queue import PageCountPriorityQueue
from adaptive_framework.scheduler.work_stealing import WorkStealingCoordinator

logger = logging.getLogger(ROOT_LOGGER_NAME + ".coordinator")


class DistributedCoordinator:
    """Primary orchestration hub integrating all Phase 2B components.

    Provides a unified API for:
        - Starting and stopping the distributed cluster.
        - Registering workers and monitoring their heartbeats.
        - Submitting partition plans to the global priority queue.
        - Running the dispatch loop (assign tasks to workers).
        - Handling failures via FailureRecoveryEngine.
        - Rebalancing via WorkStealingCoordinator.
        - Reporting cluster and benchmark metrics.

    The coordinator is designed to be mode-transparent — the same code
    runs in dev (single laptop) and presentation (3-laptop) modes; only
    cluster.yaml changes.

    Example:
        >>> coord = DistributedCoordinator.from_yaml("configs/cluster.yaml")
        >>> coord.start()
        >>> coord.submit_partitions(partitions)
        >>> coord.run_dispatch_loop(until_empty=True)
        >>> result = coord.stop()
    """

    def __init__(self, cluster_config: dict[str, Any], run_id: str = "adf_run") -> None:
        """Initialise the DistributedCoordinator.

        Args:
            cluster_config: Parsed cluster.yaml content (cluster section).
            run_id: Unique identifier for this pipeline run.
        """
        self._config = cluster_config
        self._run_id = run_id
        self._start_time: float | None = None
        self._state = FrameworkState.INITIALIZING
        self._lock = threading.Lock()

        # ── Sub-components ─────────────────────────────────────────────
        self._cluster_manager = ClusterManager(cluster_config)
        self._registry = WorkerRegistry()
        self._global_queue = PageCountPriorityQueue()

        self._heartbeat_monitor = HeartbeatMonitor(
            registry=self._registry,
            timeout_seconds=cluster_config.get("health", {}).get(
                "heartbeat_timeout_seconds", 15.0
            ),
            check_interval_seconds=cluster_config.get("health", {}).get(
                "health_check_interval_seconds", 2.0
            ),
            on_timeout=self._on_worker_timeout,
            on_reconnect=self._on_worker_reconnect,
        )

        self._dispatcher = TaskDispatcher(
            registry=self._registry,
            queue=self._global_queue,
        )

        self._recovery_engine = FailureRecoveryEngine(
            registry=self._registry,
            dispatcher=self._dispatcher,
            max_retries=cluster_config.get("failure_recovery", {}).get("max_retries", 3),
            reassignment_delay_seconds=cluster_config.get("failure_recovery", {}).get(
                "reassignment_delay_seconds", 1.0
            ),
        )

        ws_cfg = cluster_config.get("work_stealing", {})
        self._work_stealing = WorkStealingCoordinator(
            registry=self._registry,
            global_queue=self._global_queue,
            steal_threshold=ws_cfg.get("steal_threshold", 2),
            steal_fraction=ws_cfg.get("steal_fraction", 0.5),
            check_interval_seconds=ws_cfg.get("check_interval_seconds", 3.0),
        )

        ro_cfg = cluster_config.get("resource_orchestration", {})
        self._resource_orchestrator = ResourceOrchestrator(
            registry=self._registry,
            cpu_high_threshold=ro_cfg.get("cpu_high_threshold_percent", 85.0),
            cpu_low_threshold=ro_cfg.get("cpu_low_threshold_percent", 20.0),
            sample_interval_seconds=ro_cfg.get("sample_interval_seconds", 5.0),
        )

        self._benchmark = ClusterBenchmark(
            run_id=run_id,
            mode=cluster_config.get("mode", "dev"),
            num_workers=0,  # updated after registration
            num_nodes=1,
        )

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, config_path: str | Path, run_id: str = "adf_run") -> "DistributedCoordinator":
        """Build a DistributedCoordinator from a cluster.yaml file.

        Args:
            config_path: Path to cluster.yaml.
            run_id: Unique run identifier.

        Returns:
            Fully configured DistributedCoordinator.

        Raises:
            FrameworkError: If the config file is missing or malformed.
        """
        path = Path(config_path)
        if not path.exists():
            raise FrameworkError(f"Cluster config not found: {path}")
        try:
            with path.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception as exc:
            raise FrameworkError(f"Failed to parse cluster config: {exc}") from exc

        cluster_section = raw.get("cluster", raw)
        return cls(cluster_config=cluster_section, run_id=run_id)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self, init_ray: bool = True) -> None:
        """Start all coordinator components.

        Args:
            init_ray: If True, call ClusterManager.start() to init Ray.
                      Set to False for unit tests or Ray-less dev runs.
        """
        logger.info("DistributedCoordinator starting (run_id=%s).", self._run_id)

        if init_ray:
            self._cluster_manager.start()

        self._heartbeat_monitor.start()
        self._work_stealing.start()
        self._resource_orchestrator.start()
        self._benchmark.start()

        self._start_time = time.monotonic()
        with self._lock:
            self._state = FrameworkState.READY

        logger.info("DistributedCoordinator ready.")

    def stop(self, shutdown_ray: bool = True) -> ClusterBenchmarkResult | None:
        """Stop all coordinator components and return the benchmark result.

        Args:
            shutdown_ray: If True, shut down the Ray cluster.

        Returns:
            ClusterBenchmarkResult with all metrics, or None if no benchmark.
        """
        logger.info("DistributedCoordinator stopping.")
        with self._lock:
            self._state = FrameworkState.SHUTTING_DOWN

        self._heartbeat_monitor.stop()
        self._work_stealing.stop()
        self._resource_orchestrator.stop()

        # Compute benchmark result
        registry_stats = self._registry.get_statistics()
        ws_stats = self._work_stealing.get_statistics()
        rc_stats = self._recovery_engine.get_statistics()
        workers = self._registry.get_all()
        avg_cpu = registry_stats.get("avg_cpu_percent", 0.0)
        avg_ram = registry_stats.get("avg_ram_percent", 0.0)
        total_scheduler_time = (
            self._dispatcher.scheduler_time_seconds
            + self._work_stealing.scheduler_time_seconds
        )

        try:
            result = self._benchmark.stop(
                total_documents=registry_stats.get("total_completed", 0),
                total_pages=sum(w.total_completed for w in workers),
                scheduler_time_s=total_scheduler_time,
                avg_cpu_percent=avg_cpu,
                avg_ram_percent=avg_ram,
                load_balance_score=1.0,  # placeholder; set by caller from PartitionSummary
                total_steal_events=ws_stats["total_steal_events"],
                total_tasks_stolen=ws_stats["total_tasks_stolen"],
                total_recovery_events=rc_stats["total_workers_lost"],
                total_tasks_recovered=rc_stats["total_tasks_recovered"],
                worker_utilization=self._work_stealing.get_worker_utilization(),
            )
        except RuntimeError:
            result = None

        if shutdown_ray:
            self._cluster_manager.shutdown()

        with self._lock:
            self._state = FrameworkState.STOPPED

        logger.info("DistributedCoordinator stopped.")
        return result

    # ------------------------------------------------------------------ #
    # Worker management                                                    #
    # ------------------------------------------------------------------ #

    def register_worker(self, node_info: NodeInfo, worker_id: str | None = None) -> str:
        """Register a worker with the coordinator.

        Args:
            node_info: Hardware metadata of the registering node.
            worker_id: Optional explicit ID.

        Returns:
            The assigned worker_id.
        """
        record = self._registry.register(node_info=node_info, worker_id=worker_id)
        logger.info("Worker registered: %s (%s).", record.worker_id[:8], record.hostname)
        return record.worker_id

    def receive_heartbeat(self, payload: HeartbeatPayload) -> None:
        """Accept an incoming heartbeat payload from a worker.

        Args:
            payload: HeartbeatPayload from the worker.
        """
        self._heartbeat_monitor.record_heartbeat(payload)
        self._resource_orchestrator.update_queue_size(self._global_queue.size())

    # ------------------------------------------------------------------ #
    # Task management                                                      #
    # ------------------------------------------------------------------ #

    def submit_partitions(self, partitions: list[Partition]) -> int:
        """Insert all work units from a partition plan into the global queue.

        Args:
            partitions: List of Partition objects from PageCountPartitioner.

        Returns:
            Total number of work units inserted.
        """
        total = 0
        for partition in partitions:
            for wu in partition.work_units:
                self._global_queue.insert(wu)
                total += 1
        with self._lock:
            self._state = FrameworkState.RUNNING
        logger.info("Submitted %d work units to global queue.", total)
        self._resource_orchestrator.update_queue_size(self._global_queue.size())
        return total

    def pop_work_unit(self) -> "PageWorkUnit | None":
        """Pop and return the highest-priority work unit from the global queue.

        Used by the dev-cluster processing loop to drain work units one at a
        time without accessing the queue's internal state directly.

        Returns:
            The highest-priority PageWorkUnit, or None if the queue is empty.
        """
        wu = self._global_queue.pop()
        if wu is not None:
            self._resource_orchestrator.update_queue_size(self._global_queue.size())
        return wu

    def queue_size(self) -> int:
        """Return the current number of pending work units in the global queue.

        Returns:
            Integer count of queued work units.
        """
        return self._global_queue.size()

    def run_dispatch_loop(
        self,
        until_empty: bool = True,
        max_iterations: int = 10_000,
        poll_interval_seconds: float = 0.1,
    ) -> int:
        """Run the task dispatch loop.

        Continuously pops tasks from the priority queue and assigns them
        to available workers.

        Args:
            until_empty: If True, run until the queue is empty.
            max_iterations: Safety ceiling on dispatch iterations.
            poll_interval_seconds: Sleep interval when no workers available.

        Returns:
            Total number of tasks dispatched.
        """
        dispatched = 0
        iterations = 0

        while iterations < max_iterations:
            if until_empty and self._global_queue.is_empty():
                break

            result = self._dispatcher.dispatch_next()
            if result is not None:
                dispatched += 1
            else:
                # No work or no workers — back off
                time.sleep(poll_interval_seconds)

            iterations += 1

        logger.info("Dispatch loop complete: %d tasks dispatched.", dispatched)
        return dispatched

    def report_completed(self, work_unit_id: str) -> None:
        """Mark a work unit as completed.

        Args:
            work_unit_id: ID of the completed PageWorkUnit.
        """
        self._dispatcher.report_completed(work_unit_id)

    def report_failed(self, work_unit_id: str, reason: str = "Unknown error") -> None:
        """Mark a work unit as failed.

        Args:
            work_unit_id: ID of the failed PageWorkUnit.
            reason: Description of the failure.
        """
        self._dispatcher.report_failed(work_unit_id, reason)

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def _on_worker_timeout(self, worker_id: str) -> None:
        """HeartbeatMonitor callback: worker has timed out.

        Args:
            worker_id: Lost worker ID.
        """
        logger.warning("Coordinator: worker '%s' timed out. Starting recovery.", worker_id[:8])
        self._recovery_engine.handle_worker_lost(worker_id)
        with self._lock:
            if self._registry.lost_count > 0:
                self._state = FrameworkState.DEGRADED

    def _on_worker_reconnect(self, worker_id: str) -> None:
        """HeartbeatMonitor callback: lost worker has reconnected.

        Args:
            worker_id: Reconnected worker ID.
        """
        logger.info("Coordinator: worker '%s' reconnected.", worker_id[:8])
        self._recovery_engine.handle_worker_reconnected(worker_id)
        with self._lock:
            if self._registry.lost_count == 0:
                self._state = FrameworkState.RUNNING

    # ------------------------------------------------------------------ #
    # Status & reporting                                                   #
    # ------------------------------------------------------------------ #

    def get_cluster_status(self) -> ClusterStatus:
        """Return the current cluster status snapshot.

        Returns:
            ClusterStatus aggregating all worker states.
        """
        return self._registry.get_cluster_status()

    def get_status_dict(self) -> dict[str, Any]:
        """Return a comprehensive status dictionary for the dashboard.

        Returns:
            Dictionary with all component states and statistics.
        """
        with self._lock:
            state = self._state.value

        return {
            "run_id": self._run_id,
            "framework_state": state,
            "uptime_seconds": (
                time.monotonic() - self._start_time
                if self._start_time else 0.0
            ),
            "registry": self._registry.get_statistics(),
            "dispatcher": self._dispatcher.get_statistics(),
            "heartbeat_monitor": self._heartbeat_monitor.get_statistics(),
            "work_stealing": self._work_stealing.get_statistics(),
            "failure_recovery": self._recovery_engine.get_statistics(),
            "resource_orchestration": self._resource_orchestrator.get_runtime_report(),
            "cluster_manager": self._cluster_manager.to_dict(),
            "queue_size": self._global_queue.size(),
        }

    @property
    def registry(self) -> WorkerRegistry:
        """Return the WorkerRegistry (read-only reference).

        Returns:
            The coordinator's WorkerRegistry.
        """
        return self._registry

    @property
    def dispatcher(self) -> TaskDispatcher:
        """Return the TaskDispatcher (read-only reference).

        Returns:
            The coordinator's TaskDispatcher.
        """
        return self._dispatcher

    @property
    def recovery_engine(self) -> FailureRecoveryEngine:
        """Return the FailureRecoveryEngine (read-only reference).

        Returns:
            The coordinator's FailureRecoveryEngine.
        """
        return self._recovery_engine

    @property
    def work_stealing(self) -> WorkStealingCoordinator:
        """Return the WorkStealingCoordinator (read-only reference).

        Returns:
            The coordinator's WorkStealingCoordinator.
        """
        return self._work_stealing

    def __repr__(self) -> str:
        with self._lock:
            state = self._state.value
        return (
            f"DistributedCoordinator(run_id='{self._run_id}', "
            f"state={state}, "
            f"workers={self._registry.worker_count}, "
            f"queue={self._global_queue.size()})"
        )
