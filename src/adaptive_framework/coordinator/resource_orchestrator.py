"""ResourceOrchestrator — Adaptive Runtime Resource Orchestration (§2.5).

The ResourceOrchestrator implements the "Adaptive Runtime Resource Orchestration"
component described in architecture v2.0. It:

  1. Samples CPU/RAM/GPU metrics from all registered workers periodically.
  2. Detects over-utilization (CPU > high_threshold) → recommends scale-out.
  3. Detects under-utilization (CPU < low_threshold) → recommends scale-in.
  4. Recommends optimal worker count based on current workload.
  5. Recommends scheduling strategy (page_count, round_robin) based on
     dataset heterogeneity (from WorkloadAnalyzer).
  6. Produces runtime reports for the Engineering Dashboard and evaluation.

This component is advisory — it generates recommendations and reports.
Actual worker spawning/termination is performed by the ClusterManager in
response to these recommendations.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.core.constants import ROOT_LOGGER_NAME
from adaptive_framework.models.runtime import ResourceSnapshot, WorkerState

logger = logging.getLogger(ROOT_LOGGER_NAME + ".resource_orchestrator")


class ResourceRecommendation:
    """Advisory recommendation from the ResourceOrchestrator.

    Attributes:
        timestamp: When this recommendation was generated.
        recommended_workers: Suggested number of workers.
        current_workers: Current active worker count.
        action: 'scale_out', 'scale_in', or 'maintain'.
        scheduling_strategy: Suggested strategy name.
        reason: Human-readable explanation.
        avg_cpu_percent: Average CPU across all workers.
        avg_ram_percent: Average RAM across all workers.
        queue_size: Current global queue depth.
    """

    def __init__(
        self,
        recommended_workers: int,
        current_workers: int,
        action: str,
        scheduling_strategy: str,
        reason: str,
        avg_cpu_percent: float,
        avg_ram_percent: float,
        queue_size: int,
    ) -> None:
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.recommended_workers = recommended_workers
        self.current_workers = current_workers
        self.action = action
        self.scheduling_strategy = scheduling_strategy
        self.reason = reason
        self.avg_cpu_percent = avg_cpu_percent
        self.avg_ram_percent = avg_ram_percent
        self.queue_size = queue_size

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this recommendation.
        """
        return {
            "timestamp": self.timestamp,
            "recommended_workers": self.recommended_workers,
            "current_workers": self.current_workers,
            "action": self.action,
            "scheduling_strategy": self.scheduling_strategy,
            "reason": self.reason,
            "avg_cpu_percent": self.avg_cpu_percent,
            "avg_ram_percent": self.avg_ram_percent,
            "queue_size": self.queue_size,
        }

    def __repr__(self) -> str:
        return (
            f"ResourceRecommendation(action={self.action!r}, "
            f"recommended={self.recommended_workers}, "
            f"current={self.current_workers}, "
            f"cpu={self.avg_cpu_percent:.1f}%)"
        )


class ResourceOrchestrator:
    """Monitors cluster resources and generates scaling recommendations.

    Runs a background thread that samples worker metrics at a configurable
    interval and evaluates scaling thresholds.

    Args:
        registry: WorkerRegistry for current worker metrics.
        cpu_high_threshold: CPU % above which to recommend scale-out.
        cpu_low_threshold: CPU % below which to recommend scale-in.
        sample_interval_seconds: How often to sample metrics.
        history_size: Maximum snapshots/recommendations to retain.

    Example:
        >>> orchestrator = ResourceOrchestrator(
        ...     registry=registry,
        ...     cpu_high_threshold=85.0,
        ...     cpu_low_threshold=20.0,
        ... )
        >>> orchestrator.start()
        >>> rec = orchestrator.get_latest_recommendation()
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        cpu_high_threshold: float = 85.0,
        cpu_low_threshold: float = 20.0,
        sample_interval_seconds: float = 5.0,
        history_size: int = 200,
    ) -> None:
        self._registry = registry
        self._cpu_high = cpu_high_threshold
        self._cpu_low = cpu_low_threshold
        self._sample_interval = sample_interval_seconds

        self._snapshots: deque[ResourceSnapshot] = deque(maxlen=history_size)
        self._recommendations: deque[ResourceRecommendation] = deque(maxlen=history_size)

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._queue_size_ref: int = 0  # updated by coordinator

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background resource monitoring thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="ResourceOrchestratorThread",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "ResourceOrchestrator started (interval=%.1fs, "
                "cpu_high=%.1f%%, cpu_low=%.1f%%).",
                self._sample_interval, self._cpu_high, self._cpu_low,
            )

    def stop(self) -> None:
        """Stop the monitoring thread."""
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._sample_interval * 2)
            self._thread = None
        logger.info("ResourceOrchestrator stopped.")

    def update_queue_size(self, size: int) -> None:
        """Update the current global queue size reference.

        Args:
            size: Current number of items in the global priority queue.
        """
        with self._lock:
            self._queue_size_ref = size

    # ------------------------------------------------------------------ #
    # Monitoring loop                                                      #
    # ------------------------------------------------------------------ #

    def _monitor_loop(self) -> None:
        """Background thread: samples metrics and generates recommendations."""
        while self._running:
            time.sleep(self._sample_interval)
            if not self._running:
                break
            self._sample_and_evaluate()

    def _sample_and_evaluate(self) -> None:
        """Take a resource snapshot and evaluate scaling recommendations."""
        workers = self._registry.get_all()
        if not workers:
            return

        # Build aggregate snapshot
        avg_cpu = sum(w.cpu_percent for w in workers) / len(workers)
        avg_ram = sum(w.ram_percent for w in workers) / len(workers)
        gpu_vals = [w.gpu_percent for w in workers if w.gpu_percent is not None]
        avg_gpu = sum(gpu_vals) / len(gpu_vals) if gpu_vals else None

        snapshot = ResourceSnapshot(
            node_id="cluster_aggregate",
            cpu_percent=avg_cpu,
            memory_percent=avg_ram,
            gpu_percent=avg_gpu,
        )

        with self._lock:
            self._snapshots.append(snapshot)
            current_queue = self._queue_size_ref

        # Generate recommendation
        rec = self._evaluate(
            workers=workers,
            avg_cpu=avg_cpu,
            avg_ram=avg_ram,
            queue_size=current_queue,
        )
        with self._lock:
            self._recommendations.append(rec)

    def _evaluate(
        self,
        workers: list[Any],
        avg_cpu: float,
        avg_ram: float,
        queue_size: int,
    ) -> ResourceRecommendation:
        """Evaluate cluster state and produce a recommendation.

        Args:
            workers: Current worker records.
            avg_cpu: Average CPU utilization across workers.
            avg_ram: Average RAM utilization across workers.
            queue_size: Current global queue depth.

        Returns:
            ResourceRecommendation with action, worker count, and strategy.
        """
        current_workers = len([w for w in workers if w.state != WorkerState.LOST])
        recommended_workers = current_workers
        action = "maintain"
        reason = "Cluster utilization is within normal range."
        strategy = "page_count_lpt"

        if avg_cpu >= self._cpu_high:
            recommended_workers = min(current_workers + 2, current_workers * 2)
            action = "scale_out"
            reason = (
                f"Average CPU ({avg_cpu:.1f}%) exceeds high threshold "
                f"({self._cpu_high:.1f}%). Adding workers."
            )
        elif avg_cpu <= self._cpu_low and queue_size == 0:
            recommended_workers = max(1, current_workers - 1)
            action = "scale_in"
            reason = (
                f"Average CPU ({avg_cpu:.1f}%) below low threshold "
                f"({self._cpu_low:.1f}%) and queue is empty. Reducing workers."
            )

        # Strategy recommendation based on queue depth
        if queue_size > current_workers * 10:
            strategy = "page_count_lpt"
            reason += " Large queue detected — LPT partitioning recommended."

        return ResourceRecommendation(
            recommended_workers=recommended_workers,
            current_workers=current_workers,
            action=action,
            scheduling_strategy=strategy,
            reason=reason,
            avg_cpu_percent=avg_cpu,
            avg_ram_percent=avg_ram,
            queue_size=queue_size,
        )

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_latest_recommendation(self) -> ResourceRecommendation | None:
        """Return the most recent scaling recommendation.

        Returns:
            Latest ResourceRecommendation, or None if no data yet.
        """
        with self._lock:
            if not self._recommendations:
                return None
            return self._recommendations[-1]

    def get_recent_recommendations(self, n: int = 10) -> list[ResourceRecommendation]:
        """Return the most recent *n* recommendations (newest first).

        Args:
            n: Maximum records to return.

        Returns:
            List of ResourceRecommendation objects.
        """
        with self._lock:
            recs = list(self._recommendations)
        return list(reversed(recs))[:n]

    def get_recent_snapshots(self, n: int = 20) -> list[ResourceSnapshot]:
        """Return the most recent *n* cluster snapshots.

        Args:
            n: Maximum snapshots to return.

        Returns:
            List of ResourceSnapshot objects (newest first).
        """
        with self._lock:
            snaps = list(self._snapshots)
        return list(reversed(snaps))[:n]

    def get_runtime_report(self) -> dict[str, Any]:
        """Generate a full runtime resource report.

        Returns:
            Dictionary with current stats, latest recommendation, and history.
        """
        stats = self._registry.get_statistics()
        latest_rec = self.get_latest_recommendation()
        with self._lock:
            snap_count = len(self._snapshots)
            rec_count = len(self._recommendations)

        return {
            "registry_stats": stats,
            "latest_recommendation": latest_rec.to_dict() if latest_rec else None,
            "total_snapshots_taken": snap_count,
            "total_recommendations_made": rec_count,
            "cpu_high_threshold": self._cpu_high,
            "cpu_low_threshold": self._cpu_low,
            "sample_interval_seconds": self._sample_interval,
            "running": self._running,
        }

    def __repr__(self) -> str:
        latest = self.get_latest_recommendation()
        action = latest.action if latest else "none"
        return (
            f"ResourceOrchestrator("
            f"running={self._running}, "
            f"latest_action={action!r}, "
            f"snapshots={len(self._snapshots)})"
        )
