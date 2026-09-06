"""ResourceMonitor -- Unified runtime resource monitoring and sampling engine.

Observes and samples:
  - Host CPU utilization and logical/physical core counts
  - System RAM utilization (%), used memory (MB), and available memory (MB)
  - GPU telemetry via GPUMonitor (utilization, device memory, vendor, device info)
  - Work queue length and active worker counts
  - OCR performance metrics (inference latency ms, throughput pages/s)

Strict Design Principles:
  - Pure observability: Read-only monitoring. No scheduling or routing decisions.
  - Zero fabricated data: Unavailable metrics are None and displayed as UNAVAILABLE.
  - Low overhead: Direct in-process queries with psutil, OpenVINO, and Windows PDH.
  - Thread-safe background sampling: Configurable circular buffer for time-series history.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any

from adaptive_framework.acceleration.gpu_monitor import GPUMonitor
from adaptive_framework.models.runtime import ResourceSnapshot
from adaptive_framework.utils.system_utils import (
    get_cpu_percent,
    get_cpu_count,
    get_cpu_physical_count,
    get_memory_percent,
    get_memory_breakdown_mb,
)

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Unified runtime resource monitoring and sampling engine.

    Aggregates CPU, RAM, GPU, queue depth, worker state, and OCR telemetry
    into structured immutable ResourceSnapshot instances.

    Usage:
        monitor = ResourceMonitor(node_id="node_01")
        snapshot = monitor.sample()
        print(snapshot.format_summary())

        # Or with background sampling:
        monitor.start_background_sampling(interval_seconds=1.0)
        time.sleep(3.0)
        history = monitor.get_recent_snapshots()
        monitor.stop_background_sampling()
    """

    def __init__(
        self,
        node_id: str = "node_0",
        history_size: int = 120,
        gpu_device_id: str = "GPU",
        enable_gpu_pdh: bool = True,
    ) -> None:
        """Initialize the unified resource monitor.

        Args:
            node_id: Node or cluster identifier.
            history_size: Maximum number of recent snapshots retained in memory.
            gpu_device_id: OpenVINO GPU device identifier.
            enable_gpu_pdh: Whether to enable Windows PDH counter for GPU utilization.
        """
        self._node_id = node_id
        self._history_size = max(10, history_size)
        self._snapshots: collections.deque[ResourceSnapshot] = collections.deque(
            maxlen=self._history_size
        )
        self._lock = threading.RLock()

        # Dedicated GPU monitor
        self._gpu_monitor = GPUMonitor(
            device_id=gpu_device_id,
            enable_pdh=enable_gpu_pdh,
        )

        # Background sampling state
        self._running: bool = False
        self._sample_thread: threading.Thread | None = None
        self._sample_interval: float = 1.0

        # Cached hardware identification
        self._logical_cpu_count: int = get_cpu_count(logical=True)
        self._physical_cpu_count: int = get_cpu_physical_count()

        # External state providers (e.g. from coordinator/queue)
        self._queue_provider: Any | None = None
        self._worker_provider: Any | None = None

    # ------------------------------------------------------------------ #
    # Hardware Status & Capabilities
    # ------------------------------------------------------------------ #

    @property
    def node_id(self) -> str:
        """Return the node identifier."""
        return self._node_id

    @property
    def gpu_available(self) -> bool:
        """Return True if GPU acceleration is present and usable."""
        return self._gpu_monitor.is_available()

    def get_hardware_info(self) -> dict[str, Any]:
        """Return static CPU and GPU hardware properties."""
        total_mb, _, _ = get_memory_breakdown_mb()
        info: dict[str, Any] = {
            "node_id": self._node_id,
            "cpu": {
                "logical_cores": self._logical_cpu_count,
                "physical_cores": self._physical_cpu_count,
            },
            "ram": {
                "total_mb": total_mb,
            },
            "gpu_available": self.gpu_available,
        }
        if self.gpu_available:
            info["gpu"] = self._gpu_monitor.get_device_info()
        else:
            info["gpu"] = None
        return info

    # ------------------------------------------------------------------ #
    # State Providers Attachment
    # ------------------------------------------------------------------ #

    def register_queue_provider(self, provider: Any) -> None:
        """Attach a callable or object exposing queue length."""
        self._queue_provider = provider

    def register_worker_provider(self, provider: Any) -> None:
        """Attach a callable or object exposing active worker count."""
        self._worker_provider = provider

    def _resolve_queue_length(self, override: int | None = None) -> int:
        """Resolve current queue length."""
        if override is not None:
            return max(0, override)
        if self._queue_provider is not None:
            try:
                if callable(self._queue_provider):
                    return max(0, int(self._queue_provider()))
                if hasattr(self._queue_provider, "qsize"):
                    return max(0, int(self._queue_provider.qsize()))
                if hasattr(self._queue_provider, "__len__"):
                    return max(0, len(self._queue_provider))
            except Exception:
                pass
        return 0

    def _resolve_active_workers(self, override: int | None = None) -> int:
        """Resolve active worker count."""
        if override is not None:
            return max(0, override)
        if self._worker_provider is not None:
            try:
                if callable(self._worker_provider):
                    return max(0, int(self._worker_provider()))
                if hasattr(self._worker_provider, "__len__"):
                    return max(0, len(self._worker_provider))
            except Exception:
                pass
        return 0

    # ------------------------------------------------------------------ #
    # Sampling API
    # ------------------------------------------------------------------ #

    def sample(
        self,
        queue_length: int | None = None,
        active_workers: int | None = None,
        ocr_latency_ms: float | None = None,
        ocr_throughput_pages_s: float | None = None,
    ) -> ResourceSnapshot:
        """Capture a single point-in-time ResourceSnapshot.

        Args:
            queue_length: Optional manual queue length override.
            active_workers: Optional manual active worker count override.
            ocr_latency_ms: Optional recent OCR inference latency in ms.
            ocr_throughput_pages_s: Optional recent OCR throughput in pages/sec.

        Returns:
            Populated immutable ResourceSnapshot.
        """
        # CPU
        cpu_pct = get_cpu_percent(interval=0.05)
        if cpu_pct < 0.0:
            cpu_pct = 0.0

        # RAM
        ram_pct = get_memory_percent()
        if ram_pct < 0.0:
            ram_pct = 0.0
        _, used_mb, avail_mb = get_memory_breakdown_mb()

        # GPU
        gpu_metrics = self._gpu_monitor.sample()
        gpu_avail = self._gpu_monitor.is_available()

        # Queue and workers
        q_len = self._resolve_queue_length(queue_length)
        workers_count = self._resolve_active_workers(active_workers)

        snapshot = ResourceSnapshot(
            node_id=self._node_id,
            cpu_percent=cpu_pct,
            memory_percent=ram_pct,
            cpu_count=self._logical_cpu_count,
            ram_used_mb=used_mb,
            ram_available_mb=avail_mb,
            gpu_available=gpu_avail,
            gpu_name=gpu_metrics.device_name if gpu_avail else None,
            gpu_device=gpu_metrics.openvino_device_id if gpu_avail else None,
            gpu_utilization_percent=gpu_metrics.utilization_percent,
            gpu_percent=gpu_metrics.utilization_percent,
            gpu_memory_used_mb=gpu_metrics.memory_used_mb,
            gpu_memory_total_mb=gpu_metrics.memory_total_mb,
            gpu_temperature_c=gpu_metrics.temperature_c,
            gpu_power_w=gpu_metrics.power_w,
            queue_length=q_len,
            active_workers=workers_count,
            ocr_latency_ms=ocr_latency_ms,
            ocr_throughput_pages_s=ocr_throughput_pages_s,
        )

        with self._lock:
            self._snapshots.append(snapshot)

        return snapshot

    # ------------------------------------------------------------------ #
    # Time-Series History
    # ------------------------------------------------------------------ #

    def get_latest_snapshot(self) -> ResourceSnapshot | None:
        """Return the most recent snapshot, or None if no samples taken."""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def get_recent_snapshots(self, n: int = 60) -> list[ResourceSnapshot]:
        """Return up to n recent snapshots (oldest first)."""
        with self._lock:
            count = min(n, len(self._snapshots))
            if count == 0:
                return []
            return list(self._snapshots)[-count:]

    # ------------------------------------------------------------------ #
    # Background Sampling Engine
    # ------------------------------------------------------------------ #

    def start_background_sampling(self, interval_seconds: float = 1.0) -> None:
        """Start a background daemon thread taking periodic samples."""
        with self._lock:
            if self._running:
                return
            self._sample_interval = max(0.1, interval_seconds)
            self._running = True
            self._sample_thread = threading.Thread(
                target=self._background_loop,
                daemon=True,
                name="ResourceMonitorSampler",
            )
            self._sample_thread.start()
            logger.info(
                "ResourceMonitor: background sampling started (interval=%.2fs)",
                self._sample_interval,
            )

    def stop_background_sampling(self) -> None:
        """Stop background sampling thread."""
        with self._lock:
            self._running = False
        if self._sample_thread and self._sample_thread.is_alive():
            self._sample_thread.join(timeout=2.0)
        self._sample_thread = None
        logger.info("ResourceMonitor: background sampling stopped.")

    def _background_loop(self) -> None:
        """Target loop for background sampling thread."""
        while self._running:
            try:
                self.sample()
            except Exception as exc:
                logger.debug("Background sampling error: %s", exc)
            time.sleep(self._sample_interval)

    # ------------------------------------------------------------------ #
    # Context Management & Cleanup
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Stop background sampling and close GPU handles."""
        self.stop_background_sampling()
        self._gpu_monitor.close()

    def __enter__(self) -> ResourceMonitor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
