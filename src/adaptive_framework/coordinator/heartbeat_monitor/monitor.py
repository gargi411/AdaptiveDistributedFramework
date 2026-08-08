"""HeartbeatMonitor — Worker liveness detection and timeout management.

The HeartbeatMonitor runs in a dedicated thread and:
  1. Accepts incoming heartbeat payloads from workers.
  2. Tracks the last-seen timestamp per worker.
  3. Detects timeouts (configurable threshold) and emits TIMEOUT / DISCONNECTED events.
  4. Emits RECONNECTED events when a LOST worker sends a heartbeat again.
  5. Maintains a bounded event history for the Engineering Dashboard.

It does NOT directly modify the WorkerRegistry — instead it fires callbacks
that the DistributedCoordinator subscribes to, keeping concerns separated.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

from adaptive_framework.coordinator.heartbeat_monitor.heartbeat_event import (
    HeartbeatEvent,
    HeartbeatEventType,
    HeartbeatPayload,
)
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.core.constants import ROOT_LOGGER_NAME
from adaptive_framework.models.runtime import WorkerState

logger = logging.getLogger(ROOT_LOGGER_NAME + ".heartbeat_monitor")

# Type alias for timeout / reconnect callbacks
_TimeoutCallback = Callable[[str], None]
_ReconnectCallback = Callable[[str], None]


class HeartbeatMonitor:
    """Thread-safe monitor that detects worker timeouts via periodic polling.

    Design:
        A single background thread (``_poll_thread``) wakes every
        ``_check_interval`` seconds and compares each worker's
        ``last_seen_at`` against the current time.
        Workers that exceed ``_timeout_seconds`` are declared LOST.

    Callbacks:
        ``on_timeout``: Called with ``worker_id`` when a timeout is detected.
        ``on_reconnect``: Called with ``worker_id`` when a LOST worker returns.

    Args:
        registry: The WorkerRegistry to monitor.
        timeout_seconds: Seconds without a heartbeat before a worker is
            declared LOST. Default: 15.0.
        check_interval_seconds: How often the poll thread wakes. Default: 2.0.
        history_size: Maximum number of events retained. Default: 500.
        on_timeout: Optional callback invoked when a worker times out.
        on_reconnect: Optional callback invoked when a LOST worker reconnects.

    Example:
        >>> monitor = HeartbeatMonitor(
        ...     registry=registry,
        ...     timeout_seconds=15.0,
        ...     on_timeout=lambda wid: coordinator.handle_timeout(wid),
        ... )
        >>> monitor.start()
        >>> monitor.record_heartbeat(payload)  # called by coordinator per heartbeat
        >>> monitor.stop()
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        timeout_seconds: float = 15.0,
        check_interval_seconds: float = 2.0,
        history_size: int = 500,
        on_timeout: _TimeoutCallback | None = None,
        on_reconnect: _ReconnectCallback | None = None,
    ) -> None:
        """Initialise the HeartbeatMonitor.

        Args:
            registry: WorkerRegistry to monitor.
            timeout_seconds: Timeout threshold in seconds.
            check_interval_seconds: Poll interval in seconds.
            history_size: Maximum events in history ring buffer.
            on_timeout: Callback(worker_id) on timeout detection.
            on_reconnect: Callback(worker_id) on worker reconnection.
        """
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._check_interval = check_interval_seconds
        self._on_timeout = on_timeout
        self._on_reconnect = on_reconnect

        # last heartbeat timestamps per worker (monotonic)
        self._last_seen: dict[str, float] = {}
        # set of worker_ids currently in LOST state (per monitor's view)
        self._lost_workers: set[str] = set()

        # Event history ring buffer
        self._history: deque[HeartbeatEvent] = deque(maxlen=history_size)

        # Thread control
        self._lock = threading.Lock()
        self._running = False
        self._poll_thread: threading.Thread | None = None

        # Counters
        self._total_heartbeats: int = 0
        self._total_timeouts: int = 0
        self._total_reconnects: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background polling thread.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
            self._poll_thread = threading.Thread(
                target=self._poll_loop,
                name="HeartbeatMonitorThread",
                daemon=True,
            )
            self._poll_thread.start()
            logger.info(
                "HeartbeatMonitor started (timeout=%.1fs, interval=%.1fs).",
                self._timeout_seconds,
                self._check_interval,
            )

    def stop(self) -> None:
        """Stop the background polling thread gracefully."""
        with self._lock:
            self._running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=self._check_interval * 2)
            self._poll_thread = None
        logger.info("HeartbeatMonitor stopped.")

    @property
    def is_running(self) -> bool:
        """Return True if the monitor thread is active.

        Returns:
            True when the poll thread is running.
        """
        return self._running

    # ------------------------------------------------------------------ #
    # Heartbeat ingestion                                                  #
    # ------------------------------------------------------------------ #

    def record_heartbeat(self, payload: HeartbeatPayload) -> None:
        """Process an incoming heartbeat from a worker.

        Updates the worker's last-seen timestamp, applies the payload to
        the registry, and emits a RECONNECTED event if the worker was LOST.

        Args:
            payload: HeartbeatPayload received from the worker.
        """
        now = time.monotonic()
        worker_id = payload.worker_id

        # Parse state
        try:
            state = WorkerState(payload.state)
        except ValueError:
            state = WorkerState.ACTIVE

        with self._lock:
            was_lost = worker_id in self._lost_workers
            self._last_seen[worker_id] = now
            self._total_heartbeats += 1

        # Apply to registry
        self._registry.update_heartbeat(
            worker_id=worker_id,
            cpu_percent=payload.cpu_percent,
            ram_percent=payload.ram_percent,
            gpu_percent=payload.gpu_percent,
            queue_depth=payload.queue_depth,
            current_task_id=payload.current_task_id,
            state=state,
        )

        if was_lost:
            self._handle_reconnect(worker_id, payload)
        else:
            event = HeartbeatEvent(
                event_type=HeartbeatEventType.ALIVE,
                worker_id=worker_id,
                message=f"Heartbeat OK (cpu={payload.cpu_percent:.1f}%, "
                        f"ram={payload.ram_percent:.1f}%, "
                        f"queue={payload.queue_depth}).",
                payload=payload,
            )
            self._append_event(event)

    # ------------------------------------------------------------------ #
    # Poll loop                                                            #
    # ------------------------------------------------------------------ #

    def _poll_loop(self) -> None:
        """Background thread: checks all workers for heartbeat timeout."""
        while self._running:
            time.sleep(self._check_interval)
            if not self._running:
                break
            self._check_all_workers()

    def _check_all_workers(self) -> None:
        """Scan all registered workers for timeout.

        Called periodically by the poll thread.
        """
        now = time.monotonic()
        workers = self._registry.get_all()

        for worker in workers:
            worker_id = worker.worker_id
            with self._lock:
                last = self._last_seen.get(worker_id)

            if last is None:
                # Worker registered but never sent a heartbeat yet; skip
                continue

            elapsed = now - last
            with self._lock:
                is_already_lost = worker_id in self._lost_workers

            if elapsed > self._timeout_seconds and not is_already_lost:
                self._handle_timeout(worker_id, elapsed)

    def _handle_timeout(self, worker_id: str, elapsed: float) -> None:
        """Declare a worker as LOST and fire callbacks.

        Args:
            worker_id: The timed-out worker.
            elapsed: Seconds since last heartbeat.
        """
        with self._lock:
            self._lost_workers.add(worker_id)
            self._total_timeouts += 1

        self._registry.mark_lost(worker_id)

        event = HeartbeatEvent(
            event_type=HeartbeatEventType.DISCONNECTED,
            worker_id=worker_id,
            message=(
                f"Worker '{worker_id}' declared LOST "
                f"(no heartbeat for {elapsed:.1f}s; threshold={self._timeout_seconds}s)."
            ),
        )
        self._append_event(event)
        logger.warning("Worker '%s' declared LOST (%.1fs elapsed).", worker_id, elapsed)

        if self._on_timeout is not None:
            try:
                self._on_timeout(worker_id)
            except Exception as exc:
                logger.error("on_timeout callback raised: %s", exc)

    def _handle_reconnect(self, worker_id: str, payload: HeartbeatPayload) -> None:
        """Handle a heartbeat from a previously LOST worker.

        Args:
            worker_id: The reconnecting worker.
            payload: The reconnect heartbeat payload.
        """
        with self._lock:
            self._lost_workers.discard(worker_id)
            self._total_reconnects += 1

        self._registry.mark_recovered(worker_id)

        event = HeartbeatEvent(
            event_type=HeartbeatEventType.RECONNECTED,
            worker_id=worker_id,
            message=f"Worker '{worker_id}' reconnected.",
            payload=payload,
        )
        self._append_event(event)
        logger.info("Worker '%s' reconnected.", worker_id)

        if self._on_reconnect is not None:
            try:
                self._on_reconnect(worker_id)
            except Exception as exc:
                logger.error("on_reconnect callback raised: %s", exc)

    # ------------------------------------------------------------------ #
    # History                                                              #
    # ------------------------------------------------------------------ #

    def _append_event(self, event: HeartbeatEvent) -> None:
        """Append an event to the history ring buffer (thread-safe).

        Args:
            event: Event to append.
        """
        with self._lock:
            self._history.append(event)

    def get_recent_events(self, n: int = 50) -> list[HeartbeatEvent]:
        """Return the most recent *n* heartbeat events.

        Args:
            n: Maximum number of events to return.

        Returns:
            List of HeartbeatEvent objects, newest first.
        """
        with self._lock:
            events = list(self._history)
        return list(reversed(events))[:n]

    def get_lost_workers(self) -> set[str]:
        """Return the set of worker IDs currently considered LOST.

        Returns:
            Frozen copy of the lost worker set.
        """
        with self._lock:
            return set(self._lost_workers)

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_statistics(self) -> dict[str, Any]:
        """Return heartbeat monitor performance statistics.

        Returns:
            Dictionary with total heartbeats, timeouts, reconnects, and
            current lost worker count.
        """
        with self._lock:
            return {
                "total_heartbeats": self._total_heartbeats,
                "total_timeouts": self._total_timeouts,
                "total_reconnects": self._total_reconnects,
                "current_lost_workers": len(self._lost_workers),
                "history_size": len(self._history),
                "timeout_threshold_s": self._timeout_seconds,
                "check_interval_s": self._check_interval,
                "running": self._running,
            }

    def __repr__(self) -> str:
        return (
            f"HeartbeatMonitor(running={self._running}, "
            f"timeout={self._timeout_seconds}s, "
            f"lost={len(self._lost_workers)}, "
            f"heartbeats={self._total_heartbeats})"
        )
