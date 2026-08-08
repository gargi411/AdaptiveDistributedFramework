"""FailureRecoveryEngine — Implements the architecture §2.3 failure recovery flow.

Recovery Flow (from architecture_v2.0_locked.md §2.3):
    1. Worker Lost (e.g. Laptop 2 disconnects)
           ↓
    2. Detect via Heartbeat Timeout (→ HeartbeatMonitor fires on_timeout)
           ↓
    3. Return Unfinished Work Units (→ TaskDispatcher.recover_work_unit)
           ↓
    4. Re-insert into Priority Queue (→ PageCountPriorityQueue.reinsert)
           ↓
    5. Assign to Available Worker (→ TaskDispatcher.dispatch_next)

The FailureRecoveryEngine is the glue that executes steps 3–5 when
HeartbeatMonitor fires a timeout event.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from adaptive_framework.coordinator.failure_recovery.recovery_event import (
    RecoveryEvent,
    RecoveryEventType,
)
from adaptive_framework.coordinator.task_dispatcher.dispatcher import TaskDispatcher
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.core.constants import (
    DEFAULT_TASK_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY_SECONDS,
    ROOT_LOGGER_NAME,
)
from adaptive_framework.models.scheduling import PageWorkUnit

logger = logging.getLogger(ROOT_LOGGER_NAME + ".failure_recovery")


class FailureRecoveryEngine:
    """Recovers tasks from lost workers and re-queues them.

    This engine is triggered by the HeartbeatMonitor's on_timeout callback.
    It queries the TaskDispatcher for all tasks that were in-flight on the
    lost worker, re-inserts them into the priority queue with incremented
    retry_count, and fires appropriate RecoveryEvents.

    Work units that exceed ``max_retries`` are permanently failed and
    removed from the queue.

    Attributes:
        _registry: WorkerRegistry for worker state queries.
        _dispatcher: TaskDispatcher for task ownership and recovery.
        _max_retries: Maximum retry attempts per work unit.
        _reassignment_delay: Seconds to wait before re-queuing (back-off).
        _history: Bounded deque of RecoveryEvent records.
        _retry_counts: work_unit_id → retry count tracking.

    Example:
        >>> engine = FailureRecoveryEngine(
        ...     registry=registry,
        ...     dispatcher=dispatcher,
        ...     max_retries=3,
        ... )
        >>> # Called by HeartbeatMonitor:
        >>> engine.handle_worker_lost("worker_id_42")
    """

    _MAX_HISTORY = 1000

    def __init__(
        self,
        registry: WorkerRegistry,
        dispatcher: TaskDispatcher,
        max_retries: int = DEFAULT_TASK_RETRY_ATTEMPTS,
        reassignment_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        history_size: int = 500,
    ) -> None:
        """Initialise the FailureRecoveryEngine.

        Args:
            registry: WorkerRegistry for worker state queries.
            dispatcher: TaskDispatcher for task recovery operations.
            max_retries: Maximum times a work unit may be retried.
            reassignment_delay_seconds: Seconds to wait before re-queuing.
            history_size: Maximum recovery events to retain.
        """
        self._registry = registry
        self._dispatcher = dispatcher
        self._max_retries = max_retries
        self._reassignment_delay = reassignment_delay_seconds

        self._history: deque[RecoveryEvent] = deque(maxlen=history_size)
        self._retry_counts: dict[str, int] = {}

        self._total_workers_lost: int = 0
        self._total_tasks_recovered: int = 0
        self._total_tasks_permanently_failed: int = 0
        self._total_workers_recovered: int = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Primary entry point (called by HeartbeatMonitor.on_timeout)         #
    # ------------------------------------------------------------------ #

    def handle_worker_lost(self, worker_id: str) -> list[RecoveryEvent]:
        """Handle a worker LOST event: recover all its in-flight tasks.

        This method is safe to call from any thread (including the
        HeartbeatMonitor's background thread).

        Args:
            worker_id: ID of the worker that was declared LOST.

        Returns:
            List of RecoveryEvent objects generated during this recovery.
        """
        events: list[RecoveryEvent] = []

        with self._lock:
            self._total_workers_lost += 1

        # Emit WORKER_LOST event
        lost_event = RecoveryEvent(
            event_type=RecoveryEventType.WORKER_LOST,
            worker_id=worker_id,
            message=f"Worker '{worker_id}' declared LOST. Initiating task recovery.",
        )
        self._append_event(lost_event)
        events.append(lost_event)
        logger.warning("Recovery initiated for lost worker '%s'.", worker_id[:8])

        # Get all tasks that were running on this worker
        active_task_ids = self._dispatcher.get_active_tasks_for_worker(worker_id)

        if not active_task_ids:
            logger.info("No active tasks to recover from worker '%s'.", worker_id[:8])
            return events

        recovered_ids: list[str] = []
        permanently_failed_ids: list[str] = []

        for wu_id in active_task_ids:
            with self._lock:
                retry_count = self._retry_counts.get(wu_id, 0) + 1
                self._retry_counts[wu_id] = retry_count

            if retry_count > self._max_retries:
                # Permanently fail this work unit
                self._dispatcher.report_failed(
                    wu_id, f"Max retries ({self._max_retries}) exceeded after worker loss."
                )
                permanently_failed_ids.append(wu_id)

                exhaust_event = RecoveryEvent(
                    event_type=RecoveryEventType.RETRY_EXHAUSTED,
                    worker_id=worker_id,
                    work_unit_ids=[wu_id],
                    retry_count=retry_count,
                    message=(
                        f"WorkUnit '{wu_id}' permanently failed "
                        f"(max_retries={self._max_retries} exceeded)."
                    ),
                )
                self._append_event(exhaust_event)
                events.append(exhaust_event)
                logger.error(
                    "WorkUnit '%s' permanently failed (retries=%d).",
                    wu_id[:8], retry_count,
                )
            else:
                # Build a recovery stub — the real PageWorkUnit will be re-inserted
                # by the dispatcher using its internal record
                stub = PageWorkUnit(
                    document_id=f"recovered_{wu_id[:8]}",
                    file_path="/recovered",
                    start_page=1,
                    end_page=1,
                    work_unit_id=wu_id,
                    priority=0,
                    retry_count=retry_count,
                )
                # Small delay to avoid thundering herd
                time.sleep(self._reassignment_delay)
                self._dispatcher.recover_work_unit(wu_id, stub)
                recovered_ids.append(wu_id)

                retry_event = RecoveryEvent(
                    event_type=RecoveryEventType.RETRY_SCHEDULED,
                    worker_id=worker_id,
                    work_unit_ids=[wu_id],
                    retry_count=retry_count,
                    message=(
                        f"WorkUnit '{wu_id}' re-queued for retry "
                        f"(attempt {retry_count}/{self._max_retries})."
                    ),
                )
                self._append_event(retry_event)
                events.append(retry_event)

        with self._lock:
            self._total_tasks_recovered += len(recovered_ids)
            self._total_tasks_permanently_failed += len(permanently_failed_ids)

        if recovered_ids:
            recovery_event = RecoveryEvent(
                event_type=RecoveryEventType.TASKS_RECOVERED,
                worker_id=worker_id,
                work_unit_ids=recovered_ids,
                message=(
                    f"{len(recovered_ids)} task(s) recovered from worker '{worker_id}' "
                    f"and re-inserted into the priority queue."
                ),
            )
            self._append_event(recovery_event)
            events.append(recovery_event)
            logger.info(
                "%d task(s) recovered from worker '%s'.",
                len(recovered_ids), worker_id[:8],
            )

        return events

    def handle_worker_reconnected(self, worker_id: str) -> RecoveryEvent:
        """Handle a previously LOST worker reconnecting.

        Args:
            worker_id: Worker that sent a new heartbeat.

        Returns:
            WORKER_RECOVERED RecoveryEvent.
        """
        with self._lock:
            self._total_workers_recovered += 1

        event = RecoveryEvent(
            event_type=RecoveryEventType.WORKER_RECOVERED,
            worker_id=worker_id,
            message=f"Worker '{worker_id}' reconnected and is now available.",
        )
        self._append_event(event)
        logger.info("Worker '%s' recovered.", worker_id[:8])
        return event

    def handle_graceful_shutdown(self, worker_id: str) -> RecoveryEvent:
        """Handle a planned worker shutdown (not a failure).

        Args:
            worker_id: Worker shutting down gracefully.

        Returns:
            GRACEFUL_SHUTDOWN RecoveryEvent.
        """
        event = RecoveryEvent(
            event_type=RecoveryEventType.GRACEFUL_SHUTDOWN,
            worker_id=worker_id,
            message=f"Worker '{worker_id}' gracefully shut down.",
        )
        self._append_event(event)
        self._registry.remove(worker_id)
        logger.info("Worker '%s' gracefully removed from registry.", worker_id[:8])
        return event

    # ------------------------------------------------------------------ #
    # History                                                              #
    # ------------------------------------------------------------------ #

    def _append_event(self, event: RecoveryEvent) -> None:
        """Append an event to the history (thread-safe).

        Args:
            event: RecoveryEvent to append.
        """
        with self._lock:
            self._history.append(event)

    def get_recent_events(self, n: int = 50) -> list[RecoveryEvent]:
        """Return the most recent *n* recovery events (newest first).

        Args:
            n: Maximum events to return.

        Returns:
            List of RecoveryEvent objects.
        """
        with self._lock:
            events = list(self._history)
        return list(reversed(events))[:n]

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_statistics(self) -> dict[str, Any]:
        """Return recovery engine statistics.

        Returns:
            Dictionary with counters for losses, recoveries, and failures.
        """
        with self._lock:
            return {
                "total_workers_lost": self._total_workers_lost,
                "total_tasks_recovered": self._total_tasks_recovered,
                "total_tasks_permanently_failed": self._total_tasks_permanently_failed,
                "total_workers_recovered": self._total_workers_recovered,
                "max_retries": self._max_retries,
                "history_size": len(self._history),
                "pending_retries": len(self._retry_counts),
            }

    def __repr__(self) -> str:
        return (
            f"FailureRecoveryEngine("
            f"workers_lost={self._total_workers_lost}, "
            f"tasks_recovered={self._total_tasks_recovered}, "
            f"permanently_failed={self._total_tasks_permanently_failed})"
        )
