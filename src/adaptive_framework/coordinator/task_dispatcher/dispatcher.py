"""TaskDispatcher — Assigns PageWorkUnits from the priority queue to workers.

The TaskDispatcher is the bridge between the Adaptive Scheduler (Phase 2A)
and the cluster workers. It:
  1. Receives partitioned work units via the PageCountPriorityQueue.
  2. Selects the least-loaded available worker (fewest active tasks).
  3. Dispatches the work unit to that worker.
  4. Tracks ownership: which worker holds which work unit.
  5. Records assignment history for the dashboard and evaluation.
  6. Exposes retry logic for recovered work units.
  7. Instruments all dispatch operations with perf_counter for
     scheduler_overhead measurement (architecture §4.2).

Separation of concerns:
    - TaskDispatcher assigns; it does NOT process.
    - Failure recovery is handled by FailureRecoveryEngine.
    - Work stealing is handled by WorkStealingCoordinator.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict, deque
from typing import Any, Callable

from adaptive_framework.coordinator.task_dispatcher.assignment_record import (
    AssignmentRecord,
    AssignmentStatus,
)
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.core.constants import ROOT_LOGGER_NAME
from adaptive_framework.models.scheduling import PageWorkUnit
from adaptive_framework.scheduler.priority_queue import PageCountPriorityQueue

logger = logging.getLogger(ROOT_LOGGER_NAME + ".task_dispatcher")

# Callback type: called when dispatcher pops a task (for progress tracking)
_DispatchCallback = Callable[[str, str], None]  # (work_unit_id, worker_id)


class TaskDispatcher:
    """Assigns work units from the priority queue to available workers.

    Thread-safe. Intended to be called from the DistributedCoordinator's
    main dispatch loop.

    Attributes:
        _registry: Source of truth for worker availability.
        _queue: PageCountPriorityQueue supplying work units.
        _active: Mapping work_unit_id → worker_id (in-flight tasks).
        _history: Bounded deque of AssignmentRecord (all assignments ever).
        _scheduler_time_total: Cumulative time spent inside dispatch logic (§4.2).
        _lock: Thread safety lock.

    Example:
        >>> dispatcher = TaskDispatcher(registry=registry, queue=queue)
        >>> assigned = dispatcher.dispatch_next()
        >>> print(assigned)  # (work_unit, worker_record) or None
    """

    _MAX_HISTORY = 2000  # ring-buffer limit for assignment history

    def __init__(
        self,
        registry: WorkerRegistry,
        queue: PageCountPriorityQueue,
        on_dispatch: _DispatchCallback | None = None,
    ) -> None:
        """Initialise the TaskDispatcher.

        Args:
            registry: WorkerRegistry for worker availability queries.
            queue: PageCountPriorityQueue to pop work units from.
            on_dispatch: Optional callback(work_unit_id, worker_id) called
                after each successful dispatch.
        """
        self._registry = registry
        self._queue = queue
        self._on_dispatch = on_dispatch

        # work_unit_id → worker_id for in-flight tasks
        self._active: dict[str, str] = {}
        # worker_id → count of currently active work units
        self._worker_load: dict[str, int] = defaultdict(int)
        # Assignment history (newest-last deque)
        self._history: deque[AssignmentRecord] = deque(maxlen=self._MAX_HISTORY)
        # Cumulative scheduler time for overhead calculation (§4.2)
        self._scheduler_time_total: float = 0.0
        self._total_dispatched: int = 0
        self._total_completed: int = 0
        self._total_failed: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #

    def dispatch_next(self) -> tuple[PageWorkUnit, str] | None:
        """Pop the highest-priority work unit and assign it to the best worker.

        The "best" worker is the available worker with the fewest active tasks
        (ties broken by worker_id lexicographic order for determinism).

        This method is fully instrumented for scheduler_overhead measurement.

        Returns:
            (PageWorkUnit, worker_id) tuple if an assignment was made,
            or None if the queue is empty or no workers are available.
        """
        t_start = time.perf_counter()
        result = self._do_dispatch()
        t_end = time.perf_counter()
        with self._lock:
            self._scheduler_time_total += (t_end - t_start)
        return result

    def _do_dispatch(self) -> tuple[PageWorkUnit, str] | None:
        """Internal dispatch without timing instrumentation.

        Returns:
            (PageWorkUnit, worker_id) or None.
        """
        # Peek — don't pop until we know we have a worker
        if self._queue.is_empty():
            return None

        available = self._registry.get_available()
        if not available:
            return None

        # Select least-loaded worker
        def load(w: Any) -> tuple[int, str]:
            with self._lock:
                return (self._worker_load.get(w.worker_id, 0), w.worker_id)

        best_worker = min(available, key=load)

        # Now pop from the queue
        work_unit = self._queue.pop()
        if work_unit is None:
            # Race: another thread consumed the item
            return None

        worker_id = best_worker.worker_id

        # Record ownership
        with self._lock:
            self._active[work_unit.work_unit_id] = worker_id
            self._worker_load[worker_id] = self._worker_load.get(worker_id, 0) + 1
            self._total_dispatched += 1

        # Build assignment record
        rec = AssignmentRecord(
            work_unit_id=work_unit.work_unit_id,
            worker_id=worker_id,
            document_id=work_unit.document_id,
            page_count=work_unit.page_count,
        )
        with self._lock:
            self._history.append(rec)

        logger.debug(
            "Dispatched wu='%s' (pages=%d) → worker='%s'.",
            work_unit.work_unit_id[:8],
            work_unit.page_count,
            worker_id[:8],
        )

        if self._on_dispatch is not None:
            try:
                self._on_dispatch(work_unit.work_unit_id, worker_id)
            except Exception as exc:
                logger.warning("on_dispatch callback raised: %s", exc)

        return work_unit, worker_id

    def dispatch_batch(self, max_assignments: int = 10) -> list[tuple[PageWorkUnit, str]]:
        """Dispatch up to *max_assignments* work units in one call.

        Args:
            max_assignments: Maximum number of items to dispatch.

        Returns:
            List of (PageWorkUnit, worker_id) tuples.
        """
        results: list[tuple[PageWorkUnit, str]] = []
        for _ in range(max_assignments):
            result = self.dispatch_next()
            if result is None:
                break
            results.append(result)
        return results

    # ------------------------------------------------------------------ #
    # Completion / Failure reporting                                       #
    # ------------------------------------------------------------------ #

    def report_completed(self, work_unit_id: str) -> bool:
        """Mark a work unit as successfully completed.

        Args:
            work_unit_id: ID of the completed PageWorkUnit.

        Returns:
            True if the work unit was found in active assignments.
        """
        with self._lock:
            worker_id = self._active.pop(work_unit_id, None)
            if worker_id is None:
                return False
            self._worker_load[worker_id] = max(
                0, self._worker_load.get(worker_id, 1) - 1
            )
            self._total_completed += 1

        # Update history record
        self._update_history_status(work_unit_id, AssignmentStatus.COMPLETED)
        self._registry.record_completion(worker_id)
        logger.debug("WorkUnit '%s' completed by worker '%s'.", work_unit_id[:8], worker_id[:8])
        return True

    def report_failed(self, work_unit_id: str, reason: str) -> bool:
        """Mark a work unit as failed.

        Args:
            work_unit_id: ID of the failed PageWorkUnit.
            reason: Human-readable failure description.

        Returns:
            True if the work unit was found in active assignments.
        """
        with self._lock:
            worker_id = self._active.pop(work_unit_id, None)
            if worker_id is None:
                return False
            self._worker_load[worker_id] = max(
                0, self._worker_load.get(worker_id, 1) - 1
            )
            self._total_failed += 1

        self._update_history_status(work_unit_id, AssignmentStatus.FAILED, reason)
        self._registry.record_failure(worker_id)
        logger.warning(
            "WorkUnit '%s' FAILED (worker='%s', reason=%r).",
            work_unit_id[:8], worker_id[:8], reason,
        )
        return True

    def recover_work_unit(
        self, work_unit_id: str, work_unit: PageWorkUnit
    ) -> bool:
        """Re-insert a recovered work unit back into the priority queue.

        Called by FailureRecoveryEngine when a worker is declared LOST.

        Args:
            work_unit_id: ID of the work unit to recover.
            work_unit: The actual PageWorkUnit object.

        Returns:
            True if the work unit was successfully re-queued.
        """
        with self._lock:
            worker_id = self._active.pop(work_unit_id, None)
            if worker_id is not None:
                self._worker_load[worker_id] = max(
                    0, self._worker_load.get(worker_id, 1) - 1
                )

        self._update_history_status(work_unit_id, AssignmentStatus.RECOVERED)
        self._queue.reinsert(work_unit)
        logger.info(
            "WorkUnit '%s' recovered and re-queued (was on worker '%s').",
            work_unit_id[:8],
            (worker_id or "unknown")[:8],
        )
        return True

    # ------------------------------------------------------------------ #
    # Ownership queries                                                    #
    # ------------------------------------------------------------------ #

    def get_worker_for_task(self, work_unit_id: str) -> str | None:
        """Return the worker_id currently assigned to a work unit.

        Args:
            work_unit_id: Work unit to look up.

        Returns:
            worker_id or None if not in-flight.
        """
        with self._lock:
            return self._active.get(work_unit_id)

    def get_active_tasks_for_worker(self, worker_id: str) -> list[str]:
        """Return all work_unit_ids currently assigned to a worker.

        Args:
            worker_id: Worker to query.

        Returns:
            List of active work_unit_ids on that worker.
        """
        with self._lock:
            return [
                wu_id for wu_id, wid in self._active.items()
                if wid == worker_id
            ]

    @property
    def active_count(self) -> int:
        """Return number of work units currently in-flight.

        Returns:
            Count of active assignments.
        """
        with self._lock:
            return len(self._active)

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    @property
    def scheduler_time_seconds(self) -> float:
        """Cumulative time spent in dispatch logic (§4.2 overhead metric).

        Returns:
            Seconds spent in dispatch operations.
        """
        with self._lock:
            return self._scheduler_time_total

    def get_history(self, n: int = 100) -> list[AssignmentRecord]:
        """Return the most recent *n* assignment records, newest first.

        Args:
            n: Maximum records to return.

        Returns:
            List of AssignmentRecord objects.
        """
        with self._lock:
            records = list(self._history)
        return list(reversed(records))[:n]

    def get_statistics(self) -> dict[str, Any]:
        """Return dispatcher performance statistics.

        Returns:
            Dictionary with dispatch counts, scheduler time, and active load.
        """
        with self._lock:
            return {
                "total_dispatched": self._total_dispatched,
                "total_completed": self._total_completed,
                "total_failed": self._total_failed,
                "currently_active": len(self._active),
                "queue_size": self._queue.size(),
                "scheduler_time_seconds": self._scheduler_time_total,
                "history_size": len(self._history),
            }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _update_history_status(
        self,
        work_unit_id: str,
        status: AssignmentStatus,
        reason: str | None = None,
    ) -> None:
        """Update the status of the most recent assignment record for a work unit.

        Args:
            work_unit_id: Target work unit.
            status: New AssignmentStatus.
            reason: Optional failure reason.
        """
        with self._lock:
            for rec in reversed(list(self._history)):
                if rec.work_unit_id == work_unit_id and rec.status == AssignmentStatus.ASSIGNED:
                    if status == AssignmentStatus.COMPLETED:
                        rec.mark_completed()
                    elif status == AssignmentStatus.FAILED:
                        rec.mark_failed(reason or "unknown")
                    elif status == AssignmentStatus.RECOVERED:
                        rec.mark_recovered()
                    break

    def __repr__(self) -> str:
        return (
            f"TaskDispatcher(active={self.active_count}, "
            f"dispatched={self._total_dispatched}, "
            f"completed={self._total_completed}, "
            f"queue={self._queue.size()})"
        )
