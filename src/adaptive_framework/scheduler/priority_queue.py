"""Custom Priority Queue for the Adaptive Scheduler.

Implements a thread-safe max-heap priority queue where priority is determined
by page count (highest page count = highest priority). This ensures the
largest, most demanding work units are dispatched first, minimising
stragglers in the distributed pipeline (architecture v2.0 §2.2).

Algorithmic properties:
    - Insert:   O(log n)
    - Pop:      O(log n)
    - Peek:     O(1)
    - Update:   O(n) — mark-and-reinsert pattern
    - Space:    O(n)

Thread safety:
    All mutating operations are protected by a ``threading.Lock``.
    This allows the scheduler loop (in a background thread) and the
    failure recovery mechanism (in a coordinator thread) to safely
    share one queue instance.

Architecture note:
    Priority is the page_count of the PageWorkUnit.
    Higher page_count → higher priority → dispatched first.
    Ties are broken by insertion order (FIFO).
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from adaptive_framework.core.exceptions import SchedulerError
from adaptive_framework.models.scheduling import PageWorkUnit

logger = logging.getLogger("adaptive_framework.scheduler.priority_queue")


@dataclass(order=True)
class _QueueEntry:
    """Internal heap entry.

    Uses a negated priority so Python's min-heap behaves as a max-heap.

    Attributes:
        neg_priority: Negative of the page_count (for max-heap via min-heap).
        sequence: Monotonically increasing insertion counter (FIFO tie-break).
        work_unit: The actual PageWorkUnit payload (not compared).
        removed: Set to True when this entry is lazily deleted (mark pattern).
    """

    neg_priority: int
    sequence: int
    work_unit: PageWorkUnit = field(compare=False)
    removed: bool = field(default=False, compare=False)


class PageCountPriorityQueue:
    """Thread-safe max-priority queue ordered by PageWorkUnit page count.

    Attributes:
        _heap: The underlying Python heap (min-heap of _QueueEntry).
        _lock: Mutex protecting all heap mutations.
        _sequence: Monotonically increasing insertion counter.
        _entry_finder: Maps work_unit_id → active _QueueEntry (for updates).
        _max_size: Maximum queue capacity. 0 = unlimited.
        _stats: Cumulative operation counters.

    Example:
        >>> q = PageCountPriorityQueue(max_size=1000)
        >>> wu = PageWorkUnit(document_id="doc-1", file_path="/a.pdf",
        ...                   start_page=1, end_page=50)
        >>> q.insert(wu)
        >>> top = q.pop()
        >>> top.page_count
        50
    """

    def __init__(self, max_size: int = 0) -> None:
        """Initialise the priority queue.

        Args:
            max_size: Maximum number of items. 0 means unlimited.
        """
        self._heap: list[_QueueEntry] = []
        self._lock = threading.Lock()
        self._sequence: int = 0
        self._entry_finder: dict[str, _QueueEntry] = {}
        self._max_size = max_size

        # Cumulative statistics
        self._total_inserted: int = 0
        self._total_popped: int = 0
        self._total_updates: int = 0
        self._created_at: float = time.monotonic()

    # ------------------------------------------------------------------ #
    # Core operations                                                      #
    # ------------------------------------------------------------------ #

    def insert(self, work_unit: PageWorkUnit) -> None:
        """Insert a PageWorkUnit into the queue.

        Args:
            work_unit: The work unit to enqueue.

        Raises:
            SchedulerError: If the queue is full (max_size > 0).
        """
        with self._lock:
            if self._max_size > 0 and self._size_unlocked() >= self._max_size:
                raise SchedulerError(
                    f"Priority queue capacity exceeded (max_size={self._max_size})."
                )
            self._insert_unlocked(work_unit)

    def pop(self) -> PageWorkUnit | None:
        """Remove and return the highest-priority work unit.

        Returns:
            The PageWorkUnit with the highest page_count, or None if empty.
        """
        with self._lock:
            while self._heap:
                entry = heapq.heappop(self._heap)
                if entry.removed:
                    continue
                self._entry_finder.pop(entry.work_unit.work_unit_id, None)
                self._total_popped += 1
                return entry.work_unit
            return None

    def peek(self) -> PageWorkUnit | None:
        """Return the highest-priority work unit without removing it.

        Returns:
            The top PageWorkUnit, or None if the queue is empty.
        """
        with self._lock:
            while self._heap:
                entry = self._heap[0]
                if not entry.removed:
                    return entry.work_unit
                heapq.heappop(self._heap)  # discard removed entry
            return None

    def reinsert(self, work_unit: PageWorkUnit) -> None:
        """Re-insert a work unit (e.g., after worker failure).

        If the work unit is already in the queue, its entry is updated
        (old entry marked removed, new entry inserted).

        Args:
            work_unit: The work unit to re-enqueue.
        """
        with self._lock:
            self._remove_unlocked(work_unit.work_unit_id)
            self._insert_unlocked(work_unit)

    def update_priority(self, work_unit_id: str, new_priority: int) -> bool:
        """Update the priority of an existing work unit.

        Uses the lazy-deletion mark-and-reinsert pattern.

        Args:
            work_unit_id: Identifier of the work unit to update.
            new_priority: New priority value (will be used as page_count override).

        Returns:
            True if the work unit was found and updated, False otherwise.
        """
        with self._lock:
            if work_unit_id not in self._entry_finder:
                return False
            old_entry = self._entry_finder[work_unit_id]
            old_wu = old_entry.work_unit
            # Lazy-delete old entry
            old_entry.removed = True
            del self._entry_finder[work_unit_id]
            # Re-insert with new priority
            new_wu = PageWorkUnit(
                document_id=old_wu.document_id,
                file_path=old_wu.file_path,
                start_page=old_wu.start_page,
                end_page=old_wu.start_page + new_priority - 1,  # adjust end_page
                work_unit_id=old_wu.work_unit_id,
                priority=new_priority,
                status=old_wu.status,
                assigned_worker_id=old_wu.assigned_worker_id,
                retry_count=old_wu.retry_count,
            )
            self._insert_unlocked(new_wu)
            self._total_updates += 1
            return True

    def remove(self, work_unit_id: str) -> bool:
        """Remove a specific work unit from the queue by ID.

        Uses lazy deletion (O(1) mark, cleaned up on next pop/peek).

        Args:
            work_unit_id: The work unit to remove.

        Returns:
            True if found and marked for removal, False if not found.
        """
        with self._lock:
            return self._remove_unlocked(work_unit_id)

    def is_empty(self) -> bool:
        """Return True if the queue contains no active entries.

        Returns:
            True if the active (non-removed) entry count is zero.
        """
        with self._lock:
            return len(self._entry_finder) == 0

    def size(self) -> int:
        """Return the number of active (non-removed) entries.

        Returns:
            Active queue size.
        """
        with self._lock:
            return self._size_unlocked()

    def snapshot(self) -> list[PageWorkUnit]:
        """Return an ordered snapshot of all active work units.

        Work units are returned in priority order (highest first).
        This operation is O(n log n).

        Returns:
            List of PageWorkUnit objects sorted by page_count descending.
        """
        with self._lock:
            active = [e.work_unit for e in self._heap if not e.removed]
            active.sort(key=lambda wu: wu.page_count, reverse=True)
            return active

    def clear(self) -> None:
        """Remove all entries from the queue."""
        with self._lock:
            self._heap.clear()
            self._entry_finder.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Return cumulative queue statistics.

        Returns:
            Dictionary with keys: size, max_size, total_inserted,
            total_popped, total_updates, uptime_seconds.
        """
        with self._lock:
            return {
                "active_size": self._size_unlocked(),
                "max_size": self._max_size,
                "total_inserted": self._total_inserted,
                "total_popped": self._total_popped,
                "total_updates": self._total_updates,
                "uptime_seconds": round(time.monotonic() - self._created_at, 2),
                "heap_size": len(self._heap),  # includes lazy-deleted entries
            }

    def serialize(self) -> list[dict[str, Any]]:
        """Serialize the queue to a list of dicts (for inspection/persistence).

        Returns:
            Ordered list of work unit dicts (highest priority first).
        """
        return [wu.to_dict() for wu in self.snapshot()]

    # ------------------------------------------------------------------ #
    # Private unlocked helpers (must be called while holding _lock)       #
    # ------------------------------------------------------------------ #

    def _insert_unlocked(self, work_unit: PageWorkUnit) -> None:
        """Insert a work unit — caller must hold the lock.

        Args:
            work_unit: Work unit to insert.
        """
        entry = _QueueEntry(
            neg_priority=-work_unit.page_count,
            sequence=self._sequence,
            work_unit=work_unit,
        )
        self._sequence += 1
        heapq.heappush(self._heap, entry)
        self._entry_finder[work_unit.work_unit_id] = entry
        self._total_inserted += 1

    def _remove_unlocked(self, work_unit_id: str) -> bool:
        """Lazy-delete a work unit — caller must hold the lock.

        Args:
            work_unit_id: ID to mark as removed.

        Returns:
            True if found and marked, False otherwise.
        """
        entry = self._entry_finder.pop(work_unit_id, None)
        if entry is not None:
            entry.removed = True
            return True
        return False

    def _size_unlocked(self) -> int:
        """Return active size without acquiring the lock."""
        return len(self._entry_finder)

    def __repr__(self) -> str:
        return (
            f"PageCountPriorityQueue("
            f"size={self.size()}, "
            f"inserted={self._total_inserted}, "
            f"max_size={self._max_size})"
        )

    def __len__(self) -> int:
        return self.size()
