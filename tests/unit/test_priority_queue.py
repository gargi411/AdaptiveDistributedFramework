"""Unit tests for the PageCountPriorityQueue.

Tests:
    - Insert and pop restore max-priority order.
    - Peek returns highest priority without popping.
    - Size tracking is correct.
    - Reinsert replaces existing entry.
    - Update priority changes dispatch order.
    - Remove (lazy deletion) removes a specific entry.
    - is_empty() returns correct boolean.
    - Thread safety under concurrent inserts and pops.
    - Queue statistics are populated.
    - Snapshot returns ordered list.
    - Serialize returns list of dicts.
    - max_size enforcement raises SchedulerError.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from adaptive_framework.core.exceptions import SchedulerError
from adaptive_framework.models.scheduling import PageWorkUnit, WorkUnitStatus
from adaptive_framework.scheduler.priority_queue import PageCountPriorityQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wu(pages: int, doc_id: str = "") -> PageWorkUnit:
    """Build a PageWorkUnit with the given page count.

    Args:
        pages: Page count (end_page = pages).
        doc_id: Document ID (default: derived from pages).

    Returns:
        PageWorkUnit with start_page=1, end_page=pages.
    """
    return PageWorkUnit(
        document_id=doc_id or f"doc_{pages:04d}",
        file_path=f"/data/{doc_id or pages}.pdf",
        start_page=1,
        end_page=pages,
    )


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


class TestInsertAndPop:
    """Tests for insert() and pop() ordering."""

    def test_pop_returns_highest_page_count(self) -> None:
        """Highest page count is returned first."""
        q = PageCountPriorityQueue()
        q.insert(_wu(10))
        q.insert(_wu(50))
        q.insert(_wu(5))
        top = q.pop()
        assert top is not None
        assert top.page_count == 50

    def test_pop_order_is_descending(self) -> None:
        """pop() returns items in descending page_count order."""
        q = PageCountPriorityQueue()
        counts = [3, 7, 1, 15, 9]
        for c in counts:
            q.insert(_wu(c))
        popped = []
        while (item := q.pop()) is not None:
            popped.append(item.page_count)
        assert popped == sorted(counts, reverse=True)

    def test_pop_empty_queue_returns_none(self) -> None:
        """pop() on an empty queue returns None."""
        q = PageCountPriorityQueue()
        assert q.pop() is None

    def test_single_insert_pop(self) -> None:
        """Inserting one item and popping returns the same work unit."""
        q = PageCountPriorityQueue()
        wu = _wu(42)
        q.insert(wu)
        result = q.pop()
        assert result is not None
        assert result.work_unit_id == wu.work_unit_id


class TestPeek:
    """Tests for peek()."""

    def test_peek_returns_top_without_removing(self) -> None:
        """peek() returns the highest-priority item but queue remains unchanged."""
        q = PageCountPriorityQueue()
        q.insert(_wu(10))
        q.insert(_wu(50))
        top = q.peek()
        assert top is not None
        assert top.page_count == 50
        assert q.size() == 2  # not removed

    def test_peek_empty_returns_none(self) -> None:
        """peek() on empty queue returns None."""
        q = PageCountPriorityQueue()
        assert q.peek() is None


class TestSize:
    """Tests for size() and is_empty()."""

    def test_size_increments_on_insert(self) -> None:
        """size() increases by 1 per insert."""
        q = PageCountPriorityQueue()
        for i in range(5):
            q.insert(_wu(i + 1))
        assert q.size() == 5

    def test_size_decrements_on_pop(self) -> None:
        """size() decreases by 1 per pop."""
        q = PageCountPriorityQueue()
        q.insert(_wu(10))
        q.insert(_wu(20))
        q.pop()
        assert q.size() == 1

    def test_is_empty_true_when_no_items(self) -> None:
        """is_empty() returns True on a new queue."""
        q = PageCountPriorityQueue()
        assert q.is_empty()

    def test_is_empty_false_after_insert(self) -> None:
        """is_empty() returns False after an insert."""
        q = PageCountPriorityQueue()
        q.insert(_wu(5))
        assert not q.is_empty()

    def test_is_empty_true_after_all_pops(self) -> None:
        """is_empty() returns True after all items are popped."""
        q = PageCountPriorityQueue()
        q.insert(_wu(5))
        q.pop()
        assert q.is_empty()

    def test_len_matches_size(self) -> None:
        """len(q) matches q.size()."""
        q = PageCountPriorityQueue()
        q.insert(_wu(5))
        q.insert(_wu(10))
        assert len(q) == q.size()


class TestReinsert:
    """Tests for reinsert()."""

    def test_reinsert_updates_existing_entry(self) -> None:
        """reinsert() replaces an existing entry by ID."""
        q = PageCountPriorityQueue()
        wu = _wu(10)
        q.insert(wu)
        assert q.size() == 1
        q.reinsert(wu)
        assert q.size() == 1  # still one active entry

    def test_reinsert_status_preserved(self) -> None:
        """reinsert() preserves the work unit's identity (work_unit_id)."""
        q = PageCountPriorityQueue()
        wu = _wu(20)
        wu.status = WorkUnitStatus.RETRYING
        q.insert(wu)
        q.reinsert(wu)
        result = q.pop()
        assert result is not None
        assert result.work_unit_id == wu.work_unit_id


class TestUpdatePriority:
    """Tests for update_priority()."""

    def test_update_priority_changes_dispatch_order(self) -> None:
        """Updating priority to a higher value moves item to front."""
        q = PageCountPriorityQueue()
        low = _wu(5, "low")
        high = _wu(100, "high")
        q.insert(low)
        q.insert(high)
        # Artificially boost 'low' to priority 200
        q.update_priority(low.work_unit_id, 200)
        top = q.pop()
        assert top is not None
        assert top.work_unit_id == low.work_unit_id

    def test_update_nonexistent_returns_false(self) -> None:
        """update_priority() returns False for unknown IDs."""
        q = PageCountPriorityQueue()
        assert q.update_priority("nonexistent-id", 999) is False


class TestRemove:
    """Tests for remove()."""

    def test_remove_existing_entry(self) -> None:
        """remove() returns True and item is no longer returned by pop."""
        q = PageCountPriorityQueue()
        wu = _wu(50)
        q.insert(wu)
        result = q.remove(wu.work_unit_id)
        assert result is True
        assert q.pop() is None

    def test_remove_nonexistent_returns_false(self) -> None:
        """remove() returns False for an unknown work_unit_id."""
        q = PageCountPriorityQueue()
        assert q.remove("no-such-id") is False

    def test_size_decreases_after_remove(self) -> None:
        """size() decreases after remove()."""
        q = PageCountPriorityQueue()
        wu = _wu(10)
        q.insert(wu)
        q.remove(wu.work_unit_id)
        assert q.size() == 0


class TestMaxSize:
    """Tests for max_size enforcement."""

    def test_insert_beyond_max_size_raises(self) -> None:
        """Inserting beyond max_size raises SchedulerError."""
        q = PageCountPriorityQueue(max_size=2)
        q.insert(_wu(10))
        q.insert(_wu(20))
        with pytest.raises(SchedulerError, match="capacity exceeded"):
            q.insert(_wu(30))

    def test_insert_at_max_size_succeeds(self) -> None:
        """Inserting exactly max_size items succeeds."""
        q = PageCountPriorityQueue(max_size=3)
        q.insert(_wu(10))
        q.insert(_wu(20))
        q.insert(_wu(30))
        assert q.size() == 3


class TestSnapshot:
    """Tests for snapshot() and serialize()."""

    def test_snapshot_returns_descending_order(self) -> None:
        """snapshot() returns items sorted by page_count descending."""
        q = PageCountPriorityQueue()
        for pages in [3, 7, 1, 15, 9]:
            q.insert(_wu(pages))
        snap = q.snapshot()
        page_counts = [wu.page_count for wu in snap]
        assert page_counts == sorted(page_counts, reverse=True)

    def test_serialize_returns_list_of_dicts(self) -> None:
        """serialize() returns a list of dictionaries."""
        q = PageCountPriorityQueue()
        q.insert(_wu(10))
        q.insert(_wu(20))
        result = q.serialize()
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)
        assert all("page_count" in item for item in result)


class TestStatistics:
    """Tests for get_statistics()."""

    def test_statistics_keys_present(self) -> None:
        """get_statistics() returns all expected keys."""
        q = PageCountPriorityQueue()
        q.insert(_wu(10))
        q.pop()
        stats = q.get_statistics()
        assert "active_size" in stats
        assert "total_inserted" in stats
        assert "total_popped" in stats
        assert "uptime_seconds" in stats

    def test_total_inserted_tracks_correctly(self) -> None:
        """total_inserted counts all insert calls."""
        q = PageCountPriorityQueue()
        for _ in range(7):
            q.insert(_wu(5))
        stats = q.get_statistics()
        assert stats["total_inserted"] == 7


class TestClear:
    """Tests for clear()."""

    def test_clear_empties_queue(self) -> None:
        """clear() removes all entries."""
        q = PageCountPriorityQueue()
        for pages in [5, 10, 20]:
            q.insert(_wu(pages))
        q.clear()
        assert q.is_empty()
        assert q.size() == 0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_inserts_no_corruption(self) -> None:
        """Concurrent inserts from multiple threads do not corrupt the queue."""
        q = PageCountPriorityQueue()
        errors: list[Exception] = []

        def insert_many() -> None:
            try:
                for pages in range(1, 11):
                    q.insert(_wu(pages, f"t{threading.get_ident()}_{pages}"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=insert_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert q.size() == 50  # 5 threads × 10 inserts

    def test_concurrent_pop_no_corruption(self) -> None:
        """Concurrent pops return consistent results (no duplicates, no None-overflows)."""
        q = PageCountPriorityQueue()
        for i in range(20):
            q.insert(_wu(i + 1, f"doc_{i:02d}"))

        popped_ids: list[str] = []
        lock = threading.Lock()

        def pop_some() -> None:
            for _ in range(4):
                item = q.pop()
                if item is not None:
                    with lock:
                        popped_ids.append(item.work_unit_id)

        threads = [threading.Thread(target=pop_some) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(popped_ids) == len(set(popped_ids)), "Duplicate pops detected"
