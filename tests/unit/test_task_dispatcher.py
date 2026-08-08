"""Unit tests for TaskDispatcher (Module 3).

Tests cover: dispatch, completion, failure, recovery, history, and statistics.
"""

from __future__ import annotations

import pytest

from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.coordinator.task_dispatcher.assignment_record import (
    AssignmentRecord,
    AssignmentStatus,
)
from adaptive_framework.coordinator.task_dispatcher.dispatcher import TaskDispatcher
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.models.runtime import WorkerState
from adaptive_framework.models.scheduling import PageWorkUnit
from adaptive_framework.scheduler.priority_queue import PageCountPriorityQueue


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_node() -> NodeInfo:
    return NodeInfo(
        hostname="node-test", ip_address="127.0.0.1",
        cpu_count_logical=4, cpu_count_physical=2, ram_total_gb=8.0,
    )


def _make_wu(doc_id: str = "doc_001", pages: int = 10) -> PageWorkUnit:
    return PageWorkUnit(
        document_id=doc_id,
        file_path=f"/data/{doc_id}.pdf",
        start_page=1,
        end_page=pages,
    )


def _make_dispatcher(num_workers: int = 2) -> tuple[TaskDispatcher, WorkerRegistry, PageCountPriorityQueue]:
    registry = WorkerRegistry()
    queue = PageCountPriorityQueue()
    dispatcher = TaskDispatcher(registry=registry, queue=queue)

    node = _make_node()
    for i in range(num_workers):
        rec = registry.register(node, worker_id=f"w_{i:03d}")
        registry.update_heartbeat(
            rec.worker_id, 30.0, 30.0, None, 0, None, WorkerState.IDLE,
        )

    return dispatcher, registry, queue


# ── Dispatch ─────────────────────────────────────────────────────────────

class TestDispatch:

    def test_dispatch_single_task(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        wu = _make_wu(pages=20)
        queue.insert(wu)

        result = dispatcher.dispatch_next()
        assert result is not None
        dispatched_wu, worker_id = result
        assert dispatched_wu.work_unit_id == wu.work_unit_id
        assert "w_" in worker_id

    def test_dispatch_empty_queue_returns_none(self) -> None:
        dispatcher, _, _ = _make_dispatcher(1)
        result = dispatcher.dispatch_next()
        assert result is None

    def test_dispatch_no_workers_returns_none(self) -> None:
        registry = WorkerRegistry()
        queue = PageCountPriorityQueue()
        dispatcher = TaskDispatcher(registry=registry, queue=queue)
        queue.insert(_make_wu())
        result = dispatcher.dispatch_next()
        assert result is None

    def test_dispatch_selects_least_loaded_worker(self) -> None:
        dispatcher, registry, queue = _make_dispatcher(2)
        # Manually load worker 0
        dispatcher._worker_load["w_000"] = 5
        queue.insert(_make_wu(pages=30))
        result = dispatcher.dispatch_next()
        assert result is not None
        _, worker_id = result
        assert worker_id == "w_001"  # less loaded

    def test_dispatch_batch(self) -> None:
        dispatcher, _, queue = _make_dispatcher(2)
        for i in range(5):
            queue.insert(_make_wu(doc_id=f"doc_{i}", pages=10 + i))
        results = dispatcher.dispatch_batch(max_assignments=3)
        assert len(results) == 3

    def test_dispatch_highest_priority_first(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        queue.insert(_make_wu(doc_id="small", pages=5))
        queue.insert(_make_wu(doc_id="large", pages=100))

        result = dispatcher.dispatch_next()
        assert result is not None
        assert result[0].page_count == 100  # largest first

    def test_active_count_increments_on_dispatch(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        queue.insert(_make_wu())
        dispatcher.dispatch_next()
        assert dispatcher.active_count == 1

    def test_active_count_decrements_on_completion(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        wu = _make_wu()
        queue.insert(wu)
        dispatcher.dispatch_next()
        dispatcher.report_completed(wu.work_unit_id)
        assert dispatcher.active_count == 0


# ── Completion / Failure ─────────────────────────────────────────────────

class TestCompletionFailure:

    def test_report_completed_returns_true(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        wu = _make_wu()
        queue.insert(wu)
        dispatcher.dispatch_next()
        result = dispatcher.report_completed(wu.work_unit_id)
        assert result is True

    def test_report_completed_unknown_returns_false(self) -> None:
        dispatcher, _, _ = _make_dispatcher(1)
        result = dispatcher.report_completed("unknown_wu")
        assert result is False

    def test_report_failed_returns_true(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        wu = _make_wu()
        queue.insert(wu)
        dispatcher.dispatch_next()
        result = dispatcher.report_failed(wu.work_unit_id, "Test failure")
        assert result is True

    def test_report_failed_unknown_returns_false(self) -> None:
        dispatcher, _, _ = _make_dispatcher(1)
        result = dispatcher.report_failed("unknown", "no reason")
        assert result is False


# ── Recovery ─────────────────────────────────────────────────────────────

class TestRecovery:

    def test_recover_work_unit_requeues(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        wu = _make_wu(pages=42)
        queue.insert(wu)
        dispatcher.dispatch_next()

        assert queue.is_empty()
        dispatcher.recover_work_unit(wu.work_unit_id, wu)
        assert not queue.is_empty()

    def test_get_active_tasks_for_worker(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        wu1 = _make_wu(doc_id="d1", pages=10)
        wu2 = _make_wu(doc_id="d2", pages=5)
        queue.insert(wu1)
        queue.insert(wu2)
        result1 = dispatcher.dispatch_next()
        assert result1 is not None
        _, w_id = result1
        tasks = dispatcher.get_active_tasks_for_worker(w_id)
        assert wu1.work_unit_id in tasks


# ── History & Statistics ─────────────────────────────────────────────────

class TestHistory:

    def test_history_records_assignment(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        wu = _make_wu()
        queue.insert(wu)
        dispatcher.dispatch_next()
        history = dispatcher.get_history(n=10)
        assert len(history) == 1
        assert history[0].work_unit_id == wu.work_unit_id

    def test_statistics_total_dispatched(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        for i in range(3):
            queue.insert(_make_wu(doc_id=f"d{i}"))
        for _ in range(3):
            dispatcher.dispatch_next()
        stats = dispatcher.get_statistics()
        assert stats["total_dispatched"] == 3

    def test_scheduler_time_accumulates(self) -> None:
        dispatcher, _, queue = _make_dispatcher(1)
        queue.insert(_make_wu())
        dispatcher.dispatch_next()
        assert dispatcher.scheduler_time_seconds >= 0.0

    def test_assignment_record_elapsed_seconds(self) -> None:
        rec = AssignmentRecord(
            work_unit_id="wu_001",
            worker_id="w_001",
            document_id="doc_001",
            page_count=10,
        )
        rec.mark_completed()
        elapsed = rec.elapsed_seconds
        assert elapsed is not None
        assert elapsed >= 0.0

    def test_assignment_status_transitions(self) -> None:
        rec = AssignmentRecord(
            work_unit_id="wu_001",
            worker_id="w_001",
            document_id="doc_001",
            page_count=5,
        )
        assert rec.status == AssignmentStatus.ASSIGNED
        rec.mark_failed("test error")
        assert rec.status == AssignmentStatus.FAILED
        assert rec.failure_reason == "test error"
