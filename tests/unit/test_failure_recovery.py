"""Unit tests for FailureRecoveryEngine (Module 5).

Tests cover: worker lost handling, task recovery, retry exhaustion,
worker reconnection, graceful shutdown, and statistics.
"""

from __future__ import annotations

import pytest

from adaptive_framework.coordinator.failure_recovery.recovery_engine import FailureRecoveryEngine
from adaptive_framework.coordinator.failure_recovery.recovery_event import (
    RecoveryEvent,
    RecoveryEventType,
)
from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.coordinator.task_dispatcher.dispatcher import TaskDispatcher
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.models.runtime import WorkerState
from adaptive_framework.models.scheduling import PageWorkUnit
from adaptive_framework.scheduler.priority_queue import PageCountPriorityQueue


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_node() -> NodeInfo:
    return NodeInfo(hostname="n1", ip_address="127.0.0.1",
                    cpu_count_logical=2, cpu_count_physical=1, ram_total_gb=4.0)


def _make_wu(doc_id: str = "d1", pages: int = 10) -> PageWorkUnit:
    return PageWorkUnit(
        document_id=doc_id,
        file_path=f"/data/{doc_id}.pdf",
        start_page=1,
        end_page=pages,
    )


def _setup() -> tuple[FailureRecoveryEngine, WorkerRegistry, TaskDispatcher, PageCountPriorityQueue]:
    registry = WorkerRegistry()
    queue = PageCountPriorityQueue()
    dispatcher = TaskDispatcher(registry=registry, queue=queue)
    engine = FailureRecoveryEngine(
        registry=registry,
        dispatcher=dispatcher,
        max_retries=2,
        reassignment_delay_seconds=0.0,
    )
    return engine, registry, dispatcher, queue


# ── RecoveryEvent model ───────────────────────────────────────────────────

class TestRecoveryEvent:

    def test_event_to_dict(self) -> None:
        evt = RecoveryEvent(
            event_type=RecoveryEventType.WORKER_LOST,
            worker_id="w_001",
            message="Lost",
            work_unit_ids=["wu_1", "wu_2"],
        )
        d = evt.to_dict()
        assert d["event_type"] == "worker_lost"
        assert d["worker_id"] == "w_001"
        assert d["work_unit_ids"] == ["wu_1", "wu_2"]

    def test_event_repr(self) -> None:
        evt = RecoveryEvent(
            event_type=RecoveryEventType.TASKS_RECOVERED,
            worker_id="w_002",
            message="Recovered",
        )
        r = repr(evt)
        assert "tasks_recovered" in r


# ── handle_worker_lost ────────────────────────────────────────────────────

class TestHandleWorkerLost:

    def test_lost_event_emitted(self) -> None:
        engine, _, _, _ = _setup()
        events = engine.handle_worker_lost("w_unknown")
        types = [e.event_type for e in events]
        assert RecoveryEventType.WORKER_LOST in types

    def test_no_active_tasks_returns_only_lost_event(self) -> None:
        engine, _, _, _ = _setup()
        events = engine.handle_worker_lost("w_no_tasks")
        assert len(events) == 1
        assert events[0].event_type == RecoveryEventType.WORKER_LOST

    def test_recovers_active_task_to_queue(self) -> None:
        engine, registry, dispatcher, queue = _setup()
        node = _make_node()
        rec = registry.register(node, worker_id="w_001")
        registry.update_heartbeat(
            "w_001", 50.0, 40.0, None, 1, None, WorkerState.ACTIVE
        )
        # Dispatch a task to w_001
        wu = _make_wu()
        queue.insert(wu)
        dispatcher.dispatch_next()
        assert dispatcher.active_count == 1

        # Now lose the worker
        events = engine.handle_worker_lost("w_001")
        types = [e.event_type for e in events]
        # Should include RETRY_SCHEDULED and TASKS_RECOVERED
        assert RecoveryEventType.RETRY_SCHEDULED in types
        assert RecoveryEventType.TASKS_RECOVERED in types

    def test_retry_exhausted_after_max_retries(self) -> None:
        engine, registry, dispatcher, queue = _setup()
        node = _make_node()
        registry.register(node, worker_id="w_001")
        wu = _make_wu()
        queue.insert(wu)
        dispatcher.dispatch_next()

        # Pre-set retry count to max_retries so the NEXT lost event exhausts it
        with engine._lock:
            engine._retry_counts[wu.work_unit_id] = engine._max_retries

        # Now losing the worker should see retry_count > max_retries → RETRY_EXHAUSTED
        events = engine.handle_worker_lost("w_001")
        types = [e.event_type for e in events]
        assert RecoveryEventType.RETRY_EXHAUSTED in types

    def test_lost_counter_increments(self) -> None:
        engine, _, _, _ = _setup()
        engine.handle_worker_lost("w_001")
        engine.handle_worker_lost("w_002")
        stats = engine.get_statistics()
        assert stats["total_workers_lost"] == 2


# ── handle_worker_reconnected ────────────────────────────────────────────

class TestWorkerReconnected:

    def test_reconnect_emits_recovered_event(self) -> None:
        engine, _, _, _ = _setup()
        evt = engine.handle_worker_reconnected("w_001")
        assert evt.event_type == RecoveryEventType.WORKER_RECOVERED

    def test_reconnect_counter_increments(self) -> None:
        engine, _, _, _ = _setup()
        engine.handle_worker_reconnected("w_001")
        stats = engine.get_statistics()
        assert stats["total_workers_recovered"] == 1


# ── handle_graceful_shutdown ─────────────────────────────────────────────

class TestGracefulShutdown:

    def test_graceful_shutdown_emits_event(self) -> None:
        engine, registry, _, _ = _setup()
        node = _make_node()
        registry.register(node, worker_id="w_001")
        evt = engine.handle_graceful_shutdown("w_001")
        assert evt.event_type == RecoveryEventType.GRACEFUL_SHUTDOWN

    def test_graceful_shutdown_removes_from_registry(self) -> None:
        engine, registry, _, _ = _setup()
        node = _make_node()
        registry.register(node, worker_id="w_001")
        assert registry.worker_count == 1
        engine.handle_graceful_shutdown("w_001")
        assert registry.worker_count == 0


# ── History & Statistics ─────────────────────────────────────────────────

class TestHistoryAndStats:

    def test_get_recent_events_newest_first(self) -> None:
        engine, _, _, _ = _setup()
        engine.handle_worker_lost("w_001")
        engine.handle_worker_reconnected("w_001")
        events = engine.get_recent_events(10)
        assert events[0].event_type == RecoveryEventType.WORKER_RECOVERED
        assert events[-1].event_type == RecoveryEventType.WORKER_LOST

    def test_statistics_all_keys_present(self) -> None:
        engine, _, _, _ = _setup()
        stats = engine.get_statistics()
        assert "total_workers_lost" in stats
        assert "total_tasks_recovered" in stats
        assert "total_tasks_permanently_failed" in stats
        assert "total_workers_recovered" in stats
        assert "max_retries" in stats
        assert "history_size" in stats

    def test_repr(self) -> None:
        engine, _, _, _ = _setup()
        r = repr(engine)
        assert "FailureRecoveryEngine" in r
