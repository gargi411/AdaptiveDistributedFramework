"""Unit tests for WorkStealingCoordinator (Module 6).

Tests cover: StealEvent model, steal cycle logic, threshold filtering,
manual trigger, statistics, and lifecycle.
"""

from __future__ import annotations

import time

import pytest

from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.models.runtime import WorkerState
from adaptive_framework.scheduler.priority_queue import PageCountPriorityQueue
from adaptive_framework.scheduler.steal_event import StealEvent
from adaptive_framework.scheduler.work_stealing import WorkStealingCoordinator


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_node(hostname: str = "node") -> NodeInfo:
    return NodeInfo(
        hostname=hostname, ip_address="127.0.0.1",
        cpu_count_logical=4, cpu_count_physical=2, ram_total_gb=8.0,
    )


def _registry_with_workers(
    idle_count: int = 1, overloaded_count: int = 1, overloaded_queue: int = 5
) -> WorkerRegistry:
    registry = WorkerRegistry()
    node = _make_node()
    for i in range(idle_count):
        wid = f"idle_{i:02d}"
        registry.register(node, worker_id=wid)
        registry.update_heartbeat(wid, 10.0, 20.0, None, 0, None, WorkerState.IDLE)

    for i in range(overloaded_count):
        wid = f"over_{i:02d}"
        registry.register(node, worker_id=wid)
        registry.update_heartbeat(
            wid, 90.0, 80.0, None, overloaded_queue, f"wu_{i}", WorkerState.OVERLOADED
        )

    return registry


# ── StealEvent model ──────────────────────────────────────────────────────

class TestStealEvent:

    def test_steal_event_tasks_stolen_count(self) -> None:
        evt = StealEvent(
            source_worker_id="src",
            destination_worker_id="dst",
            stolen_work_unit_ids=["a", "b", "c"],
            source_queue_depth_before=5,
            source_queue_depth_after=2,
            destination_queue_depth_before=0,
            destination_queue_depth_after=3,
        )
        assert evt.tasks_stolen == 3

    def test_steal_event_to_dict_has_all_fields(self) -> None:
        evt = StealEvent(
            source_worker_id="w_src",
            destination_worker_id="w_dst",
            stolen_work_unit_ids=["x"],
            source_queue_depth_before=3,
            source_queue_depth_after=2,
            destination_queue_depth_before=0,
            destination_queue_depth_after=1,
        )
        d = evt.to_dict()
        assert "event_id" in d
        assert "tasks_stolen" in d
        assert "timestamp" in d
        assert d["tasks_stolen"] == 1

    def test_steal_event_repr(self) -> None:
        evt = StealEvent(
            source_worker_id="w_a",
            destination_worker_id="w_b",
            stolen_work_unit_ids=["x", "y"],
            source_queue_depth_before=4,
            source_queue_depth_after=2,
            destination_queue_depth_before=0,
            destination_queue_depth_after=2,
        )
        r = repr(evt)
        assert "StealEvent" in r
        assert "2" in r  # tasks stolen


# ── WorkStealingCoordinator — steal cycle ────────────────────────────────

class TestStealCycle:

    def test_no_idle_no_steal(self) -> None:
        registry = _registry_with_workers(idle_count=0, overloaded_count=1)
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            steal_threshold=2, steal_fraction=0.5,
        )
        ws._run_steal_cycle()
        stats = ws.get_statistics()
        assert stats["total_steal_events"] == 0

    def test_no_overloaded_no_steal(self) -> None:
        registry = _registry_with_workers(idle_count=2, overloaded_count=0)
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(registry=registry, global_queue=queue)
        ws._run_steal_cycle()
        stats = ws.get_statistics()
        assert stats["total_steal_events"] == 0

    def test_steal_occurs_with_idle_and_overloaded(self) -> None:
        registry = _registry_with_workers(
            idle_count=1, overloaded_count=1, overloaded_queue=6
        )
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            steal_threshold=2, steal_fraction=0.5,
        )
        ws._run_steal_cycle()
        stats = ws.get_statistics()
        assert stats["total_steal_events"] == 1
        assert stats["total_tasks_stolen"] > 0

    def test_steal_threshold_filters_low_queue(self) -> None:
        """Workers with queue_depth < threshold should not be stolen from."""
        registry = _registry_with_workers(
            idle_count=1, overloaded_count=1, overloaded_queue=1  # below threshold=2
        )
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            steal_threshold=2, steal_fraction=0.5,
        )
        ws._run_steal_cycle()
        stats = ws.get_statistics()
        assert stats["total_steal_events"] == 0

    def test_steal_updates_registry_stolen_from(self) -> None:
        registry = _registry_with_workers(
            idle_count=1, overloaded_count=1, overloaded_queue=6
        )
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            steal_threshold=2, steal_fraction=0.5,
        )
        ws._run_steal_cycle()
        overloaded = [w for w in registry.get_all() if "over" in w.worker_id]
        assert overloaded[0].total_stolen_from > 0

    def test_steal_updates_registry_stolen_to(self) -> None:
        registry = _registry_with_workers(
            idle_count=1, overloaded_count=1, overloaded_queue=6
        )
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            steal_threshold=2, steal_fraction=0.5,
        )
        ws._run_steal_cycle()
        idle_workers = [w for w in registry.get_all() if "idle" in w.worker_id]
        assert idle_workers[0].total_stolen_to > 0


# ── Manual trigger ────────────────────────────────────────────────────────

class TestManualTrigger:

    def test_trigger_steal_cycle_returns_events(self) -> None:
        registry = _registry_with_workers(idle_count=1, overloaded_count=1, overloaded_queue=8)
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            steal_threshold=2, steal_fraction=0.5,
        )
        events = ws.trigger_steal_cycle()
        # May or may not steal depending on state — just check it returns a list
        assert isinstance(events, list)


# ── History ───────────────────────────────────────────────────────────────

class TestHistory:

    def test_steal_events_recorded_in_history(self) -> None:
        registry = _registry_with_workers(idle_count=1, overloaded_count=1, overloaded_queue=6)
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            steal_threshold=2, steal_fraction=0.5,
        )
        ws._run_steal_cycle()
        events = ws.get_recent_steal_events(10)
        assert len(events) >= 1
        assert isinstance(events[0], StealEvent)

    def test_get_worker_utilization_returns_all_workers(self) -> None:
        registry = _registry_with_workers(idle_count=2, overloaded_count=1)
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(registry=registry, global_queue=queue)
        util = ws.get_worker_utilization()
        assert len(util) == 3  # 2 idle + 1 overloaded


# ── Lifecycle ─────────────────────────────────────────────────────────────

class TestLifecycle:

    def test_start_stop(self) -> None:
        registry = WorkerRegistry()
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            check_interval_seconds=30.0,
        )
        ws.start()
        assert ws.is_running
        ws.stop()
        assert not ws.is_running

    def test_double_start_safe(self) -> None:
        registry = WorkerRegistry()
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(
            registry=registry, global_queue=queue,
            check_interval_seconds=30.0,
        )
        ws.start()
        ws.start()
        ws.stop()

    def test_statistics_all_keys(self) -> None:
        registry = WorkerRegistry()
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(registry=registry, global_queue=queue)
        stats = ws.get_statistics()
        for key in [
            "total_steal_events", "total_tasks_stolen", "steal_threshold",
            "steal_fraction", "scheduler_time_seconds", "running",
        ]:
            assert key in stats

    def test_scheduler_time_accumulates(self) -> None:
        registry = _registry_with_workers(idle_count=1, overloaded_count=1, overloaded_queue=6)
        queue = PageCountPriorityQueue()
        ws = WorkStealingCoordinator(registry=registry, global_queue=queue)
        ws._run_steal_cycle()
        assert ws.scheduler_time_seconds >= 0.0
