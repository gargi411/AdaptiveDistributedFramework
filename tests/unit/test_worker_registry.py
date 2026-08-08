"""Unit tests for WorkerRegistry (Module 2).

Tests cover: registration, removal, discovery, heartbeat updates,
state transitions, statistics, and thread safety.
"""

from __future__ import annotations

import threading
import time

import pytest

from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.coordinator.worker_registry.worker_record import WorkerRecord
from adaptive_framework.models.runtime import WorkerState


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def node_info() -> NodeInfo:
    """Return a minimal NodeInfo for testing."""
    return NodeInfo(
        hostname="test-node",
        ip_address="127.0.0.1",
        cpu_count_logical=4,
        cpu_count_physical=2,
        ram_total_gb=8.0,
        gpu_count=0,
    )


@pytest.fixture()
def registry() -> WorkerRegistry:
    """Return a fresh WorkerRegistry."""
    return WorkerRegistry()


# ── Registration ─────────────────────────────────────────────────────────

class TestRegistration:

    def test_register_single_worker(self, registry: WorkerRegistry, node_info: NodeInfo) -> None:
        rec = registry.register(node_info)
        assert rec.worker_id is not None
        assert rec.hostname == "test-node"
        assert registry.worker_count == 1

    def test_register_with_explicit_id(self, registry: WorkerRegistry, node_info: NodeInfo) -> None:
        rec = registry.register(node_info, worker_id="w_001")
        assert rec.worker_id == "w_001"

    def test_register_same_id_twice_returns_existing(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        rec1 = registry.register(node_info, worker_id="w_001")
        rec2 = registry.register(node_info, worker_id="w_001")
        assert rec1.worker_id == rec2.worker_id
        assert registry.worker_count == 1  # still only 1

    def test_register_multiple_workers(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        for i in range(5):
            registry.register(node_info, worker_id=f"w_{i:03d}")
        assert registry.worker_count == 5

    def test_remove_worker(self, registry: WorkerRegistry, node_info: NodeInfo) -> None:
        registry.register(node_info, worker_id="w_001")
        removed = registry.remove("w_001")
        assert removed is not None
        assert removed.worker_id == "w_001"
        assert registry.worker_count == 0

    def test_remove_nonexistent_returns_none(self, registry: WorkerRegistry) -> None:
        result = registry.remove("nonexistent")
        assert result is None


# ── Discovery ─────────────────────────────────────────────────────────────

class TestDiscovery:

    def test_get_registered_worker(self, registry: WorkerRegistry, node_info: NodeInfo) -> None:
        registry.register(node_info, worker_id="w_001")
        rec = registry.get("w_001")
        assert rec is not None
        assert rec.worker_id == "w_001"

    def test_get_nonexistent_returns_none(self, registry: WorkerRegistry) -> None:
        assert registry.get("ghost") is None

    def test_get_all_returns_all(self, registry: WorkerRegistry, node_info: NodeInfo) -> None:
        for i in range(3):
            registry.register(node_info, worker_id=f"w_{i}")
        all_workers = registry.get_all()
        assert len(all_workers) == 3

    def test_get_available_excludes_lost(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_active")
        registry.register(node_info, worker_id="w_lost")
        registry.mark_lost("w_lost")
        available = registry.get_available()
        assert len(available) == 1
        assert available[0].worker_id == "w_active"

    def test_get_idle_returns_idle_only(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_idle")
        registry.register(node_info, worker_id="w_active")
        registry.update_heartbeat(
            "w_active",
            cpu_percent=50.0, ram_percent=40.0, gpu_percent=None,
            queue_depth=3, current_task_id="wu_001", state=WorkerState.ACTIVE,
        )
        idle = registry.get_idle()
        assert len(idle) == 1
        assert idle[0].worker_id == "w_idle"

    def test_get_lost_returns_lost_only(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_ok")
        registry.register(node_info, worker_id="w_lost")
        registry.mark_lost("w_lost")
        lost = registry.get_lost()
        assert len(lost) == 1
        assert lost[0].worker_id == "w_lost"

    def test_get_overloaded_filters_by_threshold(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_low")
        registry.register(node_info, worker_id="w_high")
        registry.update_heartbeat(
            "w_low", 20.0, 30.0, None, queue_depth=1,
            current_task_id=None, state=WorkerState.ACTIVE,
        )
        registry.update_heartbeat(
            "w_high", 90.0, 80.0, None, queue_depth=5,
            current_task_id="wu_001", state=WorkerState.OVERLOADED,
        )
        overloaded = registry.get_overloaded(steal_threshold=2)
        assert len(overloaded) == 1
        assert overloaded[0].worker_id == "w_high"


# ── State transitions ─────────────────────────────────────────────────────

class TestStateTransitions:

    def test_mark_lost_sets_state(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_001")
        result = registry.mark_lost("w_001")
        assert result is True
        rec = registry.get("w_001")
        assert rec.state == WorkerState.LOST

    def test_mark_lost_nonexistent_returns_false(self, registry: WorkerRegistry) -> None:
        assert registry.mark_lost("ghost") is False

    def test_mark_recovered_sets_idle(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_001")
        registry.mark_lost("w_001")
        registry.mark_recovered("w_001")
        rec = registry.get("w_001")
        assert rec.state == WorkerState.IDLE
        assert rec.current_task_id is None
        assert rec.queue_depth == 0

    def test_record_completion(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_001")
        registry.record_completion("w_001")
        rec = registry.get("w_001")
        assert rec.total_completed == 1

    def test_record_failure(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_001")
        registry.record_failure("w_001")
        rec = registry.get("w_001")
        assert rec.total_failed == 1


# ── Heartbeat updates ─────────────────────────────────────────────────────

class TestHeartbeatUpdates:

    def test_update_heartbeat_applies_all_fields(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_001")
        result = registry.update_heartbeat(
            "w_001",
            cpu_percent=75.5,
            ram_percent=60.0,
            gpu_percent=30.0,
            queue_depth=3,
            current_task_id="wu_007",
            state=WorkerState.ACTIVE,
        )
        assert result is True
        rec = registry.get("w_001")
        assert rec.cpu_percent == 75.5
        assert rec.ram_percent == 60.0
        assert rec.gpu_percent == 30.0
        assert rec.queue_depth == 3
        assert rec.current_task_id == "wu_007"
        assert rec.state == WorkerState.ACTIVE

    def test_update_heartbeat_nonexistent_returns_false(
        self, registry: WorkerRegistry
    ) -> None:
        result = registry.update_heartbeat(
            "ghost", 0.0, 0.0, None, 0, None, WorkerState.IDLE
        )
        assert result is False


# ── Statistics ────────────────────────────────────────────────────────────

class TestStatistics:

    def test_cluster_status_counts(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_a")
        registry.register(node_info, worker_id="w_b")
        registry.mark_lost("w_b")
        cs = registry.get_cluster_status()
        assert cs.total_workers == 2
        assert cs.lost_workers == 1

    def test_get_statistics_empty_registry(self, registry: WorkerRegistry) -> None:
        stats = registry.get_statistics()
        assert stats["total_workers"] == 0
        assert stats["avg_cpu_percent"] == 0.0

    def test_get_statistics_nonempty(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        registry.register(node_info, worker_id="w_001")
        registry.update_heartbeat("w_001", 50.0, 40.0, None, 0, None, WorkerState.IDLE)
        stats = registry.get_statistics()
        assert stats["total_workers"] == 1
        assert stats["avg_cpu_percent"] == 50.0

    def test_len(self, registry: WorkerRegistry, node_info: NodeInfo) -> None:
        assert len(registry) == 0
        registry.register(node_info)
        assert len(registry) == 1


# ── Thread safety ─────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_registrations(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        """Multiple threads registering workers must not corrupt state."""
        errors: list[Exception] = []

        def register(i: int) -> None:
            try:
                registry.register(node_info, worker_id=f"w_{i:04d}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert registry.worker_count == 50

    def test_concurrent_updates(
        self, registry: WorkerRegistry, node_info: NodeInfo
    ) -> None:
        """Concurrent heartbeat updates must not crash."""
        registry.register(node_info, worker_id="w_001")
        errors: list[Exception] = []

        def update() -> None:
            try:
                registry.update_heartbeat(
                    "w_001", 50.0, 40.0, None, 1, None, WorkerState.ACTIVE
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
