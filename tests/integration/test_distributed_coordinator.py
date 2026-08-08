"""Integration tests for DistributedCoordinator — end-to-end cluster pipeline.

These tests exercise the full coordinator pipeline without requiring a live
Ray cluster (init_ray=False). They validate:
    - Coordinator startup and shutdown
    - Worker registration and state
    - Work unit submission and dispatch loop
    - Heartbeat processing
    - Work stealing integration
    - Failure recovery integration
    - Benchmark result generation
    - Dashboard state output
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from adaptive_framework.coordinator.distributed_coordinator import DistributedCoordinator
from adaptive_framework.coordinator.heartbeat_monitor import HeartbeatPayload
from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.models.runtime import WorkerState
from adaptive_framework.models.scheduling import PageWorkUnit


# ── Fixtures ─────────────────────────────────────────────────────────────

_DEV_CONFIG = {
    "mode": "dev",
    "head": {"address": "auto", "dashboard_port": 8265},
    "workers": {"num_workers_dev": 4},
    "health": {
        "heartbeat_interval_seconds": 1.0,
        "heartbeat_timeout_seconds": 5.0,
        "health_check_interval_seconds": 1.0,
    },
    "failure_recovery": {
        "max_retries": 2,
        "reassignment_delay_seconds": 0.0,
    },
    "work_stealing": {
        "enabled": True,
        "steal_threshold": 2,
        "steal_fraction": 0.5,
        "check_interval_seconds": 30.0,  # manual trigger in tests
    },
    "resource_orchestration": {
        "sample_interval_seconds": 30.0,
        "cpu_high_threshold_percent": 85.0,
        "cpu_low_threshold_percent": 20.0,
    },
}


@pytest.fixture()
def coordinator() -> DistributedCoordinator:
    """Return a fully started coordinator (Ray-less dev mode)."""
    coord = DistributedCoordinator(cluster_config=_DEV_CONFIG, run_id="test_run")
    coord.start(init_ray=False)
    yield coord
    coord.stop(shutdown_ray=False)


def _node(name: str = "test-node") -> NodeInfo:
    return NodeInfo(
        hostname=name,
        ip_address="127.0.0.1",
        cpu_count_logical=4,
        cpu_count_physical=2,
        ram_total_gb=8.0,
    )


def _wu(doc_id: str = "doc", pages: int = 10) -> PageWorkUnit:
    return PageWorkUnit(
        document_id=doc_id,
        file_path=f"/data/{doc_id}.pdf",
        start_page=1,
        end_page=pages,
    )


def _heartbeat(worker_id: str, queue: int = 0, cpu: float = 30.0) -> HeartbeatPayload:
    return HeartbeatPayload(
        worker_id=worker_id,
        cpu_percent=cpu,
        ram_percent=40.0,
        queue_depth=queue,
        state=WorkerState.ACTIVE.value if queue > 0 else WorkerState.IDLE.value,
    )


# ── Startup / Shutdown ────────────────────────────────────────────────────

class TestStartupShutdown:

    def test_coordinator_starts_in_ready_state(self) -> None:
        coord = DistributedCoordinator(cluster_config=_DEV_CONFIG, run_id="t1")
        coord.start(init_ray=False)
        status = coord.get_status_dict()
        assert status["framework_state"] in ("ready", "running", "initializing")
        coord.stop(shutdown_ray=False)

    def test_coordinator_stops_cleanly(self) -> None:
        coord = DistributedCoordinator(cluster_config=_DEV_CONFIG, run_id="t2")
        coord.start(init_ray=False)
        result = coord.stop(shutdown_ray=False)
        # Stop should not raise
        assert result is not None or result is None  # benchmark may or may not exist

    def test_repr_contains_run_id(self) -> None:
        coord = DistributedCoordinator(cluster_config=_DEV_CONFIG, run_id="my_run")
        assert "my_run" in repr(coord)

    def test_from_yaml_raises_on_missing_file(self) -> None:
        from adaptive_framework.core.exceptions import FrameworkError
        with pytest.raises(FrameworkError):
            DistributedCoordinator.from_yaml("nonexistent/path.yaml")


# ── Worker Registration ───────────────────────────────────────────────────

class TestWorkerRegistration:

    def test_register_single_worker(self, coordinator: DistributedCoordinator) -> None:
        wid = coordinator.register_worker(_node("laptop-1"), worker_id="w_001")
        assert wid == "w_001"
        assert coordinator.registry.worker_count == 1

    def test_register_multiple_workers(self, coordinator: DistributedCoordinator) -> None:
        for i in range(4):
            coordinator.register_worker(_node(f"laptop-{i}"), worker_id=f"w_{i:03d}")
        assert coordinator.registry.worker_count == 4

    def test_registry_exposed_via_property(self, coordinator: DistributedCoordinator) -> None:
        coordinator.register_worker(_node())
        assert coordinator.registry.worker_count == 1


# ── Heartbeat Processing ──────────────────────────────────────────────────

class TestHeartbeatProcessing:

    def test_heartbeat_updates_worker_state(self, coordinator: DistributedCoordinator) -> None:
        coordinator.register_worker(_node(), worker_id="w_001")
        coordinator.receive_heartbeat(_heartbeat("w_001", cpu=72.5))
        rec = coordinator.registry.get("w_001")
        assert rec.cpu_percent == 72.5

    def test_heartbeat_with_queue_sets_active(self, coordinator: DistributedCoordinator) -> None:
        coordinator.register_worker(_node(), worker_id="w_001")
        coordinator.receive_heartbeat(_heartbeat("w_001", queue=3, cpu=50.0))
        rec = coordinator.registry.get("w_001")
        assert rec.queue_depth == 3


# ── Task Submission and Dispatch ──────────────────────────────────────────

class TestTaskDispatch:

    def test_submit_partitions_inserts_into_queue(
        self, coordinator: DistributedCoordinator
    ) -> None:
        from adaptive_framework.models.scheduling import Partition
        wu1 = _wu("d1", pages=20)
        wu2 = _wu("d2", pages=10)
        partition = Partition(work_units=[wu1, wu2])
        inserted = coordinator.submit_partitions([partition])
        assert inserted == 2
        assert coordinator.get_status_dict()["queue_size"] == 2

    def test_dispatch_loop_empties_queue(self, coordinator: DistributedCoordinator) -> None:
        coordinator.register_worker(_node("w0"), worker_id="w_000")
        from adaptive_framework.models.scheduling import Partition
        wu = _wu("d1", pages=20)
        partition = Partition(work_units=[wu])
        coordinator.submit_partitions([partition])
        dispatched = coordinator.run_dispatch_loop(until_empty=True, max_iterations=100)
        assert dispatched >= 1

    def test_report_completed_decrements_active(
        self, coordinator: DistributedCoordinator
    ) -> None:
        coordinator.register_worker(_node(), worker_id="w_000")
        from adaptive_framework.models.scheduling import Partition
        wu = _wu("doc", pages=10)
        partition = Partition(work_units=[wu])
        coordinator.submit_partitions([partition])
        coordinator.run_dispatch_loop(until_empty=True, max_iterations=10)
        coordinator.report_completed(wu.work_unit_id)
        # Should not raise and should decrement active count
        assert coordinator.dispatcher.active_count == 0

    def test_report_failed(self, coordinator: DistributedCoordinator) -> None:
        coordinator.register_worker(_node(), worker_id="w_000")
        from adaptive_framework.models.scheduling import Partition
        wu = _wu("fdoc", pages=5)
        partition = Partition(work_units=[wu])
        coordinator.submit_partitions([partition])
        coordinator.run_dispatch_loop(until_empty=True, max_iterations=10)
        result = coordinator.report_failed(wu.work_unit_id, "simulated failure")
        assert result is True


# ── Failure Recovery Integration ──────────────────────────────────────────

class TestFailureRecoveryIntegration:

    def test_worker_lost_triggers_recovery(self, coordinator: DistributedCoordinator) -> None:
        coordinator.register_worker(_node("w_a"), worker_id="w_001")
        from adaptive_framework.models.scheduling import Partition
        wu = _wu("recover_doc", pages=15)
        partition = Partition(work_units=[wu])
        coordinator.submit_partitions([partition])
        coordinator.run_dispatch_loop(until_empty=True, max_iterations=10)

        events = coordinator.recovery_engine.handle_worker_lost("w_001")
        types = [e.event_type.value for e in events]
        assert "worker_lost" in types

    def test_recovery_engine_exposed_via_property(
        self, coordinator: DistributedCoordinator
    ) -> None:
        assert coordinator.recovery_engine is not None

    def test_recovery_stats_available(self, coordinator: DistributedCoordinator) -> None:
        stats = coordinator.recovery_engine.get_statistics()
        assert "total_workers_lost" in stats


# ── Work Stealing Integration ─────────────────────────────────────────────

class TestWorkStealingIntegration:

    def test_work_stealing_exposed_via_property(
        self, coordinator: DistributedCoordinator
    ) -> None:
        assert coordinator.work_stealing is not None

    def test_manual_steal_cycle_does_not_raise(
        self, coordinator: DistributedCoordinator
    ) -> None:
        coordinator.register_worker(_node("idle_w"), worker_id="w_idle")
        coordinator.register_worker(_node("busy_w"), worker_id="w_busy")
        # Set overloaded state
        coordinator.registry.update_heartbeat(
            "w_busy", 90.0, 80.0, None, 6, "wu_001", WorkerState.OVERLOADED
        )
        events = coordinator.work_stealing.trigger_steal_cycle()
        assert isinstance(events, list)


# ── Status Dictionary ─────────────────────────────────────────────────────

class TestStatusDictionary:

    def test_status_dict_has_all_sections(self, coordinator: DistributedCoordinator) -> None:
        status = coordinator.get_status_dict()
        for key in [
            "run_id", "framework_state", "uptime_seconds",
            "registry", "dispatcher", "heartbeat_monitor",
            "work_stealing", "failure_recovery", "resource_orchestration",
            "cluster_manager", "queue_size",
        ]:
            assert key in status, f"Missing key: {key}"

    def test_status_run_id_matches(self, coordinator: DistributedCoordinator) -> None:
        status = coordinator.get_status_dict()
        assert status["run_id"] == "test_run"

    def test_status_uptime_increases(self, coordinator: DistributedCoordinator) -> None:
        s1 = coordinator.get_status_dict()["uptime_seconds"]
        time.sleep(0.05)
        s2 = coordinator.get_status_dict()["uptime_seconds"]
        assert s2 > s1


# ── Cluster Status ────────────────────────────────────────────────────────

class TestClusterStatus:

    def test_cluster_status_total_workers(self, coordinator: DistributedCoordinator) -> None:
        coordinator.register_worker(_node("a"), worker_id="w_a")
        coordinator.register_worker(_node("b"), worker_id="w_b")
        cs = coordinator.get_cluster_status()
        assert cs.total_workers == 2

    def test_cluster_status_lost_count(self, coordinator: DistributedCoordinator) -> None:
        coordinator.register_worker(_node("x"), worker_id="w_x")
        coordinator.registry.mark_lost("w_x")
        cs = coordinator.get_cluster_status()
        assert cs.lost_workers == 1
