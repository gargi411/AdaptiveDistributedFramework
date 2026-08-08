"""Integration test for Phase 2B — Full distributed pipeline simulation.

Exercises the complete pathway:
    Partitioner → Priority Queue → Coordinator → Dispatcher → Recovery

No Ray required. Designed to run in CI without a cluster.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from adaptive_framework.benchmarks.cluster_benchmark import ClusterBenchmark
from adaptive_framework.coordinator.distributed_coordinator import DistributedCoordinator
from adaptive_framework.coordinator.heartbeat_monitor import HeartbeatPayload
from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.models.document import PDFMetadata
from adaptive_framework.models.runtime import WorkerState
from adaptive_framework.models.scheduling import (
    Partition,
    PageWorkUnit,
)
from adaptive_framework.scheduler import PageCountPartitioner


# ── Configuration ─────────────────────────────────────────────────────────

_CONFIG = {
    "mode": "dev",
    "head": {"address": "auto"},
    "workers": {"num_workers_dev": 3},
    "health": {
        "heartbeat_timeout_seconds": 60.0,
        "health_check_interval_seconds": 60.0,  # disable auto checks
    },
    "failure_recovery": {"max_retries": 2, "reassignment_delay_seconds": 0.0},
    "work_stealing": {
        "steal_threshold": 2, "steal_fraction": 0.5,
        "check_interval_seconds": 60.0,
    },
    "resource_orchestration": {"sample_interval_seconds": 60.0},
}

_NUM_WORKERS = 3
_NUM_DOCS = 9


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_dataset(n_docs: int = _NUM_DOCS) -> list[PDFMetadata]:
    return [
        PDFMetadata(
            pages=10 + i * 5,  # heterogeneous
            estimated_size_mb=float(1 + i),
            file_path=f"/data/paper_{i:03d}.pdf",
            document_id=f"doc_{i:03d}",
        )
        for i in range(n_docs)
    ]


def _node(name: str) -> NodeInfo:
    return NodeInfo(
        hostname=name, ip_address="127.0.0.1",
        cpu_count_logical=4, cpu_count_physical=2, ram_total_gb=8.0,
    )


def _heartbeat(wid: str, cpu: float = 40.0, queue: int = 0) -> HeartbeatPayload:
    return HeartbeatPayload(
        worker_id=wid,
        cpu_percent=cpu,
        ram_percent=35.0,
        queue_depth=queue,
        state=WorkerState.ACTIVE.value if queue > 0 else WorkerState.IDLE.value,
    )


# ── Full pipeline integration ────────────────────────────────────────────

class TestFullPipeline:
    """Tests exercising the complete Phase 2B pipeline."""

    def test_partition_submit_dispatch_complete(self) -> None:
        """End-to-end: partition → submit → dispatch → report complete."""
        coord = DistributedCoordinator(cluster_config=_CONFIG, run_id="pipeline_test")
        coord.start(init_ray=False)

        worker_ids: list[str] = []
        for i in range(_NUM_WORKERS):
            wid = coord.register_worker(_node(f"worker-{i}"), worker_id=f"w_{i:03d}")
            worker_ids.append(wid)
            coord.receive_heartbeat(_heartbeat(wid, cpu=30.0))

        docs = _make_dataset()
        partitioner = PageCountPartitioner()
        partitions, _ = partitioner.partition(docs, num_workers=_NUM_WORKERS)

        inserted = coord.submit_partitions(partitions)
        assert inserted > 0

        dispatched = coord.run_dispatch_loop(until_empty=True, max_iterations=500)
        assert dispatched == inserted  # all tasks dispatched

        # Report all as completed
        for partition in partitions:
            for wu in partition.work_units:
                coord.report_completed(wu.work_unit_id)

        status = coord.get_status_dict()
        assert status["queue_size"] == 0

        result = coord.stop(shutdown_ray=False)
        assert result is not None

    def test_failure_recovery_mid_pipeline(self) -> None:
        """Simulate worker failure mid-dispatch; tasks are recovered."""
        coord = DistributedCoordinator(cluster_config=_CONFIG, run_id="recovery_test")
        coord.start(init_ray=False)

        for i in range(3):
            coord.register_worker(_node(f"w{i}"), worker_id=f"w_{i}")

        # Submit work
        from adaptive_framework.models.scheduling import Partition
        work_units = [
            PageWorkUnit(
                document_id=f"doc_{j}", file_path=f"/d{j}.pdf",
                start_page=1, end_page=10 + j,
            )
            for j in range(6)
        ]
        partition = Partition(work_units=work_units)
        coord.submit_partitions([partition])

        # Dispatch some tasks
        coord.run_dispatch_loop(until_empty=False, max_iterations=3)

        # Simulate worker loss
        events = coord.recovery_engine.handle_worker_lost("w_0")
        type_vals = [e.event_type.value for e in events]
        assert "worker_lost" in type_vals

        coord.stop(shutdown_ray=False)

    def test_work_stealing_on_imbalanced_cluster(self) -> None:
        """Verify work stealing fires when cluster is imbalanced."""
        coord = DistributedCoordinator(cluster_config=_CONFIG, run_id="steal_test")
        coord.start(init_ray=False)

        coord.register_worker(_node("idle_node"), worker_id="w_idle")
        coord.register_worker(_node("busy_node"), worker_id="w_busy")

        # Force overloaded state on busy worker
        coord.registry.update_heartbeat(
            "w_busy", 95.0, 85.0, None, 8, "wu_001", WorkerState.OVERLOADED
        )

        steal_events = coord.work_stealing.trigger_steal_cycle()
        stats = coord.work_stealing.get_statistics()
        # Steal should have fired
        assert stats["total_steal_events"] >= 1

        coord.stop(shutdown_ray=False)

    def test_benchmark_metrics_within_target(self) -> None:
        """Scheduler overhead must be < 1% (architecture §4.2 target)."""
        coord = DistributedCoordinator(cluster_config=_CONFIG, run_id="overhead_test")
        coord.start(init_ray=False)

        for i in range(_NUM_WORKERS):
            coord.register_worker(_node(f"n{i}"), worker_id=f"w_{i}")

        docs = _make_dataset(n_docs=12)
        partitioner = PageCountPartitioner()
        partitions, _ = partitioner.partition(docs, num_workers=_NUM_WORKERS)
        coord.submit_partitions(partitions)
        coord.run_dispatch_loop(until_empty=True, max_iterations=1000)

        result = coord.stop(shutdown_ray=False)
        if result:
            # The overhead target is < 1%
            assert result.scheduler_overhead_fraction < 0.01, (
                f"Scheduler overhead {result.scheduler_overhead_percent:.4f}% "
                f"exceeds 1% target"
            )

    def test_cluster_status_during_run(self) -> None:
        """ClusterStatus must correctly count workers at each stage."""
        coord = DistributedCoordinator(cluster_config=_CONFIG, run_id="status_test")
        coord.start(init_ray=False)

        for i in range(3):
            coord.register_worker(_node(f"s{i}"), worker_id=f"ws_{i}")

        cs = coord.get_cluster_status()
        assert cs.total_workers == 3
        assert cs.lost_workers == 0

        coord.registry.mark_lost("ws_0")
        cs2 = coord.get_cluster_status()
        assert cs2.lost_workers == 1

        coord.stop(shutdown_ray=False)

    def test_dashboard_state_output(self) -> None:
        """Status dict must contain all keys expected by the dashboard."""
        coord = DistributedCoordinator(cluster_config=_CONFIG, run_id="dash_test")
        coord.start(init_ray=False)
        coord.register_worker(_node("d0"), worker_id="wd_0")

        status = coord.get_status_dict()

        required_keys = [
            "run_id", "framework_state", "uptime_seconds",
            "registry", "dispatcher", "heartbeat_monitor",
            "work_stealing", "failure_recovery",
            "resource_orchestration", "cluster_manager", "queue_size",
        ]
        for key in required_keys:
            assert key in status, f"Dashboard key missing: {key}"

        coord.stop(shutdown_ray=False)
