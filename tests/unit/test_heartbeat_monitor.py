"""Unit tests for HeartbeatMonitor (Module 4).

Tests cover: heartbeat ingestion, timeout detection, reconnect events,
callback firing, and statistics — all without a live Ray cluster.
"""

from __future__ import annotations

import time
import threading

import pytest

from adaptive_framework.coordinator.heartbeat_monitor.heartbeat_event import (
    HeartbeatEvent,
    HeartbeatEventType,
    HeartbeatPayload,
)
from adaptive_framework.coordinator.heartbeat_monitor.monitor import HeartbeatMonitor
from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.coordinator.worker_registry.registry import WorkerRegistry
from adaptive_framework.models.runtime import WorkerState


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def registry() -> WorkerRegistry:
    reg = WorkerRegistry()
    node = NodeInfo(hostname="h1", ip_address="127.0.0.1",
                    cpu_count_logical=2, cpu_count_physical=1, ram_total_gb=4.0)
    reg.register(node, worker_id="w_001")
    reg.register(node, worker_id="w_002")
    return reg


def _make_payload(worker_id: str, cpu: float = 50.0, queue: int = 0) -> HeartbeatPayload:
    return HeartbeatPayload(
        worker_id=worker_id,
        cpu_percent=cpu,
        ram_percent=40.0,
        queue_depth=queue,
        state=WorkerState.ACTIVE.value if queue > 0 else WorkerState.IDLE.value,
    )


# ── HeartbeatPayload ──────────────────────────────────────────────────────

class TestHeartbeatPayload:

    def test_payload_to_dict_has_all_fields(self) -> None:
        p = _make_payload("w_001")
        d = p.to_dict()
        assert "worker_id" in d
        assert "cpu_percent" in d
        assert "ram_percent" in d
        assert "queue_depth" in d
        assert "state" in d
        assert "timestamp" in d

    def test_payload_defaults(self) -> None:
        p = HeartbeatPayload(worker_id="w_001", cpu_percent=10.0,
                              ram_percent=20.0, queue_depth=0)
        assert p.gpu_percent is None
        assert p.current_task_id is None
        assert p.task_progress_percent is None


# ── HeartbeatEvent ────────────────────────────────────────────────────────

class TestHeartbeatEvent:

    def test_event_to_dict(self) -> None:
        evt = HeartbeatEvent(
            event_type=HeartbeatEventType.ALIVE,
            worker_id="w_001",
            message="Test",
        )
        d = evt.to_dict()
        assert d["event_type"] == "alive"
        assert d["worker_id"] == "w_001"
        assert "event_id" in d
        assert "timestamp" in d

    def test_event_repr(self) -> None:
        evt = HeartbeatEvent(
            event_type=HeartbeatEventType.TIMEOUT,
            worker_id="w_002",
            message="Timed out",
        )
        r = repr(evt)
        assert "timeout" in r
        assert "w_002" in r


# ── HeartbeatMonitor — record_heartbeat ──────────────────────────────────

class TestHeartbeatRecording:

    def test_record_heartbeat_updates_registry(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry, timeout_seconds=30.0)
        payload = _make_payload("w_001", cpu=75.0, queue=2)
        monitor.record_heartbeat(payload)
        rec = registry.get("w_001")
        assert rec is not None
        assert rec.cpu_percent == 75.0
        assert rec.queue_depth == 2

    def test_record_heartbeat_emits_alive_event(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry, timeout_seconds=30.0)
        monitor.record_heartbeat(_make_payload("w_001"))
        events = monitor.get_recent_events(10)
        assert len(events) >= 1
        assert events[0].event_type == HeartbeatEventType.ALIVE

    def test_record_heartbeat_increments_counter(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry, timeout_seconds=30.0)
        for _ in range(5):
            monitor.record_heartbeat(_make_payload("w_001"))
        stats = monitor.get_statistics()
        assert stats["total_heartbeats"] == 5

    def test_unknown_worker_state_defaults_to_active(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry)
        payload = HeartbeatPayload(
            worker_id="w_001",
            cpu_percent=50.0, ram_percent=40.0, queue_depth=0,
            state="invalid_state",
        )
        # Should not raise
        monitor.record_heartbeat(payload)


# ── Timeout detection ────────────────────────────────────────────────────

class TestTimeoutDetection:

    def test_timeout_marks_worker_lost(self, registry: WorkerRegistry) -> None:
        """Worker not seen within timeout window is marked LOST."""
        monitor = HeartbeatMonitor(
            registry=registry,
            timeout_seconds=0.1,  # very short for test
            check_interval_seconds=0.05,
        )
        # Register last-seen 1 second ago
        monitor._last_seen["w_001"] = time.monotonic() - 1.0
        monitor._check_all_workers()
        rec = registry.get("w_001")
        assert rec.state == WorkerState.LOST

    def test_timeout_fires_on_timeout_callback(self, registry: WorkerRegistry) -> None:
        fired: list[str] = []
        monitor = HeartbeatMonitor(
            registry=registry,
            timeout_seconds=0.1,
            on_timeout=lambda wid: fired.append(wid),
        )
        monitor._last_seen["w_001"] = time.monotonic() - 1.0
        monitor._check_all_workers()
        assert "w_001" in fired

    def test_timeout_adds_disconnected_event(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry, timeout_seconds=0.1)
        monitor._last_seen["w_001"] = time.monotonic() - 1.0
        monitor._check_all_workers()
        events = monitor.get_recent_events()
        types = [e.event_type for e in events]
        assert HeartbeatEventType.DISCONNECTED in types

    def test_worker_not_timed_out_within_window(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry, timeout_seconds=30.0)
        monitor._last_seen["w_001"] = time.monotonic()  # just seen
        monitor._check_all_workers()
        rec = registry.get("w_001")
        assert rec.state != WorkerState.LOST

    def test_no_last_seen_skips_worker(self, registry: WorkerRegistry) -> None:
        """Workers with no last_seen entry are not timed out."""
        monitor = HeartbeatMonitor(registry=registry, timeout_seconds=0.1)
        # Do NOT set _last_seen for w_001
        monitor._check_all_workers()
        rec = registry.get("w_001")
        assert rec.state == WorkerState.IDLE  # unchanged


# ── Reconnect ─────────────────────────────────────────────────────────────

class TestReconnect:

    def test_reconnect_clears_lost_set(self, registry: WorkerRegistry) -> None:
        fired_reconnect: list[str] = []
        monitor = HeartbeatMonitor(
            registry=registry,
            timeout_seconds=0.1,
            on_reconnect=lambda wid: fired_reconnect.append(wid),
        )
        # Force w_001 to LOST
        monitor._lost_workers.add("w_001")
        registry.mark_lost("w_001")

        # Send a heartbeat — should trigger reconnect path
        monitor.record_heartbeat(_make_payload("w_001"))

        assert "w_001" not in monitor.get_lost_workers()
        assert "w_001" in fired_reconnect

    def test_reconnect_emits_reconnected_event(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry)
        monitor._lost_workers.add("w_001")
        registry.mark_lost("w_001")
        monitor.record_heartbeat(_make_payload("w_001"))
        events = monitor.get_recent_events()
        types = [e.event_type for e in events]
        assert HeartbeatEventType.RECONNECTED in types


# ── Lifecycle ────────────────────────────────────────────────────────────

class TestLifecycle:

    def test_start_stop(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry, check_interval_seconds=10.0)
        monitor.start()
        assert monitor.is_running
        monitor.stop()
        assert not monitor.is_running

    def test_double_start_is_safe(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry, check_interval_seconds=10.0)
        monitor.start()
        monitor.start()  # should not raise
        monitor.stop()

    def test_statistics_reflect_state(self, registry: WorkerRegistry) -> None:
        monitor = HeartbeatMonitor(registry=registry)
        for _ in range(3):
            monitor.record_heartbeat(_make_payload("w_001"))
        stats = monitor.get_statistics()
        assert stats["total_heartbeats"] == 3
        assert stats["history_size"] >= 3
