"""Unit tests for Phase G3: GPU Utilization and Resource Monitoring.

Tests:
  - GPUMonitor abstraction and hardware queries
  - Graceful handling of unavailable metrics (temperature, power)
  - ResourceSnapshot model extensions and backward compatibility
  - ResourceMonitor unified sampling engine and history ring buffer
  - DashboardStateStore GPU metric tracking
  - System utility extensions (physical core counts, memory breakdown)
"""

from __future__ import annotations

import time
import pytest

from adaptive_framework.acceleration.gpu_monitor import GPUMonitor, GPUMetrics
from adaptive_framework.acceleration.resource_monitor import ResourceMonitor
from adaptive_framework.models.runtime import ResourceSnapshot
from adaptive_framework.core.exceptions import ValidationError
from adaptive_framework.utils.system_utils import (
    get_cpu_count,
    get_cpu_physical_count,
    get_memory_breakdown_mb,
    get_cpu_percent,
    get_memory_percent,
)
from dashboard.state.dashboard_state import DashboardStateStore


# ---------------------------------------------------------------------------
# GPUMonitor Tests
# ---------------------------------------------------------------------------

class TestGPUMonitor:
    """Tests for the dedicated GPUMonitor abstraction."""

    def test_gpumonitor_init_and_availability(self) -> None:
        with GPUMonitor() as monitor:
            assert isinstance(monitor.is_available(), bool)
            info = monitor.get_device_info()
            assert isinstance(info, dict)
            if monitor.is_available():
                assert "Intel" in info.get("vendor", "")
                assert "openvino_device_id" in info
                assert info.get("total_memory_mb") is not None
                assert info["total_memory_mb"] > 0

    def test_gpumonitor_sample(self) -> None:
        with GPUMonitor() as monitor:
            sample = monitor.sample()
            assert isinstance(sample, GPUMetrics)
            assert isinstance(sample.timestamp, str)
            assert isinstance(sample.supported_metrics, dict)
            summary = sample.format_summary()
            assert "GPU" in summary
            assert "Utilization:" in summary
            assert "Memory:" in summary
            # Verify serialization
            d = sample.to_dict()
            assert d["device_name"] == sample.device_name
            assert "supported_metrics" in d

    def test_gpumonitor_temperature_and_power_are_none(self) -> None:
        """Verify strict adherence to no-fake-values rule."""
        with GPUMonitor() as monitor:
            # On Intel Iris Xe iGPU, temp and power must be None (UNAVAILABLE)
            assert monitor.get_temperature() is None
            assert monitor.get_power() is None
            sample = monitor.sample()
            assert sample.temperature_c is None
            assert sample.power_w is None
            assert sample.supported_metrics["temperature"] is False
            assert sample.supported_metrics["power"] is False

    def test_gpumonitor_memory_query(self) -> None:
        with GPUMonitor() as monitor:
            mem = monitor.get_memory()
            assert "total_mb" in mem
            assert "used_mb" in mem
            assert "free_mb" in mem
            if monitor.is_available():
                assert mem["total_mb"] is not None
                assert mem["total_mb"] > 0

    def test_gpumonitor_utilization_bounds(self) -> None:
        with GPUMonitor() as monitor:
            util = monitor.get_utilization()
            if util is not None:
                assert 0.0 <= util <= 100.0


# ---------------------------------------------------------------------------
# ResourceSnapshot Model Tests
# ---------------------------------------------------------------------------

class TestResourceSnapshotG3:
    """Tests for extended ResourceSnapshot model."""

    def test_resource_snapshot_backward_compatibility(self) -> None:
        """Ensure existing instantiation signature works identically."""
        snap = ResourceSnapshot(
            node_id="test_node",
            cpu_percent=55.0,
            memory_percent=42.0,
        )
        assert snap.node_id == "test_node"
        assert snap.cpu_percent == 55.0
        assert snap.memory_percent == 42.0
        assert snap.ram_percent == 42.0
        assert snap.gpu_percent is None
        assert snap.gpu_utilization_percent is None

    def test_resource_snapshot_g3_fields(self) -> None:
        """Verify all new G3 fields are populated and accessible."""
        snap = ResourceSnapshot(
            node_id="node_g3",
            cpu_percent=25.0,
            memory_percent=60.0,
            cpu_count=8,
            ram_used_mb=8192.0,
            ram_available_mb=8192.0,
            gpu_available=True,
            gpu_name="Intel(R) Iris(R) Xe Graphics (iGPU)",
            gpu_device="GPU.0",
            gpu_utilization_percent=12.5,
            gpu_memory_used_mb=128.0,
            gpu_memory_total_mb=7128.0,
            gpu_temperature_c=None,
            gpu_power_w=None,
            queue_length=4,
            active_workers=2,
            ocr_latency_ms=45.2,
            ocr_throughput_pages_s=22.1,
        )
        assert snap.cpu_count == 8
        assert snap.gpu_available is True
        assert snap.gpu_percent == 12.5  # Synchronized alias
        assert snap.gpu_utilization_percent == 12.5
        assert snap.ocr_latency_ms == 45.2
        assert snap.ocr_throughput_pages_s == 22.1

        summary = snap.format_summary()
        assert "Intel(R) Iris(R) Xe Graphics (iGPU)" in summary
        assert "12.5%" in summary
        assert "UNAVAILABLE" in summary  # Temp / power
        assert "45.20 ms" in summary

    def test_resource_snapshot_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            ResourceSnapshot(cpu_percent=-1.0, memory_percent=50.0)
        with pytest.raises(ValidationError):
            ResourceSnapshot(cpu_percent=105.0, memory_percent=50.0)
        with pytest.raises(ValidationError):
            ResourceSnapshot(cpu_percent=50.0, memory_percent=105.0)
        with pytest.raises(ValidationError):
            ResourceSnapshot(cpu_percent=50.0, memory_percent=50.0, gpu_percent=150.0)


# ---------------------------------------------------------------------------
# ResourceMonitor Tests
# ---------------------------------------------------------------------------

class TestResourceMonitor:
    """Tests for the unified ResourceMonitor sampling engine."""

    def test_resource_monitor_sample(self) -> None:
        with ResourceMonitor(node_id="mon_node") as monitor:
            snap = monitor.sample(queue_length=5, active_workers=3)
            assert snap.node_id == "mon_node"
            assert 0.0 <= snap.cpu_percent <= 100.0
            assert 0.0 <= snap.memory_percent <= 100.0
            assert snap.queue_length == 5
            assert snap.active_workers == 3
            assert snap.cpu_count >= 1
            assert snap.ram_used_mb >= 0.0

    def test_resource_monitor_background_sampling_and_history(self) -> None:
        with ResourceMonitor(node_id="bg_node", history_size=20) as monitor:
            monitor.start_background_sampling(interval_seconds=0.1)
            time.sleep(0.35)
            monitor.stop_background_sampling()

            recent = monitor.get_recent_snapshots()
            assert len(recent) >= 2
            latest = monitor.get_latest_snapshot()
            assert latest is not None
            assert latest == recent[-1]

    def test_resource_monitor_providers(self) -> None:
        with ResourceMonitor(node_id="prov_node") as monitor:
            monitor.register_queue_provider(lambda: 7)
            monitor.register_worker_provider(lambda: 4)

            snap = monitor.sample()
            assert snap.queue_length == 7
            assert snap.active_workers == 4

            # Explicit override takes precedence
            snap_ovr = monitor.sample(queue_length=12, active_workers=8)
            assert snap_ovr.queue_length == 12
            assert snap_ovr.active_workers == 8


# ---------------------------------------------------------------------------
# DashboardStateStore Integration Tests
# ---------------------------------------------------------------------------

class TestDashboardStateStoreG3:
    """Tests for DashboardStateStore GPU history tracking."""

    def test_dashboard_state_empty_state_has_gpu(self) -> None:
        store = DashboardStateStore()
        state = store.get()
        assert "gpu" in state
        assert state["gpu"]["available"] is False
        assert state["gpu"]["utilization_percent"] is None

    def test_dashboard_state_gpu_history_update(self) -> None:
        store = DashboardStateStore()
        payload = {
            "registry": {"avg_cpu_percent": 15.0, "avg_ram_percent": 60.0},
            "dispatcher": {"total_completed": 5},
            "queue_size": 2,
            "gpu": {
                "available": True,
                "device_name": "Intel(R) Iris(R) Xe Graphics (iGPU)",
                "utilization_percent": 8.5,
                "memory_used_mb": 256.0,
            },
        }
        store.update(payload)

        util_hist = store.get_gpu_util_history()
        mem_hist = store.get_gpu_mem_history()
        assert len(util_hist) == 1
        assert util_hist[0] == 8.5
        assert len(mem_hist) == 1
        assert mem_hist[0] == 256.0


# ---------------------------------------------------------------------------
# System Utility Extensions Tests
# ---------------------------------------------------------------------------

class TestSystemUtilsExtensions:
    """Tests for physical core count and memory breakdown utilities."""

    def test_cpu_physical_count(self) -> None:
        phys = get_cpu_physical_count()
        log = get_cpu_count(logical=True)
        assert phys >= 1
        assert log >= phys

    def test_memory_breakdown_mb(self) -> None:
        tot_mb, used_mb, avail_mb = get_memory_breakdown_mb()
        assert tot_mb > 0
        assert used_mb > 0
        assert avail_mb > 0
        assert tot_mb >= used_mb
