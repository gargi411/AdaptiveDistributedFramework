"""Unit and invariant tests for Phase G4: Adaptive CPU/GPU Work Routing.

Covers:
  - WorkloadCharacterizer heuristic scoring and latency (< 1.0 ms)
  - Deterministic routing rules A-H
  - Anti-oscillation hysteresis cooldown and overrides
  - RoutingMetricsCollector metrics and latency percentiles
  - AdaptiveRoutingStrategyProxy seamless CPU fallback on GPU failure
  - ProcessingStrategyFactory routing (digital bypass vs scanned routing)
  - Invariants 1-8 (valid devices, explanations, zero clinical data, telemetry integrity)
"""

import time
import pytest
from typing import Any

from adaptive_framework.acceleration.adaptive_router import (
    AdaptiveWorkRouter,
    RoutingDecision,
    RoutingMetricsCollector,
)
from adaptive_framework.acceleration.workload_characterizer import (
    WorkloadCharacterizer,
    WorkloadProfile,
)
from adaptive_framework.config.models import (
    AdaptiveRoutingConfig,
    CPUThresholdConfig,
    GPUThresholdConfig,
    HysteresisConfig,
    SmallWorkloadConfig,
)
from adaptive_framework.document_processing.processing_strategy import (
    AdaptiveRoutingStrategyProxy,
    DirectExtractionStrategy,
    PageExtractionResult,
    ProcessingStrategyFactory,
)
from adaptive_framework.models.page import PageType
from adaptive_framework.models.runtime import ResourceSnapshot


# =========================================================================
# 1. Workload Characterization Tests
# =========================================================================

def test_workload_characterizer_synthetic():
    """Verify WorkloadCharacterizer heuristic produces correct normalized complexity."""
    characterizer = WorkloadCharacterizer()

    # Small digital page
    p_small = characterizer.characterize_synthetic(
        document_id="doc_small",
        page_number=1,
        page_count=1,
        char_count=1500,
        image_count=0,
        image_density=0.0,
        is_scanned=False,
    )
    assert 0.0 <= p_small.estimated_complexity <= 0.35
    assert not p_small.is_scanned
    assert p_small.estimated_work_units >= 0.1

    # Dense scanned page
    p_dense = characterizer.characterize_synthetic(
        document_id="doc_dense",
        page_number=1,
        page_count=1,
        char_count=0,
        image_count=2,
        image_density=0.90,
        is_scanned=True,
    )
    assert p_dense.estimated_complexity >= 0.55
    assert p_dense.is_scanned


def test_workload_characterizer_latency():
    """Verify WorkloadCharacterizer heuristic executes with ultra-low latency (< 1.0 ms)."""
    characterizer = WorkloadCharacterizer()
    t0 = time.perf_counter()
    for i in range(100):
        _ = characterizer.characterize_synthetic(
            document_id=f"doc_{i}",
            page_number=1,
            page_count=1,
            char_count=100 * i,
            image_count=1,
            image_density=0.5,
            is_scanned=True,
        )
    elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / 100.0
    assert elapsed_ms < 1.0, f"Characterization overhead {elapsed_ms:.4f} ms exceeded 1.0 ms limit"


# =========================================================================
# 2. Router Deterministic Rules (A-H)
# =========================================================================

@pytest.fixture
def base_router() -> AdaptiveWorkRouter:
    """Fixture providing a fresh AdaptiveWorkRouter."""
    config = AdaptiveRoutingConfig(
        enabled=True,
        gpu=GPUThresholdConfig(min_workload_complexity=0.35, max_utilization_percent=85.0, min_memory_available_mb=256.0),
        cpu=CPUThresholdConfig(max_utilization_percent=80.0),
        small_workload=SmallWorkloadConfig(max_pages=1, max_complexity=0.20),
        hysteresis=HysteresisConfig(enabled=True, cooldown_seconds=2.0, utilization_delta_threshold=5.0),
    )
    return AdaptiveWorkRouter(config=config)


def test_rule_a_gpu_unavailable(base_router):
    """Rule A: Route to CPU if GPU is absent or unavailable."""
    workload = WorkloadProfile(
        document_id="doc_a",
        page_count=5,
        estimated_complexity=0.80,
        is_scanned=True,
    )
    # GPU unavailable
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=20.0,
        memory_percent=30.0,
        gpu_available=False,
    )
    decision = base_router.route(workload, snap)
    assert decision.target_device == "CPU"
    assert decision.applied_rule == "RULE_A_NO_GPU"
    assert "unavailable" in decision.reason.lower()


def test_rule_b_corrupted_telemetry(base_router):
    """Rule B: Route to CPU if telemetry has invalid negative values."""
    workload = WorkloadProfile(
        document_id="doc_b",
        page_count=5,
        estimated_complexity=0.80,
        is_scanned=True,
    )
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=20.0,
        memory_percent=30.0,
        gpu_available=True,
    )
    # Simulate corrupted negative reading arriving at runtime
    snap.cpu_percent = -5.0
    decision = base_router.route(workload, snap)
    assert decision.target_device == "CPU"
    assert decision.applied_rule == "RULE_B_INVALID_METRICS"


def test_rule_c_small_workload(base_router):
    """Rule C: Route to CPU if workload is small (1 page, low complexity <= 0.20)."""
    workload = WorkloadProfile(
        document_id="doc_c",
        page_count=1,
        estimated_complexity=0.15,
        is_scanned=True,
    )
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=20.0,
        memory_percent=30.0,
        gpu_available=True,
        gpu_utilization_percent=10.0,
        gpu_memory_used_mb=100.0,
        gpu_memory_total_mb=4000.0,
    )
    decision = base_router.route(workload, snap)
    assert decision.target_device == "CPU"
    assert decision.applied_rule == "RULE_C_SMALL_WORKLOAD"
    assert "small" in decision.reason.lower()


def test_rule_d_gpu_saturated(base_router):
    """Rule D: Route to CPU if GPU utilization >= max_utilization_percent (85%)."""
    workload = WorkloadProfile(
        document_id="doc_d",
        page_count=4,
        estimated_complexity=0.75,
        is_scanned=True,
    )
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=90.0,  # > 85%
        gpu_memory_used_mb=200.0,
        gpu_memory_total_mb=4000.0,
    )
    decision = base_router.route(workload, snap)
    assert decision.target_device == "CPU"
    assert decision.applied_rule == "RULE_D_GPU_SATURATED"


def test_rule_e_insufficient_gpu_memory(base_router):
    """Rule E: Route to CPU if free GPU memory < min_memory_available_mb (256 MB)."""
    workload = WorkloadProfile(
        document_id="doc_e",
        page_count=4,
        estimated_complexity=0.75,
        is_scanned=True,
    )
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=20.0,
        gpu_memory_used_mb=3900.0,
        gpu_memory_total_mb=4000.0,  # 100 MB free < 256 MB
    )
    decision = base_router.route(workload, snap)
    assert decision.target_device == "CPU"
    assert decision.applied_rule == "RULE_E_LOW_GPU_MEMORY"


def test_rule_f_cpu_saturated(base_router):
    """Rule F: Route to GPU if CPU utilization >= max_utilization_percent (80%)."""
    workload = WorkloadProfile(
        document_id="doc_f",
        page_count=2,
        estimated_complexity=0.30,  # Under 0.35, but CPU saturated
        is_scanned=True,
    )
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=88.0,  # > 80%
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=25.0,
        gpu_memory_used_mb=200.0,
        gpu_memory_total_mb=4000.0,
    )
    decision = base_router.route(workload, snap)
    assert decision.target_device == "GPU"
    assert decision.applied_rule == "RULE_F_CPU_SATURATED"


def test_rule_g_complexity_justifies_gpu(base_router):
    """Rule G: Route to GPU if complexity >= min_workload_complexity (0.35) and GPU healthy."""
    workload = WorkloadProfile(
        document_id="doc_g",
        page_count=3,
        estimated_complexity=0.65,
        is_scanned=True,
    )
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=15.0,
        gpu_memory_used_mb=150.0,
        gpu_memory_total_mb=4000.0,
    )
    decision = base_router.route(workload, snap)
    assert decision.target_device == "GPU"
    assert decision.applied_rule == "RULE_G_COMPLEXITY_GPU"


def test_rule_h_default_fallback_cpu(base_router):
    """Rule H: Route to CPU if complexity < 0.35 and no CPU saturation."""
    workload = WorkloadProfile(
        document_id="doc_h",
        page_count=2,
        estimated_complexity=0.28,  # Between small_workload (0.20) and min_gpu (0.35)
        is_scanned=True,
    )
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=15.0,
        gpu_memory_used_mb=150.0,
        gpu_memory_total_mb=4000.0,
    )
    decision = base_router.route(workload, snap)
    assert decision.target_device == "CPU"
    assert decision.applied_rule == "RULE_H_CPU_DEFAULT"


# =========================================================================
# 3. Anti-Oscillation Hysteresis Tests
# =========================================================================

def test_hysteresis_dampening_cooldown(base_router):
    """Verify hysteresis retains device within cooldown period (anti-oscillation)."""
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=20.0,
        gpu_memory_used_mb=150.0,
        gpu_memory_total_mb=4000.0,
    )

    # 1. Establish GPU active
    w_gpu = WorkloadProfile(document_id="w1", page_count=3, estimated_complexity=0.70, is_scanned=True)
    d1 = base_router.route(w_gpu, snap)
    assert d1.target_device == "GPU"

    # 2. Immediately present a borderline workload that would normally yield CPU (0.30 < 0.35)
    w_cpu = WorkloadProfile(document_id="w2", page_count=2, estimated_complexity=0.30, is_scanned=True)
    d2 = base_router.route(w_cpu, snap)

    # Hysteresis dampens the transition within 2.0s cooldown
    assert d2.target_device == "GPU"
    assert d2.hysteresis_applied
    assert d2.applied_rule == "HYSTERESIS_DAMPENED"


def test_hysteresis_cooldown_override_on_gpu_disappearance(base_router):
    """Verify hysteresis allows immediate switch away from GPU if GPU becomes unavailable."""
    snap_good = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=20.0,
        gpu_memory_used_mb=150.0,
        gpu_memory_total_mb=4000.0,
    )
    w_gpu = WorkloadProfile(document_id="w1", page_count=3, estimated_complexity=0.70, is_scanned=True)
    d1 = base_router.route(w_gpu, snap_good)
    assert d1.target_device == "GPU"

    # Sudden GPU loss overrides cooldown
    snap_bad = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=False,
    )
    d2 = base_router.route(w_gpu, snap_bad)
    assert d2.target_device == "CPU"
    assert d2.applied_rule == "RULE_A_NO_GPU"


def test_hysteresis_reset_state(base_router):
    """Verify reset_state clears previous device and allows fresh routing."""
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=20.0,
        gpu_memory_used_mb=150.0,
        gpu_memory_total_mb=4000.0,
    )
    w_gpu = WorkloadProfile(document_id="w1", page_count=3, estimated_complexity=0.70, is_scanned=True)
    base_router.route(w_gpu, snap)

    # Reset
    base_router.reset_state()
    w_cpu = WorkloadProfile(document_id="w2", page_count=2, estimated_complexity=0.25, is_scanned=True)
    d = base_router.route(w_cpu, snap)
    assert d.target_device == "CPU"
    assert not d.hysteresis_applied


# =========================================================================
# 4. Fallback Execution and Strategy Proxy Tests
# =========================================================================

def test_strategy_proxy_gpu_fallback():
    """Verify AdaptiveRoutingStrategyProxy catches GPU exception and executes CPU fallback."""
    router = AdaptiveWorkRouter()

    class FailingGPUStrategy:
        def process(self, page: Any, page_number: int, document_id: str, file_path: str) -> PageExtractionResult:
            raise RuntimeError("Simulated OpenVINO GPU driver exception")

    class SuccessfulCPUStrategy:
        def process(self, page: Any, page_number: int, document_id: str, file_path: str) -> PageExtractionResult:
            res = PageExtractionResult(processing_method="ocr")
            res.text = "Clean CPU extraction text"
            return res

    proxy = AdaptiveRoutingStrategyProxy(
        router=router,
        gpu_strategy=FailingGPUStrategy(),
        cpu_strategy=SuccessfulCPUStrategy(),
    )

    workload = WorkloadProfile(document_id="doc_fb", page_count=3, estimated_complexity=0.75, is_scanned=True)
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=20.0,
        memory_percent=30.0,
        gpu_available=True,
        gpu_utilization_percent=10.0,
        gpu_memory_used_mb=100.0,
        gpu_memory_total_mb=4000.0,
    )

    # Execute proxy process
    result = proxy.process(
        page=None,
        page_number=1,
        document_id="doc_fb",
        file_path="sample.pdf",
        workload_profile=workload,
        resource_snapshot=snap,
    )

    assert result.ocr_device == "CPU_FALLBACK"
    assert result.text == "Clean CPU extraction text"
    assert any("GPU execution failed" in w for w in result.warnings)

    # Verify fallback event was recorded in router metrics
    summary = router.collector.get_summary()
    assert summary["fallback_decisions"] >= 1
    assert summary["gpu_failure_count"] >= 1


def test_strategy_factory_digital_bypass():
    """Verify ProcessingStrategyFactory returns DirectExtractionStrategy for DIGITAL pages."""
    factory = ProcessingStrategyFactory()
    strategy = factory.get_strategy(PageType.DIGITAL)
    assert isinstance(strategy, DirectExtractionStrategy)
    assert strategy.strategy_name == "direct_extraction"


def test_strategy_factory_scanned_adaptive_proxy():
    """Verify ProcessingStrategyFactory returns AdaptiveRoutingStrategyProxy for SCANNED pages with openvino."""
    factory = ProcessingStrategyFactory(ocr_backend="openvino", enable_adaptive_routing=True)
    strategy = factory.get_strategy(PageType.SCANNED)
    assert isinstance(strategy, AdaptiveRoutingStrategyProxy)
    assert strategy.strategy_name == "adaptive_routing_proxy"


# =========================================================================
# 5. Router Latency & Invariants
# =========================================================================

def test_router_decision_latency_sub_millisecond(base_router):
    """Invariant 3: Verify router decision overhead is strictly < 1.0 ms."""
    workload = WorkloadProfile(document_id="doc_lat", page_count=2, estimated_complexity=0.50, is_scanned=True)
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
        gpu_utilization_percent=20.0,
        gpu_memory_used_mb=150.0,
        gpu_memory_total_mb=4000.0,
    )
    for _ in range(20):
        decision = base_router.route(workload, snap)
        assert decision.routing_latency_ms < 1.0, f"Decision latency {decision.routing_latency_ms:.3f} ms >= 1.0 ms"


def test_zero_clinical_data_in_decisions(base_router):
    """Invariant 7: Zero clinical text or patient identities in routing records."""
    workload = WorkloadProfile(document_id="anon_doc_001", page_count=2, estimated_complexity=0.60, is_scanned=True)
    snap = ResourceSnapshot(
        node_id="node_1",
        cpu_percent=30.0,
        memory_percent=40.0,
        gpu_available=True,
    )
    decision = base_router.route(workload, snap)
    d_dict = decision.to_dict()

    # Verify no clinical fields exist in payload
    forbidden_keys = ["patient", "diagnosis", "symptoms", "clinical_notes", "ocr_text"]
    for k in forbidden_keys:
        assert k not in d_dict, f"Forbidden clinical key '{k}' found in decision dictionary"

    reason_lower = decision.reason.lower()
    for forbidden in ["patient", "cancer", "diabetes", "heart", "biopsy", "prescription"]:
        assert forbidden not in reason_lower, f"Clinical term '{forbidden}' leaked in reason text"


def test_routing_metrics_collector():
    """Verify RoutingMetricsCollector correctly calculates p95, mean, and device ratios."""
    collector = RoutingMetricsCollector()
    workload = WorkloadProfile(document_id="doc_m", page_count=1, estimated_complexity=0.5, is_scanned=True)

    for i in range(10):
        dec = RoutingDecision(
            decision_id=f"dec_{i}",
            timestamp="2026-09-06T12:00:00.000",
            target_device="GPU" if i < 7 else "CPU",
            reason="Test decision",
            confidence=0.9,
            workload_profile=workload,
            resource_snapshot=None,
            routing_latency_ms=0.05 + 0.01 * i,
        )
        collector.record_decision(dec)

    summary = collector.get_summary()
    assert summary["total_decisions"] == 10
    assert summary["gpu_decisions"] == 7
    assert summary["cpu_decisions"] == 3
    assert summary["device_selection_ratio"]["gpu"] == 0.7
    assert summary["device_selection_ratio"]["cpu"] == 0.3
    assert summary["latency_ms"]["p95"] > 0.0
