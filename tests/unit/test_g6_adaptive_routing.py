"""Unit tests for Phase G6 Data-Driven Adaptive CPU/GPU Work Routing.

Covers:
  - Workload cost model normalization, scaling, and determinism
  - Backend suitability scoring under diverse hardware states (cold, warm, overload)
  - Routing decisions, confidence, and explanations
  - Hysteresis stability around boundary conditions
  - Fault tolerance & CPU fallback execution
  - Zero leakage invariant verification (no future execution / oracle inputs)
"""

import time
from unittest.mock import MagicMock

import pytest

from adaptive_framework.acceleration.adaptive_router import AdaptiveRoutingConfig
from adaptive_framework.acceleration.adaptive_router_g6 import (
    AdaptiveWorkRouterG6,
    RoutingDecisionG6,
)
from adaptive_framework.acceleration.backend_suitability import (
    BackendSuitabilityEstimator,
    BackendSuitabilityResult,
)
from adaptive_framework.acceleration.workload_characterizer import WorkloadProfile
from adaptive_framework.acceleration.workload_cost_model import (
    WorkloadCostEstimator,
    WorkloadCostProfile,
)
from adaptive_framework.document_processing.processing_strategy import (
    AdaptiveRoutingStrategyProxy,
    PageExtractionResult,
)
from adaptive_framework.models.runtime import ResourceSnapshot


# =========================================================================
# Category A: Workload Cost Tests
# =========================================================================

def test_cost_estimator_deterministic():
    """Verify identical inputs yield identical compute intensity profiles."""
    estimator = WorkloadCostEstimator()
    profile = WorkloadProfile(
        document_id="doc_test_1",
        page_number=1,
        page_count=5,
        width_pts=612.0,
        height_pts=792.0,
        image_density=0.50,
        estimated_complexity=0.60,
    )

    cost1 = estimator.estimate_cost(profile)
    cost2 = estimator.estimate_cost(profile)

    assert cost1.normalized_compute_intensity == pytest.approx(cost2.normalized_compute_intensity)
    assert cost1.total_pixels == cost2.total_pixels
    assert cost1.estimated_work_units == cost2.estimated_work_units


def test_cost_estimator_normalization_bounds():
    """Verify compute intensity is strictly bounded in [0.0, 1.0]."""
    estimator = WorkloadCostEstimator()

    # Extreme minimal profile
    p_min = WorkloadProfile(
        document_id="min_doc",
        width_pts=10.0,
        height_pts=10.0,
        image_density=0.0,
        estimated_complexity=0.0,
    )
    cost_min = estimator.estimate_cost(p_min)
    assert 0.0 <= cost_min.normalized_compute_intensity <= 1.0

    # Extreme maximal profile
    p_max = WorkloadProfile(
        document_id="max_doc",
        width_pts=5000.0,
        height_pts=5000.0,
        image_density=1.0,
        estimated_complexity=1.0,
        char_count=10000,
    )
    cost_max = estimator.estimate_cost(p_max)
    assert 0.0 <= cost_max.normalized_compute_intensity <= 1.0
    assert cost_max.normalized_compute_intensity > cost_min.normalized_compute_intensity


def test_cost_estimator_page_dimensions_scaling():
    """Verify larger page dimensions produce proportionally higher total pixel counts."""
    estimator = WorkloadCostEstimator()
    p_letter = WorkloadProfile(document_id="letter", width_pts=612.0, height_pts=792.0)
    p_a3 = WorkloadProfile(document_id="a3", width_pts=841.9, height_pts=1190.5)

    c_letter = estimator.estimate_cost(p_letter)
    c_a3 = estimator.estimate_cost(p_a3)

    assert c_a3.total_pixels > c_letter.total_pixels
    assert c_a3.normalized_compute_intensity > c_letter.normalized_compute_intensity


def test_cost_estimator_image_density_impact():
    """Verify image density increases compute intensity."""
    estimator = WorkloadCostEstimator()
    p_low_img = WorkloadProfile(document_id="low_img", image_density=0.10)
    p_high_img = WorkloadProfile(document_id="high_img", image_density=0.90)

    c_low = estimator.estimate_cost(p_low_img)
    c_high = estimator.estimate_cost(p_high_img)

    assert c_high.normalized_compute_intensity > c_low.normalized_compute_intensity


# =========================================================================
# Category B: Backend Suitability Tests
# =========================================================================

def test_suitability_cpu_preference_small_workload():
    """Small simple page favors CPU."""
    bse = BackendSuitabilityEstimator()
    cost = WorkloadCostProfile(
        document_id="simple_doc",
        page_count=1,
        total_pixels=500000,
        image_density=0.10,
        complexity_score=0.15,
        normalized_compute_intensity=0.18,
    )
    res = bse.estimate_suitability(cost, is_gpu_warm=True)
    assert res.recommended_backend == "CPU"
    assert res.cpu_score > res.gpu_score
    assert "cpu" in res.primary_reason


def test_suitability_gpu_preference_large_workload():
    """Large high-intensity page favors GPU when warm and free."""
    bse = BackendSuitabilityEstimator()
    cost = WorkloadCostProfile(
        document_id="large_doc",
        page_count=10,
        total_pixels=4000000,
        image_density=0.85,
        complexity_score=0.75,
        normalized_compute_intensity=0.80,
    )
    res = bse.estimate_suitability(cost, is_gpu_warm=True)
    assert res.recommended_backend == "GPU"
    assert res.gpu_score > res.cpu_score


def test_suitability_gpu_unavailable():
    """When GPU is unavailable in snapshot, CPU is 100% selected."""
    bse = BackendSuitabilityEstimator()
    cost = WorkloadCostProfile(
        document_id="doc",
        normalized_compute_intensity=0.90,
    )
    snap = ResourceSnapshot(node_id="n1", gpu_available=False)
    res = bse.estimate_suitability(cost, resource=snap)

    assert res.recommended_backend == "CPU"
    assert res.cpu_score == 1.0
    assert res.gpu_score == 0.0
    assert res.primary_reason == "gpu_hardware_unavailable"


def test_suitability_gpu_overloaded():
    """When GPU utilization > 80%, suitability shifts towards CPU."""
    bse = BackendSuitabilityEstimator(gpu_overload_threshold_pct=80.0)
    cost = WorkloadCostProfile(
        document_id="doc",
        normalized_compute_intensity=0.60,
    )
    snap_busy = ResourceSnapshot(
        node_id="n1",
        gpu_available=True,
        gpu_utilization_percent=95.0,
        cpu_percent=15.0,
    )
    res = bse.estimate_suitability(cost, resource=snap_busy)
    assert res.recommended_backend == "CPU"


def test_suitability_gpu_cold_start_amortization():
    """Cold GPU is penalized on small batches (<15 pages), but amortizes on large batches."""
    bse = BackendSuitabilityEstimator(cold_start_amortization_pages=15)

    # 2 pages: cold penalty should make CPU win
    cost_small = WorkloadCostProfile(
        document_id="small",
        page_count=2,
        normalized_compute_intensity=0.55,
    )
    res_cold_small = bse.estimate_suitability(cost_small, is_gpu_warm=False)
    assert res_cold_small.recommended_backend == "CPU"
    assert "cold_gpu" in res_cold_small.primary_reason

    # 25 pages: batch size amortizes cold compilation
    cost_large = WorkloadCostProfile(
        document_id="large",
        page_count=25,
        normalized_compute_intensity=0.65,
    )
    res_cold_large = bse.estimate_suitability(cost_large, is_gpu_warm=False)
    assert res_cold_large.recommended_backend == "GPU"
    assert "amortizes" in res_cold_large.primary_reason


# =========================================================================
# Category C: Routing and Hysteresis Tests
# =========================================================================

def test_router_deterministic_decision():
    """Router produces consistent decisions and valid confidence."""
    router = AdaptiveWorkRouterG6()
    workload = WorkloadProfile(
        document_id="doc1",
        page_count=5,
        estimated_complexity=0.70,
        image_density=0.80,
    )
    dec = router.route(workload, is_gpu_warm=True)
    assert isinstance(dec, RoutingDecisionG6)
    assert dec.target_device in ("CPU", "GPU")
    assert 0.0 <= dec.confidence <= 1.0
    assert dec.routing_latency_ms >= 0.0
    assert len(dec.primary_reason) > 0


def test_router_hysteresis_dampens_rapid_flip():
    """Hysteresis dampens rapid device switching during cooldown period."""
    config = AdaptiveRoutingConfig()
    config.hysteresis.enabled = True
    config.hysteresis.cooldown_seconds = 2.0
    router = AdaptiveWorkRouterG6(config=config)

    # 1. High complexity routes to GPU
    w_gpu = WorkloadProfile(document_id="d1", page_count=10, estimated_complexity=0.85, image_density=0.90)
    d1 = router.route(w_gpu)
    assert d1.target_device == "GPU"

    # 2. Immediately route borderline small workload (candidate is CPU)
    w_cpu = WorkloadProfile(document_id="d2", page_count=1, estimated_complexity=0.30, image_density=0.20)
    d2 = router.route(w_cpu)

    # Anti-oscillation should retain GPU during cooldown
    assert d2.target_device == "GPU"
    assert d2.hysteresis_applied is True
    assert "Hysteresis retained GPU" in d2.primary_reason


def test_router_hysteresis_critical_override():
    """Hysteresis is overridden immediately when GPU becomes unavailable."""
    config = AdaptiveRoutingConfig()
    config.hysteresis.enabled = True
    config.hysteresis.cooldown_seconds = 5.0
    router = AdaptiveWorkRouterG6(config=config)

    # 1. Route to GPU
    w_gpu = WorkloadProfile(document_id="d1", page_count=10, estimated_complexity=0.85, image_density=0.90)
    router.route(w_gpu)

    # 2. GPU suddenly becomes unavailable
    snap_no_gpu = ResourceSnapshot(node_id="n1", gpu_available=False)
    d_fail = router.route(w_gpu, resource=snap_no_gpu)

    # Overrides cooldown and flips to CPU safely
    assert d_fail.target_device == "CPU"
    assert d_fail.was_switched is True


# =========================================================================
# Category D: Fallback Tests
# =========================================================================

def test_fallback_under_gpu_failure():
    """Verify GPU execution failure triggers seamless CPU fallback with zero loss."""
    router = AdaptiveWorkRouterG6()

    class FailingGPUStrategy:
        def process(self, page, page_number, document_id, file_path):
            raise RuntimeError("Hardware accelerated kernel fault")

    class FallbackCPUStrategy:
        def process(self, page, page_number, document_id, file_path):
            res = PageExtractionResult(processing_method="ocr")
            res.text = "Valid patient record extracted via CPU fallback"
            return res

    proxy = AdaptiveRoutingStrategyProxy(
        gpu_strategy=FailingGPUStrategy(),
        cpu_strategy=FallbackCPUStrategy(),
        router=router,
    )

    workload = WorkloadProfile(
        document_id="test_fallback_doc",
        page_count=5,
        estimated_complexity=0.80,
        image_density=0.85,
    )
    snap = ResourceSnapshot(node_id="n1", gpu_available=True)

    result = proxy.process(
        page="dummy_page",
        page_number=1,
        document_id="test_fallback_doc",
        file_path="doc.pdf",
        workload_profile=workload,
        resource_snapshot=snap,
    )

    assert result.ocr_device == "CPU_FALLBACK"
    assert "Valid patient record extracted" in result.text
    assert "fell back to CPU OCR" in result.warnings[-1]


# =========================================================================
# Category E: Leakage Prevention Tests
# =========================================================================

def test_zero_leakage_api_boundaries():
    """Verify router and cost estimators accept NO future timings or oracle values."""
    import inspect

    # 1. WorkloadCostEstimator.estimate_cost signature
    cost_sig = inspect.signature(WorkloadCostEstimator.estimate_cost)
    forbidden_terms = ["oracle", "ground_truth", "actual_time", "label", "future", "g5_result"]
    for param in cost_sig.parameters.keys():
        for f in forbidden_terms:
            assert f not in param.lower(), f"Forbidden parameter '{param}' in estimate_cost"

    # 2. BackendSuitabilityEstimator.estimate_suitability signature
    suit_sig = inspect.signature(BackendSuitabilityEstimator.estimate_suitability)
    for param in suit_sig.parameters.keys():
        for f in forbidden_terms:
            assert f not in param.lower(), f"Forbidden parameter '{param}' in estimate_suitability"

    # 3. AdaptiveWorkRouterG6.route signature
    route_sig = inspect.signature(AdaptiveWorkRouterG6.route)
    for param in route_sig.parameters.keys():
        for f in forbidden_terms:
            assert f not in param.lower(), f"Forbidden parameter '{param}' in route"
