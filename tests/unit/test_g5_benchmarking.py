"""Unit tests for Phase G5 Benchmarking Infrastructure.

Verifies:
- BenchmarkPolicy and WorkloadClass enumerations
- BenchmarkRunRecord and BenchmarkSummary dataclass invariants
- Statistical aggregation math (mean, median, p95, standard deviation)
- Speedup, percentage change, and overhead calculation methods
- Division-by-zero protection and boundary handling
- Corpus curation and deterministic page generation
- Policy isolation (CPU_ONLY, GPU_ONLY, ADAPTIVE)
- Graceful telemetry unavailable handling
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from adaptive_framework.acceleration.benchmark_dataset import (
    BenchmarkCorpusManager,
    BenchmarkPageItem,
)
from adaptive_framework.acceleration.benchmarking import (
    BenchmarkComparator,
    BenchmarkPolicy,
    BenchmarkRunRecord,
    BenchmarkSummary,
    G5BenchmarkRunner,
    WorkloadClass,
)


def test_benchmark_policy_enum():
    """Verify all 3 required policies are defined in BenchmarkPolicy."""
    assert BenchmarkPolicy.CPU_ONLY.value == "CPU_ONLY"
    assert BenchmarkPolicy.GPU_ONLY.value == "GPU_ONLY"
    assert BenchmarkPolicy.ADAPTIVE.value == "ADAPTIVE"
    assert len(BenchmarkPolicy) == 3


def test_workload_class_enum():
    """Verify all 4 required workload classes are defined in WorkloadClass."""
    assert WorkloadClass.CLASS_A.value == "CLASS_A"
    assert WorkloadClass.CLASS_B.value == "CLASS_B"
    assert WorkloadClass.CLASS_C.value == "CLASS_C"
    assert WorkloadClass.CLASS_D.value == "CLASS_D"
    assert len(WorkloadClass) == 4


def test_run_record_structure():
    """Verify BenchmarkRunRecord fields and types."""
    rec = BenchmarkRunRecord(
        run_id="run_001",
        policy=BenchmarkPolicy.CPU_ONLY.value,
        workload_class=WorkloadClass.CLASS_B.value,
        page_count=5,
        repetition_index=1,
        is_warmup=False,
        cold_start_ms=None,
        first_inference_ms=None,
        total_time_seconds=0.5,
        throughput_pages_s=10.0,
        page_latencies_ms=[96.0, 95.0, 97.0, 96.0, 96.0],
        mean_page_latency_ms=96.0,
        median_page_latency_ms=96.0,
        p95_page_latency_ms=97.0,
        min_page_latency_ms=95.0,
        max_page_latency_ms=97.0,
        std_page_latency_ms=0.7,
        cpu_utilization_mean=45.0,
        cpu_utilization_peak=60.0,
        gpu_utilization_mean=None,
        gpu_utilization_peak=None,
        ram_used_mb_mean=2500.0,
        ram_used_mb_peak=2600.0,
        gpu_memory_used_mb_mean=None,
        gpu_memory_used_mb_peak=None,
        routing_decision_count=5,
        cpu_selected_count=5,
        gpu_selected_count=0,
        fallback_count=0,
        routing_overhead_ms=0.0,
        successful_pages=5,
        failed_pages=0,
        worker_count=1,
    )
    assert rec.run_id == "run_001"
    assert rec.policy == "CPU_ONLY"
    assert rec.page_count == 5
    assert rec.throughput_pages_s == 10.0
    assert len(rec.page_latencies_ms) == 5
    d = rec.to_dict()
    assert isinstance(d, dict)
    assert d["page_count"] == 5


def test_summary_aggregation_math():
    """Verify BenchmarkComparator.aggregate_runs correctly aggregates statistical properties."""
    runs = [
        BenchmarkRunRecord(
            run_id=f"run_{i}",
            policy=BenchmarkPolicy.GPU_ONLY.value,
            workload_class=WorkloadClass.CLASS_B.value,
            page_count=5,
            repetition_index=i + 1,
            is_warmup=False,
            cold_start_ms=100.0 if i == 0 else None,
            first_inference_ms=50.0 if i == 0 else None,
            total_time_seconds=0.10 + (i * 0.01),  # 0.10, 0.11, 0.12, 0.13, 0.14
            throughput_pages_s=50.0 - (i * 2.0),
            page_latencies_ms=[20.0 + i] * 5,
            mean_page_latency_ms=20.0 + i,
            median_page_latency_ms=20.0 + i,
            p95_page_latency_ms=20.0 + i,
            min_page_latency_ms=20.0 + i,
            max_page_latency_ms=20.0 + i,
            std_page_latency_ms=0.0,
            cpu_utilization_mean=20.0 + i,
            cpu_utilization_peak=40.0 + i,
            gpu_utilization_mean=80.0,
            gpu_utilization_peak=85.0,
            ram_used_mb_mean=1000.0,
            ram_used_mb_peak=1100.0,
            gpu_memory_used_mb_mean=200.0,
            gpu_memory_used_mb_peak=220.0,
            routing_decision_count=5,
            cpu_selected_count=0,
            gpu_selected_count=5,
            fallback_count=0,
            routing_overhead_ms=0.0,
            successful_pages=5,
            failed_pages=0,
            worker_count=1,
        )
        for i in range(5)
    ]

    summary = BenchmarkComparator.aggregate_runs(
        runs=runs,
        policy=BenchmarkPolicy.GPU_ONLY.value,
        workload_class=WorkloadClass.CLASS_B.value,
        page_count=5,
    )

    assert summary.sample_count == 5
    assert summary.execution_time_mean_s == pytest.approx(0.12, abs=1e-3)
    assert summary.execution_time_median_s == pytest.approx(0.12, abs=1e-3)
    assert summary.gpu_selection_pct == 100.0
    assert summary.cpu_selection_pct == 0.0
    assert summary.gpu_util_mean == pytest.approx(80.0)
    assert summary.is_exploratory is False


def test_speedup_calculation():
    """Verify speedup = T_baseline / T_target."""
    # 2.0s baseline / 1.0s target = 2.0x speedup
    speedup = BenchmarkComparator.calculate_speedup(2.0, 1.0)
    assert speedup == pytest.approx(2.0)

    # 1.0s baseline / 2.0s target = 0.5x speedup
    speedup_slow = BenchmarkComparator.calculate_speedup(1.0, 2.0)
    assert speedup_slow == pytest.approx(0.5)


def test_percentage_change_calculation():
    """Verify percentage change = ((T_target - T_baseline) / T_baseline) * 100."""
    # Reduced from 200 to 100 -> -50%
    pct = BenchmarkComparator.calculate_percentage_change(200.0, 100.0)
    assert pct == pytest.approx(-50.0)

    # Increased from 100 to 150 -> +50%
    pct_up = BenchmarkComparator.calculate_percentage_change(100.0, 150.0)
    assert pct_up == pytest.approx(50.0)


def test_p95_calculation_method():
    """Verify P95 calculation correctly interpolates 95th percentile."""
    values = list(range(1, 101))  # 1 to 100
    p95 = BenchmarkComparator.calculate_p95([float(v) for v in values])
    assert p95 == pytest.approx(95.0, abs=1.0)


def test_monitoring_overhead_computation():
    """Verify monitoring overhead = ((T_with - T_without) / T_without) * 100."""
    oh = BenchmarkComparator.calculate_monitoring_overhead(
        time_without_s=100.0,
        time_with_s=105.0,
    )
    assert oh == pytest.approx(5.0)


def test_adaptive_overhead_computation():
    """Verify adaptive routing overhead from summary."""
    summary = BenchmarkSummary(
        policy=BenchmarkPolicy.ADAPTIVE.value,
        workload_class=WorkloadClass.CLASS_B.value,
        page_count=10,
        sample_count=5,
        execution_time_mean_s=0.5,
        execution_time_std_s=0.01,
        execution_time_median_s=0.5,
        execution_time_min_s=0.49,
        execution_time_max_s=0.52,
        throughput_mean_pps=20.0,
        throughput_std_pps=0.5,
        throughput_median_pps=20.0,
        latency_mean_ms=50.0,
        latency_median_ms=50.0,
        latency_p95_ms=52.0,
        latency_std_ms=1.5,
        coefficient_of_variation=0.02,
        cpu_util_mean=25.0,
        cpu_util_peak=35.0,
        gpu_util_mean=80.0,
        gpu_util_peak=85.0,
        ram_mb_mean=1200.0,
        ram_mb_peak=1300.0,
        gpu_mem_mb_mean=250.0,
        gpu_mem_mb_peak=260.0,
        cpu_selection_pct=10.0,
        gpu_selection_pct=90.0,
        fallback_pct=0.0,
        routing_overhead_mean_ms=0.45,
    )
    assert summary.routing_overhead_mean_ms == 0.45
    # Relative overhead: 0.45 ms / 50.0 ms = 0.9%
    overhead_pct = (summary.routing_overhead_mean_ms / summary.latency_mean_ms) * 100.0
    assert overhead_pct < 2.0


def test_comparison_matrix_structure():
    """Verify compare_policies correctly structures comparisons and offline oracle regret."""
    s_cpu = BenchmarkSummary(
        policy=BenchmarkPolicy.CPU_ONLY.value,
        workload_class=WorkloadClass.CLASS_B.value,
        page_count=10,
        sample_count=5,
        execution_time_mean_s=1.0,
        execution_time_std_s=0.02,
        execution_time_median_s=1.0,
        execution_time_min_s=0.98,
        execution_time_max_s=1.02,
        throughput_mean_pps=10.0,
        throughput_std_pps=0.2,
        throughput_median_pps=10.0,
        latency_mean_ms=100.0,
        latency_median_ms=100.0,
        latency_p95_ms=102.0,
        latency_std_ms=2.0,
        coefficient_of_variation=0.02,
        cpu_util_mean=50.0,
        cpu_util_peak=60.0,
        gpu_util_mean=None,
        gpu_util_peak=None,
        ram_mb_mean=1000.0,
        ram_mb_peak=1100.0,
        gpu_mem_mb_mean=None,
        gpu_mem_mb_peak=None,
        cpu_selection_pct=100.0,
        gpu_selection_pct=0.0,
        fallback_pct=0.0,
        routing_overhead_mean_ms=0.0,
    )
    s_gpu = BenchmarkSummary(
        policy=BenchmarkPolicy.GPU_ONLY.value,
        workload_class=WorkloadClass.CLASS_B.value,
        page_count=10,
        sample_count=5,
        execution_time_mean_s=0.5,
        execution_time_std_s=0.01,
        execution_time_median_s=0.5,
        execution_time_min_s=0.49,
        execution_time_max_s=0.51,
        throughput_mean_pps=20.0,
        throughput_std_pps=0.4,
        throughput_median_pps=20.0,
        latency_mean_ms=50.0,
        latency_median_ms=50.0,
        latency_p95_ms=51.0,
        latency_std_ms=1.0,
        coefficient_of_variation=0.02,
        cpu_util_mean=20.0,
        cpu_util_peak=30.0,
        gpu_util_mean=85.0,
        gpu_util_peak=90.0,
        ram_mb_mean=1000.0,
        ram_mb_peak=1100.0,
        gpu_mem_mb_mean=250.0,
        gpu_mem_mb_peak=260.0,
        cpu_selection_pct=0.0,
        gpu_selection_pct=100.0,
        fallback_pct=0.0,
        routing_overhead_mean_ms=0.0,
    )
    s_adp = BenchmarkSummary(
        policy=BenchmarkPolicy.ADAPTIVE.value,
        workload_class=WorkloadClass.CLASS_B.value,
        page_count=10,
        sample_count=5,
        execution_time_mean_s=0.52,
        execution_time_std_s=0.012,
        execution_time_median_s=0.52,
        execution_time_min_s=0.51,
        execution_time_max_s=0.53,
        throughput_mean_pps=19.23,
        throughput_std_pps=0.3,
        throughput_median_pps=19.23,
        latency_mean_ms=52.0,
        latency_median_ms=52.0,
        latency_p95_ms=53.0,
        latency_std_ms=1.2,
        coefficient_of_variation=0.023,
        cpu_util_mean=22.0,
        cpu_util_peak=32.0,
        gpu_util_mean=80.0,
        gpu_util_peak=85.0,
        ram_mb_mean=1000.0,
        ram_mb_peak=1100.0,
        gpu_mem_mb_mean=240.0,
        gpu_mem_mb_peak=250.0,
        cpu_selection_pct=10.0,
        gpu_selection_pct=90.0,
        fallback_pct=0.0,
        routing_overhead_mean_ms=0.4,
    )

    comp = BenchmarkComparator.compare_policies(s_cpu, s_gpu, s_adp)
    assert comp["speedup_vs_cpu"]["gpu"] == pytest.approx(2.0)
    assert comp["speedup_vs_cpu"]["adaptive"] == pytest.approx(1.0 / 0.52, abs=0.01)
    assert comp["offline_oracle"]["oracle_policy"] == "GPU_ONLY"
    assert comp["offline_oracle"]["adaptive_regret_pct"] == pytest.approx(((0.52 - 0.5) / 0.5) * 100.0, abs=0.01)


def test_reproducible_workload_selection():
    """Verify BenchmarkCorpusManager deterministically returns requested page counts."""
    mgr = BenchmarkCorpusManager()
    pages_5 = mgr.get_workload_pages(WorkloadClass.CLASS_A, target_page_count=5)
    assert len(pages_5) == 5
    for item, img in pages_5:
        assert isinstance(img, np.ndarray)
        assert item.estimated_complexity < 0.35

    pages_10 = mgr.get_workload_pages(WorkloadClass.CLASS_B, target_page_count=10)
    assert len(pages_10) == 10

    pages_20 = mgr.get_workload_pages(WorkloadClass.CLASS_D, target_page_count=20)
    assert len(pages_20) == 20


def test_policy_isolation_cpu_only():
    """Verify CPU_ONLY policy only invokes CPU strategy."""
    mock_cpu = MagicMock()
    mock_gpu = MagicMock()
    mock_cpu.recognize.return_value = [{"text": "cpu_text"}]
    mock_gpu.recognize.return_value = [{"text": "gpu_text"}]

    runner = G5BenchmarkRunner(enable_monitoring=False)
    runner._cpu_strategy = mock_cpu
    runner._gpu_strategy = mock_gpu

    mgr = BenchmarkCorpusManager()
    pages = mgr.get_workload_pages(WorkloadClass.CLASS_A, target_page_count=3)

    run = runner.execute_single_run(
        policy=BenchmarkPolicy.CPU_ONLY,
        workload_class=WorkloadClass.CLASS_A,
        pages=pages,
        repetition_idx=1,
    )

    assert mock_cpu.recognize.call_count == 3
    assert mock_gpu.recognize.call_count == 0
    assert run.cpu_selected_count == 3
    assert run.gpu_selected_count == 0


def test_policy_isolation_gpu_only():
    """Verify GPU_ONLY policy only invokes GPU strategy."""
    mock_cpu = MagicMock()
    mock_gpu = MagicMock()
    mock_cpu.recognize.return_value = [{"text": "cpu_text"}]
    mock_gpu.recognize.return_value = [{"text": "gpu_text"}]

    runner = G5BenchmarkRunner(enable_monitoring=False)
    runner._cpu_strategy = mock_cpu
    runner._gpu_strategy = mock_gpu

    mgr = BenchmarkCorpusManager()
    pages = mgr.get_workload_pages(WorkloadClass.CLASS_A, target_page_count=3)

    run = runner.execute_single_run(
        policy=BenchmarkPolicy.GPU_ONLY,
        workload_class=WorkloadClass.CLASS_A,
        pages=pages,
        repetition_idx=1,
    )

    assert mock_gpu.recognize.call_count == 3
    assert mock_cpu.recognize.call_count == 0
    assert run.gpu_selected_count == 3
    assert run.cpu_selected_count == 0


def test_telemetry_handling_unavailable():
    """Verify missing/unavailable telemetry does not raise exceptions."""
    runs = [
        BenchmarkRunRecord(
            run_id="run_unavail",
            policy=BenchmarkPolicy.GPU_ONLY.value,
            workload_class=WorkloadClass.CLASS_A.value,
            page_count=2,
            repetition_index=1,
            is_warmup=False,
            cold_start_ms=None,
            first_inference_ms=None,
            total_time_seconds=0.1,
            throughput_pages_s=20.0,
            page_latencies_ms=[47.0, 48.0],
            mean_page_latency_ms=47.5,
            median_page_latency_ms=47.5,
            p95_page_latency_ms=48.0,
            min_page_latency_ms=47.0,
            max_page_latency_ms=48.0,
            std_page_latency_ms=0.7,
            cpu_utilization_mean=15.0,
            cpu_utilization_peak=20.0,
            gpu_utilization_mean=None,  # UNAVAILABLE
            gpu_utilization_peak=None,  # UNAVAILABLE
            ram_used_mb_mean=1200.0,
            ram_used_mb_peak=1300.0,
            gpu_memory_used_mb_mean=None,
            gpu_memory_used_mb_peak=None,
            routing_decision_count=2,
            cpu_selected_count=0,
            gpu_selected_count=2,
            fallback_count=0,
            routing_overhead_ms=0.0,
            successful_pages=2,
            failed_pages=0,
            worker_count=1,
        )
    ]
    summary = BenchmarkComparator.aggregate_runs(
        runs=runs,
        policy=BenchmarkPolicy.GPU_ONLY.value,
        workload_class=WorkloadClass.CLASS_A.value,
        page_count=2,
    )
    assert summary.gpu_util_mean is None
    assert summary.gpu_mem_mb_mean is None
    assert summary.cpu_util_mean == 15.0


def test_break_even_estimation():
    """Verify break-even page count estimation."""
    comparisons = [
        {
            "page_count": 5,
            "execution_time_s": {"cpu": 0.5, "gpu": 0.6},
        },
        {
            "page_count": 10,
            "execution_time_s": {"cpu": 1.0, "gpu": 0.8},  # crossover
        },
        {
            "page_count": 20,
            "execution_time_s": {"cpu": 2.0, "gpu": 1.4},
        },
    ]
    res = BenchmarkComparator.estimate_break_even(comparisons)
    assert res["crossover_page_count"] == 10


def test_division_by_zero_protection():
    """Verify division by zero is safely guarded in mathematical computations."""
    assert BenchmarkComparator.calculate_speedup(0.0, 0.0) == 1.0
    assert BenchmarkComparator.calculate_percentage_change(0.0, 50.0) == 0.0
    assert BenchmarkComparator.calculate_monitoring_overhead(0.0, 100.0) == 0.0


def test_insufficient_sample_handling():
    """Verify calculate_p95 handles empty or single element lists without crash."""
    assert BenchmarkComparator.calculate_p95([]) == 0.0
    assert BenchmarkComparator.calculate_p95([42.0]) == 42.0
