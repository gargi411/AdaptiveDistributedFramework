"""Core Benchmarking Engine for Phase G5: Controlled CPU vs GPU Performance Benchmarking.

Provides:
  - BenchmarkPolicy: CPU_ONLY, GPU_ONLY, ADAPTIVE
  - WorkloadClass: CLASS_A (small), CLASS_B (medium), CLASS_C (large/dense), CLASS_D (mixed)
  - BenchmarkRunRecord: Raw metrics from an individual benchmark run
  - BenchmarkSummary: Statistical aggregation (mean, median, std, p95, CV)
  - BenchmarkComparator: Speedup, percentage changes, break-even analysis, oracle regret
  - G5BenchmarkRunner: Controlled experiment execution engine
"""

from __future__ import annotations

import logging
import math
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =========================================================================
# Enumerations
# =========================================================================

class BenchmarkPolicy(str, Enum):
    """Execution policy for the OCR benchmark."""
    CPU_ONLY = "CPU_ONLY"
    GPU_ONLY = "GPU_ONLY"
    ADAPTIVE = "ADAPTIVE"


class WorkloadClass(str, Enum):
    """Workload complexity classification."""
    CLASS_A = "CLASS_A"  # Small / simple scanned pages (complexity ~0.15 - 0.25)
    CLASS_B = "CLASS_B"  # Medium-complexity scanned pages (complexity ~0.45 - 0.60)
    CLASS_C = "CLASS_C"  # Large / high-complexity dense scanned pages (complexity ~0.65 - 0.85)
    CLASS_D = "CLASS_D"  # Mixed-complexity workload


# =========================================================================
# Data Models
# =========================================================================

@dataclass
class BenchmarkRunRecord:
    """Detailed observation record of a single benchmark execution run."""
    run_id: str
    policy: str
    workload_class: str
    page_count: int
    repetition_index: int
    is_warmup: bool
    cold_start_ms: float | None
    first_inference_ms: float | None
    total_time_seconds: float
    throughput_pages_s: float
    page_latencies_ms: list[float]
    mean_page_latency_ms: float
    median_page_latency_ms: float
    p95_page_latency_ms: float
    min_page_latency_ms: float
    max_page_latency_ms: float
    std_page_latency_ms: float
    cpu_utilization_mean: float
    cpu_utilization_peak: float
    gpu_utilization_mean: float | None
    gpu_utilization_peak: float | None
    ram_used_mb_mean: float
    ram_used_mb_peak: float
    gpu_memory_used_mb_mean: float | None
    gpu_memory_used_mb_peak: float | None
    routing_decision_count: int = 0
    cpu_selected_count: int = 0
    gpu_selected_count: int = 0
    fallback_count: int = 0
    routing_overhead_ms: float = 0.0
    successful_pages: int = 0
    failed_pages: int = 0
    worker_count: int = 1
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with rounded floats for clean logging."""
        d = asdict(self)
        d["total_time_seconds"] = round(self.total_time_seconds, 4)
        d["throughput_pages_s"] = round(self.throughput_pages_s, 2)
        d["mean_page_latency_ms"] = round(self.mean_page_latency_ms, 2)
        d["median_page_latency_ms"] = round(self.median_page_latency_ms, 2)
        d["p95_page_latency_ms"] = round(self.p95_page_latency_ms, 2)
        d["std_page_latency_ms"] = round(self.std_page_latency_ms, 2)
        d["routing_overhead_ms"] = round(self.routing_overhead_ms, 4)
        if self.cold_start_ms is not None:
            d["cold_start_ms"] = round(self.cold_start_ms, 2)
        if self.first_inference_ms is not None:
            d["first_inference_ms"] = round(self.first_inference_ms, 2)
        return d


@dataclass
class BenchmarkSummary:
    """Aggregated statistical summary for a specific configuration."""
    policy: str
    workload_class: str
    page_count: int
    sample_count: int
    execution_time_mean_s: float
    execution_time_std_s: float
    execution_time_median_s: float
    execution_time_min_s: float
    execution_time_max_s: float
    throughput_mean_pps: float
    throughput_std_pps: float
    throughput_median_pps: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_std_ms: float
    coefficient_of_variation: float
    cpu_util_mean: float
    cpu_util_peak: float
    gpu_util_mean: float | None
    gpu_util_peak: float | None
    ram_mb_mean: float
    ram_mb_peak: float
    gpu_mem_mb_mean: float | None
    gpu_mem_mb_peak: float | None
    cpu_selection_pct: float = 0.0
    gpu_selection_pct: float = 0.0
    fallback_pct: float = 0.0
    routing_overhead_mean_ms: float = 0.0
    cold_start_ms: float | None = None
    first_inference_ms: float | None = None
    is_exploratory: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with sensible research precision."""
        d = asdict(self)
        for k in [
            "execution_time_mean_s",
            "execution_time_std_s",
            "execution_time_median_s",
            "execution_time_min_s",
            "execution_time_max_s",
        ]:
            d[k] = round(d[k], 4)
        for k in ["throughput_mean_pps", "throughput_std_pps", "throughput_median_pps"]:
            d[k] = round(d[k], 2)
        for k in ["latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_std_ms"]:
            d[k] = round(d[k], 2)
        d["coefficient_of_variation"] = round(d["coefficient_of_variation"], 4)
        d["routing_overhead_mean_ms"] = round(d["routing_overhead_mean_ms"], 4)
        return d


# =========================================================================
# Benchmark Comparator & Mathematical Utilities
# =========================================================================

class BenchmarkComparator:
    """Calculates research metrics: speedups, percentage changes, break-even, and regret."""

    @staticmethod
    def calculate_speedup(baseline_time_s: float, target_time_s: float) -> float:
        """Calculate execution-time speedup: S = T_baseline / T_target.

        Args:
            baseline_time_s: Time of baseline (e.g. CPU).
            target_time_s: Time of target (e.g. GPU or Adaptive).

        Returns:
            Speedup factor (1.0 = equal, >1.0 = target is faster, <1.0 = target is slower).
        """
        if target_time_s <= 0.0 or baseline_time_s <= 0.0:
            return 1.0
        return baseline_time_s / target_time_s

    @staticmethod
    def calculate_percentage_change(baseline: float, target: float) -> float:
        """Calculate percentage change: ((target - baseline) / baseline) * 100.

        Negative means target is lower/faster, positive means target is higher/slower.
        """
        if baseline == 0.0:
            return 0.0
        return ((target - baseline) / baseline) * 100.0

    @staticmethod
    def calculate_monitoring_overhead(time_without_s: float, time_with_s: float) -> float:
        """Calculate monitoring overhead percentage."""
        if time_without_s <= 0.0:
            return 0.0
        return ((time_with_s - time_without_s) / time_without_s) * 100.0

    @staticmethod
    def calculate_p95(values: list[float]) -> float:
        """Calculate 95th percentile from a list of values."""
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        s = sorted(values)
        idx = int(math.ceil(0.95 * len(s))) - 1
        idx = max(0, min(len(s) - 1, idx))
        return s[idx]

    @classmethod
    def aggregate_runs(
        cls,
        runs: list[BenchmarkRunRecord],
        policy: str,
        workload_class: str,
        page_count: int,
    ) -> BenchmarkSummary:
        """Aggregate a collection of measured runs into statistical summary."""
        measured = [r for r in runs if not r.is_warmup]
        if not measured:
            # Fallback to all runs if no warmup flag distinguished
            measured = runs

        n = len(measured)
        times = [r.total_time_seconds for r in measured]
        throughputs = [r.throughput_pages_s for r in measured]

        # Flatten all page latencies
        all_latencies: list[float] = []
        for r in measured:
            all_latencies.extend(r.page_latencies_ms)

        mean_time = statistics.mean(times) if times else 0.0
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        median_time = statistics.median(times) if times else 0.0
        min_time = min(times) if times else 0.0
        max_time = max(times) if times else 0.0

        mean_thru = statistics.mean(throughputs) if throughputs else 0.0
        std_thru = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
        median_thru = statistics.median(throughputs) if throughputs else 0.0

        mean_lat = statistics.mean(all_latencies) if all_latencies else 0.0
        median_lat = statistics.median(all_latencies) if all_latencies else 0.0
        std_lat = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0
        p95_lat = cls.calculate_p95(all_latencies)

        cv = (std_time / mean_time) if mean_time > 0.0 else 0.0

        # Hardware metrics
        cpu_utils = [r.cpu_utilization_mean for r in measured]
        cpu_peaks = [r.cpu_utilization_peak for r in measured]
        ram_means = [r.ram_used_mb_mean for r in measured]
        ram_peaks = [r.ram_used_mb_peak for r in measured]

        gpu_utils = [r.gpu_utilization_mean for r in measured if r.gpu_utilization_mean is not None]
        gpu_peaks = [r.gpu_utilization_peak for r in measured if r.gpu_utilization_peak is not None]
        gpu_mems = [r.gpu_memory_used_mb_mean for r in measured if r.gpu_memory_used_mb_mean is not None]
        gpu_mem_peaks = [r.gpu_memory_used_mb_peak for r in measured if r.gpu_memory_used_mb_peak is not None]

        # Adaptive specific
        total_decisions = sum(r.routing_decision_count for r in measured)
        total_cpu_sel = sum(r.cpu_selected_count for r in measured)
        total_gpu_sel = sum(r.gpu_selected_count for r in measured)
        total_fallback = sum(r.fallback_count for r in measured)
        mean_routing_oh = (
            statistics.mean([r.routing_overhead_ms for r in measured]) if measured else 0.0
        )

        cold_starts = [r.cold_start_ms for r in runs if r.cold_start_ms is not None]
        first_infers = [r.first_inference_ms for r in runs if r.first_inference_ms is not None]

        return BenchmarkSummary(
            policy=policy,
            workload_class=workload_class,
            page_count=page_count,
            sample_count=n,
            execution_time_mean_s=mean_time,
            execution_time_std_s=std_time,
            execution_time_median_s=median_time,
            execution_time_min_s=min_time,
            execution_time_max_s=max_time,
            throughput_mean_pps=mean_thru,
            throughput_std_pps=std_thru,
            throughput_median_pps=median_thru,
            latency_mean_ms=mean_lat,
            latency_median_ms=median_lat,
            latency_p95_ms=p95_lat,
            latency_std_ms=std_lat,
            coefficient_of_variation=cv,
            cpu_util_mean=statistics.mean(cpu_utils) if cpu_utils else 0.0,
            cpu_util_peak=max(cpu_peaks) if cpu_peaks else 0.0,
            gpu_util_mean=statistics.mean(gpu_utils) if gpu_utils else None,
            gpu_util_peak=max(gpu_peaks) if gpu_peaks else None,
            ram_mb_mean=statistics.mean(ram_means) if ram_means else 0.0,
            ram_mb_peak=max(ram_peaks) if ram_peaks else 0.0,
            gpu_mem_mb_mean=statistics.mean(gpu_mems) if gpu_mems else None,
            gpu_mem_mb_peak=max(gpu_mem_peaks) if gpu_mem_peaks else None,
            cpu_selection_pct=(total_cpu_sel / total_decisions * 100.0) if total_decisions > 0 else 0.0,
            gpu_selection_pct=(total_gpu_sel / total_decisions * 100.0) if total_decisions > 0 else 0.0,
            fallback_pct=(total_fallback / total_decisions * 100.0) if total_decisions > 0 else 0.0,
            routing_overhead_mean_ms=mean_routing_oh,
            cold_start_ms=cold_starts[0] if cold_starts else None,
            first_inference_ms=first_infers[0] if first_infers else None,
            is_exploratory=(n < 5),
        )

    @classmethod
    def compare_policies(
        cls,
        cpu_summary: BenchmarkSummary,
        gpu_summary: BenchmarkSummary,
        adaptive_summary: BenchmarkSummary,
    ) -> dict[str, Any]:
        """Perform head-to-head comparison between CPU, GPU, and Adaptive policies."""
        t_cpu = cpu_summary.execution_time_mean_s
        t_gpu = gpu_summary.execution_time_mean_s
        t_adapt = adaptive_summary.execution_time_mean_s

        s_gpu = cls.calculate_speedup(t_cpu, t_gpu)
        s_adaptive = cls.calculate_speedup(t_cpu, t_adapt)

        pct_gpu = cls.calculate_percentage_change(t_cpu, t_gpu)
        pct_adaptive = cls.calculate_percentage_change(t_cpu, t_adapt)

        # Offline Oracle Analysis
        t_oracle = min(t_cpu, t_gpu)
        oracle_policy = "GPU_ONLY" if t_gpu < t_cpu else "CPU_ONLY"
        oracle_regret = ((t_adapt - t_oracle) / t_oracle) if t_oracle > 0 else 0.0

        # Throughput speedups
        thru_cpu = cpu_summary.throughput_mean_pps
        thru_gpu = gpu_summary.throughput_mean_pps
        thru_adapt = adaptive_summary.throughput_mean_pps

        s_thru_gpu = thru_gpu / thru_cpu if thru_cpu > 0 else 1.0
        s_thru_adapt = thru_adapt / thru_cpu if thru_cpu > 0 else 1.0

        return {
            "workload_class": cpu_summary.workload_class,
            "page_count": cpu_summary.page_count,
            "execution_time_s": {
                "cpu": round(t_cpu, 4),
                "gpu": round(t_gpu, 4),
                "adaptive": round(t_adapt, 4),
            },
            "throughput_pps": {
                "cpu": round(thru_cpu, 2),
                "gpu": round(thru_gpu, 2),
                "adaptive": round(thru_adapt, 2),
            },
            "latency_median_ms": {
                "cpu": round(cpu_summary.latency_median_ms, 2),
                "gpu": round(gpu_summary.latency_median_ms, 2),
                "adaptive": round(adaptive_summary.latency_median_ms, 2),
            },
            "latency_p95_ms": {
                "cpu": round(cpu_summary.latency_p95_ms, 2),
                "gpu": round(gpu_summary.latency_p95_ms, 2),
                "adaptive": round(adaptive_summary.latency_p95_ms, 2),
            },
            "speedup_vs_cpu": {
                "gpu": round(s_gpu, 3),
                "adaptive": round(s_adaptive, 3),
            },
            "throughput_speedup_vs_cpu": {
                "gpu": round(s_thru_gpu, 3),
                "adaptive": round(s_thru_adapt, 3),
            },
            "percentage_time_change_vs_cpu": {
                "gpu": round(pct_gpu, 2),
                "adaptive": round(pct_adaptive, 2),
            },
            "offline_oracle": {
                "oracle_policy": oracle_policy,
                "oracle_time_s": round(t_oracle, 4),
                "adaptive_regret": round(oracle_regret, 4),
                "adaptive_regret_pct": round(oracle_regret * 100.0, 2),
            },
            "adaptive_routing": {
                "cpu_selected_pct": round(adaptive_summary.cpu_selection_pct, 1),
                "gpu_selected_pct": round(adaptive_summary.gpu_selection_pct, 1),
                "fallback_pct": round(adaptive_summary.fallback_pct, 1),
                "overhead_mean_ms": round(adaptive_summary.routing_overhead_mean_ms, 4),
            },
        }

    @classmethod
    def estimate_break_even(cls, comparisons: list[dict[str, Any]]) -> dict[str, Any]:
        """Empirically identify break-even workload size/complexity between CPU and GPU."""
        crossover_size: int | None = None
        for c in sorted(comparisons, key=lambda x: x["page_count"]):
            t_cpu = c["execution_time_s"]["cpu"]
            t_gpu = c["execution_time_s"]["gpu"]
            if t_gpu < t_cpu:
                crossover_size = c["page_count"]
                break

        return {
            "crossover_page_count": crossover_size,
            "interpretation": (
                f"GPU surpasses CPU at or above {crossover_size} pages on evaluated hardware"
                if crossover_size is not None
                else "CPU remained faster or competitive across all evaluated page sizes on evaluated hardware"
            ),
        }


# =========================================================================
# G5 Benchmark Runner
# =========================================================================

class G5BenchmarkRunner:
    """Coordinates and executes controlled Phase G5 benchmark experiments.

    Enforces:
      - Identical inputs across CPU_ONLY, GPU_ONLY, and ADAPTIVE policies
      - Separation of cold start from warm runs
      - System resource monitoring using G3 ResourceMonitor
      - Controlled GPU failure fallback validation
      - Monitoring and routing overhead isolation
    """

    def __init__(
        self,
        corpus_manager: Any | None = None,
        resource_monitor: Any | None = None,
        enable_monitoring: bool = True,
    ) -> None:
        from adaptive_framework.acceleration.benchmark_dataset import BenchmarkCorpusManager
        from adaptive_framework.acceleration.resource_monitor import ResourceMonitor

        self._corpus = corpus_manager or BenchmarkCorpusManager()
        self._monitor = resource_monitor if resource_monitor is not None else (
            ResourceMonitor(node_id="g5_bench") if enable_monitoring else None
        )
        self._enable_monitoring = enable_monitoring

        # Shared warm strategies for benchmark runs
        self._cpu_strategy: Any = None
        self._gpu_strategy: Any = None
        self._characterizer: Any = None

    def initialize_strategies(self) -> None:
        """Initialize and warm execution strategies."""
        from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy
        from adaptive_framework.acceleration.workload_characterizer import WorkloadCharacterizer

        if self._cpu_strategy is None:
            self._cpu_strategy = OpenVINOOCRStrategy(device="CPU")
            self._cpu_strategy.initialize()
        if self._gpu_strategy is None:
            self._gpu_strategy = OpenVINOOCRStrategy(device="GPU")
            self._gpu_strategy.initialize()
        if self._characterizer is None:
            self._characterizer = WorkloadCharacterizer()

    def run_cold_start_benchmark(
        self,
        policy: BenchmarkPolicy | str,
        sample_img: Any,
    ) -> dict[str, float]:
        """Measure cold initialization and first inference latency on a fresh strategy.

        Args:
            policy: BenchmarkPolicy (CPU_ONLY or GPU_ONLY).
            sample_img: A sample NumPy image array.

        Returns:
            Dictionary with cold_start_ms, first_inference_ms, and warm_ms.
        """
        from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy

        pol_str = policy.value if isinstance(policy, BenchmarkPolicy) else str(policy)
        dev = "GPU" if "GPU" in pol_str else "CPU"

        # Instantiate brand new strategy instance without auto_initialize
        strat = OpenVINOOCRStrategy(device=dev, auto_initialize=False)

        t_init_0 = time.perf_counter()
        strat.initialize()
        cold_start_ms = (time.perf_counter() - t_init_0) * 1000.0

        t_first_0 = time.perf_counter()
        _ = strat.recognize(sample_img)
        first_infer_ms = (time.perf_counter() - t_first_0) * 1000.0

        t_warm_0 = time.perf_counter()
        _ = strat.recognize(sample_img)
        warm_ms = (time.perf_counter() - t_warm_0) * 1000.0

        return {
            "cold_start_ms": round(cold_start_ms, 2),
            "first_inference_ms": round(first_infer_ms, 2),
            "warm_ms": round(warm_ms, 2),
        }

    def execute_single_run(
        self,
        policy: BenchmarkPolicy | str,
        workload_class: WorkloadClass | str,
        pages: list[tuple[Any, Any]],
        repetition_idx: int,
        is_warmup: bool = False,
        router: Any | None = None,
    ) -> BenchmarkRunRecord:
        """Execute a single repetition of OCR across the specified pages."""
        self.initialize_strategies()
        from adaptive_framework.document_processing.processing_strategy import (
            AdaptiveRoutingStrategyProxy,
        )

        pol_str = policy.value if isinstance(policy, BenchmarkPolicy) else str(policy)
        w_class_str = (
            workload_class.value if isinstance(workload_class, WorkloadClass) else str(workload_class)
        )

        # Set up adaptive proxy if needed
        proxy: Any = None
        if pol_str == BenchmarkPolicy.ADAPTIVE.value:
            from adaptive_framework.acceleration.adaptive_router import AdaptiveWorkRouter

            if router is None:
                router = AdaptiveWorkRouter()
            proxy = AdaptiveRoutingStrategyProxy(
                gpu_strategy=self._gpu_strategy,
                cpu_strategy=self._cpu_strategy,
                router=router,
                resource_monitor=self._monitor,
            )

        page_latencies_ms: list[float] = []
        cpu_utils: list[float] = []
        gpu_utils: list[float] = []
        ram_used: list[float] = []
        gpu_mems: list[float] = []

        successful_pages = 0
        failed_pages = 0

        t_run_start = time.perf_counter()

        for idx, (item, img) in enumerate(pages):
            # Sample telemetry before page processing
            snap = self._monitor.sample(queue_length=0, active_workers=1) if self._monitor else None
            if snap:
                cpu_utils.append(snap.cpu_percent)
                ram_used.append(snap.ram_used_mb)
                if snap.gpu_utilization_percent is not None:
                    gpu_utils.append(snap.gpu_utilization_percent)
                if snap.gpu_memory_used_mb is not None:
                    gpu_mems.append(snap.gpu_memory_used_mb)

            t_page_0 = time.perf_counter()

            try:
                if pol_str == BenchmarkPolicy.CPU_ONLY.value:
                    _ = self._cpu_strategy.recognize(img)
                elif pol_str == BenchmarkPolicy.GPU_ONLY.value:
                    _ = self._gpu_strategy.recognize(img)
                else:  # ADAPTIVE
                    profile = self._characterizer.characterize_synthetic(
                        document_id=item.source_pdf,
                        page_number=item.page_number,
                        page_count=len(pages),
                        width_pts=item.width_pts,
                        height_pts=item.height_pts,
                        image_density=item.image_density,
                        complexity_override=item.estimated_complexity,
                    )
                    _ = proxy.process(
                        page=img,
                        page_number=item.page_number,
                        document_id=item.source_pdf,
                        file_path=item.source_pdf,
                        workload_profile=profile,
                        resource_snapshot=snap,
                    )
                successful_pages += 1
            except Exception as exc:
                logger.debug("Page execution exception: %s", exc)
                failed_pages += 1

            page_lat_ms = (time.perf_counter() - t_page_0) * 1000.0
            page_latencies_ms.append(page_lat_ms)

        total_time_s = time.perf_counter() - t_run_start
        throughput = len(pages) / total_time_s if total_time_s > 0 else 0.0

        # Latency statistics
        mean_lat = statistics.mean(page_latencies_ms) if page_latencies_ms else 0.0
        median_lat = statistics.median(page_latencies_ms) if page_latencies_ms else 0.0
        p95_lat = BenchmarkComparator.calculate_p95(page_latencies_ms)
        min_lat = min(page_latencies_ms) if page_latencies_ms else 0.0
        max_lat = max(page_latencies_ms) if page_latencies_ms else 0.0
        std_lat = statistics.stdev(page_latencies_ms) if len(page_latencies_ms) > 1 else 0.0

        # Telemetry aggregation
        cpu_mean = statistics.mean(cpu_utils) if cpu_utils else 0.0
        cpu_peak = max(cpu_utils) if cpu_utils else 0.0
        gpu_mean = statistics.mean(gpu_utils) if gpu_utils else None
        gpu_peak = max(gpu_utils) if gpu_utils else None
        ram_mean = statistics.mean(ram_used) if ram_used else 0.0
        ram_peak = max(ram_used) if ram_used else 0.0
        gpu_mem_mean = statistics.mean(gpu_mems) if gpu_mems else None
        gpu_mem_peak = max(gpu_mems) if gpu_mems else None

        # Adaptive metrics
        routing_decisions = 0
        cpu_sel = 0
        gpu_sel = 0
        fallbacks = 0
        routing_oh = 0.0

        if pol_str == BenchmarkPolicy.ADAPTIVE.value and router is not None:
            metrics = router.metrics.get_summary()
            routing_decisions = metrics.get("total_decisions", 0)
            cpu_sel = metrics.get("cpu_decisions", 0)
            gpu_sel = metrics.get("gpu_decisions", 0)
            fallbacks = metrics.get("fallback_decisions", 0)
            routing_oh = metrics.get("latency_ms", {}).get("mean", 0.0)
        elif pol_str == BenchmarkPolicy.CPU_ONLY.value:
            cpu_sel = len(pages)
        else:
            gpu_sel = len(pages)

        return BenchmarkRunRecord(
            run_id=str(uuid.uuid4()),
            policy=pol_str,
            workload_class=w_class_str,
            page_count=len(pages),
            repetition_index=repetition_idx,
            is_warmup=is_warmup,
            cold_start_ms=None,
            first_inference_ms=None,
            total_time_seconds=total_time_s,
            throughput_pages_s=throughput,
            page_latencies_ms=page_latencies_ms,
            mean_page_latency_ms=mean_lat,
            median_page_latency_ms=median_lat,
            p95_page_latency_ms=p95_lat,
            min_page_latency_ms=min_lat,
            max_page_latency_ms=max_lat,
            std_page_latency_ms=std_lat,
            cpu_utilization_mean=cpu_mean,
            cpu_utilization_peak=cpu_peak,
            gpu_utilization_mean=gpu_mean,
            gpu_utilization_peak=gpu_peak,
            ram_used_mb_mean=ram_mean,
            ram_used_mb_peak=ram_peak,
            gpu_memory_used_mb_mean=gpu_mem_mean,
            gpu_memory_used_mb_peak=gpu_mem_peak,
            routing_decision_count=routing_decisions,
            cpu_selected_count=cpu_sel,
            gpu_selected_count=gpu_sel,
            fallback_count=fallbacks,
            routing_overhead_ms=routing_oh,
            successful_pages=successful_pages,
            failed_pages=failed_pages,
            worker_count=1,
        )

    def run_workload_experiment(
        self,
        policy: BenchmarkPolicy | str,
        workload_class: WorkloadClass | str,
        page_count: int,
        repetitions: int = 5,
        warmup_runs: int = 2,
    ) -> list[BenchmarkRunRecord]:
        """Execute warmup runs followed by repeated measured runs for a policy."""
        pages = self._corpus.get_workload_pages(workload_class, page_count)
        runs: list[BenchmarkRunRecord] = []

        # 1. Warmup runs
        for w_idx in range(warmup_runs):
            run = self.execute_single_run(
                policy=policy,
                workload_class=workload_class,
                pages=pages,
                repetition_idx=w_idx + 1,
                is_warmup=True,
            )
            runs.append(run)

        # 2. Measured repetitions
        for r_idx in range(repetitions):
            run = self.execute_single_run(
                policy=policy,
                workload_class=workload_class,
                pages=pages,
                repetition_idx=r_idx + 1,
                is_warmup=False,
            )
            runs.append(run)

        return runs

    def run_monitoring_overhead_control(
        self,
        workload_class: WorkloadClass | str = WorkloadClass.CLASS_B,
        page_count: int = 10,
        repetitions: int = 3,
    ) -> dict[str, Any]:
        """Measure the execution time difference with monitoring disabled vs enabled."""
        pages = self._corpus.get_workload_pages(workload_class, page_count)

        # 1. Run without monitoring
        saved_monitor = self._monitor
        self._monitor = None
        times_without: list[float] = []
        for i in range(repetitions):
            r = self.execute_single_run(
                policy=BenchmarkPolicy.CPU_ONLY,
                workload_class=workload_class,
                pages=pages,
                repetition_idx=i + 1,
            )
            times_without.append(r.total_time_seconds)

        # 2. Run with monitoring
        self._monitor = saved_monitor
        times_with: list[float] = []
        for i in range(repetitions):
            r = self.execute_single_run(
                policy=BenchmarkPolicy.CPU_ONLY,
                workload_class=workload_class,
                pages=pages,
                repetition_idx=i + 1,
            )
            times_with.append(r.total_time_seconds)

        mean_without = statistics.mean(times_without) if times_without else 0.0
        mean_with = statistics.mean(times_with) if times_with else 0.0
        overhead_pct = BenchmarkComparator.calculate_monitoring_overhead(mean_without, mean_with)

        return {
            "repetitions": repetitions,
            "page_count": page_count,
            "mean_time_without_monitoring_s": round(mean_without, 4),
            "mean_time_with_monitoring_s": round(mean_with, 4),
            "monitoring_overhead_pct": round(overhead_pct, 2),
        }

    def run_failure_fallback_test(
        self,
        sample_img: Any | None = None,
    ) -> dict[str, Any]:
        """Verify controlled GPU failure fallback behavior and latency."""
        from adaptive_framework.acceleration.adaptive_router import AdaptiveWorkRouter
        from adaptive_framework.acceleration.workload_characterizer import WorkloadProfile
        from adaptive_framework.document_processing.processing_strategy import (
            AdaptiveRoutingStrategyProxy,
            PageExtractionResult,
        )

        import numpy as np

        if sample_img is None:
            sample_img = np.full((800, 600, 3), 255, dtype=np.uint8)

        router = AdaptiveWorkRouter()

        class InjectedFailingGPU:
            def process(self, page: Any, page_number: int, document_id: str, file_path: str) -> PageExtractionResult:
                raise RuntimeError("Controlled injected OpenVINO GPU driver failure")

        class SafeCPUFallback:
            def process(self, page: Any, page_number: int, document_id: str, file_path: str) -> PageExtractionResult:
                res = PageExtractionResult(processing_method="ocr")
                res.text = "Controlled CPU fallback extraction"
                return res

        proxy = AdaptiveRoutingStrategyProxy(
            router=router,
            gpu_strategy=InjectedFailingGPU(),
            cpu_strategy=SafeCPUFallback(),
        )

        from adaptive_framework.acceleration.resource_monitor import ResourceSnapshot

        workload = WorkloadProfile(
            document_id="doc_fallback_test",
            page_count=5,
            estimated_complexity=0.75,  # Complexity >= 0.35 selects GPU
            is_scanned=True,
        )

        snapshot = ResourceSnapshot(
            node_id="test_node",
            cpu_percent=20.0,
            memory_percent=35.0,
            ram_used_mb=2000.0,
            ram_available_mb=14000.0,
            gpu_available=True,
            gpu_name="Intel(R) Iris(R) Xe Graphics",
            gpu_device="GPU.0",
            gpu_utilization_percent=25.0,
            gpu_memory_used_mb=300.0,
            gpu_memory_total_mb=7000.0,
            queue_length=0,
            active_workers=1,
        )

        t0 = time.perf_counter()
        result = proxy.process(
            page=sample_img,
            page_number=1,
            document_id="doc_fallback_test",
            file_path="fallback_test.pdf",
            workload_profile=workload,
            resource_snapshot=snapshot,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        history = router.metrics.get_recent_decisions(10)
        fallback_decisions = [d for d in history if d.fallback_occurred]

        return {
            "success": result.ocr_device == "CPU_FALLBACK",
            "ocr_device": result.ocr_device,
            "fallback_latency_ms": round(elapsed_ms, 2),
            "fallback_recorded": len(fallback_decisions) > 0,
            "fallback_reason": fallback_decisions[-1].reason if fallback_decisions else "",
        }

    def close(self) -> None:
        """Close telemetry monitors and clean up."""
        if self._monitor is not None:
            self._monitor.close()
