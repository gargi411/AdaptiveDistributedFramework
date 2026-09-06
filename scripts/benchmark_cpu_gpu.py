"""CLI benchmark execution script for Phase G5.

Executes controlled CPU vs GPU performance benchmarking across three policies:
- CPU_ONLY
- GPU_ONLY
- ADAPTIVE

Under standardized workloads (Class A, B, C, D) and page counts (5, 10, 20).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_framework.acceleration.benchmark_dataset import (
    BenchmarkCorpusManager,
)
from adaptive_framework.acceleration.benchmarking import (
    BenchmarkComparator,
    BenchmarkPolicy,
    BenchmarkRunRecord,
    BenchmarkSummary,
    G5BenchmarkRunner,
    WorkloadClass,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("g5_benchmark_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase G5: Controlled CPU vs GPU Performance Benchmarking",
    )
    parser.add_argument(
        "--policy",
        choices=["CPU_ONLY", "GPU_ONLY", "ADAPTIVE", "ALL"],
        default="ALL",
        help="Policy to evaluate (default: ALL)",
    )
    parser.add_argument(
        "--workload",
        choices=["CLASS_A", "CLASS_B", "CLASS_C", "CLASS_D", "ALL"],
        default="ALL",
        help="Workload class to evaluate (default: ALL)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Workload page counts to evaluate (default: 5 10 20)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of measured repetitions per condition (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of discarded warmup repetitions per condition (default: 2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/monitoring",
        help="Directory to save JSON artifacts (default: outputs/monitoring)",
    )
    parser.add_argument(
        "--measure-overhead",
        action="store_true",
        help="Run explicit monitoring overhead measurement",
    )
    parser.add_argument(
        "--test-fallback",
        action="store_true",
        help="Run explicit simulated GPU failure fallback test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run complete benchmark matrix (equivalent to default settings)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Phase G5 Benchmark Runner...")
    corpus_mgr = BenchmarkCorpusManager()
    runner = G5BenchmarkRunner(corpus_manager=corpus_mgr)

    # Determine policies
    if args.policy == "ALL" or args.all:
        policies = [
            BenchmarkPolicy.CPU_ONLY,
            BenchmarkPolicy.GPU_ONLY,
            BenchmarkPolicy.ADAPTIVE,
        ]
    else:
        policies = [BenchmarkPolicy[args.policy]]

    # Determine workloads
    if args.workload == "ALL" or args.all:
        workloads = [
            WorkloadClass.CLASS_A,
            WorkloadClass.CLASS_B,
            WorkloadClass.CLASS_C,
            WorkloadClass.CLASS_D,
        ]
    else:
        workloads = [WorkloadClass[args.workload]]

    sizes = args.sizes
    repetitions = args.repetitions
    warmup = args.warmup

    logger.info(
        f"Benchmark Configuration:\n"
        f"  Policies: {[p.value for p in policies]}\n"
        f"  Workloads: {[w.value for w in workloads]}\n"
        f"  Sizes: {sizes}\n"
        f"  Warmup reps: {warmup}\n"
        f"  Measured reps: {repetitions}\n"
        f"  Workers: {args.workers}\n"
        f"  Output Dir: {out_dir.resolve()}\n"
    )

    all_runs: List[BenchmarkRunRecord] = []
    all_summaries: List[BenchmarkSummary] = []

    # Export corpus metadata first
    corpus_meta_path = out_dir / "g5_corpus_metadata.json"
    corpus_mgr.export_corpus_manifest(corpus_meta_path)
    logger.info(f"Saved corpus metadata to {corpus_meta_path}")

    # Cold start measurements for CPU and GPU
    logger.info("Measuring cold start and first inference latency...")
    # Generate a single sample image from class A
    sample_pages = corpus_mgr.get_workload_pages(WorkloadClass.CLASS_A, target_page_count=1)
    sample_img = sample_pages[0][1]

    cold_starts: Dict[str, Dict[str, float]] = {}
    try:
        cold_starts["CPU"] = runner.run_cold_start_benchmark(BenchmarkPolicy.CPU_ONLY, sample_img)
        logger.info(f"CPU Cold Start: {cold_starts['CPU']}")
    except Exception as exc:
        logger.warning(f"Failed to measure CPU cold start: {exc}")

    try:
        cold_starts["GPU"] = runner.run_cold_start_benchmark(BenchmarkPolicy.GPU_ONLY, sample_img)
        logger.info(f"GPU Cold Start: {cold_starts['GPU']}")
    except Exception as exc:
        logger.warning(f"Failed to measure GPU cold start: {exc}")

    cold_path = out_dir / "g5_cold_start.json"
    with open(cold_path, "w", encoding="utf-8") as f:
        json.dump(cold_starts, f, indent=2)

    # Main Matrix Execution
    for w_class in workloads:
        for size in sizes:
            for policy in policies:
                logger.info(
                    f"Executing: Policy={policy.value} | Workload={w_class.value} | Pages={size}"
                )
                runs = runner.run_workload_experiment(
                    policy=policy,
                    workload_class=w_class,
                    page_count=size,
                    repetitions=repetitions,
                    warmup_runs=warmup,
                )
                all_runs.extend(runs)

                # Attach cold start data if applicable
                summary = BenchmarkComparator.aggregate_runs(
                    runs=runs,
                    policy=policy.value,
                    workload_class=w_class.value,
                    page_count=size,
                )
                if policy == BenchmarkPolicy.GPU_ONLY and "GPU" in cold_starts:
                    summary.cold_start_ms = cold_starts["GPU"]["cold_start_ms"]
                    summary.first_inference_ms = cold_starts["GPU"]["first_inference_ms"]
                elif policy == BenchmarkPolicy.CPU_ONLY and "CPU" in cold_starts:
                    summary.cold_start_ms = cold_starts["CPU"]["cold_start_ms"]
                    summary.first_inference_ms = cold_starts["CPU"]["first_inference_ms"]

                all_summaries.append(summary)

                logger.info(
                    f"Completed {policy.value}: Mean Time = {summary.execution_time_mean_s:.4f} s | "
                    f"Throughput = {summary.throughput_mean_pps:.2f} p/s | "
                    f"Latency Median = {summary.latency_median_ms:.2f} ms"
                )

    # Save run records
    runs_data = [r.to_dict() for r in all_runs]
    runs_path = out_dir / "g5_benchmark_runs.json"
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump(runs_data, f, indent=2)
    logger.info(f"Saved run records to {runs_path}")

    # Save summaries
    summaries_data = [s.to_dict() for s in all_summaries]
    summaries_path = out_dir / "g5_benchmark_summary.json"
    with open(summaries_path, "w", encoding="utf-8") as f:
        json.dump(summaries_data, f, indent=2)
    logger.info(f"Saved summaries to {summaries_path}")

    # Build Comparative Matrix
    comparisons: List[Dict[str, Any]] = []
    for w_class in workloads:
        for size in sizes:
            cpu_s = next(
                (s for s in all_summaries if s.workload_class == w_class.value and s.page_count == size and s.policy == BenchmarkPolicy.CPU_ONLY.value),
                None,
            )
            gpu_s = next(
                (s for s in all_summaries if s.workload_class == w_class.value and s.page_count == size and s.policy == BenchmarkPolicy.GPU_ONLY.value),
                None,
            )
            adp_s = next(
                (s for s in all_summaries if s.workload_class == w_class.value and s.page_count == size and s.policy == BenchmarkPolicy.ADAPTIVE.value),
                None,
            )
            if cpu_s and gpu_s and adp_s:
                cmp_res = BenchmarkComparator.compare_policies(cpu_s, gpu_s, adp_s)
                comparisons.append(cmp_res)

    break_even = BenchmarkComparator.estimate_break_even(comparisons)
    comp_artifact = {
        "comparisons": comparisons,
        "break_even_analysis": break_even,
    }

    comparison_path = out_dir / "g5_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comp_artifact, f, indent=2)
    logger.info(f"Saved comparisons to {comparison_path}")

    # Monitoring Overhead Measurement
    if args.measure_overhead or args.all:
        logger.info("Executing Controlled Monitoring Overhead Measurement...")
        overhead_res = runner.run_monitoring_overhead_control(
            workload_class=WorkloadClass.CLASS_B,
            page_count=10,
            repetitions=5,
        )
        overhead_path = out_dir / "g5_monitoring_overhead.json"
        with open(overhead_path, "w", encoding="utf-8") as f:
            json.dump(overhead_res, f, indent=2)
        logger.info(f"Saved monitoring overhead results to {overhead_path}")

    # Simulated Fallback Test
    if args.test_fallback or args.all:
        logger.info("Executing Controlled Fallback Under GPU Failure Test...")
        fallback_res = runner.run_failure_fallback_test(sample_img)
        fallback_path = out_dir / "g5_fallback_test.json"
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(fallback_res, f, indent=2)
        logger.info(f"Saved fallback test results to {fallback_path}")

    runner.close()
    logger.info("Phase G5 Benchmarking Run Complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
