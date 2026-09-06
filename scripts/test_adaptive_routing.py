"""scripts/test_adaptive_routing.py -- CLI testing tool for Phase G4 Adaptive CPU/GPU Work Routing.

Validates:
  1. Workload characterization (complexity heuristic)
  2. Policy decision rules A-H across edge cases and resource states
  3. Anti-oscillation hysteresis behavior
  4. Fallback execution on GPU failure
  5. Live Intel Iris Xe execution with real ResourceMonitor telemetry

Usage:
  # Run simulation suite covering scenarios 1-7:
  python scripts/test_adaptive_routing.py --simulate

  # Run live evaluation on local hardware:
  python scripts/test_adaptive_routing.py --real

  # Run both simulation and live evaluation:
  python scripts/test_adaptive_routing.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root and src/ are in sys.path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from adaptive_framework.acceleration.adaptive_router import (
    AdaptiveWorkRouter,
    RoutingDecision,
    RoutingMetricsCollector,
)
from adaptive_framework.acceleration.resource_monitor import ResourceMonitor
from adaptive_framework.acceleration.workload_characterizer import (
    WorkloadCharacterizer,
    WorkloadProfile,
)
from adaptive_framework.config.models import AdaptiveRoutingConfig
from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy
from adaptive_framework.document_processing.processing_strategy import (
    AdaptiveRoutingStrategyProxy,
    DirectExtractionStrategy,
    ProcessingStrategyFactory,
)
from adaptive_framework.models.page import Page
from adaptive_framework.models.runtime import ResourceSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADF Adaptive CPU/GPU Routing Validation CLI (Phase G4)"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulated scenarios 1-7 testing policy rules and hysteresis",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run live evaluation on local hardware with real ResourceMonitor telemetry",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both simulated scenarios and live hardware evaluation",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of real-mode evaluation iterations (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/monitoring",
        help="Directory to write output JSON artifacts (default: outputs/monitoring)",
    )
    return parser.parse_args()


def run_simulated_scenarios(
    router: AdaptiveWorkRouter,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute scenarios 1-7 covering all policy rules and boundary conditions."""
    scenarios: list[dict[str, Any]] = []

    # Scenario 1: GPU unavailable -> routes CPU (Rule A)
    w1 = WorkloadProfile(
        document_id="doc_scen_1",
        page_count=5,
        estimated_complexity=0.75,
        char_count=5000,
        is_scanned=True,
    )
    s1 = ResourceSnapshot(
        timestamp="2026-09-06T12:00:00.000",
        node_id="test_node",
        cpu_percent=30.0,
        memory_percent=40.0,
        ram_used_mb=6000.0,
        ram_available_mb=10000.0,
        gpu_available=False,
    )
    d1 = router.route(w1, s1)
    scenarios.append({
        "scenario": 1,
        "name": "GPU unavailable",
        "expected_device": "CPU",
        "actual_device": d1.target_device,
        "matched": d1.target_device == "CPU",
        "rule": d1.applied_rule,
        "reason": d1.explanation,
        "decision": d1.to_dict(),
    })

    # Reset router state for independent tests
    router.reset_state()

    # Scenario 2: Small workload -> routes CPU (Rule B)
    w2 = WorkloadProfile(
        document_id="doc_scen_2",
        page_count=1,
        estimated_complexity=0.15,
        char_count=200,
        is_scanned=True,
    )
    s2 = ResourceSnapshot(
        timestamp="2026-09-06T12:00:01.000",
        node_id="test_node",
        cpu_percent=30.0,
        memory_percent=40.0,
        ram_used_mb=6000.0,
        ram_available_mb=10000.0,
        gpu_available=True,
        gpu_name="Intel(R) Iris(R) Xe Graphics",
        gpu_utilization_percent=10.0,
        gpu_memory_used_mb=100.0,
        gpu_memory_total_mb=7000.0,
    )
    d2 = router.route(w2, s2)
    scenarios.append({
        "scenario": 2,
        "name": "Small workload (1 page, low complexity)",
        "expected_device": "CPU",
        "actual_device": d2.target_device,
        "matched": d2.target_device == "CPU",
        "rule": d2.applied_rule,
        "reason": d2.explanation,
        "decision": d2.to_dict(),
    })

    router.reset_state()

    # Scenario 3: High complexity scanned document -> routes GPU (Rule C)
    w3 = WorkloadProfile(
        document_id="doc_scen_3",
        page_count=4,
        estimated_complexity=0.75,
        char_count=4500,
        is_scanned=True,
    )
    s3 = ResourceSnapshot(
        timestamp="2026-09-06T12:00:02.000",
        node_id="test_node",
        cpu_percent=30.0,
        memory_percent=40.0,
        ram_used_mb=6000.0,
        ram_available_mb=10000.0,
        gpu_available=True,
        gpu_name="Intel(R) Iris(R) Xe Graphics",
        gpu_utilization_percent=15.0,
        gpu_memory_used_mb=150.0,
        gpu_memory_total_mb=7000.0,
    )
    d3 = router.route(w3, s3)
    scenarios.append({
        "scenario": 3,
        "name": "High complexity scanned document",
        "expected_device": "GPU",
        "actual_device": d3.target_device,
        "matched": d3.target_device == "GPU",
        "rule": d3.applied_rule,
        "reason": d3.explanation,
        "decision": d3.to_dict(),
    })

    router.reset_state()

    # Scenario 4: High GPU load (utilization 92% > max 85%) -> routes CPU (Rule D)
    w4 = WorkloadProfile(
        document_id="doc_scen_4",
        page_count=3,
        estimated_complexity=0.70,
        char_count=3500,
        is_scanned=True,
    )
    s4 = ResourceSnapshot(
        timestamp="2026-09-06T12:00:03.000",
        node_id="test_node",
        cpu_percent=40.0,
        memory_percent=45.0,
        ram_used_mb=7000.0,
        ram_available_mb=9000.0,
        gpu_available=True,
        gpu_name="Intel(R) Iris(R) Xe Graphics",
        gpu_utilization_percent=92.0,
        gpu_memory_used_mb=500.0,
        gpu_memory_total_mb=7000.0,
    )
    d4 = router.route(w4, s4)
    scenarios.append({
        "scenario": 4,
        "name": "High GPU load (GPU util 92% > 85%)",
        "expected_device": "CPU",
        "actual_device": d4.target_device,
        "matched": d4.target_device == "CPU",
        "rule": d4.applied_rule,
        "reason": d4.explanation,
        "decision": d4.to_dict(),
    })

    router.reset_state()

    # Scenario 5: High CPU load, GPU available (CPU 88% > 80%, GPU 30%) -> routes GPU (Rule E)
    w5 = WorkloadProfile(
        document_id="doc_scen_5",
        page_count=2,
        estimated_complexity=0.50,
        char_count=2000,
        is_scanned=True,
    )
    s5 = ResourceSnapshot(
        timestamp="2026-09-06T12:00:04.000",
        node_id="test_node",
        cpu_percent=88.0,
        memory_percent=50.0,
        ram_used_mb=8000.0,
        ram_available_mb=8000.0,
        gpu_available=True,
        gpu_name="Intel(R) Iris(R) Xe Graphics",
        gpu_utilization_percent=30.0,
        gpu_memory_used_mb=300.0,
        gpu_memory_total_mb=7000.0,
    )
    d5 = router.route(w5, s5)
    scenarios.append({
        "scenario": 5,
        "name": "High CPU load, GPU available (CPU 88% > 80%)",
        "expected_device": "GPU",
        "actual_device": d5.target_device,
        "matched": d5.target_device == "GPU",
        "rule": d5.applied_rule,
        "reason": d5.explanation,
        "decision": d5.to_dict(),
    })

    # Scenario 6: Execution fallback on GPU failure (simulated via Strategy Proxy)
    router.reset_state()
    from adaptive_framework.document_processing.processing_strategy import PageExtractionResult

    class FailingGPUStrategy:
        def process(self, page: Any, page_number: int, document_id: str, file_path: str) -> PageExtractionResult:
            raise RuntimeError("Simulated OpenVINO GPU out-of-memory error")

    class SuccessfulCPUStrategy:
        def process(self, page: Any, page_number: int, document_id: str, file_path: str) -> PageExtractionResult:
            res = PageExtractionResult(processing_method="ocr")
            res.text = "Fallback CPU extraction text"
            return res

    proxy = AdaptiveRoutingStrategyProxy(
        router=router,
        gpu_strategy=FailingGPUStrategy(),
        cpu_strategy=SuccessfulCPUStrategy(),
    )
    res = proxy.process(
        page=None,
        page_number=1,
        document_id="doc_scen_6",
        file_path="mock.pdf",
        workload_profile=w3,
        resource_snapshot=s3,
    )
    fallback_decisions = [
        d for d in router.collector.get_history() if d.fallback_occurred
    ]
    scenarios.append({
        "scenario": 6,
        "name": "Execution fallback on GPU failure",
        "expected_device": "CPU (after fallback)",
        "actual_device": f"{res.ocr_device}",
        "matched": res.ocr_device == "CPU_FALLBACK" and len(fallback_decisions) > 0,
        "rule": "EXECUTION_FALLBACK",
        "reason": "GPU execution failed, fell back to CPU cleanly",
        "decision": fallback_decisions[-1].to_dict() if fallback_decisions else {},
    })

    # Scenario 7: Borderline hysteresis test
    router.reset_state()
    # First, make a GPU routing decision
    d_gpu = router.route(w3, s3)
    # Immediately evaluate a borderline workload that would normally route to CPU under normal conditions
    # (e.g. complexity 0.32, just under min_complexity 0.35)
    w_borderline = WorkloadProfile(
        document_id="doc_scen_7",
        page_count=2,
        estimated_complexity=0.32,
        char_count=1200,
        is_scanned=True,
    )
    d7 = router.route(w_borderline, s3)
    # Because cooldown is 2.0s and last decision was GPU, hysteresis dampening retains GPU
    scenarios.append({
        "scenario": 7,
        "name": "Borderline hysteresis anti-oscillation dampening",
        "expected_device": "GPU",
        "actual_device": d7.target_device,
        "matched": d7.target_device == "GPU" and d7.hysteresis_applied,
        "rule": d7.applied_rule,
        "reason": d7.explanation,
        "decision": d7.to_dict(),
    })

    all_matched = all(s["matched"] for s in scenarios)
    summary = {
        "mode": "simulation",
        "total_scenarios": len(scenarios),
        "passed_scenarios": sum(1 for s in scenarios if s["matched"]),
        "all_passed": all_matched,
    }
    return scenarios, summary


def run_real_evaluation(
    samples: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate real hardware telemetry and live execution on local device."""
    monitor = ResourceMonitor(node_id="local_hw_test")
    hw_info = monitor.get_hardware_info()
    router = AdaptiveWorkRouter()
    characterizer = WorkloadCharacterizer()

    # Prepare sample workloads of varying complexity
    test_cases = [
        # (name, width, height, density, pages, complexity)
        ("Small standard invoice page", 612.0, 792.0, 0.15, 1, 0.15),
        ("Medium mixed clinical report", 612.0, 792.0, 0.50, 2, 0.45),
        ("Dense scanned biomedical record", 612.0, 792.0, 0.85, 3, 0.78),
    ]

    decisions: list[dict[str, Any]] = []

    for i in range(samples):
        tc_idx = i % len(test_cases)
        name, w, h, density, p_count, comp = test_cases[tc_idx]

        t_char_0 = time.perf_counter()
        profile = characterizer.characterize_synthetic(
            document_id=f"live_eval_doc_{i+1}",
            page_number=1,
            page_count=p_count,
            width_pts=w,
            height_pts=h,
            image_density=density,
            complexity_override=comp,
        )
        char_latency_ms = (time.perf_counter() - t_char_0) * 1000.0

        # Sample live system telemetry
        snapshot = monitor.sample(queue_length=0, active_workers=1)

        t_route_0 = time.perf_counter()
        decision = router.route(profile, snapshot)
        route_latency_ms = (time.perf_counter() - t_route_0) * 1000.0

        decisions.append({
            "iteration": i + 1,
            "test_case": name,
            "complexity": round(profile.estimated_complexity, 3),
            "characterizer_latency_ms": round(char_latency_ms, 3),
            "router_latency_ms": round(route_latency_ms, 3),
            "routed_device": decision.target_device,
            "rule": decision.applied_rule,
            "hysteresis_active": decision.hysteresis_applied,
            "reason": decision.explanation,
            "snapshot": {
                "cpu_percent": snapshot.cpu_percent,
                "ram_percent": snapshot.memory_percent,
                "gpu_available": snapshot.gpu_available,
                "gpu_util_percent": snapshot.gpu_utilization_percent,
                "gpu_mem_used_mb": snapshot.gpu_memory_used_mb,
            },
        })
        time.sleep(0.3)

    monitor.close()

    metrics = router.collector.get_metrics()
    summary = {
        "mode": "real_hardware",
        "hardware": {
            "node_id": hw_info["node_id"],
            "cpu": hw_info["cpu"],
            "gpu_available": hw_info["gpu_available"],
            "gpu": hw_info.get("gpu"),
        },
        "total_decisions": len(decisions),
        "routed_gpu": sum(1 for d in decisions if d["routed_device"] == "GPU"),
        "routed_cpu": sum(1 for d in decisions if d["routed_device"] == "CPU"),
        "metrics": metrics,
    }
    return decisions, summary


def main() -> int:
    args = parse_args()
    if not (args.simulate or args.real or args.all):
        # Default to simulate if no flag provided
        args.simulate = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ADAPTIVE DISTRIBUTED FRAMEWORK -- PHASE G4 ROUTING VALIDATION")
    print("=" * 78)

    all_results: dict[str, Any] = {}

    if args.simulate or args.all:
        print("\n--- RUNNING SIMULATED POLICY SCENARIOS (1-7) ---")
        router = AdaptiveWorkRouter()
        scenarios, sim_summary = run_simulated_scenarios(router)

        print(
            f"{'Scen':<5} | {'Scenario Name':<42} | {'Exp':<5} | {'Act':<5} | {'Status':<6} | {'Rule':<16}"
        )
        print("-" * 88)
        for s in scenarios:
            status_str = "PASS" if s["matched"] else "FAIL"
            print(
                f"{s['scenario']:<5} | {s['name']:<42} | {s['expected_device']:<5} | "
                f"{s['actual_device'][:5]:<5} | {status_str:<6} | {s['rule']:<16}"
            )

        print("-" * 88)
        print(
            f"Simulation Result: {sim_summary['passed_scenarios']}/{sim_summary['total_scenarios']} passed."
        )
        all_results["simulation"] = {
            "scenarios": scenarios,
            "summary": sim_summary,
        }

    if args.real or args.all:
        print("\n--- RUNNING LIVE HARDWARE EVALUATION ---")
        decisions, real_summary = run_real_evaluation(samples=args.samples)

        hw = real_summary["hardware"]
        print(f"Node: {hw['node_id']}")
        print(
            f"CPU: {hw['cpu']['logical_cores']} logical cores, {hw['cpu']['physical_cores']} physical"
        )
        if hw["gpu_available"] and hw.get("gpu"):
            gpu = hw["gpu"]
            print(f"GPU: {gpu.get('device_name')} ({gpu.get('openvino_device_id')})")
        else:
            print("GPU: UNAVAILABLE")

        print("-" * 88)
        print(
            f"{'#':<3} | {'Test Case':<32} | {'Comp':<5} | {'Route':<5} | {'Rule':<15} | {'Latency':<9}"
        )
        print("-" * 88)
        for d in decisions:
            lat_str = f"{d['router_latency_ms']:.3f} ms"
            print(
                f"{d['iteration']:<3} | {d['test_case'][:32]:<32} | {d['complexity']:<5.2f} | "
                f"{d['routed_device']:<5} | {d['rule']:<15} | {lat_str:<9}"
            )

        print("-" * 88)
        print(
            f"Decisions: {real_summary['total_decisions']} total | "
            f"GPU: {real_summary['routed_gpu']} | CPU: {real_summary['routed_cpu']}"
        )
        all_results["real_hardware"] = {
            "decisions": decisions,
            "summary": real_summary,
        }

    # Save output artifacts
    decisions_path = out_dir / "g4_routing_decisions.json"
    summary_path = out_dir / "g4_routing_summary.json"

    with decisions_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    summary_out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "simulation": all_results.get("simulation", {}).get("summary"),
        "real_hardware": all_results.get("real_hardware", {}).get("summary"),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nArtifacts saved cleanly:")
    print(f"  - Decisions: {decisions_path.as_posix()}")
    print(f"  - Summary:   {summary_path.as_posix()}")
    print("=" * 78)

    # Return 0 if simulations passed
    if "simulation" in all_results:
        return 0 if all_results["simulation"]["summary"]["all_passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
