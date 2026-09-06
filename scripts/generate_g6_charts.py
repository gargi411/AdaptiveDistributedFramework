"""Publication-Quality Chart Generator for Phase G6 Results.

Generates 6 figures saved to outputs/monitoring/g6_charts/:
1. chart1_latency_comparison.png (CPU vs GPU vs G4 vs G6 latency)
2. chart2_adaptive_regret.png (Offline oracle regret percentage)
3. chart3_oracle_agreement.png (Oracle policy agreement percentage)
4. chart4_routing_decisions.png (Device selection breakdown)
5. chart5_resource_utilization.png (CPU and GPU utilization profile)
6. chart6_cold_vs_warm.png (Cold compilation vs warm execution comparison)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("g6_chart_generator")

# Visual style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})


def load_g6_data(g6_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    heldout_p = g6_dir / "g6_heldout_results.json"
    comp_p = g6_dir / "g6_comparison.json"
    cold_p = g6_dir / "g6_cold_start.json"

    with open(heldout_p, "r", encoding="utf-8") as f:
        heldout = json.load(f)
    with open(comp_p, "r", encoding="utf-8") as f:
        comp = json.load(f)
    with open(cold_p, "r", encoding="utf-8") as f:
        cold = json.load(f)

    return heldout, comp, cold


def generate_chart1_latency(heldout: dict[str, Any], out_dir: Path) -> None:
    """Chart 1: Execution Latency (Mean & Median) Across Policies on Held-Out Set."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    policies = ["CPU_ONLY", "GPU_ONLY", "ADAPTIVE_G4", "ADAPTIVE_G6"]

    means = [heldout[p]["latency_mean_ms"] for p in policies]
    medians = [heldout[p]["latency_median_ms"] for p in policies]

    x = np.arange(len(policies))
    width = 0.35

    rects1 = ax.bar(x - width/2, means, width, label="Mean Latency (ms)")
    rects2 = ax.bar(x + width/2, medians, width, label="Median Latency (ms)")

    ax.set_title("OCR Page Latency Comparison (Held-Out Test Set)")
    ax.set_ylabel("Latency (milliseconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(["CPU Only", "GPU Only", "Adaptive G4\n(Rule-Based)", "Adaptive G6\n(Data-Driven)"])
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    for rects in (rects1, rects2):
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}", (r.get_x() + r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    p = out_dir / "chart1_latency_comparison.png"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Generated {p}")


def generate_chart2_regret(comp: dict[str, Any], out_dir: Path) -> None:
    """Chart 2: Offline Oracle Regret Comparison (G4 vs G6)."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    oracle_data = comp.get("offline_oracle", {})

    policies = ["Adaptive G4\n(Rule-Based)", "Adaptive G6\n(Data-Driven)"]
    regrets = [oracle_data.get("g4_regret_pct", 0.0), oracle_data.get("g6_regret_pct", 0.0)]

    x = np.arange(len(policies))
    width = 0.45

    bars = ax.bar(x, regrets, width)
    ax.set_title("Offline Oracle Regret on Held-Out Workload")
    ax.set_ylabel("Regret vs Oracle (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    for r in bars:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", (r.get_x() + r.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    p = out_dir / "chart2_adaptive_regret.png"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Generated {p}")


def generate_chart3_oracle_agreement(comp: dict[str, Any], out_dir: Path) -> None:
    """Chart 3: Oracle Agreement Percentage (G4 vs G6)."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    oracle_data = comp.get("offline_oracle", {})

    policies = ["Adaptive G4\n(Rule-Based)", "Adaptive G6\n(Data-Driven)"]
    agreements = [oracle_data.get("oracle_agreement_g4_pct", 0.0), oracle_data.get("oracle_agreement_g6_pct", 0.0)]

    x = np.arange(len(policies))
    width = 0.45

    bars = ax.bar(x, agreements, width)
    ax.set_title("Oracle Decision Agreement Rate (Held-Out Set)")
    ax.set_ylabel("Agreement Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    for r in bars:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    p = out_dir / "chart3_oracle_agreement.png"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Generated {p}")


def generate_chart4_routing_decisions(heldout: dict[str, Any], out_dir: Path) -> None:
    """Chart 4: Routing Decisions Distribution (CPU vs GPU)."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    policies = ["ADAPTIVE_G4", "ADAPTIVE_G6"]
    cpu_pcts = [heldout[p]["cpu_percentage"] for p in policies]
    gpu_pcts = [heldout[p]["gpu_percentage"] for p in policies]

    x = np.arange(len(policies))
    width = 0.45

    p1 = ax.bar(x, cpu_pcts, width, label="Routed to CPU")
    p2 = ax.bar(x, gpu_pcts, width, bottom=cpu_pcts, label="Routed to GPU")

    ax.set_title("Backend Selection Breakdown on Held-Out Test Set")
    ax.set_ylabel("Percentage of Routing Decisions (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Adaptive G4\n(Rule-Based)", "Adaptive G6\n(Data-Driven)"])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    for i, (cp, gp) in enumerate(zip(cpu_pcts, gpu_pcts)):
        if cp > 0:
            ax.annotate(f"CPU: {cp:.1f}%", (x[i], cp / 2), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if gp > 0:
            ax.annotate(f"GPU: {gp:.1f}%", (x[i], cp + gp / 2), ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    fig.tight_layout()
    p = out_dir / "chart4_routing_decisions.png"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Generated {p}")


def generate_chart5_resource_util(heldout: dict[str, Any], out_dir: Path) -> None:
    """Chart 5: Host CPU and GPU Utilization Across Policies."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    policies = ["CPU_ONLY", "GPU_ONLY", "ADAPTIVE_G4", "ADAPTIVE_G6"]
    cpu_u = [heldout[p]["mean_cpu_util"] for p in policies]
    gpu_u = [heldout[p]["mean_gpu_util"] for p in policies]

    x = np.arange(len(policies))
    width = 0.35

    r1 = ax.bar(x - width/2, cpu_u, width, label="Host CPU Utilization (%)")
    r2 = ax.bar(x + width/2, gpu_u, width, label="Iris Xe GPU Utilization (%)")

    ax.set_title("Resource Utilization Profile Across Execution Policies")
    ax.set_ylabel("Utilization (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["CPU Only", "GPU Only", "Adaptive G4", "Adaptive G6"])
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    for rects in (r1, r2):
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    p = out_dir / "chart5_resource_utilization.png"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Generated {p}")


def generate_chart6_cold_vs_warm(cold: dict[str, Any], out_dir: Path) -> None:
    """Chart 6: Cold-Start Compilation vs Warm Execution Time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    devices = ["CPU", "GPU"]
    cold_comp = [cold[d]["cold_start_ms"] for d in devices]
    first_inf = [cold[d]["first_inference_ms"] for d in devices]
    warm_inf = [cold[d]["warm_ms"] for d in devices]

    x = np.arange(len(devices))
    width = 0.5

    # Subplot 1: Compilation Latency
    b1 = ax1.bar(x, cold_comp, width)
    ax1.set_title("Cold-Start Model Compilation Latency")
    ax1.set_ylabel("Latency (milliseconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(devices)
    ax1.grid(True, linestyle="--", alpha=0.7, axis="y")
    for r in b1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}ms", (r.get_x() + r.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    # Subplot 2: Inference Latency (First vs Warm)
    w2 = 0.35
    b2a = ax2.bar(x - w2/2, first_inf, w2, label="First Inference")
    b2b = ax2.bar(x + w2/2, warm_inf, w2, label="Warm Inference")
    ax2.set_title("First vs Warm Inference Latency")
    ax2.set_ylabel("Latency (milliseconds)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(devices)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7, axis="y")
    for rects in (b2a, b2b):
        for r in rects:
            h = r.get_height()
            ax2.annotate(f"{h:.1f}ms", (r.get_x() + r.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    p = out_dir / "chart6_cold_vs_warm.png"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Generated {p}")


def main() -> int:
    g6_dir = Path("outputs/monitoring/g6")
    charts_dir = Path("outputs/monitoring/g6_charts")
    charts_dir.mkdir(parents=True, exist_ok=True)

    try:
        heldout, comp, cold = load_g6_data(g6_dir)
    except FileNotFoundError as exc:
        logger.error(f"Cannot generate G6 charts: {exc}")
        return 1

    logger.info("Generating publication-quality charts for Phase G6...")
    generate_chart1_latency(heldout, charts_dir)
    generate_chart2_regret(comp, charts_dir)
    generate_chart3_oracle_agreement(comp, charts_dir)
    generate_chart4_routing_decisions(heldout, charts_dir)
    generate_chart5_resource_util(heldout, charts_dir)
    generate_chart6_cold_vs_warm(cold, charts_dir)

    logger.info(f"All 6 Phase G6 charts generated in {charts_dir.resolve()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
