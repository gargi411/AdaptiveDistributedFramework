"""Publication-quality chart generator for Phase G5 Benchmarking Results.

Generates 5 distinct figures saved to outputs/monitoring/g5_charts/:
1. chart1_execution_time_vs_size.png (Execution time vs page count across policies)
2. chart2_throughput_vs_size.png (Throughput pages/sec vs page count across policies)
3. chart3_latency_distribution.png (Median vs P95 latency comparison)
4. chart4_gpu_utilization_profile.png (Host CPU & GPU utilization metrics)
5. chart5_adaptive_routing_distribution.png (CPU vs GPU page routing breakdown)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("g5_chart_generator")

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

POLICY_COLORS = {
    "CPU_ONLY": "#2b5c8f",     # Deep blue
    "GPU_ONLY": "#00a86b",     # Jade green
    "ADAPTIVE": "#e67e22",     # Amber orange
}


def load_data(monitoring_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = monitoring_dir / "g5_benchmark_summary.json"
    comp_path = monitoring_dir / "g5_comparison.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if not comp_path.exists():
        raise FileNotFoundError(f"Comparison file not found: {comp_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summaries = json.load(f)
    with open(comp_path, "r", encoding="utf-8") as f:
        comparisons = json.load(f)

    return summaries, comparisons


def generate_chart1_execution_time(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    """Chart 1: Execution Time vs Workload Size for Class B (Clinical Standard)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Filter for Class B
    class_b = [s for s in summaries if s["workload_class"] == "CLASS_B"]
    sizes = sorted(list(set(s["page_count"] for s in class_b)))

    for policy in ["CPU_ONLY", "GPU_ONLY", "ADAPTIVE"]:
        subset = [s for s in class_b if s["policy"] == policy]
        subset.sort(key=lambda x: x["page_count"])
        if subset:
            x_vals = [s["page_count"] for s in subset]
            y_vals = [s["execution_time_mean_s"] for s in subset]
            y_err = [s["execution_time_std_s"] for s in subset]
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                label=policy,
                marker="o",
                capsize=5,
                color=POLICY_COLORS.get(policy, "#333333"),
            )

    ax.set_title("Total Execution Time vs Page Count (Class B: Clinical Scanned)")
    ax.set_xlabel("Number of Scanned Pages")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_xticks(sizes)
    ax.legend(title="Execution Policy")
    ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    chart_path = out_dir / "chart1_execution_time_vs_size.png"
    fig.savefig(chart_path)
    plt.close(fig)
    logger.info(f"Generated {chart_path}")


def generate_chart2_throughput(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    """Chart 2: Throughput (pages/sec) vs Workload Size."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    class_b = [s for s in summaries if s["workload_class"] == "CLASS_B"]
    sizes = sorted(list(set(s["page_count"] for s in class_b)))

    for policy in ["CPU_ONLY", "GPU_ONLY", "ADAPTIVE"]:
        subset = [s for s in class_b if s["policy"] == policy]
        subset.sort(key=lambda x: x["page_count"])
        if subset:
            x_vals = [s["page_count"] for s in subset]
            y_vals = [s["throughput_mean_pps"] for s in subset]
            ax.plot(
                x_vals,
                y_vals,
                label=policy,
                marker="s",
                color=POLICY_COLORS.get(policy, "#333333"),
            )

    ax.set_title("OCR Processing Throughput vs Workload Size (Class B)")
    ax.set_xlabel("Number of Scanned Pages")
    ax.set_ylabel("Throughput (pages/second)")
    ax.set_xticks(sizes)
    ax.legend(title="Execution Policy")
    ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    chart_path = out_dir / "chart2_throughput_vs_size.png"
    fig.savefig(chart_path)
    plt.close(fig)
    logger.info(f"Generated {chart_path}")


def generate_chart3_latency_distribution(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    """Chart 3: Median vs P95 OCR Latency across policies (20-page workload)."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    subset = [s for s in summaries if s["workload_class"] == "CLASS_B" and s["page_count"] == 20]
    policies = ["CPU_ONLY", "GPU_ONLY", "ADAPTIVE"]

    medians = []
    p95s = []
    for pol in policies:
        entry = next((s for s in subset if s["policy"] == pol), None)
        if entry:
            medians.append(entry["latency_median_ms"])
            p95s.append(entry["latency_p95_ms"])
        else:
            medians.append(0.0)
            p95s.append(0.0)

    x = np.arange(len(policies))
    width = 0.35

    rects1 = ax.bar(x - width/2, medians, width, label="Median Latency (ms)", color="#4a90e2")
    rects2 = ax.bar(x + width/2, p95s, width, label="P95 Latency (ms)", color="#d9534f")

    ax.set_title("OCR Latency Profile: Median vs P95 (Class B, 20 Pages)")
    ax.set_ylabel("Latency (milliseconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    chart_path = out_dir / "chart3_latency_distribution.png"
    fig.savefig(chart_path)
    plt.close(fig)
    logger.info(f"Generated {chart_path}")


def generate_chart4_gpu_profile(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    """Chart 4: CPU Utilization Profile Across Policies (Class B Workload)."""
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    class_b = [s for s in summaries if s["workload_class"] == "CLASS_B"]
    sizes = sorted(list(set(s["page_count"] for s in class_b)))

    x = np.arange(len(sizes))
    width = 0.25

    cpu_cpu = [next(s["cpu_util_mean"] for s in class_b if s["policy"] == "CPU_ONLY" and s["page_count"] == sz) for sz in sizes]
    gpu_cpu = [next(s["cpu_util_mean"] for s in class_b if s["policy"] == "GPU_ONLY" and s["page_count"] == sz) for sz in sizes]
    adp_cpu = [next(s["cpu_util_mean"] for s in class_b if s["policy"] == "ADAPTIVE" and s["page_count"] == sz) for sz in sizes]

    r1 = ax1.bar(x - width, cpu_cpu, width, label="CPU_ONLY (CPU %)", color=POLICY_COLORS["CPU_ONLY"], alpha=0.85)
    r2 = ax1.bar(x, gpu_cpu, width, label="GPU_ONLY (CPU %)", color=POLICY_COLORS["GPU_ONLY"], alpha=0.85)
    r3 = ax1.bar(x + width, adp_cpu, width, label="ADAPTIVE (CPU %)", color=POLICY_COLORS["ADAPTIVE"], alpha=0.85)

    ax1.set_title("CPU Utilization Profile Across Policies (Class B Workload)")
    ax1.set_xlabel("Number of Pages")
    ax1.set_ylabel("Host CPU Utilization (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(sz) for sz in sizes])
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.7, axis="y")

    fig.tight_layout()
    chart_path = out_dir / "chart4_gpu_utilization_profile.png"
    fig.savefig(chart_path)
    plt.close(fig)
    logger.info(f"Generated {chart_path}")


def generate_chart5_adaptive_routing(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    """Chart 5: Adaptive Routing Distribution across Workload Classes (20 pages)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    adaptive_subset = [s for s in summaries if s["policy"] == "ADAPTIVE" and s["page_count"] == 20]
    classes = ["CLASS_A", "CLASS_B", "CLASS_C", "CLASS_D"]

    cpu_pcts = []
    gpu_pcts = []

    for c in classes:
        entry = next((s for s in adaptive_subset if s["workload_class"] == c), None)
        if entry:
            cpu_pcts.append(entry["cpu_selection_pct"])
            gpu_pcts.append(entry["gpu_selection_pct"])
        else:
            cpu_pcts.append(0.0)
            gpu_pcts.append(0.0)

    x = np.arange(len(classes))
    width = 0.5

    p1 = ax.bar(x, cpu_pcts, width, label="Routed to CPU (%)", color="#2b5c8f")
    p2 = ax.bar(x, gpu_pcts, width, bottom=cpu_pcts, label="Routed to GPU (%)", color="#00a86b")

    ax.set_title("Adaptive Route Distribution by Workload Class (20 Pages)")
    ax.set_ylabel("Percentage of Total Decisions (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Class A\n(Simple)", "Class B\n(Standard)", "Class C\n(Dense)", "Class D\n(Mixed)"])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    for i, (cp, gp) in enumerate(zip(cpu_pcts, gpu_pcts)):
        if cp > 0:
            ax.annotate(f"CPU: {cp:.1f}%", (x[i], cp / 2), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if gp > 0:
            ax.annotate(f"GPU: {gp:.1f}%", (x[i], cp + gp / 2), ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    fig.tight_layout()
    chart_path = out_dir / "chart5_adaptive_routing_distribution.png"
    fig.savefig(chart_path)
    plt.close(fig)
    logger.info(f"Generated {chart_path}")


def main() -> int:
    monitoring_dir = Path("outputs/monitoring")
    charts_dir = monitoring_dir / "g5_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    try:
        summaries, comparisons = load_data(monitoring_dir)
    except FileNotFoundError as e:
        logger.error(f"Cannot generate charts: {e}")
        return 1

    logger.info("Generating publication-quality charts for Phase G5...")
    generate_chart1_execution_time(summaries, charts_dir)
    generate_chart2_throughput(summaries, charts_dir)
    generate_chart3_latency_distribution(summaries, charts_dir)
    generate_chart4_gpu_profile(summaries, charts_dir)
    generate_chart5_adaptive_routing(summaries, charts_dir)

    logger.info(f"All 5 charts successfully generated in {charts_dir.resolve()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
