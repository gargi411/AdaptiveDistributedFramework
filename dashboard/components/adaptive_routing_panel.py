"""adaptive_routing_panel.py -- Adaptive CPU/GPU Routing Panel for Engineering Dashboard."""

from __future__ import annotations

from typing import Any
import streamlit as st


def render_adaptive_routing_panel(state: dict[str, Any]) -> None:
    """Render the Adaptive CPU/GPU Work Routing monitoring panel.

    Displays routing status, decisions distribution, fallback/failure metrics,
    and a table of recent routing decisions with factor-specific rationales.

    Args:
        state: Current DashboardStateStore snapshot dictionary.
    """
    st.markdown("## Adaptive Work Routing (Phase G4)")

    routing = state.get("adaptive_routing", {})
    gpu = state.get("gpu", {})
    registry = state.get("registry", {})
    queue_size = state.get("queue_size", 0)

    enabled = routing.get("enabled", True)
    total_decisions = routing.get("total_decisions", 0)
    cpu_decisions = routing.get("cpu_decisions", 0)
    gpu_decisions = routing.get("gpu_decisions", 0)
    fallback_decisions = routing.get("fallback_decisions", 0)
    gpu_failures = routing.get("gpu_failure_count", 0)

    cpu_pct = registry.get("avg_cpu_percent", 0.0)
    gpu_pct = gpu.get("utilization_percent")
    gpu_pct_str = f"{gpu_pct:.1f}%" if gpu_pct is not None else "UNAVAILABLE"

    # Status header
    status_label = "[ENABLED]" if enabled else "[DISABLED]"
    active_policy = routing.get("active_policy", routing.get("policy_version", "v1.0"))
    st.markdown(
        f"**Routing Engine:** `{status_label}`&nbsp;&nbsp;&nbsp;"
        f"**Active Policy:** `{active_policy}`&nbsp;&nbsp;&nbsp;"
        f"**Current CPU Load:** `{cpu_pct:.1f}%`&nbsp;&nbsp;&nbsp;"
        f"**Current GPU Load:** `{gpu_pct_str}`&nbsp;&nbsp;&nbsp;"
        f"**Queue Depth:** `{queue_size}`"
    )

    st.divider()

    # Metric Cards
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Decisions", total_decisions)
    with cols[1]:
        st.metric("CPU Decisions", cpu_decisions)
    with cols[2]:
        st.metric("GPU Decisions", gpu_decisions)
    with cols[3]:
        st.metric("Execution Fallbacks", fallback_decisions)
    with cols[4]:
        st.metric("GPU Failures", gpu_failures)

    # Selection ratio bar if decisions exist
    if total_decisions > 0:
        ratio = routing.get("device_selection_ratio", {})
        cpu_r = ratio.get("cpu", 0.0) * 100.0
        gpu_r = ratio.get("gpu", 0.0) * 100.0
        st.caption(f"Observed Device Distribution: **CPU: {cpu_r:.1f}%** | **GPU: {gpu_r:.1f}%**")

    # Recent decisions table
    recent = routing.get("recent_decisions", [])
    with st.expander(f"Recent Routing Decisions ({len(recent)})", expanded=True):
        if not recent:
            st.info("No routing decisions recorded in the active session yet.")
        else:
            table_data: list[dict[str, Any]] = []
            for d in reversed(recent[-20:]):  # newest first, up to 20
                wl = d.get("workload", {})
                res = d.get("resources") or {}
                row = {
                    "Time": d.get("timestamp", "")[-8:],
                    "Document": wl.get("document_id", "-"),
                    "Page": wl.get("page_number", 1),
                    "Policy": d.get("policy_name", "G4"),
                    "Complexity": f"{wl.get('complexity', 0.0):.2f}",
                    "CPU Score": f"{d.get('cpu_score', 0.0):.2f}" if "cpu_score" in d else "-",
                    "GPU Score": f"{d.get('gpu_score', 0.0):.2f}" if "gpu_score" in d else "-",
                    "Decision": d.get("target_device", "-"),
                    "Confidence": f"{d.get('confidence', 1.0):.2f}",
                    "Reason": d.get("reason", "-"),
                }
                table_data.append(row)
            st.dataframe(table_data, use_container_width=True)

