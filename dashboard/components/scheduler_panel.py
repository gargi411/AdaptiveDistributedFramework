"""scheduler_panel.py -- Scheduler monitoring panel for the Engineering Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_scheduler_panel(state: dict[str, Any]) -> None:
    """Render the Scheduler Monitoring section.

    Displays scheduling strategy, section 4.2 scheduler overhead, partition
    balance score, throughput, and resource orchestration recommendations.

    Args:
        state: Current DashboardStateStore snapshot dictionary.
    """
    st.markdown("## Scheduler Monitoring")

    dispatcher = state.get("dispatcher", {})
    ws_stats = state.get("work_stealing", {})
    ro_report = state.get("resource_orchestration", {})
    latest_rec = ro_report.get("latest_recommendation")

    # -- Section 4.2 Scheduler Overhead -------------------------------------
    disp_time = dispatcher.get("scheduler_time_seconds", 0.0)
    ws_time = ws_stats.get("scheduler_time_seconds", 0.0)
    total_scheduler_time = disp_time + ws_time

    uptime = state.get("uptime_seconds", 0.0) or 1.0
    overhead_frac = total_scheduler_time / uptime
    overhead_pct = overhead_frac * 100.0
    target_met = overhead_pct < 1.0

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "Scheduling Strategy",
            latest_rec.get("scheduling_strategy", "page_count_lpt") if latest_rec else "page_count_lpt",
        )
    with cols[1]:
        st.metric(
            "Scheduler Overhead",
            f"{overhead_pct:.4f}%",
            delta="[OK] <1% target" if target_met else "[WARN] >1% target",
            delta_color="normal" if target_met else "inverse",
        )
    with cols[2]:
        total_completed = dispatcher.get("total_completed", 0)
        throughput = total_completed / max(uptime, 1.0)
        st.metric("Throughput", f"{throughput:.2f} tasks/s")
    with cols[3]:
        queue_size = state.get("queue_size", 0)
        st.metric("Queue Depth", queue_size)

    st.divider()

    # -- Resource Orchestration Recommendation ------------------------------
    if latest_rec:
        action = latest_rec.get("action", "maintain")
        action_label = {
            "scale_out": "[SCALE OUT]",
            "scale_in": "[SCALE IN]",
            "maintain": "[MAINTAIN]",
        }.get(action, action.upper())
        st.markdown(f"**Orchestrator Recommendation:** `{action_label}`")

        rcols = st.columns(4)
        with rcols[0]:
            st.metric("Current Workers", latest_rec.get("current_workers", 0))
        with rcols[1]:
            st.metric("Recommended Workers", latest_rec.get("recommended_workers", 0))
        with rcols[2]:
            st.metric("Avg CPU", f"{latest_rec.get('avg_cpu_percent', 0):.1f}%")
        with rcols[3]:
            st.metric("Avg RAM", f"{latest_rec.get('avg_ram_percent', 0):.1f}%")

        st.caption(f"{latest_rec.get('reason', '-')}")

    # -- Scheduler Overhead breakdown ----------------------------------------
    with st.expander("Overhead Breakdown (Section 4.2)", expanded=False):
        st.markdown(
            """
            **Definition** (Architecture v2.0, Section 4.2):
            ```
            Scheduler Overhead (%) = (Scheduler Time / Total Execution Time) x 100
            ```
            **Target:** < 1%
            """
        )
        bcols = st.columns(3)
        with bcols[0]:
            st.metric("Dispatch Time", f"{disp_time:.6f}s")
        with bcols[1]:
            st.metric("Work-Stealing Time", f"{ws_time:.6f}s")
        with bcols[2]:
            st.metric("Total Scheduler Time", f"{total_scheduler_time:.6f}s")

        overhead_bar = min(overhead_frac, 1.0)
        bar_label = (
            f"Overhead: {overhead_pct:.4f}% {'[OK]' if target_met else '[WARN]'}"
        )
        st.progress(overhead_bar, text=bar_label)
