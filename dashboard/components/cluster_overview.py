"""cluster_overview.py — Cluster Overview panel for the Engineering Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_cluster_overview(state: dict[str, Any]) -> None:
    """Render the Cluster Overview section.

    Displays head node status, worker counts, cluster health,
    total CPU/RAM, and GPU availability as metric cards.

    Args:
        state: Current DashboardStateStore snapshot dictionary.
    """
    st.markdown("## 🌐 Cluster Overview")

    registry = state.get("registry", {})
    cm = state.get("cluster_manager", {})
    framework_state = state.get("framework_state", "unknown")
    uptime = state.get("uptime_seconds", 0.0)

    # ── Health indicator ────────────────────────────────────────────────
    health_color = {
        "ready": "🟢",
        "running": "🟢",
        "degraded": "🟠",
        "initializing": "🔵",
        "shutting_down": "🔴",
        "stopped": "⚫",
    }.get(framework_state, "⚪")

    st.markdown(
        f"**Framework State:** {health_color} `{framework_state.upper()}`"
        f"&nbsp;&nbsp;&nbsp;**Uptime:** `{_fmt_uptime(uptime)}`"
        f"&nbsp;&nbsp;&nbsp;**Mode:** `{cm.get('mode', 'dev').upper()}`"
    )

    st.divider()

    # ── Metric cards ────────────────────────────────────────────────────
    cols = st.columns(6)

    total_workers = registry.get("total_workers", 0)
    active_workers = registry.get("active_workers", 0)
    idle_workers = registry.get("idle_workers", 0)
    lost_workers = registry.get("lost_workers", 0)
    avg_cpu = registry.get("avg_cpu_percent", 0.0)
    avg_ram = registry.get("avg_ram_percent", 0.0)

    with cols[0]:
        st.metric("Total Workers", total_workers)

    with cols[1]:
        delta_active = active_workers - (total_workers - active_workers - idle_workers)
        st.metric("Active", active_workers, delta=f"{active_workers}/{total_workers}")

    with cols[2]:
        st.metric("Idle", idle_workers)

    with cols[3]:
        st.metric(
            "Lost / Degraded",
            lost_workers,
            delta=f"-{lost_workers}" if lost_workers else None,
            delta_color="inverse" if lost_workers else "normal",
        )

    with cols[4]:
        st.metric("Avg CPU", f"{avg_cpu:.1f}%")

    with cols[5]:
        st.metric("Avg RAM", f"{avg_ram:.1f}%")

    # ── Head node card ─────────────────────────────────────────────────
    head = cm.get("head_node")
    if head:
        with st.expander("🖥️ Head Node Details", expanded=False):
            hcols = st.columns(4)
            with hcols[0]:
                st.write(f"**Hostname:** `{head.get('hostname', '—')}`")
                st.write(f"**IP:** `{head.get('ip_address', '—')}`")
            with hcols[1]:
                st.write(f"**CPUs:** `{head.get('cpu_count_logical', '—')}`")
                st.write(f"**RAM:** `{head.get('ram_total_gb', 0):.1f} GB`")
            with hcols[2]:
                st.write(f"**GPUs:** `{head.get('gpu_count', 0)}`")
                st.write(f"**OS:** `{head.get('os_platform', '—')}`")
            with hcols[3]:
                st.write(f"**Node Uptime:** `{_fmt_uptime(cm.get('uptime_seconds', 0))}`")
                st.write(f"**Ray Alive:** `{'Yes ✅' if cm.get('alive') else 'No ❌'}`")


def _fmt_uptime(seconds: float) -> str:
    """Format seconds as H:MM:SS.

    Args:
        seconds: Elapsed seconds.

    Returns:
        Formatted string.
    """
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s"
