"""performance_charts.py — Real-time performance charts for the Engineering Dashboard.

Uses Plotly for rich, interactive charts rendered inside Streamlit.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def render_performance_charts(
    state: dict[str, Any],
    cpu_history: list[float],
    ram_history: list[float],
    queue_history: list[int],
    throughput_history: list[float],
) -> None:
    """Render the Performance Graphs section.

    Displays real-time sparkline charts for CPU, RAM, queue size,
    and throughput using Plotly. Falls back to st.line_chart if
    Plotly is not installed.

    Args:
        state: Current DashboardStateStore snapshot.
        cpu_history: List of average CPU % values (oldest first).
        ram_history: List of average RAM % values (oldest first).
        queue_history: List of queue size values (oldest first).
        throughput_history: List of completed task counts (oldest first).
    """
    st.markdown("## 📈 Performance Graphs")

    if not _PLOTLY_AVAILABLE:
        _render_fallback_charts(cpu_history, ram_history, queue_history, throughput_history)
        return

    _render_plotly_charts(state, cpu_history, ram_history, queue_history, throughput_history)


def _render_plotly_charts(
    state: dict[str, Any],
    cpu_history: list[float],
    ram_history: list[float],
    queue_history: list[int],
    throughput_history: list[float],
) -> None:
    """Render interactive Plotly charts.

    Args:
        state: Dashboard state snapshot.
        cpu_history: CPU % history.
        ram_history: RAM % history.
        queue_history: Queue size history.
        throughput_history: Throughput history.
    """
    n = max(len(cpu_history), 1)
    x = list(range(n))

    # ── Row 1: CPU + RAM ──────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig_cpu = go.Figure()
        if cpu_history:
            fig_cpu.add_trace(go.Scatter(
                x=x[:len(cpu_history)], y=cpu_history,
                mode="lines+markers",
                name="CPU %",
                line=dict(color="#4285f4", width=2),
                fill="tozeroy",
                fillcolor="rgba(66,133,244,0.15)",
            ))
        fig_cpu.add_hline(y=85, line_dash="dash", line_color="#ea4335",
                          annotation_text="High threshold")
        fig_cpu.update_layout(
            title="Average CPU Utilization",
            yaxis=dict(range=[0, 100], title="%"),
            xaxis_title="Sample #",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark",
        )
        st.plotly_chart(fig_cpu, use_container_width=True)

    with col2:
        fig_ram = go.Figure()
        if ram_history:
            fig_ram.add_trace(go.Scatter(
                x=x[:len(ram_history)], y=ram_history,
                mode="lines+markers",
                name="RAM %",
                line=dict(color="#34a853", width=2),
                fill="tozeroy",
                fillcolor="rgba(52,168,83,0.15)",
            ))
        fig_ram.update_layout(
            title="Average RAM Utilization",
            yaxis=dict(range=[0, 100], title="%"),
            xaxis_title="Sample #",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark",
        )
        st.plotly_chart(fig_ram, use_container_width=True)

    # ── Row 2: Queue Size + Throughput ───────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        fig_q = go.Figure()
        if queue_history:
            fig_q.add_trace(go.Bar(
                x=x[:len(queue_history)], y=queue_history,
                name="Queue Size",
                marker_color="#fbbc04",
            ))
        fig_q.update_layout(
            title="Global Queue Size",
            yaxis_title="Tasks",
            xaxis_title="Sample #",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark",
        )
        st.plotly_chart(fig_q, use_container_width=True)

    with col4:
        fig_tp = go.Figure()
        if len(throughput_history) > 1:
            # Compute delta (rate)
            deltas = [
                max(0.0, throughput_history[i] - throughput_history[i - 1])
                for i in range(1, len(throughput_history))
            ]
            fig_tp.add_trace(go.Scatter(
                x=list(range(len(deltas))), y=deltas,
                mode="lines+markers",
                name="Tasks/interval",
                line=dict(color="#ff6d00", width=2),
                fill="tozeroy",
                fillcolor="rgba(255,109,0,0.15)",
            ))
        fig_tp.update_layout(
            title="Task Throughput (Tasks / Interval)",
            yaxis_title="Tasks",
            xaxis_title="Interval #",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark",
        )
        st.plotly_chart(fig_tp, use_container_width=True)

    # ── Row 3: Worker Utilization Bar Chart ──────────────────────────
    workers: list[dict[str, Any]] = state.get("workers", [])
    if workers:
        st.markdown("**Worker Utilization Comparison**")
        names = [w.get("hostname", w.get("worker_id", "?")[:8]) for w in workers]
        cpu_vals = [w.get("cpu_percent", 0.0) for w in workers]
        ram_vals = [w.get("ram_percent", 0.0) for w in workers]

        fig_util = go.Figure(data=[
            go.Bar(name="CPU %", x=names, y=cpu_vals, marker_color="#4285f4"),
            go.Bar(name="RAM %", x=names, y=ram_vals, marker_color="#34a853"),
        ])
        fig_util.update_layout(
            barmode="group",
            title="Per-Worker CPU & RAM",
            yaxis=dict(range=[0, 100], title="%"),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark",
        )
        st.plotly_chart(fig_util, use_container_width=True)


def _render_fallback_charts(
    cpu_history: list[float],
    ram_history: list[float],
    queue_history: list[int],
    throughput_history: list[float],
) -> None:
    """Render basic line charts without Plotly.

    Args:
        cpu_history: CPU % history.
        ram_history: RAM % history.
        queue_history: Queue size history.
        throughput_history: Throughput history.
    """
    st.info("Install plotly for interactive charts: `pip install plotly`")

    import pandas as pd

    if cpu_history or ram_history:
        df = pd.DataFrame({"CPU %": cpu_history, "RAM %": ram_history})
        st.line_chart(df, height=200)

    if queue_history:
        df_q = pd.DataFrame({"Queue Size": queue_history})
        st.line_chart(df_q, height=150)

    if len(throughput_history) > 1:
        deltas = [
            max(0.0, throughput_history[i] - throughput_history[i - 1])
            for i in range(1, len(throughput_history))
        ]
        df_tp = pd.DataFrame({"Throughput": deltas})
        st.line_chart(df_tp, height=150)
