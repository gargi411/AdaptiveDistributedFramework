"""task_queue_panel.py — Task Queue visualization for the Engineering Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_task_queue_panel(state: dict[str, Any]) -> None:
    """Render the Task Queue monitoring section.

    Displays pending / running / completed / failed task counts as
    progress bars and a summary table of recent assignment history.

    Args:
        state: Current DashboardStateStore snapshot dictionary.
    """
    st.markdown("## 📋 Task Queue")

    dispatcher = state.get("dispatcher", {})
    queue_size = state.get("queue_size", 0)
    total_dispatched = dispatcher.get("total_dispatched", 0)
    total_completed = dispatcher.get("total_completed", 0)
    total_failed = dispatcher.get("total_failed", 0)
    currently_active = dispatcher.get("currently_active", 0)

    # Derived
    pending = queue_size
    running = currently_active
    completed = total_completed
    failed = total_failed
    grand_total = max(pending + running + completed + failed, 1)

    # ── Summary metrics ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("⏳ Pending", pending)
    with c2:
        st.metric("⚡ Running", running)
    with c3:
        st.metric("✅ Completed", completed)
    with c4:
        st.metric("❌ Failed", failed)

    st.divider()

    # ── Progress bars ───────────────────────────────────────────────────
    st.markdown("**Overall Progress**")
    if grand_total > 0:
        complete_frac = completed / grand_total
        st.progress(complete_frac, text=f"Completed: {complete_frac*100:.1f}%")

        failed_frac = failed / grand_total
        if failed_frac > 0:
            st.progress(
                failed_frac,
                text=f"Failed: {failed_frac*100:.1f}%",
            )

    # ── Dispatcher statistics ───────────────────────────────────────────
    with st.expander("📊 Dispatcher Statistics", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Dispatched", total_dispatched)
        with cols[1]:
            sched_time = dispatcher.get("scheduler_time_seconds", 0.0)
            st.metric("Scheduler Time", f"{sched_time:.4f}s")
        with cols[2]:
            history_size = dispatcher.get("history_size", 0)
            st.metric("History Records", history_size)

    # ── Assignment history ──────────────────────────────────────────────
    assignments: list[dict[str, Any]] = state.get("assignment_history", [])
    if assignments:
        st.markdown("**Recent Assignments**")
        rows = []
        for a in assignments[:20]:
            rows.append({
                "Work Unit": a.get("work_unit_id", "")[:12],
                "Worker": a.get("worker_id", "")[:12],
                "Pages": a.get("page_count", "—"),
                "Status": a.get("status", "—").upper(),
                "Retries": a.get("retry_count", 0),
                "Elapsed (s)": f"{a.get('elapsed_seconds') or 0:.2f}",
                "Assigned At": a.get("assigned_at", "—")[:19],
            })
        if rows:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
