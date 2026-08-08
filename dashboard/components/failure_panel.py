"""failure_panel.py — Failure Recovery panel for the Engineering Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_failure_panel(state: dict[str, Any]) -> None:
    """Render the Failure Recovery section.

    Displays disconnected workers, recovered workers, recovered tasks,
    retry counts, and a history of recovery events.

    Args:
        state: Current DashboardStateStore snapshot dictionary.
    """
    st.markdown("## 🚨 Failure Recovery")

    rc_stats = state.get("failure_recovery", {})
    recovery_events: list[dict[str, Any]] = state.get("recovery_events", [])

    workers_lost = rc_stats.get("total_workers_lost", 0)
    tasks_recovered = rc_stats.get("total_tasks_recovered", 0)
    perm_failed = rc_stats.get("total_tasks_permanently_failed", 0)
    workers_recovered = rc_stats.get("total_workers_recovered", 0)
    max_retries = rc_stats.get("max_retries", 3)

    # ── Aggregate metrics ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Workers Lost",
            workers_lost,
            delta=f"-{workers_lost}" if workers_lost else None,
            delta_color="inverse" if workers_lost else "normal",
        )
    with c2:
        st.metric("Workers Recovered", workers_recovered)
    with c3:
        st.metric("Tasks Recovered", tasks_recovered)
    with c4:
        st.metric(
            "Permanently Failed",
            perm_failed,
            delta=f"Max Retries: {max_retries}",
            delta_color="off",
        )

    # ── Recovery flow explanation ───────────────────────────────────────
    with st.expander("📖 Recovery Flow (Architecture §2.3)", expanded=False):
        st.markdown(
            """
            ```
            Worker Lost (e.g. Laptop 2 disconnects)
                    ↓
            Detect via Heartbeat Timeout
                    ↓
            Return Unfinished Work Units
                    ↓
            Re-insert into Priority Queue
                    ↓
            Assign to Available Worker
            ```
            """
        )

    st.divider()

    # ── Recovery event log ──────────────────────────────────────────────
    if not recovery_events:
        st.success("✅ No failure events recorded. All workers healthy.")
        return

    st.markdown(f"**Recovery Events** — {len(recovery_events)} total")

    _type_icons = {
        "worker_lost": "💀",
        "tasks_recovered": "♻️",
        "worker_recovered": "✅",
        "retry_scheduled": "🔁",
        "retry_exhausted": "❌",
        "graceful_shutdown": "🛑",
    }

    rows = []
    for evt in recovery_events[:30]:
        etype = evt.get("event_type", "unknown")
        rows.append({
            "Timestamp": evt.get("timestamp", "")[:19],
            "Type": f"{_type_icons.get(etype, '⚪')} {etype.replace('_', ' ').title()}",
            "Worker": evt.get("worker_id", "")[:12],
            "Tasks Affected": len(evt.get("work_unit_ids", [])),
            "Retry #": evt.get("retry_count", 0),
            "Message": evt.get("message", "")[:60],
        })

    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
