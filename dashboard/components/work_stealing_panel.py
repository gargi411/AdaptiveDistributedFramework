"""work_stealing_panel.py -- Work Stealing visualization for the Engineering Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_work_stealing_panel(state: dict[str, Any]) -> None:
    """Render the Work Stealing Visualization section.

    Displays aggregate steal statistics and a chronological history table
    of all steal events (source, destination, count, timestamp).

    Args:
        state: Current DashboardStateStore snapshot dictionary.
    """
    st.markdown("## Work Stealing")

    ws_stats = state.get("work_stealing", {})
    steal_events: list[dict[str, Any]] = state.get("steal_events", [])

    # -- Aggregate metrics --------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Steal Events", ws_stats.get("total_steal_events", 0))
    with c2:
        st.metric("Tasks Stolen", ws_stats.get("total_tasks_stolen", 0))
    with c3:
        st.metric("Steal Threshold", ws_stats.get("steal_threshold", "-"))
    with c4:
        fraction = ws_stats.get("steal_fraction", 0.5)
        st.metric("Steal Fraction", f"{fraction:.0%}")

    st.divider()

    # -- Steal event history ------------------------------------------------
    total_steal_events = ws_stats.get("total_steal_events", 0)
    if not steal_events and total_steal_events == 0:
        st.info(
            "No work-stealing events recorded. "
            "Events appear when an idle worker steals tasks from an overloaded peer."
        )
        return

    if not steal_events and total_steal_events > 0:
        st.info(
            f"{total_steal_events} steal event(s) occurred during processing "
            "(detailed event log not available in post-run view)."
        )
    else:
        st.markdown(f"**Recent Steal Events** -- {len(steal_events)} total")

        rows = []
        for evt in steal_events[:30]:
            rows.append({
                "Timestamp": evt.get("timestamp", "")[:19],
                "From Worker": evt.get("source_worker_id", "")[:12],
                "To Worker": evt.get("destination_worker_id", "")[:12],
                "Tasks Stolen": evt.get("tasks_stolen", 0),
                "Src Q Before": evt.get("source_queue_depth_before", "-"),
                "Src Q After": evt.get("source_queue_depth_after", "-"),
                "Dst Q Before": evt.get("destination_queue_depth_before", "-"),
                "Dst Q After": evt.get("destination_queue_depth_after", "-"),
            })

        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # -- Per-worker steal summary -------------------------------------------
    workers: list[dict[str, Any]] = state.get("workers", [])
    if workers:
        with st.expander("Per-Worker Steal Summary", expanded=False):
            wrows = [
                {
                    "Worker": w.get("hostname", "-"),
                    "ID": w.get("worker_id", "")[:12],
                    "State": w.get("state", "-"),
                    "Stolen From": w.get("total_stolen_from", 0),
                    "Stolen To": w.get("total_stolen_to", 0),
                    "Queue Depth": w.get("queue_depth", 0),
                }
                for w in workers
            ]
            import pandas as pd
            st.dataframe(pd.DataFrame(wrows), use_container_width=True, hide_index=True)
