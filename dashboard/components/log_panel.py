"""log_panel.py — Cluster Logs panel for the Engineering Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_log_panel(state: dict[str, Any]) -> None:
    """Render the Cluster Logs section.

    Streams recent heartbeat events, recovery events, and work-stealing
    events as a unified log, colour-coded by severity and type.

    Args:
        state: Current DashboardStateStore snapshot dictionary.
    """
    st.markdown("## 📝 Cluster Logs")

    heartbeat_events: list[dict[str, Any]] = state.get("heartbeat_events", [])
    recovery_events: list[dict[str, Any]] = state.get("recovery_events", [])
    steal_events: list[dict[str, Any]] = state.get("steal_events", [])

    # Merge and sort by timestamp (desc)
    all_events: list[dict[str, Any]] = []

    for evt in heartbeat_events:
        all_events.append({
            "ts": evt.get("timestamp", ""),
            "category": "HEARTBEAT",
            "type": evt.get("event_type", ""),
            "worker": evt.get("worker_id", "")[:12],
            "message": evt.get("message", ""),
        })

    for evt in recovery_events:
        all_events.append({
            "ts": evt.get("timestamp", ""),
            "category": "RECOVERY",
            "type": evt.get("event_type", ""),
            "worker": evt.get("worker_id", "")[:12],
            "message": evt.get("message", ""),
        })

    for evt in steal_events:
        all_events.append({
            "ts": evt.get("timestamp", ""),
            "category": "STEAL",
            "type": "work_steal",
            "worker": evt.get("source_worker_id", "")[:8]
                       + "→" + evt.get("destination_worker_id", "")[:8],
            "message": (
                f"Stole {evt.get('tasks_stolen', 0)} task(s) "
                f"from {evt.get('source_worker_id', '')[:8]} "
                f"to {evt.get('destination_worker_id', '')[:8]}"
            ),
        })

    all_events.sort(key=lambda e: e["ts"], reverse=True)
    display_events = all_events[:100]

    if not display_events:
        st.info("No events yet. The cluster log will populate as the framework runs.")
        return

    # ── Filter controls ─────────────────────────────────────────────────
    categories = ["ALL", "HEARTBEAT", "RECOVERY", "STEAL"]
    selected = st.selectbox("Filter by category", categories, index=0)

    if selected != "ALL":
        display_events = [e for e in display_events if e["category"] == selected]

    st.caption(f"Showing {len(display_events)} event(s) (newest first)")

    # ── Log entries ─────────────────────────────────────────────────────
    _cat_colors = {
        "HEARTBEAT": "#1a73e8",
        "RECOVERY": "#ea4335",
        "STEAL": "#fbbc04",
    }
    _type_icons = {
        "alive": "💚",
        "timeout": "⏰",
        "disconnected": "💀",
        "reconnected": "🔄",
        "worker_lost": "💀",
        "tasks_recovered": "♻️",
        "worker_recovered": "✅",
        "retry_scheduled": "🔁",
        "retry_exhausted": "❌",
        "work_steal": "🔀",
    }

    for evt in display_events:
        cat = evt["category"]
        etype = evt["type"]
        icon = _type_icons.get(etype, "•")
        color = _cat_colors.get(cat, "#888")
        ts_short = evt["ts"][:19]
        worker = evt["worker"]
        msg = evt["message"][:120]

        st.markdown(
            f"<div style='font-family: monospace; font-size: 0.82em; "
            f"border-left: 3px solid {color}; padding-left: 8px; margin-bottom: 4px;'>"
            f"<span style='color: #888;'>{ts_short}</span> "
            f"<span style='color: {color}; font-weight: bold;'>[{cat}]</span> "
            f"{icon} <span style='color: #ccc;'>{worker}</span> — {msg}"
            f"</div>",
            unsafe_allow_html=True,
        )
