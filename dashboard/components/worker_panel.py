"""worker_panel.py -- Per-worker monitoring panel for the Engineering Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


_STATE_LABEL = {
    "idle": "IDLE",
    "active": "ACTIVE",
    "overloaded": "OVERLOADED",
    "lost": "LOST",
    "shutting_down": "STOPPING",
}


def render_worker_panel(state: dict[str, Any]) -> None:
    """Render the Worker Monitoring section.

    Displays a card per worker with ID, hostname, state, CPU/RAM bars,
    queue depth, current task, heartbeat status, and utilization score.

    Args:
        state: Current DashboardStateStore snapshot dictionary.
    """
    st.markdown("## Worker Monitoring")

    workers: list[dict[str, Any]] = state.get("workers", [])

    # Fall back to registry summary if workers list not populated
    if not workers:
        registry = state.get("registry", {})
        total = registry.get("total_workers", 0)
        if total == 0:
            st.info("No workers registered. Start the cluster and workers will appear here.")
            return
        st.warning(
            f"{total} worker(s) registered -- detailed per-worker metrics will appear "
            "after the first heartbeat cycle."
        )
        return

    # Sort: ACTIVE -> IDLE -> OVERLOADED -> LOST -> SHUTTING_DOWN
    _order = {"active": 0, "idle": 1, "overloaded": 2, "shutting_down": 3, "lost": 4}
    workers_sorted = sorted(workers, key=lambda w: _order.get(w.get("state", "idle"), 99))

    # Two columns of worker cards
    cols = st.columns(2)
    for i, worker in enumerate(workers_sorted):
        with cols[i % 2]:
            _render_worker_card(worker)


def _render_worker_card(worker: dict[str, Any]) -> None:
    """Render a single worker card.

    Args:
        worker: Worker dictionary from WorkerRecord.to_dict().
    """
    wid = worker.get("worker_id", "unknown")
    hostname = worker.get("hostname", "-")
    state = worker.get("state", "idle")
    state_label = _STATE_LABEL.get(state, state.upper())
    cpu = worker.get("cpu_percent", 0.0)
    ram = worker.get("ram_percent", 0.0)
    gpu = worker.get("gpu_percent")
    queue_depth = worker.get("queue_depth", 0)
    current_task = worker.get("current_task_id") or "-"
    completed = worker.get("total_completed", 0)
    failed = worker.get("total_failed", 0)
    stolen_from = worker.get("total_stolen_from", 0)
    stolen_to = worker.get("total_stolen_to", 0)
    util_score = worker.get("utilization_score", 0.0)
    last_seen = worker.get("last_seen_at", "-")

    short_id = wid[:12] if len(wid) > 12 else wid

    state_color = (
        "#00cc44" if state == "active"
        else "#ff4444" if state == "lost"
        else "#ffaa00"
    )

    with st.container(border=True):
        # Header row
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            st.markdown(f"**{hostname}** `{short_id}`")
        with hcol2:
            st.markdown(
                f"<span style='color: {state_color}'>"
                f"**{state_label}**</span>",
                unsafe_allow_html=True,
            )

        # Resource bars
        st.progress(cpu / 100.0, text=f"CPU {cpu:.1f}%")
        st.progress(ram / 100.0, text=f"RAM {ram:.1f}%")
        if gpu is not None:
            st.progress(gpu / 100.0, text=f"GPU {gpu:.1f}%")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Queue", queue_depth)
        with m2:
            st.metric("Done", completed)
        with m3:
            st.metric("Failed", failed)
        with m4:
            st.metric("Util.", f"{util_score:.2f}")

        # Details
        with st.expander("Details", expanded=False):
            st.text(f"IP:            {worker.get('ip_address', '-')}")
            st.text(f"CPUs:          {worker.get('cpu_count', '-')}")
            st.text(f"RAM Total:     {worker.get('ram_total_gb', 0):.1f} GB")
            st.text(f"GPUs:          {worker.get('gpu_count', 0)}")
            st.text(f"Current Task:  {current_task}")
            st.text(f"Stolen From:   {stolen_from}")
            st.text(f"Stolen To:     {stolen_to}")
            st.text(f"Registered:    {worker.get('registered_at', '-')}")
            st.text(f"Last Seen:     {last_seen}")
