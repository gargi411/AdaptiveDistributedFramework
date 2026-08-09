"""Engineering Dashboard — Adaptive Distributed Framework v2.0

A live Streamlit monitoring dashboard for the distributed processing cluster.
Designed for research demonstration and engineering visibility.

Usage:
    # From the project root:
    streamlit run dashboard/app.py

    # Or with a custom state file (file-based mode for multi-process):
    streamlit run dashboard/app.py -- --state-file outputs/dashboard_state.json

Configuration:
    Reads cluster mode (dev / presentation) from configs/cluster.yaml.
    Refreshes automatically every REFRESH_INTERVAL_SECONDS seconds.

Sections:
    1. Cluster Overview
    2. Worker Monitoring
    3. Scheduler Monitoring
    4. Task Queue
    5. Work Stealing
    6. Failure Recovery
    7. Performance Charts
    8. Cluster Logs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import streamlit as st

# -- Resolve project root for imports ---------------------------------------
_DASHBOARD_DIR = Path(__file__).parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent

# Insert project root so `from dashboard.components.xxx` resolves,
# and insert src/ so adaptive_framework imports also resolve.
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# -- Dashboard components ---------------------------------------------------
from dashboard.components.cluster_overview import render_cluster_overview
from dashboard.components.dataset_panel import render_dataset_panel
from dashboard.components.worker_panel import render_worker_panel
from dashboard.components.task_queue_panel import render_task_queue_panel
from dashboard.components.scheduler_panel import render_scheduler_panel
from dashboard.components.work_stealing_panel import render_work_stealing_panel
from dashboard.components.failure_panel import render_failure_panel
from dashboard.components.log_panel import render_log_panel
from dashboard.components.performance_charts import render_performance_charts
from dashboard.state.dashboard_state import DashboardStateStore

# -- Constants --------------------------------------------------------------
REFRESH_INTERVAL_SECONDS: float = 3.0
DEFAULT_STATE_FILE: str = "outputs/dashboard_state.json"
FRAMEWORK_VERSION: str = "2.0.0"

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="ADF Engineering Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/adaptive-distributed-framework",
        "About": (
            "Adaptive Distributed Parallel Processing Framework v2.0\n"
            "Engineering Monitoring Dashboard - Phase 3.5"
        ),
    },
)

# -- Custom CSS (dark, Kubernetes-inspired) ---------------------------------
st.markdown(
    """
    <style>
    /* Dark header bar */
    [data-testid="stHeader"] {
        background-color: #0d1117;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    /* Main area background */
    .main .block-container {
        background-color: #0d1117;
        padding-top: 1rem;
    }
    /* Metric card borders */
    [data-testid="metric-container"] {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* Divider */
    hr {
        border-color: #30363d;
    }
    /* Container borders */
    [data-testid="stVerticalBlock"] > div[style] {
        border-radius: 8px;
    }
    /* Section headers */
    h2 {
        color: #e6edf3;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #30363d;
        padding-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -- Sidebar ----------------------------------------------------------------

def _render_sidebar(store: DashboardStateStore, state: dict) -> dict:
    """Render sidebar controls and return active section configuration.

    Args:
        store: The DashboardStateStore.
        state: Current state snapshot.

    Returns:
        Dict of display settings from sidebar controls.
    """
    with st.sidebar:
        st.markdown("# ADF Dashboard")
        st.markdown(f"**Version:** `{FRAMEWORK_VERSION}`")

        cm = state.get("cluster_manager", {})
        mode = cm.get("mode", "dev")
        mode_label = "DEV (Single Laptop)" if mode == "dev" else "PRESENTATION (3 Laptops)"
        st.info(f"**Cluster Mode:** {mode_label}")

        st.divider()

        st.markdown("**Navigation**")
        sections = st.multiselect(
            "Visible Sections",
            options=[
                "Cluster Overview",
                "Dataset Health",
                "Worker Monitoring",
                "Scheduler Monitoring",
                "Task Queue",
                "Work Stealing",
                "Failure Recovery",
                "Performance Charts",
                "Cluster Logs",
            ],
            default=[
                "Cluster Overview",
                "Dataset Health",
                "Worker Monitoring",
                "Scheduler Monitoring",
                "Task Queue",
                "Work Stealing",
                "Failure Recovery",
                "Performance Charts",
                "Cluster Logs",
            ],
        )

        st.divider()
        st.markdown("**Auto-Refresh**")
        auto_refresh = st.toggle("Enable Auto-Refresh", value=True)
        refresh_interval = st.slider(
            "Refresh Interval (s)",
            min_value=1,
            max_value=30,
            value=int(REFRESH_INTERVAL_SECONDS),
        )

        st.divider()
        st.markdown("**Cluster Info**")
        st.text(f"Run ID: {state.get('run_id', '-')}")
        uptime = state.get("uptime_seconds", 0.0)
        h, rem = divmod(int(uptime), 3600)
        m, s = divmod(rem, 60)
        st.text(f"Uptime: {h}h {m:02d}m {s:02d}s")
        last_updated = store.last_updated_seconds_ago
        if last_updated == float("inf"):
            st.text("Last Update: -")
        else:
            st.text(f"Last Update: {last_updated:.1f}s ago")

        ts = state.get("timestamp", "")
        if ts:
            st.text(f"Timestamp: {ts[:19]}")

        if st.button("Force Refresh", use_container_width=True):
            st.rerun()

    return {
        "sections": sections,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
    }


# -- State loading ----------------------------------------------------------

@st.cache_resource
def _get_store(state_file: str) -> DashboardStateStore:
    """Load (or create) the DashboardStateStore. Cached across reruns.

    Args:
        state_file: Path to state JSON file.

    Returns:
        DashboardStateStore instance.
    """
    path = Path(state_file)
    if path.exists():
        return DashboardStateStore.from_file(path)
    return DashboardStateStore(file_path=path)


def _load_state_from_file(store: DashboardStateStore, state_file: str) -> None:
    """Reload state from JSON file if it has been updated.

    Restores chart histories from the ``_chart_history`` key so the
    Performance Graphs panel shows the full run history, not just the
    current (post-run zero) snapshot.

    Args:
        store: The DashboardStateStore to update.
        state_file: Path to state JSON file.
    """
    path = Path(state_file)
    if not path.exists():
        return
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        # Restore full chart history if pipeline wrote it
        ch = data.get("_chart_history", {})
        if ch:
            with store._lock:  # type: ignore[attr-defined]
                store._cpu_history = [float(x) for x in ch.get("cpu", [])]  # type: ignore[attr-defined]
                store._ram_history = [float(x) for x in ch.get("ram", [])]  # type: ignore[attr-defined]
                store._queue_history = [int(x) for x in ch.get("queue", [])]  # type: ignore[attr-defined]
                store._throughput_history = [float(x) for x in ch.get("throughput", [])]  # type: ignore[attr-defined]
                store._data = data  # type: ignore[attr-defined]
                store._last_updated = time.monotonic()  # type: ignore[attr-defined]
        else:
            store.update(data)
    except Exception:
        pass


# -- Main ------------------------------------------------------------------

def main() -> None:
    """Main dashboard entrypoint."""
    # -- Parse CLI args -------------------------------------------------------
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    args, _ = parser.parse_known_args()
    state_file = args.state_file

    # -- Load state store -----------------------------------------------------
    store = _get_store(state_file)
    _load_state_from_file(store, state_file)
    state = store.get()

    # -- Sidebar --------------------------------------------------------------
    cfg = _render_sidebar(store, state)
    sections = cfg["sections"]
    auto_refresh = cfg["auto_refresh"]
    refresh_interval = cfg["refresh_interval"]

    # -- Page title -----------------------------------------------------------
    fw_state = state.get("framework_state", "initializing")
    health_label = (
        "[RUNNING]" if fw_state in ("ready", "running")
        else "[DEGRADED]" if fw_state == "degraded"
        else "[STOPPED]"
    )
    st.markdown(
        f"<h1 style='color:#e6edf3; font-size:1.6rem; margin-bottom:0;'>"
        f"Adaptive Distributed Framework -- Engineering Dashboard "
        f"<span style='font-size:0.9rem; color:#888;'>{health_label}</span>"
        f"</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Version {FRAMEWORK_VERSION} | Phase 3.5 | "
        f"Mode: `{state.get('cluster_manager', {}).get('mode', 'dev').upper()}`"
    )

    # -- Render sections ------------------------------------------------------
    if "Cluster Overview" in sections:
        render_cluster_overview(state)
        st.divider()

    if "Dataset Health" in sections:
        render_dataset_panel(state)
        st.divider()

    if "Worker Monitoring" in sections:
        render_worker_panel(state)
        st.divider()

    two_col_left = []
    two_col_right = []
    if "Scheduler Monitoring" in sections:
        two_col_left.append("scheduler")
    if "Task Queue" in sections:
        two_col_right.append("taskqueue")

    if two_col_left or two_col_right:
        cols = st.columns(2)
        with cols[0]:
            if "scheduler" in two_col_left:
                render_scheduler_panel(state)
        with cols[1]:
            if "taskqueue" in two_col_right:
                render_task_queue_panel(state)
        st.divider()

    if "Work Stealing" in sections:
        render_work_stealing_panel(state)
        st.divider()

    if "Failure Recovery" in sections:
        render_failure_panel(state)
        st.divider()

    if "Performance Charts" in sections:
        render_performance_charts(
            state=state,
            cpu_history=store.get_cpu_history(),
            ram_history=store.get_ram_history(),
            queue_history=store.get_queue_history(),
            throughput_history=store.get_throughput_history(),
        )
        st.divider()

    if "Cluster Logs" in sections:
        render_log_panel(state)

    # -- Auto-refresh ---------------------------------------------------------
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
