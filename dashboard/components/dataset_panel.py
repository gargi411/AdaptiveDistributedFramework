"""Dataset Health Panel — Streamlit component for Phase 3.5.

Renders a two-column dataset health summary sourced from the
``dataset_summary`` key in the dashboard state dictionary. This key is
populated by ``start_dev_cluster.py`` from ``DocumentRegistry.summary()``.

Usage::

    from dashboard.components.dataset_panel import render_dataset_panel

    render_dataset_panel(state)
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_dataset_panel(state: dict[str, Any]) -> None:
    """Render the Dataset Health section.

    Displays two columns of metrics:
        Left  — dataset composition (PDFs, pages, source types)
        Right — pipeline progress (queue, completed, failed, averages)

    Args:
        state: Current dashboard state dictionary from ``DashboardStateStore``.
    """
    ds: dict[str, Any] = state.get("dataset_summary", {})

    st.markdown("## 📄 Dataset Health")

    # ── Top status bar ──────────────────────────────────────────────────
    total_pdfs = ds.get("total_pdfs", 0)
    total_pages = ds.get("total_pages", 0)
    metadata_cached = ds.get("metadata_cached", False)

    if total_pdfs == 0:
        st.info("No dataset loaded yet. Run `python -m scripts.start_dev_cluster` to begin.")
        return

    cache_badge = "✅ Cached" if metadata_cached else "🔄 Extracted"

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("PDFs Loaded", total_pdfs)
    with col_b:
        st.metric("Pages Loaded", f"{total_pages:,}")
    with col_c:
        st.metric("Metadata", cache_badge)

    st.divider()

    # ── Two-column detail breakdown ─────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("**Source Composition**")
        digital = ds.get("digital_pdfs", 0)
        scanned = ds.get("scanned_pdfs", 0)
        unknown = ds.get("unknown_pdfs", 0)
        corrupted = 0  # Populated when failed > 0 and total < expected

        st.metric("Digital PDFs", digital)
        st.metric("Scanned PDFs", scanned)
        st.metric("Unknown / Mixed", unknown)
        st.metric("Corrupted / Failed Extraction", corrupted)

        st.divider()
        avg_pages = ds.get("avg_pages", 0.0)
        avg_size = ds.get("avg_size_mb", 0.0)
        st.metric("Average Pages / PDF", f"{avg_pages:.1f}")
        st.metric("Average PDF Size", f"{avg_size:.2f} MB")

    with right:
        st.markdown("**Pipeline Progress**")
        pending = ds.get("pending", 0)
        queued = ds.get("queued", 0)
        in_progress = ds.get("in_progress", 0)
        completed = ds.get("completed", 0)
        failed = ds.get("failed", 0)

        st.metric("Pending", pending)
        st.metric("Queued", queued)
        st.metric("In Progress", in_progress)
        st.metric(
            "Completed",
            completed,
            delta=completed if completed > 0 else None,
            delta_color="normal",
        )
        st.metric(
            "Failed",
            failed,
            delta=failed if failed > 0 else None,
            delta_color="inverse",
        )

    # ── Progress bar ────────────────────────────────────────────────────
    if total_pdfs > 0:
        progress = completed / total_pdfs
        st.divider()
        st.markdown(f"**Processing Progress** — {completed}/{total_pdfs} documents completed")
        st.progress(progress, text=f"{progress * 100:.1f}%")
