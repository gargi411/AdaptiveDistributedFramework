"""start_dev_cluster.py — Launch a full dev-mode pipeline on a single laptop.

Phase 3.5 update: replaced synthetic document generation with real biomedical
PDFs from ``dataset/raw/pmc_pdfs/`` via ``load_real_dataset()``.

This script:
    1. Reads configs/simulation_dev.yaml
    2. Loads real biomedical PDFs -> builds DocumentRegistry
    3. Partitions documents across workers using PageCountPartitioner (LPT)
    4. Starts local coordinator and registers simulated workers
    5. Submits real PageWorkUnits to the priority queue
    6. Runs a background processing thread (DocumentProcessingWorker per worker)
       that drains the queue, processes every page, and builds UnifiedDocuments
    7. Writes dashboard state (including Dataset Health) to outputs/dashboard_state.json
    8. Runs a foreground monitoring loop (Ctrl+C to stop)

Usage:
    python -m scripts.start_dev_cluster [--config configs/simulation_dev.yaml]
    python -m scripts.start_dev_cluster --workers 4

Then open the dashboard in another terminal:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

# ── Resolve project root ────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from adaptive_framework.coordinator.distributed_coordinator import DistributedCoordinator
from adaptive_framework.coordinator.heartbeat_monitor import HeartbeatPayload
from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.dataset_builder.document_registry import (
    DocumentRegistry,
    DocumentStatus,
)
from adaptive_framework.document_processing.processing_worker import DocumentProcessingWorker
from adaptive_framework.document_processing.unified_document_builder import UnifiedDocumentBuilder
from adaptive_framework.models.page import Page
from adaptive_framework.models.runtime import WorkerState
from adaptive_framework.scheduler.page_count_partitioner import PageCountPartitioner
from adaptive_framework.scheduler.partition_summary import PartitionSummary
from dashboard.state.dashboard_state import DashboardStateStore

logger = logging.getLogger("adaptive_framework.scripts")

_SHUTDOWN_REQUESTED = False


def _handle_sigint(signum: int, frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    print("\n\n[Ctrl+C] Shutdown requested. Stopping coordinator gracefully...")
    _SHUTDOWN_REQUESTED = True


def _load_config(config_path: str) -> dict:
    """Load the cluster config YAML.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Cluster configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"[ERROR] Config not found: {path}. Falling back to defaults.")
        return {
            "mode": "dev",
            "head": {"address": "auto"},
            "workers": {"num_workers_dev": 4},
        }
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("cluster", raw)


def _simulate_heartbeat(
    coordinator: DistributedCoordinator,
    worker_ids: list[str],
) -> None:
    """Send synthetic heartbeats from all simulated workers.

    Uses real CPU/RAM metrics from the host machine with per-worker variation.

    Args:
        coordinator: The active DistributedCoordinator.
        worker_ids: List of registered worker IDs.
    """
    import random
    import psutil

    real_cpu = psutil.cpu_percent(interval=None)
    real_ram = psutil.virtual_memory().percent

    for i, wid in enumerate(worker_ids):
        cpu = min(100.0, max(0.0, real_cpu + random.uniform(-10, 15)))
        ram = min(100.0, max(0.0, real_ram + random.uniform(-5, 10)))
        queue_depth = max(0, random.randint(0, 5))
        state_choice = (
            WorkerState.ACTIVE if queue_depth > 0 else WorkerState.IDLE
        )

        payload = HeartbeatPayload(
            worker_id=wid,
            cpu_percent=cpu,
            ram_percent=ram,
            gpu_percent=None,
            queue_depth=queue_depth,
            current_task_id=f"wu_{i:04d}" if queue_depth > 0 else None,
            state=state_choice.value,
        )
        coordinator.receive_heartbeat(payload)


def _log_partition_assignments(partitions: list, registry: DocumentRegistry) -> None:
    """Log worker-to-document assignments to stdout and logger.

    Args:
        partitions: List of Partition objects from PageCountPartitioner.
        registry: Populated DocumentRegistry for filename lookups.
    """
    print("\nGenerating PageWorkUnits...")
    for partition in partitions:
        worker_id = partition.worker_id or "unassigned"
        total_pages = partition.total_pages
        num_docs = partition.total_work_units
        print(
            f"  {worker_id:<12} ->  {total_pages:>5} pages  ({num_docs} doc(s))"
        )
        for wu in partition.work_units:
            path = registry.get_path(wu.document_id)
            fname = path.name if path else wu.document_id[:16]
            logger.info(
                "  %-14s -> %s  (pages %d-%d)",
                worker_id, fname, wu.start_page, wu.end_page,
            )


# ── Processing loop ─────────────────────────────────────────────────────


def _run_processing_loop(
    coordinator: DistributedCoordinator,
    doc_registry: DocumentRegistry,
    worker_ids: list[str],
    run_id: str,
    num_workers: int,
) -> None:
    """Background thread: drain the priority queue and process every work unit.

    One DocumentProcessingWorker is created per simulated worker ID.
    Work units are popped from the coordinator's priority queue and dispatched
    in round-robin order. Pages are collected per document; when all pages for
    a document are assembled the UnifiedDocumentBuilder produces a
    UnifiedDocument which is stored back in the DocumentRegistry.

    Args:
        coordinator: Active DistributedCoordinator holding the global queue.
        doc_registry: DocumentRegistry to update status and store results.
        worker_ids: Registered simulated worker IDs (round-robin assignment).
        run_id: Pipeline run identifier passed to UnifiedDocumentBuilder.
        num_workers: Number of simulated workers.
    """
    # One worker instance per worker_id, stored as (id, worker) pairs
    worker_pairs: list[tuple[str, DocumentProcessingWorker]] = [
        (wid, DocumentProcessingWorker(worker_id=wid, node_id=f"node-{i:04d}"))
        for i, wid in enumerate(worker_ids)
    ]
    builder = UnifiedDocumentBuilder()

    # Accumulate pages per document: doc_id -> list[Page]
    pages_by_doc: dict[str, list[Page]] = collections.defaultdict(list)
    # Track expected page range per document: doc_id -> (start, end)
    expected_by_doc: dict[str, tuple[int, int]] = {}

    worker_idx = 0  # round-robin counter

    print("\n[Processing] Starting document processing loop...")
    processed_units = 0
    completed_docs = 0

    while not _SHUTDOWN_REQUESTED:
        wu = coordinator.pop_work_unit()

        if wu is None:
            # Queue empty — check if all docs are done
            if coordinator.queue_size() == 0 and processed_units > 0:
                # Flush any remaining accumulated pages
                _flush_completed_docs(
                    pages_by_doc, expected_by_doc,
                    doc_registry, builder, run_id,
                    force=True,
                )
                completed = len(doc_registry.completed())
                total = len(doc_registry)
                if completed >= total:
                    print(
                        f"\n[Processing] All {completed} documents completed "
                        f"({total} registered). Processing loop done."
                    )
                    break
            time.sleep(0.2)
            continue

        # Mark document as IN_PROGRESS on first work unit for this doc
        doc_id = wu.document_id
        current_status = doc_registry.get_status(doc_id)
        if current_status == DocumentStatus.QUEUED:
            try:
                doc_registry.set_status(doc_id, DocumentStatus.IN_PROGRESS)
            except KeyError:
                pass

        # Track expected page range for completion detection
        if doc_id not in expected_by_doc:
            meta = doc_registry.get_metadata(doc_id)
            if meta:
                expected_by_doc[doc_id] = (1, meta.pages)

        # Assign to next worker (round-robin)
        wid, worker = worker_pairs[worker_idx % num_workers]
        worker_idx += 1

        fname = Path(wu.file_path).name
        print(
            f"  [{wid}] {fname:<40} "
            f"pages {wu.start_page}-{wu.end_page} ... ",
            end="",
            flush=True,
        )

        t0 = time.perf_counter()
        result = worker.process_work_unit(wu)
        elapsed = time.perf_counter() - t0

        if result.success:
            pages_by_doc[doc_id].extend(result.pages)
            print(f"done  ({elapsed:.2f}s,  {result.pages_succeeded} pages)")
        else:
            print(f"FAILED  ({result.error})")
            logger.warning(
                "Work unit failed for doc '%s': %s", doc_id[:8], result.error
            )

        processed_units += 1

        # Try to flush completed documents after every work unit
        newly_completed = _flush_completed_docs(
            pages_by_doc, expected_by_doc,
            doc_registry, builder, run_id,
            force=False,
        )
        completed_docs += newly_completed

    logger.info(
        "Processing loop exited. processed_units=%d  completed_docs=%d",
        processed_units,
        completed_docs,
    )


def _flush_completed_docs(
    pages_by_doc: dict[str, list[Page]],
    expected_by_doc: dict[str, tuple[int, int]],
    doc_registry: DocumentRegistry,
    builder: UnifiedDocumentBuilder,
    run_id: str,
    force: bool = False,
) -> int:
    """Build UnifiedDocuments for any document whose pages are all collected.

    Args:
        pages_by_doc: Accumulated pages per document_id.
        expected_by_doc: Expected (start_page, end_page) per document_id.
        doc_registry: Registry to store results and update status.
        builder: UnifiedDocumentBuilder instance.
        run_id: Pipeline run identifier.
        force: If True, flush all docs regardless of completeness check.

    Returns:
        Number of UnifiedDocuments newly built in this call.
    """
    newly_built = 0
    completed_ids = []

    for doc_id, pages in pages_by_doc.items():
        if not pages:
            continue

        # Check if we have all pages
        meta = doc_registry.get_metadata(doc_id)
        expected_total = meta.pages if meta else 1

        collected = len(pages)
        if not force and collected < expected_total:
            continue  # still waiting for more work units

        # Build UnifiedDocument
        file_path = str(doc_registry.get_path(doc_id) or "")
        try:
            unified_doc = builder.build(
                document_id=doc_id,
                file_path=file_path,
                pages=pages,
                run_id=run_id,
            )
            doc_registry.set_unified_document(doc_id, unified_doc)
            fname = Path(file_path).name
            logger.info(
                "UnifiedDocument built: %s (%d pages, %d chars)",
                fname,
                unified_doc.statistics.total_pages,
                unified_doc.statistics.total_chars,
            )
            newly_built += 1
        except Exception as exc:
            logger.error(
                "UnifiedDocumentBuilder failed for doc '%s': %s", doc_id[:8], exc
            )
            try:
                doc_registry.set_status(doc_id, DocumentStatus.FAILED)
            except KeyError:
                pass

        completed_ids.append(doc_id)

    # Remove flushed docs from accumulator
    for doc_id in completed_ids:
        pages_by_doc.pop(doc_id, None)

    return newly_built


def main() -> None:
    """Entry point for the dev cluster (real dataset mode)."""
    parser = argparse.ArgumentParser(
        description="ADF Dev Cluster - Real Dataset Mode (Phase 3.5)"
    )
    parser.add_argument(
        "--config",
        default="configs/simulation_dev.yaml",
        help="Path to cluster YAML config",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of simulated workers (default from config)",
    )
    parser.add_argument(
        "--state-file",
        default="outputs/dashboard_state.json",
        help="Path to write dashboard state JSON",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset_builder.yaml",
        help="Path to dataset YAML config",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Force metadata re-extraction even if cache exists",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Load cluster config ─────────────────────────────────────────────
    config = _load_config(args.config)
    num_workers = args.workers or config.get("workers", {}).get("num_workers_dev", 4)

    print(f"[ADF] Dev Cluster - Real Dataset Mode (Phase 3.5)")
    print(f"[ADF] Config:   {args.config}")
    print(f"[ADF] Workers:  {num_workers}")
    print(f"[ADF] State:    {args.state_file}")
    print(f"[ADF] Dataset:  {args.dataset_config}")
    print()

    # ── Load real biomedical dataset ────────────────────────────────────
    from scripts.load_real_dataset import load_real_dataset

    doc_registry: DocumentRegistry = load_real_dataset(
        config_path=args.dataset_config,
        force_refresh=args.force_refresh,
    )

    # ── Partition across workers ────────────────────────────────────────
    print("\nBuilding BiomedicalDataset...")
    print("Scheduler initialized.")

    dataset = doc_registry.all_metadata()
    partitioner = PageCountPartitioner()
    partitions, part_stats = partitioner.partition(dataset, num_workers=num_workers)

    _log_partition_assignments(partitions, doc_registry)

    # ── Dashboard state store ───────────────────────────────────────────
    state_store = DashboardStateStore(file_path=args.state_file)

    # ── Build coordinator ───────────────────────────────────────────────
    run_id = f"real_dataset_{int(time.time())}"
    coordinator = DistributedCoordinator(
        cluster_config=config,
        run_id=run_id,
    )
    coordinator.start(init_ray=False)

    # Submit real work units to the coordinator queue
    total_wus = coordinator.submit_partitions(partitions)

    # ── Register simulated workers ──────────────────────────────────────
    worker_ids: list[str] = []
    for i in range(num_workers):
        node_info = NodeInfo.from_current_host(is_head=False)
        node_info.hostname = f"worker-{i:02d}"
        node_info.node_id = f"node-{i:04d}"
        wid = coordinator.register_worker(
            node_info=node_info, worker_id=f"worker_{i:04d}"
        )
        worker_ids.append(wid)

    print("\nCoordinator started.")
    print("Dashboard updated.")
    print("------------------------------------------------")
    print(f"\n[ADF] {total_wus} work units submitted to queue.")
    print(f"[ADF] All {num_workers} workers registered.")
    print(f"[ADF] Dashboard: streamlit run dashboard/app.py")
    print(f"[ADF] Press Ctrl+C to stop.\n")

    # ── Partition summary table ─────────────────────────────────────────
    summary_table = PartitionSummary(partitions, part_stats)
    print(summary_table.format_table())

    # ── Start processing in background thread ───────────────────────────
    proc_thread = threading.Thread(
        target=_run_processing_loop,
        args=(coordinator, doc_registry, worker_ids, run_id, num_workers),
        daemon=True,
        name="processing-loop",
    )
    proc_thread.start()

    # ── Main monitoring loop ────────────────────────────────────────────
    heartbeat_interval = config.get("health", {}).get(
        "heartbeat_interval_seconds", 5.0
    )

    while not _SHUTDOWN_REQUESTED:
        _simulate_heartbeat(coordinator, worker_ids)

        status = coordinator.get_status_dict()
        registry_recs = coordinator.registry
        workers_list = [rec.to_dict() for rec in registry_recs.get_all()]
        status["workers"] = workers_list

        # Live dataset summary from DocumentRegistry (updates as docs complete)
        reg_summary = doc_registry.summary()
        dataset_summary = reg_summary.to_dict()
        status["dataset_summary"] = dataset_summary

        state_store.update(status)

        reg_stats = status.get("registry", {})
        print(
            f"\r[ADF] "
            f"Active: {reg_stats.get('active_workers', 0)} | "
            f"Idle: {reg_stats.get('idle_workers', 0)} | "
            f"CPU: {reg_stats.get('avg_cpu_percent', 0):.1f}% | "
            f"Queue: {coordinator.queue_size()} | "
            f"Completed: {dataset_summary.get('completed', 0)}/{dataset_summary.get('total_pdfs', 0)} | "
            f"Pages: {dataset_summary.get('total_pages', 0)}",
            end="",
            flush=True,
        )

        # Exit monitoring loop once all documents are completed
        total_docs = dataset_summary.get("total_pdfs", 0)
        completed_docs = dataset_summary.get("completed", 0)
        if total_docs > 0 and completed_docs >= total_docs:
            print(
                f"\n\n[ADF] All {completed_docs} documents processed successfully!"
            )
            # Let the processing thread finish
            proc_thread.join(timeout=10.0)
            break

        time.sleep(heartbeat_interval)

    # ── Graceful shutdown ───────────────────────────────────────────────
    if _SHUTDOWN_REQUESTED:
        print(f"\n\n[ADF] Stopping coordinator...")
    result = coordinator.stop(shutdown_ray=False)
    if result:
        print(result.format_summary())
    print("[ADF] Dev cluster stopped.")

    # ── Final summary ───────────────────────────────────────────────────
    final = doc_registry.summary()
    print()
    print("=" * 56)
    print("  ADF Phase 3.5 - Final Processing Summary")
    print("=" * 56)
    print(f"  Total documents : {final.total_pdfs}")
    print(f"  Total pages     : {final.total_pages}")
    print(f"  Completed       : {final.completed}")
    print(f"  Failed          : {final.failed}")
    print(f"  Digital PDFs    : {final.digital_pdfs}")
    print(f"  Scanned PDFs    : {final.scanned_pdfs}")
    print(f"  UnifiedDocuments: {final.completed}")
    print("=" * 56)
    print()


if __name__ == "__main__":
    main()
