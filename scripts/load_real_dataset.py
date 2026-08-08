"""load_real_dataset.py — Phase 3.5 dataset pipeline orchestrator.

Orchestrates the full local-dataset pipeline and returns a populated
``DocumentRegistry``. Intended to be imported by ``start_dev_cluster.py``
as a single function call, keeping cluster startup logic clean.

Pipeline executed:
    1. Read ``configs/dataset_builder.yaml`` (``dataset`` section)
    2. ``DatasetScanner.scan()``         → list[Path]
    3. ``DatasetLoader.load()``          → list[PDFMetadata]
    4. ``DocumentRegistry.register_batch()``  → populated registry

Usage::

    from scripts.load_real_dataset import load_real_dataset

    registry = load_real_dataset()
    metadata  = registry.all_metadata()   # → list[PDFMetadata] for scheduler
    summary   = registry.summary()        # → RegistrySummary for dashboard
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

# ── Resolve project root for imports ────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from adaptive_framework.dataset_builder.dataset_loader import DatasetLoader
from adaptive_framework.dataset_builder.dataset_scanner import DatasetScanner
from adaptive_framework.dataset_builder.document_registry import (
    DocumentRegistry,
    DocumentStatus,
)
from adaptive_framework.models.document import PDFMetadata

logger = logging.getLogger("adaptive_framework.scripts.load_real_dataset")

# Default config file (relative to project root)
_DEFAULT_CONFIG = "configs/dataset_builder.yaml"


def _load_dataset_config(config_path: str) -> dict:
    """Load the ``dataset`` section from a YAML config file.

    Args:
        config_path: Path to the YAML config file (relative to CWD or absolute).

    Returns:
        Dictionary with the ``dataset`` config keys. Falls back to safe defaults
        if the file is missing or the ``dataset`` key is absent.
    """
    defaults: dict = {
        "root": "dataset/raw/pmc_pdfs",
        "recursive": False,
        "extensions": ["pdf"],
        "metadata_cache_dir": "dataset/metadata",
        "force_refresh": False,
        "text_sample_pages": 3,
    }
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "Config not found: '%s'. Using defaults.", config_path
        )
        return defaults
    try:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = raw.get("dataset", {})
        # Merge with defaults (config values take precedence)
        return {**defaults, **cfg}
    except Exception as exc:
        logger.warning("Failed to parse '%s': %s. Using defaults.", config_path, exc)
        return defaults


def load_real_dataset(
    config_path: str = _DEFAULT_CONFIG,
    force_refresh: bool | None = None,
) -> DocumentRegistry:
    """Run the full dataset pipeline and return a populated DocumentRegistry.

    Steps:
        1. Load ``dataset`` configuration from YAML.
        2. Run ``DatasetScanner`` to find all PDFs.
        3. Run ``DatasetLoader`` to extract / load cached metadata.
        4. Register all documents in a new ``DocumentRegistry``.
        5. Log each loaded document (filename, pages, language, source type).

    Args:
        config_path: Path to the YAML config file. Defaults to
            ``configs/dataset_builder.yaml``.
        force_refresh: Override the ``force_refresh`` setting from YAML.
            ``None`` means use the YAML value.

    Returns:
        ``DocumentRegistry`` with all discovered PDFs registered as PENDING.

    Raises:
        SystemExit: If no PDFs are found in the configured directory.
    """
    cfg = _load_dataset_config(config_path)

    root: str = cfg["root"]
    recursive: bool = bool(cfg.get("recursive", False))
    extensions: list[str] = cfg.get("extensions", ["pdf"])
    cache_dir: str = cfg["metadata_cache_dir"]
    _force_refresh: bool = (
        force_refresh if force_refresh is not None
        else bool(cfg.get("force_refresh", False))
    )
    text_sample_pages: int = int(cfg.get("text_sample_pages", 3))

    # ── Step 1: Scan ────────────────────────────────────────────────────
    print("\n------------------------------------------------")
    print("Scanning dataset...")

    scanner = DatasetScanner(root=root, recursive=recursive, extensions=extensions)
    pdf_paths = scanner.scan()

    if not pdf_paths:
        print(f"[ERROR] No PDFs found in '{root}'. Aborting.")
        logger.error("No PDFs found in '%s'. Aborting load_real_dataset.", root)
        sys.exit(1)

    print(f"{len(pdf_paths)} PDFs found")

    # ── Step 2: Load / extract metadata ─────────────────────────────────
    print("\nGenerating metadata...")

    loader = DatasetLoader(
        cache_dir=Path(cache_dir),
        text_sample_pages=text_sample_pages,
    )
    metadata_list: list[PDFMetadata] = loader.load(
        pdf_paths, force_refresh=_force_refresh
    )

    if not metadata_list:
        print("[ERROR] Metadata extraction returned 0 records. Aborting.")
        logger.error("DatasetLoader returned 0 records. Aborting.")
        sys.exit(1)

    cache_used = loader.is_cache_valid() and not _force_refresh
    print(f"Metadata loaded.{'  (from cache)' if cache_used else ''}")

    # ── Step 3: Log each document ────────────────────────────────────────
    logger.info("=== Dataset Load Summary (%d documents) ===", len(metadata_list))
    for meta in metadata_list:
        fname = Path(meta.file_path).name
        is_digital = "True" if meta.is_digital() else ("False" if meta.is_scanned() else "Unknown")
        lang = meta.language or "unknown"
        logger.info(
            "Loaded: %s | Pages: %d | Language: %s | Digital: %s",
            fname, meta.pages, lang, is_digital,
        )

    # ── Step 4: Build Document Registry ─────────────────────────────────
    print("\nBuilding Document Registry...")

    registry = DocumentRegistry(metadata_cached=cache_used)
    registry.register_batch(metadata_list)

    # Mark all as QUEUED (they are about to enter the scheduler)
    for doc_id in registry.all_ids():
        registry.set_status(doc_id, DocumentStatus.QUEUED)

    summary = registry.summary()
    print(
        f"Registry: {summary.total_pdfs} documents registered "
        f"({summary.digital_pdfs} digital, {summary.scanned_pdfs} scanned, "
        f"{summary.unknown_pdfs} unknown)"
    )
    print(f"         Total pages: {summary.total_pages}")
    print(f"         Avg pages:   {summary.avg_pages:.1f}")
    print(f"         Avg size:    {summary.avg_size_mb:.2f} MB")

    return registry
