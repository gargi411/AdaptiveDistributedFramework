"""Dataset Builder package — PubMed Central Open Access + Phase 3.5 Local Scanner.

Public API:
    PubMedDatasetBuilder  — Concrete IDatasetBuilder implementation.
    DatasetBuilderConfig  — Configuration dataclass.
    PDFDownloader         — Low-level PDF downloader.
    DownloadStats         — Session download statistics.
    PubMedCentralClient   — NCBI E-utilities HTTP client.
    ArticleRecord         — PubMed article metadata.

    ── Phase 3.5 additions ──────────────────────────────────────────────
    DatasetScanner        — Scan a directory for readable PDF files.
    DatasetLoader         — Cache-aware metadata extraction pipeline.
    DocumentRegistry      — Single source of truth for pipeline state.
    DocumentStatus        — Lifecycle status enum (PENDING → COMPLETED).
    RegistrySummary       — Aggregate statistics for the dashboard.

Usage::

    from adaptive_framework.dataset_builder import (
        DatasetScanner,
        DatasetLoader,
        DocumentRegistry,
        DocumentStatus,
    )

    scanner = DatasetScanner(root="dataset/raw/pmc_pdfs")
    paths = scanner.scan()

    loader = DatasetLoader(cache_dir="dataset/metadata")
    metadata_list = loader.load(paths)

    registry = DocumentRegistry(metadata_cached=loader.is_cache_valid())
    registry.register_batch(metadata_list)
"""

from adaptive_framework.dataset_builder.dataset_builder import (
    DatasetBuilderConfig,
    PubMedDatasetBuilder,
)
from adaptive_framework.dataset_builder.dataset_loader import DatasetLoader
from adaptive_framework.dataset_builder.dataset_scanner import DatasetScanner
from adaptive_framework.dataset_builder.document_registry import (
    DocumentRegistry,
    DocumentStatus,
    RegistrySummary,
)
from adaptive_framework.dataset_builder.download_stats import DownloadStats
from adaptive_framework.dataset_builder.downloader import PDFDownloader
from adaptive_framework.dataset_builder.pubmed_client import (
    ArticleRecord,
    PubMedCentralClient,
)

__all__ = [
    # Phase 2A — PubMed downloader
    "PubMedDatasetBuilder",
    "DatasetBuilderConfig",
    "PDFDownloader",
    "DownloadStats",
    "PubMedCentralClient",
    "ArticleRecord",
    # Phase 3.5 — Local dataset integration
    "DatasetScanner",
    "DatasetLoader",
    "DocumentRegistry",
    "DocumentStatus",
    "RegistrySummary",
]