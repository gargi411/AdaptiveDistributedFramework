"""Concrete Dataset Builder — PubMed Central Open Access.

Implements the ``IDatasetBuilder`` interface for the Adaptive Distributed
Framework. Orchestrates:

    1. Keyword search via ``PubMedCentralClient``
    2. PDF downloads via ``PDFDownloader``
    3. metadata.csv generation
    4. Progress reporting (console progress bars via print-based iterator)

This module intentionally does NOT perform metadata extraction (page count,
DPI, source type). That responsibility belongs to the MetadataExtractor
(Module 2 / Phase 2A).

Dataset structure produced::

    dataset/
    └── raw/
        ├── PMC1234567.pdf
        ├── PMC7654321.pdf
        └── ...
    metadata.csv          ← article-level metadata (PubMed fields)

The ``metadata.csv`` stores PubMed-level fields. The full PDF-level
``PDFMetadata`` objects are built by the ``MetadataExtractor`` after download.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_framework.core.exceptions import DatasetError
from adaptive_framework.dataset_builder.download_stats import DownloadStats
from adaptive_framework.dataset_builder.downloader import PDFDownloader
from adaptive_framework.dataset_builder.pubmed_client import (
    ArticleRecord,
    PubMedCentralClient,
)
from adaptive_framework.interfaces.i_dataset_builder import IDatasetBuilder
from adaptive_framework.models.document import PDFMetadata

logger = logging.getLogger("adaptive_framework.dataset_builder")

# Default paths
_DEFAULT_DATASET_ROOT = Path("dataset")
_DEFAULT_RAW_SUBDIR = "raw"
_METADATA_CSV_FILENAME = "metadata.csv"

# Development-mode download cap (override with max_downloads in search config)
_DEV_MODE_MAX_DOWNLOADS: int = 10
_RESEARCH_MODE_MAX_DOWNLOADS: int = 500


@dataclass
class DatasetBuilderConfig:
    """Configuration for the PubMedDatasetBuilder.

    Attributes:
        output_dir: Root output directory (e.g. ``Path('dataset')``)
        keywords: Biomedical keyword list to query.
        max_downloads: Maximum PDFs to download per run.
        only_open_access: Restrict search to Open Access articles.
        max_retries: Download retry limit per article.
        dev_mode: If True, caps downloads at _DEV_MODE_MAX_DOWNLOADS.
        tool_name: Sent to NCBI to identify this application.
        email: Contact email sent to NCBI (required by ToS).

    Example:
        >>> cfg = DatasetBuilderConfig(
        ...     keywords=["cancer", "diabetes"],
        ...     max_downloads=50,
        ... )
    """

    keywords: list[str] = field(default_factory=lambda: ["cancer"])
    output_dir: Path = field(default_factory=lambda: _DEFAULT_DATASET_ROOT)
    max_downloads: int = 100
    only_open_access: bool = True
    max_retries: int = 3
    dev_mode: bool = False
    tool_name: str = "AdaptiveDistributedFramework"
    email: str = "adf@research.edu"

    def __post_init__(self) -> None:
        if self.dev_mode:
            self.max_downloads = min(self.max_downloads, _DEV_MODE_MAX_DOWNLOADS)


class PubMedDatasetBuilder(IDatasetBuilder):
    """Concrete implementation of IDatasetBuilder for PubMed Central OA.

    Orchestrates the full dataset acquisition pipeline:
        1. Run keyword search via E-utilities.
        2. Download PDFs via PDFDownloader (with resume / dedup).
        3. Write ``metadata.csv`` with article-level PubMed fields.
        4. Return a list of ``PDFMetadata`` stubs (page count = 0, filled
           later by MetadataExtractor).

    This implementation respects the interface contract:
        - ``build()`` → list[PDFMetadata]
        - ``validate()`` → bool
        - ``get_document_count()`` → int

    Attributes:
        _config: Dataset builder configuration.
        _client: PubMed Central E-utilities client.
        _downloader: PDF file downloader.

    Example:
        >>> cfg = DatasetBuilderConfig(keywords=["cancer"], max_downloads=20,
        ...                             dev_mode=True)
        >>> builder = PubMedDatasetBuilder(config=cfg)
        >>> metadata_list = builder.build(source_dir=Path("dataset/raw"))
    """

    def __init__(
        self,
        config: DatasetBuilderConfig | None = None,
        client: PubMedCentralClient | None = None,
    ) -> None:
        """Initialise the PubMedDatasetBuilder.

        Args:
            config: Dataset builder configuration. Uses safe defaults if None.
            client: Optional pre-configured PubMedCentralClient (for DI/testing).
        """
        self._config: DatasetBuilderConfig = config or DatasetBuilderConfig()
        self._raw_dir: Path = self._config.output_dir / _DEFAULT_RAW_SUBDIR
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        self._client: PubMedCentralClient = client or PubMedCentralClient(
            tool=self._config.tool_name,
            email=self._config.email,
        )
        self._downloader: PDFDownloader = PDFDownloader(
            output_dir=self._raw_dir,
            max_retries=self._config.max_retries,
            progress_callback=self._on_progress,
        )
        self._last_stats: DownloadStats | None = None

    # ------------------------------------------------------------------ #
    # IDatasetBuilder interface implementation                             #
    # ------------------------------------------------------------------ #

    def build(self, source_dir: Path | None = None) -> list[PDFMetadata]:  # type: ignore[override]
        """Download PDFs and build PDFMetadata stubs.

        Overrides the ``IDatasetBuilder.build()`` signature to make
        ``source_dir`` optional — when None, downloads from PubMed Central
        into the configured output directory.

        When ``source_dir`` is given (non-None), scans the directory for
        existing PDFs and builds metadata stubs without downloading.

        Args:
            source_dir: Optional path to an existing directory of PDFs.
                        If None, performs the full download pipeline.

        Returns:
            List of PDFMetadata stubs (pages=1 placeholder; updated by
            MetadataExtractor in the next pipeline stage).

        Raises:
            DatasetError: If the download pipeline encounters a fatal error.
        """
        if source_dir is not None:
            # Scan-only mode: build stubs from existing files
            return self._scan_existing(source_dir)

        # Full download mode
        logger.info(
            "Starting PubMed Central dataset build.",
            extra={
                "keywords": self._config.keywords,
                "max_downloads": self._config.max_downloads,
                "dev_mode": self._config.dev_mode,
            },
        )

        try:
            articles = self._search_all_keywords()
            stats = self._download_articles(articles)
            self._last_stats = stats
            metadata_list = self._scan_existing(self._raw_dir)

            # Write article-level metadata CSV
            self._write_metadata_csv(articles, metadata_list)

            logger.info(
                "Dataset build complete.",
                extra=stats.to_dict(),
            )
            self._print_summary(stats)
            return metadata_list

        except Exception as exc:
            raise DatasetError(
                f"Dataset build failed: {exc}"
            ) from exc

    def validate(self, source_dir: Path) -> bool:
        """Check whether source_dir contains at least one PDF.

        Args:
            source_dir: Directory to check.

        Returns:
            True if the directory exists and contains >= 1 PDF.
        """
        if not source_dir.is_dir():
            return False
        return any(source_dir.glob("*.pdf"))

    def get_document_count(self, source_dir: Path) -> int:
        """Count PDF files in source_dir.

        Args:
            source_dir: Directory to count PDFs in.

        Returns:
            Number of .pdf files (non-recursive).
        """
        if not source_dir.is_dir():
            return 0
        return sum(1 for _ in source_dir.glob("*.pdf"))

    def get_last_stats(self) -> DownloadStats | None:
        """Return statistics from the most recent download session.

        Returns:
            DownloadStats from the last ``build()`` call, or None if build()
            has not been called yet.
        """
        return self._last_stats

    # ------------------------------------------------------------------ #
    # Private: search                                                      #
    # ------------------------------------------------------------------ #

    def _search_all_keywords(self) -> list[ArticleRecord]:
        """Run the keyword search and deduplicate results.

        Returns:
            Deduplicated list of ArticleRecord objects, capped at
            ``config.max_downloads``.
        """
        seen_ids: set[str] = set()
        all_articles: list[ArticleRecord] = []
        per_keyword_limit = max(
            1, self._config.max_downloads // max(1, len(self._config.keywords))
        )

        for keyword in self._config.keywords:
            logger.info("Searching for keyword: '%s'", keyword)
            results = self._client.search(
                query=keyword,
                max_results=per_keyword_limit,
                only_open_access=self._config.only_open_access,
            )
            for rec in results:
                if rec.pmc_id not in seen_ids:
                    seen_ids.add(rec.pmc_id)
                    all_articles.append(rec)
                    if len(all_articles) >= self._config.max_downloads:
                        break
            if len(all_articles) >= self._config.max_downloads:
                break

        logger.info(
            "Search complete: %d unique articles found.",
            len(all_articles),
        )
        return all_articles[: self._config.max_downloads]

    # ------------------------------------------------------------------ #
    # Private: download                                                    #
    # ------------------------------------------------------------------ #

    def _download_articles(self, articles: list[ArticleRecord]) -> DownloadStats:
        """Download PDFs for the given article list.

        Iterates sequentially, updating a shared DownloadStats object.
        Prints a simple text progress bar to stdout.

        Args:
            articles: List of ArticleRecord objects to download.

        Returns:
            DownloadStats summarising the download session.
        """
        stats = DownloadStats(total_requested=len(articles))
        total = len(articles)

        for i, rec in enumerate(articles, start=1):
            self._print_progress(i, total, rec.pmc_id)
            self._downloader.download(
                url=rec.pdf_url,
                article_id=rec.pmc_id,
                stats=stats,
            )

        sys.stdout.write("\n")
        sys.stdout.flush()
        return stats

    # ------------------------------------------------------------------ #
    # Private: scan + metadata                                             #
    # ------------------------------------------------------------------ #

    def _scan_existing(self, source_dir: Path) -> list[PDFMetadata]:
        """Build PDFMetadata stubs from existing PDFs in source_dir.

        Page count and resolution_dpi are set to placeholder values (1 and None)
        because this builder does not open PDF files. The MetadataExtractor
        will update these fields in the next pipeline stage.

        Args:
            source_dir: Directory containing PDF files.

        Returns:
            List of PDFMetadata stubs (one per .pdf file in source_dir).
        """
        if not source_dir.is_dir():
            return []

        stubs: list[PDFMetadata] = []
        for pdf_path in sorted(source_dir.glob("*.pdf")):
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            stub = PDFMetadata(
                pages=1,  # Placeholder — updated by MetadataExtractor
                estimated_size_mb=round(size_mb, 4),
                file_path=str(pdf_path.resolve()),
                source_type=None,
                language=None,
            )
            stubs.append(stub)

        logger.info("Scanned %d PDF files in %s.", len(stubs), source_dir)
        return stubs

    def _write_metadata_csv(
        self,
        articles: list[ArticleRecord],
        stubs: list[PDFMetadata],
    ) -> None:
        """Write article-level metadata to metadata.csv.

        Merges PubMed-level article fields with the downloaded file info.

        Args:
            articles: List of ArticleRecord objects from PubMed search.
            stubs: Corresponding PDFMetadata stubs from the filesystem.
        """
        csv_path = self._config.output_dir / _METADATA_CSV_FILENAME
        fieldnames = [
            "pmc_id", "pmid", "title", "journal", "year",
            "pdf_url", "is_oa", "file_path", "estimated_size_mb",
        ]

        # Build pmc_id → stub lookup
        stub_by_id: dict[str, PDFMetadata] = {}
        for stub in stubs:
            fname = Path(stub.file_path).stem  # e.g. 'PMC1234567'
            stub_by_id[fname] = stub

        try:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in articles:
                    stub = stub_by_id.get(rec.pmc_id)
                    row: dict[str, Any] = rec.to_dict()
                    row["file_path"] = stub.file_path if stub else ""
                    row["estimated_size_mb"] = stub.estimated_size_mb if stub else 0.0
                    writer.writerow(row)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as exc:
            logger.warning("Could not write metadata CSV: %s", exc)

    # ------------------------------------------------------------------ #
    # Private: progress reporting                                          #
    # ------------------------------------------------------------------ #

    def _on_progress(
        self,
        article_id: str,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        """Progress callback invoked per download chunk.

        Prints an inline progress bar to stdout.

        Args:
            article_id: Current article being downloaded.
            downloaded_bytes: Bytes downloaded so far.
            total_bytes: Expected total bytes (0 if unknown).
        """
        if total_bytes > 0:
            pct = downloaded_bytes / total_bytes * 100
            bar_len = 30
            filled = int(bar_len * downloaded_bytes / total_bytes)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  {article_id[:20]:<20}"
            )
            sys.stdout.flush()

    def _print_progress(self, current: int, total: int, article_id: str) -> None:
        """Print a sequential download progress line.

        Args:
            current: Current download number (1-indexed).
            total: Total downloads in this session.
            article_id: PMC ID of the current article.
        """
        sys.stdout.write(f"\r  [{current:>4}/{total}] Downloading {article_id:<20}")
        sys.stdout.flush()

    def _print_summary(self, stats: DownloadStats) -> None:
        """Print a formatted download summary to stdout.

        Args:
            stats: Completed download session statistics.
        """
        print(f"""
╔══════════════════════════════════════════════════╗
║           Dataset Builder — Summary              ║
╠══════════════════════════════════════════════════╣
║  Requested   : {stats.total_requested:>6}                          ║
║  Downloaded  : {stats.total_downloaded:>6}                          ║
║  Skipped     : {stats.total_skipped:>6}                          ║
║  Failed      : {stats.total_failed:>6}                          ║
║  Success     : {stats.success_rate * 100:>5.1f}%                         ║
║  Total Data  : {stats.total_mb_downloaded:>6.1f} MB                        ║
║  Elapsed     : {stats.elapsed_seconds:>6.1f} s                         ║
║  Rate        : {stats.download_rate_mb_per_second:>6.2f} MB/s                      ║
╚══════════════════════════════════════════════════╝
""")
