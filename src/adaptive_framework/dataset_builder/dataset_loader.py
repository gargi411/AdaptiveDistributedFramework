"""Dataset Loader — Phase 3.5.

Bridges the ``DatasetScanner`` (filesystem paths) and the ``DocumentRegistry``
(pipeline state). Uses the existing ``MetadataExtractor`` and ``MetadataStore``
for extraction and caching respectively.

Pipeline:
    list[Path]  (from DatasetScanner)
        ↓
    MetadataStore.load_json()  ← cache hit → skip extraction
        ↓ cache miss
    MetadataExtractor.extract_batch()
        ↓
    MetadataStore.save_csv() + save_json()  ← write cache
        ↓
    list[PDFMetadata]  (consumed by DocumentRegistry)

Design:
    - ``DatasetLoader`` is stateless; all caching is delegated to
      ``MetadataStore``.
    - ``force_refresh=True`` bypasses the cache and re-extracts all metadata.
    - No OCR. No page processing. Extraction reads only PDF headers/structure.
    - Never raises on individual file failures; logs warnings and continues.

Usage::

    from pathlib import Path
    from adaptive_framework.dataset_builder.dataset_loader import DatasetLoader

    loader = DatasetLoader(cache_dir=Path("dataset/metadata"))
    metadata_list = loader.load(paths, force_refresh=False)
"""

from __future__ import annotations

import logging
from pathlib import Path

from adaptive_framework.metadata_generator.metadata_extractor import MetadataExtractor
from adaptive_framework.metadata_generator.metadata_store import MetadataStore
from adaptive_framework.models.document import PDFMetadata

logger = logging.getLogger("adaptive_framework.dataset_builder.loader")


class DatasetLoader:
    """Load and cache ``PDFMetadata`` for a list of PDF paths.

    Wraps ``MetadataExtractor`` (extraction) and ``MetadataStore`` (caching)
    to provide an idempotent, cache-aware metadata loading step.

    Attributes:
        _cache_dir: Directory for ``metadata_extracted.csv`` / ``.json``.
        _extractor: ``MetadataExtractor`` instance for PDF introspection.
        _store: ``MetadataStore`` instance for cache I/O.

    Example:
        >>> loader = DatasetLoader(cache_dir=Path("dataset/metadata"))
        >>> metadata = loader.load([Path("dataset/raw/pmc_pdfs/paper.pdf")])
        >>> metadata[0].pages
        12
    """

    def __init__(
        self,
        cache_dir: Path | str = Path("dataset/metadata"),
        text_sample_pages: int = 3,
    ) -> None:
        """Initialise the DatasetLoader.

        Args:
            cache_dir: Directory for metadata cache files. Created automatically
                if it does not exist.
            text_sample_pages: Pages to sample per document when detecting
                whether it is digital or scanned. Forwarded to ``MetadataExtractor``.
        """
        self._cache_dir = Path(cache_dir)
        self._extractor = MetadataExtractor(text_sample_pages=text_sample_pages)
        self._store = MetadataStore(output_dir=self._cache_dir)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(
        self,
        paths: list[Path],
        force_refresh: bool = False,
    ) -> list[PDFMetadata]:
        """Load ``PDFMetadata`` for the given PDF paths.

        On the first call (or when ``force_refresh=True``), runs
        ``MetadataExtractor.extract_batch()`` on all paths and writes the
        result to the cache. On subsequent calls, loads from the JSON cache
        without re-opening any PDF files.

        The returned list preserves the same order as ``paths``.

        Args:
            paths: PDF file paths to load metadata for.
            force_refresh: If True, ignore any existing cache and re-extract
                all metadata from the source PDF files.

        Returns:
            List of ``PDFMetadata`` records, one per successfully loaded PDF.
            Files that cannot be opened are skipped (logged as warnings).
        """
        if not paths:
            logger.warning("DatasetLoader.load() called with an empty path list.")
            return []

        logger.info(
            "DatasetLoader: %d path(s) to load. force_refresh=%s",
            len(paths),
            force_refresh,
        )

        # ── Cache hit path ──────────────────────────────────────────────
        if not force_refresh:
            cached = self._try_load_cache(paths)
            if cached is not None:
                logger.info(
                    "Metadata cache hit: %d records loaded from cache.", len(cached)
                )
                return cached

        # ── Cache miss / force refresh ──────────────────────────────────
        logger.info("Extracting metadata from %d PDF file(s)...", len(paths))
        metadata_list = self._extractor.extract_batch(paths, skip_invalid=True)

        if not metadata_list:
            logger.warning(
                "MetadataExtractor returned 0 records for %d input paths.", len(paths)
            )
            return []

        # ── Persist cache ───────────────────────────────────────────────
        csv_path = self._store.save_csv(metadata_list)
        json_path = self._store.save_json(metadata_list)
        logger.info(
            "Metadata cached: CSV='%s', JSON='%s'", csv_path.name, json_path.name
        )

        return metadata_list

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path.

        Returns:
            Absolute path to the metadata cache directory.
        """
        return self._cache_dir

    def is_cache_valid(self) -> bool:
        """Check whether a valid metadata cache exists.

        Returns:
            True if the JSON cache file exists and is non-empty.
        """
        json_path = self._cache_dir / "metadata_extracted.json"
        return json_path.exists() and json_path.stat().st_size > 0

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _try_load_cache(self, paths: list[Path]) -> list[PDFMetadata] | None:
        """Attempt to load metadata from the JSON cache.

        Validates that the cache covers the same set of file paths as the
        current scan. Returns None if the cache is absent or stale.

        Args:
            paths: Current set of PDF paths from DatasetScanner.

        Returns:
            List of ``PDFMetadata`` if the cache is valid and covers all paths,
            or None if cache is missing or stale.
        """
        cached_records = self._store.load_json()
        if not cached_records:
            logger.debug("No existing metadata cache found.")
            return None

        # Validate cache coverage: every scanned path must appear in the cache
        cached_paths = {Path(m.file_path).resolve() for m in cached_records}
        current_paths = {p.resolve() for p in paths}

        missing = current_paths - cached_paths
        if missing:
            logger.info(
                "Cache is stale: %d new path(s) not in cache. Re-extracting.",
                len(missing),
            )
            return None

        # Return only the records for the current set of paths (handles
        # the case where cache has MORE entries than the current scan)
        path_set = {str(p.resolve()) for p in paths}
        filtered = [m for m in cached_records if m.file_path in path_set]
        return filtered if filtered else None
