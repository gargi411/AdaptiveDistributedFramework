"""tests/unit/test_dataset_loader.py — Unit tests for DatasetLoader.

Uses real PDFs from dataset/raw/pmc_pdfs/ for integration-level checks,
and temporary directories + mocks for cache logic tests.

Never downloads anything. No network access required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Resolve project root ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_REAL_PDF_DIR = _PROJECT_ROOT / "dataset" / "raw" / "pmc_pdfs"

import sys
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from adaptive_framework.dataset_builder.dataset_loader import DatasetLoader
from adaptive_framework.dataset_builder.dataset_scanner import DatasetScanner
from adaptive_framework.models.document import PDFMetadata


# =============================================================
# Helpers
# =============================================================


def _make_metadata(
    file_path: str = "/fake/paper.pdf",
    pages: int = 10,
    doc_id: str = "test-doc-id",
) -> PDFMetadata:
    """Build a minimal PDFMetadata for testing."""
    return PDFMetadata(
        document_id=doc_id,
        pages=pages,
        estimated_size_mb=0.5,
        file_path=file_path,
        source_type="digital",
        language="en",
    )


# =============================================================
# Fixtures
# =============================================================


@pytest.fixture
def real_pdf_paths() -> list[Path]:
    """Return paths from the real dataset directory. Skip if unavailable."""
    if not _REAL_PDF_DIR.exists():
        pytest.skip(f"Real PDF directory not found: {_REAL_PDF_DIR}")
    scanner = DatasetScanner(root=_REAL_PDF_DIR)
    paths = scanner.scan()
    if not paths:
        pytest.skip("No PDFs found in real dataset directory.")
    return paths


@pytest.fixture
def loader_with_tmp_cache(tmp_path: Path) -> DatasetLoader:
    """DatasetLoader with a fresh temporary cache directory."""
    return DatasetLoader(cache_dir=tmp_path / "metadata")


# =============================================================
# Tests — real PDFs
# =============================================================


class TestDatasetLoaderReal:
    """Tests that operate on the real biomedical PDFs."""

    def test_load_returns_20_records(
        self, real_pdf_paths: list[Path], tmp_path: Path
    ) -> None:
        """Loading all 20 real PDFs should produce 20 metadata records."""
        loader = DatasetLoader(
            cache_dir=tmp_path / "meta",
            text_sample_pages=1,  # Fast extraction
        )
        records = loader.load(real_pdf_paths, force_refresh=True)
        assert len(records) == 20, (
            f"Expected 20 records, got {len(records)}"
        )

    def test_all_records_have_pages_ge_1(
        self, real_pdf_paths: list[Path], tmp_path: Path
    ) -> None:
        """Every extracted metadata record must have pages >= 1."""
        loader = DatasetLoader(
            cache_dir=tmp_path / "meta",
            text_sample_pages=1,
        )
        records = loader.load(real_pdf_paths, force_refresh=True)
        for rec in records:
            assert rec.pages >= 1, (
                f"Record '{Path(rec.file_path).name}' has pages={rec.pages}"
            )

    def test_all_records_have_non_empty_file_path(
        self, real_pdf_paths: list[Path], tmp_path: Path
    ) -> None:
        """Every record must have a non-empty file_path."""
        loader = DatasetLoader(cache_dir=tmp_path / "meta", text_sample_pages=1)
        records = loader.load(real_pdf_paths, force_refresh=True)
        for rec in records:
            assert rec.file_path, f"Empty file_path in record: {rec}"

    def test_cache_written_after_extraction(
        self, real_pdf_paths: list[Path], tmp_path: Path
    ) -> None:
        """After force_refresh extraction, CSV and JSON cache files must exist."""
        cache_dir = tmp_path / "meta"
        loader = DatasetLoader(cache_dir=cache_dir, text_sample_pages=1)
        loader.load(real_pdf_paths, force_refresh=True)
        assert (cache_dir / "metadata_extracted.csv").exists()
        assert (cache_dir / "metadata_extracted.json").exists()

    def test_cache_hit_avoids_re_extraction(
        self, real_pdf_paths: list[Path], tmp_path: Path
    ) -> None:
        """Second load call (force_refresh=False) must hit the cache."""
        cache_dir = tmp_path / "meta"
        loader = DatasetLoader(cache_dir=cache_dir, text_sample_pages=1)

        # First call: extract and cache
        first_records = loader.load(real_pdf_paths, force_refresh=True)
        assert len(first_records) > 0

        # Second call: should load from cache without calling MetadataExtractor
        with patch.object(
            loader._extractor, "extract_batch", wraps=loader._extractor.extract_batch
        ) as mock_extract:
            cached_records = loader.load(real_pdf_paths, force_refresh=False)
            mock_extract.assert_not_called(), (
                "extract_batch was called on cache hit — cache logic is broken."
            )

        assert len(cached_records) == len(first_records)

    def test_force_refresh_bypasses_cache(
        self, real_pdf_paths: list[Path], tmp_path: Path
    ) -> None:
        """force_refresh=True must call extract_batch even when cache exists."""
        cache_dir = tmp_path / "meta"
        loader = DatasetLoader(cache_dir=cache_dir, text_sample_pages=1)

        # Create a cache first
        loader.load(real_pdf_paths, force_refresh=True)

        # Second call with force_refresh=True must re-extract
        with patch.object(
            loader._extractor, "extract_batch", wraps=loader._extractor.extract_batch
        ) as mock_extract:
            loader.load(real_pdf_paths, force_refresh=True)
            mock_extract.assert_called_once()

    def test_is_cache_valid_after_extraction(
        self, real_pdf_paths: list[Path], tmp_path: Path
    ) -> None:
        """``is_cache_valid()`` must return True after a successful extraction."""
        loader = DatasetLoader(cache_dir=tmp_path / "meta", text_sample_pages=1)
        assert not loader.is_cache_valid(), "Cache should not exist before first load."
        loader.load(real_pdf_paths, force_refresh=True)
        assert loader.is_cache_valid(), "Cache should exist after extraction."


# =============================================================
# Tests — edge cases (no real PDFs needed)
# =============================================================


class TestDatasetLoaderEdgeCases:
    """Tests for edge cases that do not require real PDF files."""

    def test_empty_paths_returns_empty_list(
        self, loader_with_tmp_cache: DatasetLoader
    ) -> None:
        """Calling load() with an empty list must return an empty list."""
        result = loader_with_tmp_cache.load([], force_refresh=False)
        assert result == []

    def test_stale_cache_triggers_re_extraction(self, tmp_path: Path) -> None:
        """Cache with fewer paths than current scan must trigger re-extraction."""
        cache_dir = tmp_path / "meta"
        cache_dir.mkdir(parents=True)

        # Pre-populate cache with ONE document
        one_meta = _make_metadata(file_path="/some/old/path.pdf", pages=5)
        json_path = cache_dir / "metadata_extracted.json"
        json_path.write_text(json.dumps([one_meta.to_dict()]), encoding="utf-8")

        # Loader should detect that current paths (two) differ from cache (one)
        loader = DatasetLoader(cache_dir=cache_dir)
        current_paths = [
            Path("/some/old/path.pdf"),
            Path("/new/doc.pdf"),
        ]

        with patch.object(
            loader._extractor,
            "extract_batch",
            return_value=[one_meta],
        ) as mock_extract:
            loader.load(current_paths, force_refresh=False)
            mock_extract.assert_called_once(), (
                "Stale cache should trigger re-extraction."
            )

    def test_cache_dir_property(self, tmp_path: Path) -> None:
        """``cache_dir`` property must return the configured directory."""
        loader = DatasetLoader(cache_dir=tmp_path / "mycache")
        assert loader.cache_dir == tmp_path / "mycache"
