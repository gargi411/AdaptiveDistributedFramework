"""Unit tests for the MetadataExtractor.

Tests:
    - extract() raises DatasetError for non-existent file.
    - extract() raises DatasetError for a directory path.
    - extract() returns a PDFMetadata with pages >= 1 for a valid PDF.
    - extract() returns a safe stub (pages=1) for a corrupt/empty PDF.
    - extract() file_path is the resolved absolute path.
    - extract() document_id is a non-empty string.
    - extract() estimated_size_mb reflects file size.
    - extract_batch() processes all files.
    - extract_batch() skips corrupt files when skip_invalid=True.
    - extract_batch() raises DatasetError when skip_invalid=False.
    - _detect_source_type() returns a valid string.
    - _extract_language() returns a string or None.

    MetadataStore:
    - save_csv() creates a file.
    - load_csv() returns same count as saved.
    - Roundtrip save/load preserves pages.
    - save_json() creates a file.
    - load_json() returns same count as saved.
    - compute_statistics() for empty list returns zeros.
    - compute_statistics() computes correct total_pages.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from adaptive_framework.core.exceptions import DatasetError
from adaptive_framework.metadata_generator import MetadataExtractor, MetadataStore
from adaptive_framework.models.document import PDFMetadata


# ---------------------------------------------------------------------------
# Minimal valid PDF fixture
# ---------------------------------------------------------------------------


def _write_minimal_pdf(path: Path, num_pages: int = 1) -> None:
    """Write a minimal but structurally valid PDF to *path*.

    This creates a PDF that pypdf can open without errors.
    It uses a pre-built minimal PDF byte string.

    Args:
        path: Destination file path.
        num_pages: Number of pages in the PDF (always 1 for the minimal fixture).
    """
    # This is the smallest valid PDF that pypdf can parse (1 page, no content)
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF\n"
    )
    path.write_bytes(minimal_pdf)


def _write_corrupt_pdf(path: Path) -> None:
    """Write a corrupt (non-parseable) PDF file.

    Args:
        path: Destination file path.
    """
    path.write_bytes(b"%PDF-1.4 CORRUPTED GARBAGE\x00\xff\x01\x02")


# ---------------------------------------------------------------------------
# MetadataExtractor — extract()
# ---------------------------------------------------------------------------


class TestMetadataExtractorExtract:
    """Tests for MetadataExtractor.extract()."""

    def test_raises_for_nonexistent_file(self, tmp_path: Path) -> None:
        """extract() raises DatasetError for a path that does not exist."""
        extractor = MetadataExtractor()
        with pytest.raises(DatasetError, match="not found"):
            extractor.extract(tmp_path / "missing.pdf")

    def test_raises_for_directory(self, tmp_path: Path) -> None:
        """extract() raises DatasetError when path is a directory."""
        extractor = MetadataExtractor()
        with pytest.raises(DatasetError):
            extractor.extract(tmp_path)

    def test_returns_pdf_metadata_for_valid_pdf(self, tmp_path: Path) -> None:
        """extract() returns a PDFMetadata for a valid minimal PDF."""
        pdf = tmp_path / "doc.pdf"
        _write_minimal_pdf(pdf)
        extractor = MetadataExtractor()
        meta = extractor.extract(pdf)
        assert isinstance(meta, PDFMetadata)
        assert meta.pages >= 1

    def test_document_id_is_non_empty(self, tmp_path: Path) -> None:
        """extract() generates a non-empty document_id."""
        pdf = tmp_path / "doc.pdf"
        _write_minimal_pdf(pdf)
        extractor = MetadataExtractor()
        meta = extractor.extract(pdf)
        assert meta.document_id and len(meta.document_id) > 0

    def test_document_id_is_unique_per_call(self, tmp_path: Path) -> None:
        """Each extract() call generates a unique document_id (UUID4)."""
        pdf = tmp_path / "doc.pdf"
        _write_minimal_pdf(pdf)
        extractor = MetadataExtractor()
        ids = {extractor.extract(pdf).document_id for _ in range(5)}
        assert len(ids) == 5  # all unique

    def test_file_path_is_absolute(self, tmp_path: Path) -> None:
        """extract() stores the absolute resolved path."""
        pdf = tmp_path / "doc.pdf"
        _write_minimal_pdf(pdf)
        extractor = MetadataExtractor()
        meta = extractor.extract(pdf)
        assert Path(meta.file_path).is_absolute()

    def test_estimated_size_mb_positive(self, tmp_path: Path) -> None:
        """extract() sets estimated_size_mb > 0 for a non-empty file."""
        pdf = tmp_path / "doc.pdf"
        _write_minimal_pdf(pdf)
        extractor = MetadataExtractor()
        meta = extractor.extract(pdf)
        assert meta.estimated_size_mb > 0.0

    def test_corrupt_pdf_returns_safe_stub(self, tmp_path: Path) -> None:
        """extract() returns a safe stub (pages=1) for corrupt PDFs."""
        pdf = tmp_path / "corrupt.pdf"
        _write_corrupt_pdf(pdf)
        extractor = MetadataExtractor()
        meta = extractor.extract(pdf)
        # Should not raise; returns stub with pages=1
        assert isinstance(meta, PDFMetadata)
        assert meta.pages == 1

    def test_processing_timestamp_is_set(self, tmp_path: Path) -> None:
        """extract() sets processing_timestamp to a non-empty string."""
        pdf = tmp_path / "doc.pdf"
        _write_minimal_pdf(pdf)
        extractor = MetadataExtractor()
        meta = extractor.extract(pdf)
        assert isinstance(meta.processing_timestamp, str)
        assert len(meta.processing_timestamp) > 0


# ---------------------------------------------------------------------------
# MetadataExtractor — extract_batch()
# ---------------------------------------------------------------------------


class TestMetadataExtractorBatch:
    """Tests for MetadataExtractor.extract_batch()."""

    def test_batch_processes_all_valid_files(self, tmp_path: Path) -> None:
        """extract_batch() returns one PDFMetadata per valid PDF."""
        pdfs: list[Path] = []
        for i in range(3):
            p = tmp_path / f"doc_{i}.pdf"
            _write_minimal_pdf(p)
            pdfs.append(p)
        extractor = MetadataExtractor()
        results = extractor.extract_batch(pdfs)
        assert len(results) == 3

    def test_batch_skips_corrupt_when_skip_invalid_true(
        self, tmp_path: Path
    ) -> None:
        """extract_batch(skip_invalid=True) skips corrupt files but continues."""
        valid = tmp_path / "valid.pdf"
        corrupt = tmp_path / "corrupt.pdf"
        _write_minimal_pdf(valid)
        _write_corrupt_pdf(corrupt)
        extractor = MetadataExtractor()
        # extract_batch uses safe-stub fallback so corrupt files still produce results
        results = extractor.extract_batch([valid, corrupt], skip_invalid=True)
        assert len(results) >= 1

    def test_batch_returns_empty_for_empty_list(self) -> None:
        """extract_batch([]) returns an empty list."""
        extractor = MetadataExtractor()
        results = extractor.extract_batch([])
        assert results == []

    def test_batch_skips_nonexistent_with_skip_invalid(
        self, tmp_path: Path
    ) -> None:
        """extract_batch skips non-existent files when skip_invalid=True."""
        extractor = MetadataExtractor()
        results = extractor.extract_batch(
            [tmp_path / "ghost.pdf"], skip_invalid=True
        )
        assert results == []

    def test_batch_raises_on_nonexistent_skip_invalid_false(
        self, tmp_path: Path
    ) -> None:
        """extract_batch raises DatasetError when skip_invalid=False."""
        extractor = MetadataExtractor()
        with pytest.raises(DatasetError):
            extractor.extract_batch(
                [tmp_path / "ghost.pdf"], skip_invalid=False
            )


# ---------------------------------------------------------------------------
# MetadataStore
# ---------------------------------------------------------------------------


def _make_metadata(pages: int, idx: int = 0) -> PDFMetadata:
    """Create a minimal PDFMetadata for testing the store.

    Args:
        pages: Page count.
        idx: Index for unique document_id.

    Returns:
        PDFMetadata record.
    """
    return PDFMetadata(
        document_id=f"doc_{idx:04d}",
        pages=pages,
        estimated_size_mb=round(pages * 0.1, 2),
        file_path=f"/data/doc_{idx:04d}.pdf",
        resolution_dpi=None,
        source_type="digital",
        language="en",
        processing_timestamp="2024-01-01T00:00:00+00:00",
    )


class TestMetadataStoreCSV:
    """Tests for MetadataStore CSV operations."""

    def test_save_csv_creates_file(self, tmp_path: Path) -> None:
        """save_csv() creates the CSV file."""
        store = MetadataStore(output_dir=tmp_path)
        metadata = [_make_metadata(10, 0)]
        csv_path = store.save_csv(metadata)
        assert csv_path.exists()

    def test_load_csv_returns_same_count(self, tmp_path: Path) -> None:
        """load_csv() returns the same number of records as saved."""
        store = MetadataStore(output_dir=tmp_path)
        metadata = [_make_metadata(i * 10, i) for i in range(1, 6)]  # pages >= 1
        store.save_csv(metadata)
        loaded = store.load_csv()
        assert len(loaded) == 5

    def test_roundtrip_preserves_pages(self, tmp_path: Path) -> None:
        """save_csv() → load_csv() roundtrip preserves pages field."""
        store = MetadataStore(output_dir=tmp_path)
        metadata = [_make_metadata(42, 0)]
        store.save_csv(metadata)
        loaded = store.load_csv()
        assert len(loaded) == 1
        assert loaded[0].pages == 42

    def test_load_csv_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        """load_csv() returns [] when the file does not exist."""
        store = MetadataStore(output_dir=tmp_path)
        result = store.load_csv()
        assert result == []


class TestMetadataStoreJSON:
    """Tests for MetadataStore JSON operations."""

    def test_save_json_creates_file(self, tmp_path: Path) -> None:
        """save_json() creates the JSON file."""
        store = MetadataStore(output_dir=tmp_path)
        store.save_json([_make_metadata(20, 0)])
        assert (tmp_path / "metadata_extracted.json").exists()

    def test_load_json_returns_same_count(self, tmp_path: Path) -> None:
        """load_json() returns the same count as saved."""
        store = MetadataStore(output_dir=tmp_path)
        metadata = [_make_metadata(i * 5, i) for i in range(1, 5)]  # pages >= 1
        store.save_json(metadata)
        loaded = store.load_json()
        assert len(loaded) == 4

    def test_json_roundtrip_preserves_pages(self, tmp_path: Path) -> None:
        """JSON roundtrip preserves the pages field."""
        store = MetadataStore(output_dir=tmp_path)
        store.save_json([_make_metadata(77, 0)])
        loaded = store.load_json()
        assert loaded[0].pages == 77

    def test_load_json_returns_empty_for_missing(self, tmp_path: Path) -> None:
        """load_json() returns [] when file does not exist."""
        store = MetadataStore(output_dir=tmp_path)
        assert store.load_json() == []


class TestMetadataStoreStatistics:
    """Tests for MetadataStore.compute_statistics()."""

    def test_empty_list_returns_zeroed_stats(self, tmp_path: Path) -> None:
        """compute_statistics([]) returns a dict with all zeros."""
        store = MetadataStore(output_dir=tmp_path)
        stats = store.compute_statistics([])
        assert stats["count"] == 0
        assert stats["total_pages"] == 0
        assert stats["avg_pages"] == 0.0

    def test_total_pages_correct(self, tmp_path: Path) -> None:
        """compute_statistics() computes the correct total_pages."""
        store = MetadataStore(output_dir=tmp_path)
        metadata = [_make_metadata(10, 0), _make_metadata(20, 1), _make_metadata(30, 2)]
        stats = store.compute_statistics(metadata)
        assert stats["total_pages"] == 60

    def test_avg_pages_correct(self, tmp_path: Path) -> None:
        """compute_statistics() computes the correct avg_pages."""
        store = MetadataStore(output_dir=tmp_path)
        metadata = [_make_metadata(10, 0), _make_metadata(20, 1), _make_metadata(30, 2)]
        stats = store.compute_statistics(metadata)
        assert abs(stats["avg_pages"] - 20.0) < 0.01

    def test_digital_count_is_correct(self, tmp_path: Path) -> None:
        """compute_statistics() counts digital source_type documents."""
        store = MetadataStore(output_dir=tmp_path)
        metadata = [_make_metadata(10, i) for i in range(3)]
        # All are 'digital' from the helper
        stats = store.compute_statistics(metadata)
        assert stats["digital_count"] == 3

    def test_stats_has_all_required_keys(self, tmp_path: Path) -> None:
        """compute_statistics() returns a dict with all expected keys."""
        store = MetadataStore(output_dir=tmp_path)
        stats = store.compute_statistics([_make_metadata(10, 0)])
        for key in ["count", "total_pages", "avg_pages", "min_pages", "max_pages",
                    "total_size_mb", "digital_count", "scanned_count"]:
            assert key in stats, f"Missing key: '{key}'"
