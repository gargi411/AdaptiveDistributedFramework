"""tests/unit/test_dataset_scanner.py — Unit tests for DatasetScanner.

Uses the real PDFs already present in dataset/raw/pmc_pdfs/.
Never downloads anything. No network access required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Resolve project root so tests can import from src/
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_REAL_PDF_DIR = _PROJECT_ROOT / "dataset" / "raw" / "pmc_pdfs"

import sys
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from adaptive_framework.dataset_builder.dataset_scanner import DatasetScanner


# =============================================================
# Fixtures
# =============================================================


@pytest.fixture
def real_pdf_dir() -> Path:
    """Return the path to the real biomedical PDF directory.

    Skips all tests in this module if the directory doesn't exist or is empty.
    """
    if not _REAL_PDF_DIR.exists() or not _REAL_PDF_DIR.is_dir():
        pytest.skip(f"Real PDF directory not found: {_REAL_PDF_DIR}")
    pdfs = list(_REAL_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"No PDFs found in: {_REAL_PDF_DIR}")
    return _REAL_PDF_DIR


@pytest.fixture
def tmp_pdf_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with dummy PDF, TXT, and PNG files."""
    # Create dummy PDFs (content doesn't matter for scanner tests)
    (tmp_path / "paper_a.pdf").write_bytes(b"%PDF-1.4 dummy-content-a")
    (tmp_path / "paper_b.pdf").write_bytes(b"%PDF-1.4 dummy-content-b")
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / "figure.png").write_bytes(b"\x89PNG dummy")
    (tmp_path / "empty.pdf").write_bytes(b"")  # zero-byte — should be excluded
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Return an empty temporary directory."""
    empty = tmp_path / "empty_scan_dir"
    empty.mkdir()
    return empty


# =============================================================
# Tests — real PDF directory
# =============================================================


class TestDatasetScannerReal:
    """Tests that operate on the real dataset/raw/pmc_pdfs/ directory."""

    def test_scan_returns_20_pdfs(self, real_pdf_dir: Path) -> None:
        """Exactly 20 PDFs should be present in the real dataset directory."""
        scanner = DatasetScanner(root=real_pdf_dir)
        paths = scanner.scan()
        assert len(paths) == 20, (
            f"Expected 20 PDFs, got {len(paths)}. "
            f"Paths: {[p.name for p in paths]}"
        )

    def test_all_returned_paths_are_pdf(self, real_pdf_dir: Path) -> None:
        """Every path in the result must have a .pdf extension."""
        scanner = DatasetScanner(root=real_pdf_dir)
        paths = scanner.scan()
        non_pdf = [p for p in paths if p.suffix.lower() != ".pdf"]
        assert non_pdf == [], f"Non-PDF paths returned: {non_pdf}"

    def test_all_paths_exist_and_are_files(self, real_pdf_dir: Path) -> None:
        """Every returned path must point to an existing, non-empty file."""
        scanner = DatasetScanner(root=real_pdf_dir)
        paths = scanner.scan()
        for p in paths:
            assert p.exists(), f"Path does not exist: {p}"
            assert p.is_file(), f"Path is not a file: {p}"
            assert p.stat().st_size > 0, f"Zero-byte file: {p}"

    def test_paths_are_sorted_alphabetically(self, real_pdf_dir: Path) -> None:
        """Returned paths must be in ascending alphabetical order."""
        scanner = DatasetScanner(root=real_pdf_dir)
        paths = scanner.scan()
        assert paths == sorted(paths), "Paths are not sorted alphabetically."

    def test_all_paths_are_absolute(self, real_pdf_dir: Path) -> None:
        """All returned paths must be absolute (resolved)."""
        scanner = DatasetScanner(root=real_pdf_dir)
        paths = scanner.scan()
        for p in paths:
            assert p.is_absolute(), f"Path is not absolute: {p}"

    def test_root_property_returns_resolved_path(self, real_pdf_dir: Path) -> None:
        """``scanner.root`` should return the resolved absolute path."""
        scanner = DatasetScanner(root=real_pdf_dir)
        assert scanner.root.is_absolute()
        assert scanner.root.exists()


# =============================================================
# Tests — synthetic/temporary directories
# =============================================================


class TestDatasetScannerSynthetic:
    """Tests using temporary directories for edge-case coverage."""

    def test_excludes_non_pdf_files(self, tmp_pdf_dir: Path) -> None:
        """Non-PDF files (.txt, .png) must not appear in the result."""
        scanner = DatasetScanner(root=tmp_pdf_dir)
        paths = scanner.scan()
        names = {p.name for p in paths}
        assert "notes.txt" not in names
        assert "figure.png" not in names

    def test_excludes_zero_byte_pdfs(self, tmp_pdf_dir: Path) -> None:
        """Zero-byte files must be excluded even if they have .pdf extension."""
        scanner = DatasetScanner(root=tmp_pdf_dir)
        paths = scanner.scan()
        names = {p.name for p in paths}
        assert "empty.pdf" not in names

    def test_includes_valid_pdfs(self, tmp_pdf_dir: Path) -> None:
        """Valid PDFs (non-empty .pdf files) must be included."""
        scanner = DatasetScanner(root=tmp_pdf_dir)
        paths = scanner.scan()
        names = {p.name for p in paths}
        assert "paper_a.pdf" in names
        assert "paper_b.pdf" in names

    def test_empty_directory_returns_empty_list(self, empty_dir: Path) -> None:
        """Scanning an empty directory returns an empty list (no exception)."""
        scanner = DatasetScanner(root=empty_dir)
        paths = scanner.scan()
        assert paths == []

    def test_nonexistent_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Scanning a directory that does not exist returns an empty list."""
        scanner = DatasetScanner(root=tmp_path / "does_not_exist")
        paths = scanner.scan()
        assert paths == []

    def test_custom_extension_filter(self, tmp_path: Path) -> None:
        """Scanner respects the ``extensions`` parameter."""
        (tmp_path / "data.csv").write_text("col1,col2")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "paper.pdf").write_bytes(b"%PDF content")
        scanner = DatasetScanner(root=tmp_path, extensions=["csv"])
        paths = scanner.scan()
        names = {p.name for p in paths}
        assert "data.csv" in names
        assert "paper.pdf" not in names

    def test_extension_case_insensitive(self, tmp_path: Path) -> None:
        """File extension comparison must be case-insensitive."""
        (tmp_path / "paper.PDF").write_bytes(b"%PDF content uppercase ext")
        scanner = DatasetScanner(root=tmp_path, extensions=["pdf"])
        paths = scanner.scan()
        assert len(paths) == 1

    def test_recursive_false_does_not_scan_subdirs(self, tmp_path: Path) -> None:
        """Non-recursive scan must ignore PDFs in subdirectories."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "top.pdf").write_bytes(b"%PDF top")
        (subdir / "nested.pdf").write_bytes(b"%PDF nested")
        scanner = DatasetScanner(root=tmp_path, recursive=False)
        paths = scanner.scan()
        names = {p.name for p in paths}
        assert "top.pdf" in names
        assert "nested.pdf" not in names

    def test_recursive_true_scans_subdirs(self, tmp_path: Path) -> None:
        """Recursive scan must include PDFs in sub-directories."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "top.pdf").write_bytes(b"%PDF top")
        (subdir / "nested.pdf").write_bytes(b"%PDF nested")
        scanner = DatasetScanner(root=tmp_path, recursive=True)
        paths = scanner.scan()
        names = {p.name for p in paths}
        assert "top.pdf" in names
        assert "nested.pdf" in names
