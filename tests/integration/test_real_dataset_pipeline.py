"""tests/integration/test_real_dataset_pipeline.py — Integration tests.

End-to-end pipeline: DatasetScanner → DatasetLoader → DocumentRegistry
→ PageCountPartitioner → PageWorkUnits.

Uses the real 20 PDFs from dataset/raw/pmc_pdfs/.
No downloads. No network. No Ray.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ── Resolve project root ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_REAL_PDF_DIR = _PROJECT_ROOT / "dataset" / "raw" / "pmc_pdfs"

import sys
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from adaptive_framework.dataset_builder.dataset_loader import DatasetLoader
from adaptive_framework.dataset_builder.dataset_scanner import DatasetScanner
from adaptive_framework.dataset_builder.document_registry import (
    DocumentRegistry,
    DocumentStatus,
)
from adaptive_framework.scheduler.page_count_partitioner import PageCountPartitioner


# =============================================================
# Fixtures
# =============================================================


@pytest.fixture(scope="module")
def real_pipeline(tmp_path_factory: pytest.TempPathFactory):
    """Run the full pipeline once for the entire test module.

    Returns:
        Tuple of (registry, partitions, stats) for the 20 real PDFs.
    """
    if not _REAL_PDF_DIR.exists():
        pytest.skip(f"Real PDF directory not found: {_REAL_PDF_DIR}")

    cache_dir = tmp_path_factory.mktemp("metadata_cache")

    # Step 1: Scan
    scanner = DatasetScanner(root=_REAL_PDF_DIR)
    paths = scanner.scan()
    if not paths:
        pytest.skip("No PDFs found in real dataset directory.")

    # Step 2: Load metadata
    loader = DatasetLoader(cache_dir=cache_dir, text_sample_pages=1)
    metadata_list = loader.load(paths, force_refresh=True)

    # Step 3: Register
    registry = DocumentRegistry(metadata_cached=False)
    registry.register_batch(metadata_list)

    # Step 4: Partition
    partitioner = PageCountPartitioner()
    partitions, stats = partitioner.partition(metadata_list, num_workers=4)

    return registry, partitions, stats


# =============================================================
# Pipeline shape tests
# =============================================================


class TestRealPipelineShape:
    """Verify the structural correctness of the full pipeline output."""

    def test_scanner_finds_20_pdfs(self) -> None:
        """DatasetScanner must find exactly 20 PDFs."""
        if not _REAL_PDF_DIR.exists():
            pytest.skip(f"Real PDF directory not found: {_REAL_PDF_DIR}")
        scanner = DatasetScanner(root=_REAL_PDF_DIR)
        paths = scanner.scan()
        assert len(paths) == 20

    def test_registry_has_20_documents(self, real_pipeline) -> None:
        """DocumentRegistry must contain exactly 20 documents."""
        registry, _, _ = real_pipeline
        assert len(registry) == 20

    def test_all_registry_documents_initially_pending(self) -> None:
        """All documents start as PENDING before any scheduling."""
        if not _REAL_PDF_DIR.exists():
            pytest.skip(f"Real PDF directory not found: {_REAL_PDF_DIR}")
        cache_dir = Path(pytest.tmp_path_factory if hasattr(pytest, "tmp_path_factory") else "/tmp/adf_test_cache")

        scanner = DatasetScanner(root=_REAL_PDF_DIR)
        paths = scanner.scan()
        loader = DatasetLoader(text_sample_pages=1)
        metadata_list = loader.load(paths, force_refresh=True)

        fresh_registry = DocumentRegistry()
        ids = fresh_registry.register_batch(metadata_list)
        for doc_id in ids:
            assert fresh_registry.get_status(doc_id) == DocumentStatus.PENDING

    def test_partition_count_is_4(self, real_pipeline) -> None:
        """Partitioner with 4 workers must create at most 4 partitions."""
        _, partitions, _ = real_pipeline
        assert 1 <= len(partitions) <= 4

    def test_partition_total_pages_equals_registry_total(
        self, real_pipeline
    ) -> None:
        """Sum of pages across all partitions must equal registry total pages."""
        registry, partitions, _ = real_pipeline
        expected_pages = registry.summary().total_pages
        actual_pages = sum(p.total_pages for p in partitions)
        assert actual_pages == expected_pages, (
            f"Partition total ({actual_pages}) ≠ registry total ({expected_pages})"
        )

    def test_all_work_units_have_valid_file_paths(self, real_pipeline) -> None:
        """Every PageWorkUnit must reference an existing PDF file."""
        _, partitions, _ = real_pipeline
        for partition in partitions:
            for wu in partition.work_units:
                path = Path(wu.file_path)
                assert path.exists(), f"Work unit references missing file: {wu.file_path}"
                assert path.suffix.lower() == ".pdf"

    def test_all_work_units_page_ranges_valid(self, real_pipeline) -> None:
        """All work units must have start_page <= end_page and start_page >= 1."""
        _, partitions, _ = real_pipeline
        for partition in partitions:
            for wu in partition.work_units:
                assert wu.start_page >= 1, f"start_page < 1: {wu}"
                assert wu.end_page >= wu.start_page, f"end_page < start_page: {wu}"

    def test_no_duplicate_work_unit_ids(self, real_pipeline) -> None:
        """All work unit IDs across all partitions must be unique."""
        _, partitions, _ = real_pipeline
        all_wu_ids = [wu.work_unit_id for p in partitions for wu in p.work_units]
        assert len(all_wu_ids) == len(set(all_wu_ids)), "Duplicate work unit IDs."

    def test_total_work_units_equals_document_count(self, real_pipeline) -> None:
        """Total work units must equal total registered documents (1 WU per doc)."""
        registry, partitions, _ = real_pipeline
        total_wus = sum(p.total_work_units for p in partitions)
        assert total_wus == len(registry)


# =============================================================
# Statistics tests
# =============================================================


class TestRealPipelineStatistics:
    """Verify PartitionStatistics and RegistrySummary correctness."""

    def test_partition_stats_total_pages_consistent(self, real_pipeline) -> None:
        """PartitionStatistics.total_pages must match sum of partition pages."""
        _, partitions, stats = real_pipeline
        assert stats.total_pages == sum(p.total_pages for p in partitions)

    def test_partition_stats_total_partitions(self, real_pipeline) -> None:
        """PartitionStatistics.total_partitions must match len(partitions)."""
        _, partitions, stats = real_pipeline
        assert stats.total_partitions == len(partitions)

    def test_partition_stats_std_is_non_negative(self, real_pipeline) -> None:
        """Standard deviation of pages per partition must be non-negative."""
        _, _, stats = real_pipeline
        assert stats.std_pages_per_partition >= 0.0

    def test_registry_summary_digital_plus_scanned_plus_unknown_eq_total(
        self, real_pipeline
    ) -> None:
        """digital + scanned + unknown must equal total_pdfs."""
        registry, _, _ = real_pipeline
        s = registry.summary()
        assert s.digital_pdfs + s.scanned_pdfs + s.unknown_pdfs == s.total_pdfs

    def test_registry_summary_all_pages_positive(self, real_pipeline) -> None:
        """total_pages and avg_pages must be positive for a real dataset."""
        registry, _, _ = real_pipeline
        s = registry.summary()
        assert s.total_pages > 0
        assert s.avg_pages > 0.0

    def test_registry_summary_to_dict_is_json_serializable(
        self, real_pipeline
    ) -> None:
        """RegistrySummary.to_dict() must produce JSON-serializable output."""
        import json
        registry, _, _ = real_pipeline
        d = registry.summary().to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0


# =============================================================
# DocumentRegistry Phase 4 entry point
# =============================================================


class TestRegistryPhase4EntryPoint:
    """Verify the Phase 4 contract: get_unified_document()."""

    def test_unified_document_is_none_before_processing(
        self, real_pipeline
    ) -> None:
        """Before Phase 3 processing, all unified_document slots must be None."""
        registry, _, _ = real_pipeline
        for doc_id in registry.all_ids():
            assert registry.get_unified_document(doc_id) is None

    def test_set_and_get_unified_document(self, real_pipeline) -> None:
        """set_unified_document() → get_unified_document() round-trip."""
        registry, _, _ = real_pipeline
        first_id = registry.all_ids()[0]
        fake_unified_doc = {"source": "test", "pages": []}
        registry.set_unified_document(first_id, fake_unified_doc)
        retrieved = registry.get_unified_document(first_id)
        assert retrieved is fake_unified_doc
        assert registry.get_status(first_id) == DocumentStatus.COMPLETED
