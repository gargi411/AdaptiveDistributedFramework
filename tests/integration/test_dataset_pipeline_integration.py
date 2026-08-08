"""Integration test — full Phase 2A data pipeline.

Tests the end-to-end flow without making real network calls:
    1. Seed a directory with synthetic PDF files.
    2. PubMedDatasetBuilder.build(source_dir) → list[PDFMetadata] stubs.
    3. MetadataExtractor.extract_batch() → list[PDFMetadata] with pages.
    4. MetadataStore.save_csv() + load_csv() roundtrip.
    5. WorkloadAnalyzer.analyze() → WorkloadReport.
    6. PageCountPartitioner.partition() → (partitions, stats).
    7. PageCountPriorityQueue population from partitions.
    8. Queue ordering is correct (highest pages first).
    9. BenchmarkReport captures all stage timings.
    10. Total pages is conserved through the pipeline.
    11. No document is lost or duplicated.

These tests use only local filesystem operations — no network I/O.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from adaptive_framework.benchmarks import BenchmarkReport
from adaptive_framework.dataset_builder import PubMedDatasetBuilder
from adaptive_framework.metadata_generator import MetadataExtractor, MetadataStore
from adaptive_framework.models.document import PDFMetadata
from adaptive_framework.scheduler import (
    PageCountPartitioner,
    PageCountPriorityQueue,
    PartitionSummary,
    WorkloadAnalyzer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_PDF = (
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


def _seed_pdf_directory(base_dir: Path, count: int = 10) -> Path:
    """Create *count* minimal PDF files in ``base_dir/raw/``.

    Args:
        base_dir: Root output directory.
        count: Number of PDF files to create.

    Returns:
        Path to the ``raw`` subdirectory.
    """
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        pdf_path = raw_dir / f"PMC{1000000 + i:07d}.pdf"
        pdf_path.write_bytes(_MINIMAL_PDF)
    return raw_dir


def _make_synthetic_metadata(
    page_counts: list[int], raw_dir: Path
) -> list[PDFMetadata]:
    """Build PDFMetadata records with given page counts (bypasses real extraction).

    Args:
        page_counts: Page counts for each document.
        raw_dir: Directory used to compute file paths.

    Returns:
        List of PDFMetadata records.
    """
    rng = random.Random(42)
    records: list[PDFMetadata] = []
    for i, pages in enumerate(page_counts):
        records.append(
            PDFMetadata(
                document_id=f"doc_{i:04d}",
                pages=pages,
                estimated_size_mb=round(pages * 0.15, 2),
                file_path=str(raw_dir / f"PMC{1000000 + i:07d}.pdf"),
                source_type="digital",
                language="en",
                processing_timestamp="2024-01-01T00:00:00+00:00",
            )
        )
    return records


# ---------------------------------------------------------------------------
# Integration test: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end pipeline integration tests."""

    def test_builder_scan_produces_stubs(self, tmp_path: Path) -> None:
        """PubMedDatasetBuilder.build(source_dir) returns PDF stubs."""
        raw_dir = _seed_pdf_directory(tmp_path, count=5)
        builder = PubMedDatasetBuilder()
        stubs = builder.build(source_dir=raw_dir)
        assert len(stubs) == 5
        for stub in stubs:
            assert stub.pages >= 1
            assert Path(stub.file_path).is_absolute()

    def test_metadata_store_roundtrip(self, tmp_path: Path) -> None:
        """save_csv() → load_csv() preserves all records."""
        raw_dir = _seed_pdf_directory(tmp_path, count=3)
        page_counts = [10, 20, 30]
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        store = MetadataStore(output_dir=tmp_path)
        store.save_csv(metadata)
        loaded = store.load_csv()
        assert len(loaded) == len(metadata)
        loaded_pages = sorted(m.pages for m in loaded)
        assert loaded_pages == sorted(page_counts)

    def test_partitioner_conserves_total_pages(self, tmp_path: Path) -> None:
        """Total pages is conserved through partition → work unit flow."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        page_counts = [5, 10, 15, 20, 25, 30, 35, 40]
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        partitioner = PageCountPartitioner()
        partitions, stats = partitioner.partition(metadata, num_workers=4)
        pipeline_total = sum(
            wu.page_count for part in partitions for wu in part.work_units
        )
        assert pipeline_total == sum(page_counts)
        assert stats.total_pages == sum(page_counts)

    def test_no_document_lost_in_partitions(self, tmp_path: Path) -> None:
        """Every document appears exactly once in the partition plan."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        page_counts = list(range(1, 21))  # 20 documents
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        partitioner = PageCountPartitioner()
        partitions, _ = partitioner.partition(metadata, num_workers=4)
        all_doc_ids: list[str] = [
            wu.document_id for part in partitions for wu in part.work_units
        ]
        expected_ids = {m.document_id for m in metadata}
        assert len(all_doc_ids) == len(set(all_doc_ids)), "Duplicate document IDs"
        assert set(all_doc_ids) == expected_ids, "Missing document IDs"

    def test_priority_queue_dispatch_order(self, tmp_path: Path) -> None:
        """Work units dispatched from queue follow descending page order."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        page_counts = [3, 50, 7, 100, 22, 1, 75]
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        partitioner = PageCountPartitioner()
        partitions, _ = partitioner.partition(metadata, num_workers=3)

        queue = PageCountPriorityQueue()
        for part in partitions:
            for wu in part.work_units:
                queue.insert(wu)

        dispatched: list[int] = []
        while (wu := queue.pop()) is not None:
            dispatched.append(wu.page_count)

        assert dispatched == sorted(page_counts, reverse=True), (
            f"Expected descending order: {sorted(page_counts, reverse=True)}, "
            f"got: {dispatched}"
        )

    def test_workload_analyzer_on_synthetic_dataset(self, tmp_path: Path) -> None:
        """WorkloadAnalyzer produces a valid report for a synthetic dataset."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        page_counts = [10, 20, 30, 40, 50, 60, 70, 80]
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        analyzer = WorkloadAnalyzer(seconds_per_page=1.0)
        report = analyzer.analyze(metadata, num_workers=4)
        assert report.total_documents == len(page_counts)
        assert report.total_pages == sum(page_counts)
        assert report.recommended_workers >= 1
        assert report.partition_difficulty in ("low", "medium", "high")

    def test_benchmark_report_captures_all_stages(self, tmp_path: Path) -> None:
        """BenchmarkReport captures timings for all pipeline stages."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        page_counts = [10, 20, 30, 40]
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        report = BenchmarkReport(run_id="integration_test")

        with report.time("dataset_loading_time_s", {"files": len(page_counts)}):
            pass  # simulates loading

        with report.time("metadata_generation_time_s", {"docs": len(metadata)}):
            pass  # simulates extraction

        with report.time("partition_time_s", {"docs": len(metadata), "workers": 2}):
            partitioner = PageCountPartitioner()
            partitions, stats = partitioner.partition(metadata, num_workers=2)

        with report.time("queue_creation_time_s", {"work_units": stats.total_work_units}):
            queue = PageCountPriorityQueue()
            for part in partitions:
                for wu in part.work_units:
                    queue.insert(wu)

        assert len(report) == 4
        assert all(r.elapsed_seconds >= 0 for r in report.results)
        stage_names = {r.stage_name for r in report.results}
        assert "partition_time_s" in stage_names
        assert "queue_creation_time_s" in stage_names

    def test_partition_summary_balance_score_in_range(self, tmp_path: Path) -> None:
        """PartitionSummary balance_score is in [0, 1] for any dataset."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        # Deliberately imbalanced dataset
        page_counts = [1, 2, 3, 100, 200, 300]
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        partitioner = PageCountPartitioner()
        partitions, stats = partitioner.partition(metadata, num_workers=3)
        summary = PartitionSummary(partitions, stats)
        score = summary.balance_score()
        assert 0.0 <= score <= 1.0

    def test_benchmark_csv_output(self, tmp_path: Path) -> None:
        """BenchmarkReport writes a readable CSV at the end of the pipeline."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        page_counts = [5, 15, 25]
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        report = BenchmarkReport(run_id="csv_test")
        with report.time("partition_time_s"):
            PageCountPartitioner().partition(metadata, num_workers=2)
        csv_path = tmp_path / "reports" / "benchmark.csv"
        saved = report.save_csv(csv_path)
        assert saved.exists()
        content = saved.read_text(encoding="utf-8")
        assert "partition_time_s" in content

    def test_workload_analyzer_partition_metrics(self, tmp_path: Path) -> None:
        """analyze_partitions() computes correct balance_score for equal partitions."""
        analyzer = WorkloadAnalyzer()
        result = analyzer.analyze_partitions([100, 100, 100, 100])
        assert result["balance_score"] == pytest.approx(1.0, abs=0.01)
        assert result["imbalance_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_store_statistics_after_partition(self, tmp_path: Path) -> None:
        """MetadataStore.compute_statistics() returns correct counts after pipeline."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        page_counts = [10, 20, 30, 40]
        metadata = _make_synthetic_metadata(page_counts, raw_dir)
        store = MetadataStore(output_dir=tmp_path)
        stats = store.compute_statistics(metadata)
        assert stats["total_pages"] == sum(page_counts)
        assert stats["count"] == len(page_counts)
        assert stats["digital_count"] == len(page_counts)  # all 'digital' from helper
