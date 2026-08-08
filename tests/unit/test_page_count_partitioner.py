"""Unit tests for the PageCountPartitioner (Research Algorithm #1).

Tests verify:
    - get_strategy_name() returns 'page_count_lpt'
    - Partition count matches num_workers for large datasets.
    - Total pages in all partitions == sum of all document pages.
    - No document is lost or duplicated.
    - LPT balance: std deviation is near-zero for uniform page counts.
    - LPT balance: handles heterogeneous page counts correctly.
    - Empty dataset raises SchedulerError.
    - num_workers < 1 raises SchedulerError.
    - Single-document dataset: one partition.
    - Fewer documents than workers: partition count capped.
    - PartitionStatistics fields are accurate.
    - Each partition has at least one work unit.
    - work units have correct page ranges (start=1, end=doc.pages).
    - PartitionSummary.balance_score() is in [0, 1].
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from adaptive_framework.core.exceptions import SchedulerError
from adaptive_framework.models.document import PDFMetadata
from adaptive_framework.scheduler.page_count_partitioner import PageCountPartitioner
from adaptive_framework.scheduler.partition_summary import PartitionSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(pages: int, idx: int = 0) -> PDFMetadata:
    """Create a minimal PDFMetadata record.

    Args:
        pages: Page count.
        idx: Index used to generate unique document_id.

    Returns:
        PDFMetadata with the given page count.
    """
    return PDFMetadata(
        pages=pages,
        estimated_size_mb=round(pages * 0.1, 2),
        file_path=f"/data/doc_{idx:04d}.pdf",
    )


def _make_dataset(page_counts: list[int]) -> list[PDFMetadata]:
    """Build a dataset from a list of page counts.

    Args:
        page_counts: List of page counts, one per document.

    Returns:
        List of PDFMetadata objects.
    """
    return [_make_metadata(pages, i) for i, pages in enumerate(page_counts)]


# ---------------------------------------------------------------------------
# Strategy contract
# ---------------------------------------------------------------------------


class TestStrategyContract:
    """Tests for the IPartitionStrategy interface contract."""

    def test_get_strategy_name(self) -> None:
        """get_strategy_name() returns 'page_count_lpt'."""
        p = PageCountPartitioner()
        assert p.get_strategy_name() == "page_count_lpt"

    def test_empty_dataset_raises_scheduler_error(self) -> None:
        """Empty dataset raises SchedulerError."""
        p = PageCountPartitioner()
        with pytest.raises(SchedulerError, match="empty"):
            p.partition([], num_workers=4)

    def test_zero_workers_raises_scheduler_error(self) -> None:
        """num_workers < 1 raises SchedulerError."""
        p = PageCountPartitioner()
        dataset = _make_dataset([10, 20, 30])
        with pytest.raises(SchedulerError, match="num_workers"):
            p.partition(dataset, num_workers=0)

    def test_negative_workers_raises_scheduler_error(self) -> None:
        """Negative num_workers raises SchedulerError."""
        p = PageCountPartitioner()
        dataset = _make_dataset([10, 20])
        with pytest.raises(SchedulerError):
            p.partition(dataset, num_workers=-1)


# ---------------------------------------------------------------------------
# Correctness: no document loss
# ---------------------------------------------------------------------------


class TestCorrectnessNoDocumentLoss:
    """Tests that all documents are placed in exactly one partition."""

    def test_total_pages_equals_sum_of_all_docs(self) -> None:
        """Sum of partition pages == sum of all document pages."""
        page_counts = [5, 10, 15, 20, 25, 30, 35, 40]
        dataset = _make_dataset(page_counts)
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=3)
        assert stats.total_pages == sum(page_counts)

    def test_no_document_duplicated(self) -> None:
        """Each document_id appears in exactly one work unit."""
        dataset = _make_dataset([10, 20, 30, 40, 50])
        p = PageCountPartitioner()
        partitions, _ = p.partition(dataset, num_workers=2)
        all_doc_ids = []
        for part in partitions:
            for wu in part.work_units:
                all_doc_ids.append(wu.document_id)
        unique_ids = set(all_doc_ids)
        assert len(all_doc_ids) == len(unique_ids), "Duplicate work units detected"
        expected_ids = {m.document_id for m in dataset}
        assert unique_ids == expected_ids, "Document IDs mismatch"

    def test_total_work_units_equals_document_count(self) -> None:
        """Total work units across all partitions equals len(dataset)."""
        page_counts = [5, 10, 15, 20, 25]
        dataset = _make_dataset(page_counts)
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=3)
        total_wu = sum(len(part.work_units) for part in partitions)
        assert total_wu == len(dataset)
        assert stats.total_work_units == len(dataset)


# ---------------------------------------------------------------------------
# Partition count
# ---------------------------------------------------------------------------


class TestPartitionCount:
    """Tests for the number of partitions created."""

    def test_partition_count_matches_workers_for_large_dataset(self) -> None:
        """With more docs than workers, partition count == num_workers."""
        dataset = _make_dataset([10] * 20)  # 20 docs
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=4)
        assert stats.total_partitions == 4

    def test_single_document_single_partition(self) -> None:
        """One document always produces one partition."""
        dataset = _make_dataset([42])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=8)
        assert stats.total_partitions == 1
        assert partitions[0].total_pages == 42

    def test_fewer_docs_than_workers_capped(self) -> None:
        """More workers than documents: partition count <= document count."""
        dataset = _make_dataset([10, 20, 30])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=10)
        assert stats.total_partitions <= 3


# ---------------------------------------------------------------------------
# Load balance
# ---------------------------------------------------------------------------


class TestLoadBalance:
    """Tests for partition balance quality."""

    def test_uniform_page_counts_near_zero_std(self) -> None:
        """Uniform page counts should yield near-zero std deviation."""
        dataset = _make_dataset([100] * 8)  # 8 × 100 = 800 pages
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=4)
        # Each partition should have exactly 200 pages
        assert stats.std_pages_per_partition < 5.0, (
            f"Expected near-zero std, got {stats.std_pages_per_partition}"
        )

    def test_avg_pages_is_correct(self) -> None:
        """Average pages per partition should be total / num_partitions."""
        dataset = _make_dataset([10, 20, 30, 40])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=2)
        expected_avg = stats.total_pages / stats.total_partitions
        assert abs(stats.avg_pages_per_partition - expected_avg) < 0.01

    def test_min_max_pages_are_valid(self) -> None:
        """min_pages_per_partition <= avg <= max_pages_per_partition."""
        dataset = _make_dataset([5, 10, 15, 20, 25, 30, 35])
        p = PageCountPartitioner()
        _, stats = p.partition(dataset, num_workers=3)
        assert stats.min_pages_per_partition <= stats.avg_pages_per_partition
        assert stats.avg_pages_per_partition <= stats.max_pages_per_partition

    def test_lpt_approximation_quality(self) -> None:
        """LPT approximation keeps max partition within 4/3 × avg.

        The LPT heuristic guarantees max ≤ (4/3 - 1/(3m)) × OPT.
        For practical purposes, we check max ≤ 1.5 × avg.
        """
        import random
        rng = random.Random(42)
        page_counts = [rng.randint(1, 200) for _ in range(50)]
        dataset = _make_dataset(page_counts)
        p = PageCountPartitioner()
        _, stats = p.partition(dataset, num_workers=4)
        ratio = stats.max_pages_per_partition / stats.avg_pages_per_partition
        assert ratio <= 1.5, f"LPT approximation ratio {ratio:.3f} exceeds 1.5"


# ---------------------------------------------------------------------------
# Work unit integrity
# ---------------------------------------------------------------------------


class TestWorkUnitIntegrity:
    """Tests that PageWorkUnit objects are correctly constructed."""

    def test_work_unit_start_page_is_one(self) -> None:
        """Each work unit starts at page 1 (whole-document work units)."""
        dataset = _make_dataset([20, 30, 40])
        p = PageCountPartitioner()
        partitions, _ = p.partition(dataset, num_workers=2)
        for part in partitions:
            for wu in part.work_units:
                assert wu.start_page == 1

    def test_work_unit_end_page_equals_doc_pages(self) -> None:
        """work unit end_page equals the document's page count."""
        pages_list = [20, 30, 40]
        dataset = _make_dataset(pages_list)
        p = PageCountPartitioner()
        partitions, _ = p.partition(dataset, num_workers=2)
        all_end_pages = sorted(
            wu.end_page
            for part in partitions
            for wu in part.work_units
        )
        assert all_end_pages == sorted(pages_list)

    def test_work_unit_priority_equals_page_count(self) -> None:
        """work unit priority is set to page_count (for scheduler queue)."""
        dataset = _make_dataset([25, 50])
        p = PageCountPartitioner()
        partitions, _ = p.partition(dataset, num_workers=2)
        for part in partitions:
            for wu in part.work_units:
                assert wu.priority == wu.page_count


# ---------------------------------------------------------------------------
# PartitionSummary integration
# ---------------------------------------------------------------------------


class TestPartitionSummary:
    """Tests for PartitionSummary using partitioner output."""

    def test_balance_score_in_range(self) -> None:
        """balance_score() is always in [0.0, 1.0]."""
        dataset = _make_dataset([5, 100, 200, 3, 50])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=3)
        summary = PartitionSummary(partitions, stats)
        score = summary.balance_score()
        assert 0.0 <= score <= 1.0

    def test_format_table_is_string(self) -> None:
        """format_table() returns a non-empty string."""
        dataset = _make_dataset([10, 20, 30])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=2)
        summary = PartitionSummary(partitions, stats)
        table = summary.format_table()
        assert isinstance(table, str)
        assert len(table) > 0

    def test_format_compact_contains_page_count(self) -> None:
        """format_compact() contains total page count."""
        dataset = _make_dataset([100, 200])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=2)
        summary = PartitionSummary(partitions, stats)
        compact = summary.format_compact()
        assert "300" in compact or "TotalPages" in compact

    def test_variance_is_non_negative(self) -> None:
        """variance() returns a non-negative float."""
        dataset = _make_dataset([10, 20, 30, 40])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=2)
        summary = PartitionSummary(partitions, stats)
        assert summary.variance() >= 0.0

    def test_estimated_completion_time_positive(self) -> None:
        """estimated_completion_time() returns a positive float."""
        dataset = _make_dataset([100, 200, 300])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=3)
        summary = PartitionSummary(partitions, stats)
        t = summary.estimated_completion_time(pages_per_second_per_worker=10.0)
        assert t > 0.0

    def test_to_dict_structure(self) -> None:
        """to_dict() returns a dict with 'statistics' and 'partitions' keys."""
        dataset = _make_dataset([10, 20])
        p = PageCountPartitioner()
        partitions, stats = p.partition(dataset, num_workers=2)
        summary = PartitionSummary(partitions, stats)
        d = summary.to_dict()
        assert "statistics" in d
        assert "partitions" in d
        assert "balance_score" in d
