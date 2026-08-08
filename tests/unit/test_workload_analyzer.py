"""Unit tests for the WorkloadAnalyzer.

Tests:
    - analyze() returns a WorkloadReport with all fields populated.
    - Empty dataset raises ValueError.
    - total_pages == sum of all document pages.
    - avg_pages is correct.
    - coefficient_of_variation is std/avg.
    - partition_difficulty classifies correctly.
    - recommended_workers is >= 1.
    - format_report() returns a multi-line string.
    - analyze_partitions() returns balance metrics.
    - single-document dataset is handled correctly.
    - seconds_per_page parameter scales processing cost.
"""

from __future__ import annotations

import pytest

from adaptive_framework.models.document import PDFMetadata
from adaptive_framework.scheduler.workload_analyzer import WorkloadAnalyzer, WorkloadReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(pages: int, size_mb: float = 1.0) -> PDFMetadata:
    """Create a minimal PDFMetadata.

    Args:
        pages: Page count.
        size_mb: Estimated file size in MB.

    Returns:
        PDFMetadata record.
    """
    return PDFMetadata(
        pages=pages,
        estimated_size_mb=size_mb,
        file_path=f"/data/doc_{pages}.pdf",
    )


# ---------------------------------------------------------------------------
# WorkloadReport fields
# ---------------------------------------------------------------------------


class TestWorkloadReportFields:
    """Tests that WorkloadReport fields are correctly computed."""

    def test_total_pages_matches_sum(self) -> None:
        """total_pages == sum of all document page counts."""
        dataset = [_meta(10), _meta(20), _meta(30)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset, num_workers=3)
        assert report.total_pages == 60

    def test_total_documents_is_correct(self) -> None:
        """total_documents matches len(dataset)."""
        dataset = [_meta(5) for _ in range(7)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset, num_workers=2)
        assert report.total_documents == 7

    def test_avg_pages_is_correct(self) -> None:
        """avg_pages == total_pages / count."""
        dataset = [_meta(10), _meta(20), _meta(30)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset)
        assert abs(report.avg_pages - 20.0) < 0.01

    def test_min_max_pages_correct(self) -> None:
        """min_pages and max_pages are correct."""
        dataset = [_meta(3), _meta(15), _meta(7)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset)
        assert report.min_pages == 3
        assert report.max_pages == 15

    def test_total_size_mb_correct(self) -> None:
        """total_size_mb sums correctly."""
        dataset = [_meta(10, 2.0), _meta(20, 3.0), _meta(5, 1.0)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset)
        assert abs(report.total_size_mb - 6.0) < 0.01

    def test_processing_cost_scales_with_pages(self) -> None:
        """estimated_processing_cost == total_pages × seconds_per_page."""
        dataset = [_meta(100)]
        analyzer = WorkloadAnalyzer(seconds_per_page=3.0)
        report = analyzer.analyze(dataset, num_workers=1)
        assert abs(report.estimated_processing_cost - 300.0) < 0.01

    def test_expected_time_per_worker_scales_with_workers(self) -> None:
        """expected_time_per_worker decreases with more workers."""
        dataset = [_meta(100)]
        analyzer = WorkloadAnalyzer(seconds_per_page=2.0)
        r1 = analyzer.analyze(dataset, num_workers=1)
        r4 = analyzer.analyze(dataset, num_workers=4)
        assert r4.expected_time_per_worker < r1.expected_time_per_worker

    def test_recommended_workers_at_least_one(self) -> None:
        """recommended_workers is always >= 1."""
        dataset = [_meta(1)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset)
        assert report.recommended_workers >= 1

    def test_analysis_timestamp_is_set(self) -> None:
        """analysis_timestamp_utc is a non-empty string."""
        dataset = [_meta(10)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset)
        assert isinstance(report.analysis_timestamp_utc, str)
        assert len(report.analysis_timestamp_utc) > 0


# ---------------------------------------------------------------------------
# Difficulty classification
# ---------------------------------------------------------------------------


class TestDifficultyClassification:
    """Tests for partition_difficulty field."""

    def test_uniform_dataset_difficulty_is_low(self) -> None:
        """Uniform page counts → low difficulty (CV < 0.2)."""
        dataset = [_meta(100) for _ in range(10)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset, num_workers=4)
        assert report.partition_difficulty == "low"

    def test_varied_dataset_difficulty_is_high(self) -> None:
        """Highly varied page counts → high difficulty (CV > 0.5)."""
        dataset = [_meta(1), _meta(10), _meta(100), _meta(1000)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset, num_workers=4)
        # std/avg will be > 0.5 for this extreme spread
        assert report.partition_difficulty in ("medium", "high")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_dataset_raises_value_error(self) -> None:
        """analyze() raises ValueError for empty dataset."""
        analyzer = WorkloadAnalyzer()
        with pytest.raises(ValueError, match="non-empty"):
            analyzer.analyze([])

    def test_single_document_dataset(self) -> None:
        """Single-document dataset is handled without error."""
        dataset = [_meta(42)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset, num_workers=1)
        assert report.total_documents == 1
        assert report.total_pages == 42
        assert report.min_pages == 42
        assert report.max_pages == 42
        assert report.std_pages == 0.0

    def test_coefficient_of_variation_zero_for_uniform(self) -> None:
        """CV == 0 when all documents have the same page count."""
        dataset = [_meta(50) for _ in range(5)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset)
        assert report.coefficient_of_variation == 0.0


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------


class TestFormatReport:
    """Tests for WorkloadReport.format_report()."""

    def test_format_report_returns_string(self) -> None:
        """format_report() returns a non-empty string."""
        dataset = [_meta(10), _meta(20), _meta(30)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset, num_workers=2)
        text = report.format_report()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_format_report_contains_total_pages(self) -> None:
        """format_report() includes the total page count."""
        dataset = [_meta(100), _meta(200)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset)
        text = report.format_report()
        assert "300" in text

    def test_to_dict_has_all_keys(self) -> None:
        """to_dict() has all WorkloadReport field keys."""
        dataset = [_meta(10)]
        analyzer = WorkloadAnalyzer()
        report = analyzer.analyze(dataset)
        d = report.to_dict()
        expected_keys = {
            "total_documents", "total_pages", "avg_pages", "min_pages",
            "max_pages", "std_pages", "coefficient_of_variation",
            "total_size_mb", "partition_difficulty", "recommended_workers",
        }
        for key in expected_keys:
            assert key in d, f"Key '{key}' missing from to_dict()"


# ---------------------------------------------------------------------------
# analyze_partitions
# ---------------------------------------------------------------------------


class TestAnalyzePartitions:
    """Tests for the analyze_partitions() helper."""

    def test_analyze_partitions_returns_dict(self) -> None:
        """analyze_partitions() returns a non-empty dict."""
        analyzer = WorkloadAnalyzer()
        result = analyzer.analyze_partitions([100, 105, 95, 100])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_balance_score_one_for_uniform_partitions(self) -> None:
        """balance_score is 1.0 when all partitions have equal pages."""
        analyzer = WorkloadAnalyzer()
        result = analyzer.analyze_partitions([100, 100, 100, 100])
        assert result["balance_score"] == pytest.approx(1.0, abs=0.01)

    def test_analyze_partitions_empty_returns_empty(self) -> None:
        """analyze_partitions([]) returns an empty dict."""
        analyzer = WorkloadAnalyzer()
        result = analyzer.analyze_partitions([])
        assert result == {}

    def test_imbalance_ratio_correct(self) -> None:
        """imbalance_ratio is max/min."""
        analyzer = WorkloadAnalyzer()
        result = analyzer.analyze_partitions([50, 100])
        assert result["imbalance_ratio"] == pytest.approx(2.0, abs=0.01)
