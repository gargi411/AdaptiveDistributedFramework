"""Unit tests for data models.

Tests:
    - PDFMetadata validation
    - PageWorkUnit page_count property
    - EvaluationResult.passes_overhead_target
    - WorkerStatus.is_available
    - ClusterStatus.is_degraded
    - ValidationError raised for invalid values
"""

from __future__ import annotations

import pytest

from adaptive_framework.core.exceptions import ValidationError
from adaptive_framework.models.document import PDFMetadata, PageResult, DocumentResult
from adaptive_framework.models.scheduling import PageWorkUnit, WorkUnitStatus
from adaptive_framework.models.runtime import (
    ClusterStatus,
    FrameworkState,
    FrameworkStatus,
    WorkerState,
    WorkerStatus,
)
from adaptive_framework.models.evaluation import EvaluationResult


class TestPDFMetadata:
    """Tests for PDFMetadata model."""

    def test_valid_minimal_metadata(self) -> None:
        """PDFMetadata with required fields should construct successfully."""
        meta = PDFMetadata(pages=10, estimated_size_mb=1.5, file_path="/data/paper.pdf")
        assert meta.pages == 10
        assert meta.estimated_size_mb == 1.5
        assert meta.source_type is None

    def test_pages_zero_raises(self) -> None:
        """pages=0 must raise ValidationError."""
        with pytest.raises(ValidationError, match="pages"):
            PDFMetadata(pages=0, estimated_size_mb=1.0, file_path="/data/paper.pdf")

    def test_invalid_source_type_raises(self) -> None:
        """Invalid source_type must raise ValidationError."""
        with pytest.raises(ValidationError, match="source_type"):
            PDFMetadata(
                pages=5, estimated_size_mb=1.0, file_path="/data/paper.pdf",
                source_type="unknown"
            )

    def test_is_scanned_returns_true(self) -> None:
        """is_scanned() must return True when source_type='scanned'."""
        meta = PDFMetadata(
            pages=5, estimated_size_mb=1.0, file_path="/data/paper.pdf",
            source_type="scanned"
        )
        assert meta.is_scanned() is True
        assert meta.is_digital() is False

    def test_to_dict_contains_required_fields(self) -> None:
        """to_dict() must contain all architecture §2.4 fields."""
        meta = PDFMetadata(pages=10, estimated_size_mb=1.5, file_path="/data/paper.pdf")
        d = meta.to_dict()
        for key in ("document_id", "pages", "estimated_size_mb", "file_path",
                    "processing_timestamp"):
            assert key in d, f"Missing key '{key}' in PDFMetadata.to_dict()"


class TestPageWorkUnit:
    """Tests for PageWorkUnit model."""

    def test_page_count_property(self) -> None:
        """page_count must equal end_page - start_page + 1."""
        wu = PageWorkUnit(
            document_id="doc-1", file_path="/data/a.pdf",
            start_page=3, end_page=12
        )
        assert wu.page_count == 10

    def test_start_page_zero_raises(self) -> None:
        """start_page=0 must raise ValidationError (1-indexed)."""
        with pytest.raises(ValidationError, match="start_page"):
            PageWorkUnit(
                document_id="doc-1", file_path="/data/a.pdf",
                start_page=0, end_page=5
            )

    def test_end_page_less_than_start_raises(self) -> None:
        """end_page < start_page must raise ValidationError."""
        with pytest.raises(ValidationError):
            PageWorkUnit(
                document_id="doc-1", file_path="/data/a.pdf",
                start_page=10, end_page=5
            )

    def test_is_terminal_completed(self) -> None:
        """COMPLETED status must be terminal."""
        wu = PageWorkUnit(
            document_id="doc-1", file_path="/data/a.pdf",
            start_page=1, end_page=5,
            status=WorkUnitStatus.COMPLETED
        )
        assert wu.is_terminal() is True

    def test_is_terminal_pending(self) -> None:
        """PENDING status must not be terminal."""
        wu = PageWorkUnit(
            document_id="doc-1", file_path="/data/a.pdf",
            start_page=1, end_page=5
        )
        assert wu.is_terminal() is False


class TestEvaluationResult:
    """Tests for EvaluationResult.passes_overhead_target."""

    def _make_result(self, overhead_percent: float) -> EvaluationResult:
        return EvaluationResult(
            run_id="test_run", node_count=4,
            total_documents=10, total_pages=500,
            speedup=3.8, throughput_pages_per_second=4.17,
            avg_cpu_percent=68.0, total_energy_joules=240.0,
            scheduler_overhead_percent=overhead_percent,
            baseline_nodes=1,
            baseline_wall_time_seconds=456.0,
            measured_wall_time_seconds=120.0,
            scheduler_time_seconds=0.8,
        )

    def test_overhead_below_target_passes(self) -> None:
        """0.67% overhead must pass the < 1% target."""
        result = self._make_result(0.67)
        assert result.passes_overhead_target is True

    def test_overhead_above_target_fails(self) -> None:
        """2.5% overhead must fail the < 1% target."""
        result = self._make_result(2.5)
        assert result.passes_overhead_target is False

    def test_overhead_exactly_one_percent_fails(self) -> None:
        """Exactly 1.0% overhead must fail (target is strictly < 1%)."""
        result = self._make_result(1.0)
        assert result.passes_overhead_target is False
