"""Unit tests for the BenchmarkRunner (BenchmarkReport, BenchmarkTimer, BenchmarkResult).

Tests:
    - BenchmarkTimer context manager measures elapsed time.
    - BenchmarkTimer.result is populated after exit.
    - BenchmarkReport.time() accumulates results.
    - BenchmarkReport.record() and record_manual() add results.
    - BenchmarkReport.total_elapsed_seconds() sums correctly.
    - BenchmarkReport.save_csv() writes a valid CSV file.
    - BenchmarkReport.get_result() retrieves by stage name.
    - BenchmarkResult.to_dict() has required keys.
    - BenchmarkReport.to_dict() has run_id and results keys.
    - len(report) == number of recorded results.
    - print_summary() does not raise.
    - CSV append mode produces more rows than overwrite mode.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from adaptive_framework.benchmarks import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkTimer,
)


# ---------------------------------------------------------------------------
# BenchmarkTimer
# ---------------------------------------------------------------------------


class TestBenchmarkTimer:
    """Tests for the BenchmarkTimer context manager."""

    def test_result_is_none_before_enter(self) -> None:
        """result is None before the context block executes."""
        timer = BenchmarkTimer("test_stage")
        assert timer.result is None

    def test_result_is_populated_after_exit(self) -> None:
        """result is populated with a BenchmarkResult after context exit."""
        timer = BenchmarkTimer("test_stage")
        with timer:
            pass
        assert timer.result is not None
        assert isinstance(timer.result, BenchmarkResult)

    def test_elapsed_is_positive(self) -> None:
        """elapsed_seconds is positive for a non-trivial operation."""
        timer = BenchmarkTimer("sleep_stage")
        with timer:
            time.sleep(0.01)
        assert timer.result is not None
        assert timer.result.elapsed_seconds >= 0.005  # at least ~5ms

    def test_stage_name_is_stored(self) -> None:
        """BenchmarkResult stores the stage name correctly."""
        timer = BenchmarkTimer("my_stage")
        with timer:
            pass
        assert timer.result is not None
        assert timer.result.stage_name == "my_stage"

    def test_metadata_is_passed_through(self) -> None:
        """Metadata dict is stored in the result."""
        timer = BenchmarkTimer("stage_with_meta", metadata={"workers": 4})
        with timer:
            pass
        assert timer.result is not None
        assert timer.result.metadata.get("workers") == 4


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_to_dict_has_stage_name(self) -> None:
        """to_dict() contains 'stage_name'."""
        result = BenchmarkResult(stage_name="my_stage", elapsed_seconds=0.5)
        d = result.to_dict()
        assert "stage_name" in d
        assert d["stage_name"] == "my_stage"

    def test_to_dict_has_elapsed_seconds(self) -> None:
        """to_dict() contains 'elapsed_seconds'."""
        result = BenchmarkResult(stage_name="s", elapsed_seconds=1.23)
        d = result.to_dict()
        assert "elapsed_seconds" in d

    def test_to_dict_includes_metadata_keys(self) -> None:
        """to_dict() flattens metadata keys into the output dict."""
        result = BenchmarkResult(
            stage_name="s", elapsed_seconds=0.1, metadata={"files": 50, "workers": 4}
        )
        d = result.to_dict()
        assert d.get("files") == 50
        assert d.get("workers") == 4

    def test_repr_contains_stage_name(self) -> None:
        """__repr__ includes the stage name."""
        result = BenchmarkResult(stage_name="partition_time", elapsed_seconds=0.03)
        assert "partition_time" in repr(result)


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReportAccumulation:
    """Tests for accumulating results."""

    def test_time_context_adds_result(self) -> None:
        """report.time() adds one result per use."""
        report = BenchmarkReport(run_id="run_001")
        with report.time("stage_a"):
            pass
        assert len(report) == 1

    def test_multiple_time_blocks_accumulate(self) -> None:
        """Multiple report.time() calls accumulate correctly."""
        report = BenchmarkReport()
        for name in ["stage_a", "stage_b", "stage_c"]:
            with report.time(name):
                pass
        assert len(report) == 3

    def test_record_manual_adds_result(self) -> None:
        """record_manual() adds a BenchmarkResult."""
        report = BenchmarkReport()
        report.record_manual("manual_stage", elapsed_seconds=0.042)
        assert len(report) == 1

    def test_record_result_adds_result(self) -> None:
        """record() adds a pre-built BenchmarkResult."""
        report = BenchmarkReport()
        result = BenchmarkResult(stage_name="pre_built", elapsed_seconds=0.1)
        report.record(result)
        assert len(report) == 1

    def test_total_elapsed_seconds_sums(self) -> None:
        """total_elapsed_seconds() sums all elapsed values."""
        report = BenchmarkReport()
        report.record_manual("a", 1.0)
        report.record_manual("b", 2.0)
        report.record_manual("c", 3.0)
        assert abs(report.total_elapsed_seconds() - 6.0) < 0.01

    def test_get_result_returns_correct_stage(self) -> None:
        """get_result() returns the result for the given stage name."""
        report = BenchmarkReport()
        report.record_manual("target_stage", 0.5)
        report.record_manual("other_stage", 0.9)
        result = report.get_result("target_stage")
        assert result is not None
        assert result.stage_name == "target_stage"

    def test_get_result_none_for_missing(self) -> None:
        """get_result() returns None if stage not found."""
        report = BenchmarkReport()
        assert report.get_result("nonexistent") is None


class TestBenchmarkReportCSV:
    """Tests for CSV serialization."""

    def test_save_csv_creates_file(self, tmp_path: Path) -> None:
        """save_csv() creates the output file."""
        report = BenchmarkReport(run_id="test_run")
        report.record_manual("partition_time_s", 0.0023, {"documents": 50})
        csv_path = tmp_path / "benchmark.csv"
        result_path = report.save_csv(csv_path)
        assert result_path.exists()

    def test_save_csv_has_header(self, tmp_path: Path) -> None:
        """Saved CSV contains a header row."""
        report = BenchmarkReport()
        report.record_manual("stage_a", 0.1)
        csv_path = tmp_path / "bench.csv"
        report.save_csv(csv_path)
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "stage_name" in header
        assert "elapsed_seconds" in header

    def test_save_csv_row_count(self, tmp_path: Path) -> None:
        """CSV has one data row per result."""
        report = BenchmarkReport()
        for i in range(5):
            report.record_manual(f"stage_{i}", float(i) * 0.1)
        csv_path = tmp_path / "bench.csv"
        report.save_csv(csv_path)
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # 1 header + 5 data rows
        assert len(rows) == 6

    def test_save_csv_append_mode(self, tmp_path: Path) -> None:
        """Appending to an existing CSV adds rows without overwriting."""
        report = BenchmarkReport()
        report.record_manual("first", 0.1)
        csv_path = tmp_path / "bench.csv"
        report.save_csv(csv_path, append=False)

        report2 = BenchmarkReport()
        report2.record_manual("second", 0.2)
        report2.record_manual("third", 0.3)
        report2.save_csv(csv_path, append=True)

        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # 1 header + 1 original + 2 appended = 4
        assert len(rows) == 4

    def test_save_csv_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_csv() creates parent directories automatically."""
        report = BenchmarkReport()
        report.record_manual("stage", 0.1)
        nested = tmp_path / "a" / "b" / "c" / "bench.csv"
        report.save_csv(nested)
        assert nested.exists()


class TestBenchmarkReportSerialization:
    """Tests for to_dict() and repr."""

    def test_to_dict_has_run_id(self) -> None:
        """to_dict() contains 'run_id'."""
        report = BenchmarkReport(run_id="run_abc")
        d = report.to_dict()
        assert d["run_id"] == "run_abc"

    def test_to_dict_has_results(self) -> None:
        """to_dict() contains 'results' list."""
        report = BenchmarkReport()
        report.record_manual("stage", 0.5)
        d = report.to_dict()
        assert "results" in d
        assert len(d["results"]) == 1

    def test_repr_contains_stages_count(self) -> None:
        """__repr__ shows the stage count."""
        report = BenchmarkReport(run_id="x")
        report.record_manual("a", 0.1)
        report.record_manual("b", 0.2)
        assert "stages=2" in repr(report)

    def test_print_summary_does_not_raise(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_summary() executes without raising."""
        report = BenchmarkReport(run_id="r")
        report.record_manual("dataset_loading_time_s", 1.23)
        report.record_manual("partition_time_s", 0.002)
        report.print_summary()
        captured = capsys.readouterr()
        assert "BENCHMARK REPORT" in captured.out


class TestBenchmarkAlias:
    """Tests for the BenchmarkRunner alias."""

    def test_benchmark_runner_is_alias_for_report(self) -> None:
        """BenchmarkRunner is identical to BenchmarkReport."""
        assert BenchmarkRunner is BenchmarkReport
