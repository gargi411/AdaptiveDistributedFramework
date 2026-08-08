"""Unit tests for ClusterBenchmark and ClusterBenchmarkResult.

Tests cover: metric computation, scheduler overhead target, speedup,
parallel efficiency, CSV/JSON export, and summary formatting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from adaptive_framework.benchmarks.cluster_benchmark import (
    ClusterBenchmark,
    ClusterBenchmarkResult,
)
from adaptive_framework.core.constants import SCHEDULER_OVERHEAD_TARGET_FRACTION


# ── ClusterBenchmarkResult ────────────────────────────────────────────────

class TestClusterBenchmarkResult:

    def _make_result(
        self,
        scheduler_time: float = 0.0005,
        wall_time: float = 10.0,
        num_nodes: int = 3,
        total_pages: int = 1000,
    ) -> ClusterBenchmarkResult:
        speedup = 10.0 / wall_time  # baseline = 10s
        peff = speedup / max(num_nodes, 1)
        overhead = scheduler_time / wall_time
        return ClusterBenchmarkResult(
            run_id="test_run",
            mode="dev",
            num_workers=4,
            num_nodes=num_nodes,
            total_documents=10,
            total_pages=total_pages,
            total_wall_time_s=wall_time,
            scheduler_time_s=scheduler_time,
            scheduler_overhead_fraction=overhead,
            throughput_pages_per_s=total_pages / wall_time,
            speedup=speedup,
            parallel_efficiency=peff,
            avg_cpu_percent=65.0,
            avg_ram_percent=55.0,
            load_balance_score=0.94,
        )

    def test_overhead_percent_calculation(self) -> None:
        result = self._make_result(scheduler_time=0.005, wall_time=10.0)
        assert abs(result.scheduler_overhead_percent - 0.05) < 1e-6

    def test_meets_overhead_target_when_below_1pct(self) -> None:
        result = self._make_result(scheduler_time=0.05, wall_time=100.0)
        # 0.05 / 100.0 = 0.05% < 1%
        assert result.meets_overhead_target is True

    def test_fails_overhead_target_when_above_1pct(self) -> None:
        result = self._make_result(scheduler_time=1.5, wall_time=10.0)
        # 1.5 / 10 = 15% > 1%
        assert result.meets_overhead_target is False

    def test_throughput_pages_per_s(self) -> None:
        result = self._make_result(total_pages=500, wall_time=5.0)
        assert abs(result.throughput_pages_per_s - 100.0) < 0.01

    def test_cluster_efficiency_in_range(self) -> None:
        result = self._make_result()
        ce = result.cluster_efficiency
        assert 0.0 <= ce <= 1.0

    def test_to_dict_has_all_expected_keys(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        for key in [
            "run_id", "mode", "num_workers", "num_nodes",
            "total_documents", "total_pages",
            "total_wall_time_s", "scheduler_time_s",
            "scheduler_overhead_fraction", "scheduler_overhead_percent",
            "meets_overhead_target",
            "throughput_pages_per_s", "speedup", "parallel_efficiency",
            "cluster_efficiency",
            "avg_cpu_percent", "avg_ram_percent", "load_balance_score",
            "total_steal_events", "total_tasks_stolen",
            "total_recovery_events", "total_tasks_recovered",
            "completed_at",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_format_summary_contains_run_id(self) -> None:
        result = self._make_result()
        summary = result.format_summary()
        assert "test_run" in summary
        assert "Scheduler Overhead" in summary

    def test_repr_contains_throughput(self) -> None:
        result = self._make_result(total_pages=100, wall_time=1.0)
        r = repr(result)
        assert "throughput" in r.lower()

    def test_speedup_and_parallel_efficiency_1node(self) -> None:
        # With 1 node, speedup=1 and efficiency=1
        result = self._make_result(num_nodes=1, wall_time=10.0)
        # speedup = 10/10 = 1.0
        assert abs(result.speedup - 1.0) < 0.01
        assert abs(result.parallel_efficiency - 1.0) < 0.01


# ── ClusterBenchmark (timer) ──────────────────────────────────────────────

class TestClusterBenchmark:

    def test_stop_without_start_raises(self) -> None:
        bench = ClusterBenchmark(run_id="t1")
        with pytest.raises(RuntimeError, match="start"):
            bench.stop(
                total_documents=0, total_pages=0,
                scheduler_time_s=0.0,
                avg_cpu_percent=0.0, avg_ram_percent=0.0,
                load_balance_score=1.0,
            )

    def test_wall_time_measured(self) -> None:
        bench = ClusterBenchmark(run_id="t2")
        bench.start()
        time.sleep(0.05)
        result = bench.stop(
            total_documents=1,
            total_pages=100,
            scheduler_time_s=0.001,
            avg_cpu_percent=50.0,
            avg_ram_percent=40.0,
            load_balance_score=0.9,
        )
        assert result.total_wall_time_s >= 0.05

    def test_result_run_id_matches(self) -> None:
        bench = ClusterBenchmark(run_id="my_run_42")
        bench.start()
        result = bench.stop(
            total_documents=5, total_pages=200,
            scheduler_time_s=0.002,
            avg_cpu_percent=60.0, avg_ram_percent=55.0,
            load_balance_score=0.85,
        )
        assert result.run_id == "my_run_42"

    def test_context_manager_usage(self) -> None:
        bench = ClusterBenchmark(run_id="ctx_test")
        with bench:
            time.sleep(0.01)
        # stop() must be called explicitly — context manager just calls start()
        result = bench.stop(
            total_documents=2, total_pages=50,
            scheduler_time_s=0.0005,
            avg_cpu_percent=30.0, avg_ram_percent=25.0,
            load_balance_score=0.95,
        )
        assert result.total_wall_time_s >= 0.01

    def test_single_node_baseline_speedup(self) -> None:
        # If single_node_baseline=10s and wall_time=2.5s → speedup=4x
        bench = ClusterBenchmark(
            run_id="speedup_test",
            num_nodes=4,
            single_node_baseline_s=10.0,
        )
        bench.start()
        result = bench.stop(
            total_documents=1, total_pages=100,
            scheduler_time_s=0.001,
            avg_cpu_percent=70.0, avg_ram_percent=60.0,
            load_balance_score=0.92,
        )
        # Speedup = 10.0 / wall_time (which is very small → speedup >> 4)
        assert result.speedup > 1.0

    def test_overhead_target_reflected_in_result(self) -> None:
        bench = ClusterBenchmark(run_id="oh_test")
        bench.start()
        time.sleep(0.1)  # ensure wall_time >> scheduler_time so overhead < 1%
        result = bench.stop(
            total_documents=1, total_pages=1000,
            scheduler_time_s=0.0001,  # 0.1ms vs >= 100ms wall time → << 1%
            avg_cpu_percent=50.0, avg_ram_percent=40.0,
            load_balance_score=0.90,
        )
        assert result.meets_overhead_target is True


# ── Export: CSV ───────────────────────────────────────────────────────────

class TestCSVExport:

    def test_save_csv_creates_file(self, tmp_path: Path) -> None:
        bench = ClusterBenchmark(run_id="csv_test")
        bench.start()
        result = bench.stop(
            total_documents=1, total_pages=100,
            scheduler_time_s=0.001,
            avg_cpu_percent=50.0, avg_ram_percent=40.0,
            load_balance_score=0.88,
        )
        csv_path = tmp_path / "benchmark.csv"
        out = ClusterBenchmark.save_csv(result, path=csv_path)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "run_id" in content
        assert "csv_test" in content

    def test_save_csv_append_mode(self, tmp_path: Path) -> None:
        bench1 = ClusterBenchmark(run_id="run_01")
        bench1.start()
        r1 = bench1.stop(
            total_documents=1, total_pages=50,
            scheduler_time_s=0.001,
            avg_cpu_percent=40.0, avg_ram_percent=35.0,
            load_balance_score=0.9,
        )
        bench2 = ClusterBenchmark(run_id="run_02")
        bench2.start()
        r2 = bench2.stop(
            total_documents=2, total_pages=100,
            scheduler_time_s=0.002,
            avg_cpu_percent=60.0, avg_ram_percent=50.0,
            load_balance_score=0.85,
        )
        csv_path = tmp_path / "multi.csv"
        ClusterBenchmark.save_csv(r1, path=csv_path, append=False)
        ClusterBenchmark.save_csv(r2, path=csv_path, append=True)
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        # header + 2 data rows
        assert len(lines) == 3


# ── Export: JSON ──────────────────────────────────────────────────────────

class TestJSONExport:

    def test_save_json_creates_valid_json(self, tmp_path: Path) -> None:
        bench = ClusterBenchmark(run_id="json_test")
        bench.start()
        result = bench.stop(
            total_documents=3, total_pages=300,
            scheduler_time_s=0.003,
            avg_cpu_percent=55.0, avg_ram_percent=45.0,
            load_balance_score=0.92,
        )
        json_path = tmp_path / "benchmark.json"
        ClusterBenchmark.save_json(result, path=json_path)
        assert json_path.exists()
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data["run_id"] == "json_test"
        assert "scheduler_overhead_percent" in data
        assert "cluster_efficiency" in data
