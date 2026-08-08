"""Unit tests for health check module — Batch 2."""

from __future__ import annotations

import pytest

from adaptive_framework.document_processing.health_check import (
    ComponentHealth,
    HealthChecker,
    SystemHealthReport,
)


class TestComponentHealth:
    def test_available_icon(self):
        ch = ComponentHealth(name="PyMuPDF", available=True)
        assert ch.status_icon == "✓"

    def test_unavailable_icon(self):
        ch = ComponentHealth(name="PaddleOCR", available=False, error="Not installed")
        assert ch.status_icon == "✗"

    def test_to_dict(self):
        ch = ComponentHealth(
            name="Docling",
            available=True,
            version="1.0.0",
            detail="Ready.",
        )
        d = ch.to_dict()
        assert d["name"] == "Docling"
        assert d["available"] is True
        assert d["version"] == "1.0.0"

    def test_repr(self):
        ch = ComponentHealth(name="GPU/CUDA", available=False)
        r = repr(ch)
        assert "GPU/CUDA" in r
        assert "✗" in r


class TestSystemHealthReport:
    def _make_report(
        self,
        pymupdf_available: bool = True,
        paddleocr_available: bool = False,
        docling_available: bool = False,
        gpu_available: bool = False,
        memory_available: bool = True,
    ) -> SystemHealthReport:
        return SystemHealthReport(
            python_version="3.11.0",
            platform_info="Windows AMD64",
            pymupdf=ComponentHealth(
                name="PyMuPDF", available=pymupdf_available
            ),
            paddleocr=ComponentHealth(
                name="PaddleOCR",
                available=paddleocr_available,
                warning="Not installed" if not paddleocr_available else None,
            ),
            docling=ComponentHealth(
                name="Docling",
                available=docling_available,
                warning="Not installed" if not docling_available else None,
            ),
            gpu=ComponentHealth(
                name="GPU/CUDA",
                available=gpu_available,
                warning="No GPU" if not gpu_available else None,
            ),
            memory=ComponentHealth(
                name="Memory", available=memory_available
            ),
        )

    def test_ready_when_pymupdf_available(self):
        report = self._make_report(pymupdf_available=True)
        assert report.ready

    def test_not_ready_when_pymupdf_missing(self):
        report = self._make_report(pymupdf_available=False)
        assert not report.ready

    def test_components_list_length(self):
        report = self._make_report()
        assert len(report.components) == 5

    def test_warnings_collected(self):
        report = self._make_report(
            paddleocr_available=False,
            docling_available=False,
        )
        assert len(report.warnings) >= 2

    def test_errors_when_pymupdf_missing(self):
        report = self._make_report(pymupdf_available=False)
        # Add error to pymupdf component
        report.pymupdf.error = "Import failed"
        errors = report.errors
        assert any("PyMuPDF" in e for e in errors)

    def test_to_dict(self):
        report = self._make_report()
        d = report.to_dict()
        assert "ready" in d
        assert "components" in d
        assert "PyMuPDF" in d["components"]

    def test_repr(self):
        report = self._make_report()
        r = repr(report)
        assert "SystemHealthReport" in r
        assert "READY" in r


class TestHealthChecker:
    def test_run_all_checks_returns_report(self):
        checker = HealthChecker()
        report = checker.run_all_checks()
        assert isinstance(report, SystemHealthReport)
        assert len(report.components) == 5

    def test_report_has_python_version(self):
        checker = HealthChecker()
        report = checker.run_all_checks()
        assert "3." in report.python_version

    def test_pymupdf_check_type(self):
        checker = HealthChecker()
        ch = checker._check_pymupdf()
        assert isinstance(ch, ComponentHealth)
        assert ch.name == "PyMuPDF"

    def test_memory_check_returns_component(self):
        checker = HealthChecker()
        ch = checker._check_memory()
        assert isinstance(ch, ComponentHealth)
        assert ch.name == "Memory"

    def test_paddleocr_check_does_not_crash(self):
        """PaddleOCR may not be installed — must return ComponentHealth, not raise."""
        checker = HealthChecker()
        ch = checker._check_paddleocr()
        assert isinstance(ch, ComponentHealth)

    def test_gpu_check_does_not_crash(self):
        """GPU check must not raise even if no GPU or torch installed."""
        checker = HealthChecker()
        ch = checker._check_gpu()
        assert isinstance(ch, ComponentHealth)

    def test_docling_check_does_not_crash(self):
        """Docling may not be installed — must return ComponentHealth."""
        checker = HealthChecker()
        ch = checker._check_docling()
        assert isinstance(ch, ComponentHealth)
