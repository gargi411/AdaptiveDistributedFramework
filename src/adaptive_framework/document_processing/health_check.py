"""System health check — validates all document processing dependencies.

Checks whether PyMuPDF, PaddleOCR, Docling, GPU/CUDA, and system memory
are available and reports a SystemHealthReport consumed by the dashboard.

Dashboard display:
    System Status
    ✓ PyMuPDF
    ✓ PaddleOCR    (or ✗ Not installed — using stub)
    ✓ Docling      (or ✗ Not installed — using stub)
    ✓ CUDA / GPU
    ✓ Memory ≥ 4 GB
    → Ready
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    """Health status for a single dependency component.

    Attributes:
        name: Human-readable component name.
        available: True if the component is usable.
        version: Version string if available. None otherwise.
        detail: Extra information (e.g. path, CUDA version).
        warning: Non-fatal warning message. None if no warning.
        error: Error message if not available. None if available.
    """

    name: str
    available: bool
    version: str | None = None
    detail: str | None = None
    warning: str | None = None
    error: str | None = None

    @property
    def status_icon(self) -> str:
        """Return ✓ if available, ✗ if not."""
        return "✓" if self.available else "✗"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "detail": self.detail,
            "warning": self.warning,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"ComponentHealth({self.status_icon} {self.name!r}, "
            f"version={self.version!r})"
        )


@dataclass
class SystemHealthReport:
    """Aggregated system health report for all document processing dependencies.

    Produced by HealthChecker.run_all_checks().
    Consumed by the dashboard 'System Status' panel.

    Attributes:
        python_version: Python version string.
        platform_info: OS and architecture string.
        pymupdf: PyMuPDF (fitz) health.
        paddleocr: PaddleOCR health.
        docling: Docling health.
        gpu: GPU / CUDA health.
        memory: System memory health.
        components: All component health objects.
        checked_at: ISO 8601 UTC timestamp.
        ready: True if critical components (PyMuPDF) are available.
    """

    python_version: str
    platform_info: str
    pymupdf: ComponentHealth
    paddleocr: ComponentHealth
    docling: ComponentHealth
    gpu: ComponentHealth
    memory: ComponentHealth
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def ready(self) -> bool:
        """Return True if critical dependencies are available.

        PyMuPDF is critical (PDF loading).
        PaddleOCR and Docling degrade gracefully to stubs.
        """
        return self.pymupdf.available

    @property
    def components(self) -> list[ComponentHealth]:
        """Return all component health objects as a list."""
        return [
            self.pymupdf,
            self.paddleocr,
            self.docling,
            self.gpu,
            self.memory,
        ]

    @property
    def warnings(self) -> list[str]:
        """Return all non-fatal warnings across components."""
        return [
            c.warning
            for c in self.components
            if c.warning is not None
        ]

    @property
    def errors(self) -> list[str]:
        """Return all error messages for unavailable components."""
        return [
            f"{c.name}: {c.error}"
            for c in self.components
            if not c.available and c.error
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary for dashboard JSON."""
        return {
            "ready": self.ready,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "checked_at": self.checked_at,
            "components": {c.name: c.to_dict() for c in self.components},
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        status = "READY" if self.ready else "NOT READY"
        return (
            f"SystemHealthReport({status}, "
            f"pymupdf={self.pymupdf.available}, "
            f"paddleocr={self.paddleocr.available}, "
            f"docling={self.docling.available}, "
            f"gpu={self.gpu.available})"
        )


class HealthChecker:
    """Checks availability and version of all document processing dependencies.

    Usage:
        >>> checker = HealthChecker()
        >>> report = checker.run_all_checks()
        >>> print(report.ready)
        True

    All checks are non-destructive and fast (< 500 ms total).
    """

    # Minimum required system memory in GB
    MIN_MEMORY_GB: float = 2.0

    def run_all_checks(self) -> SystemHealthReport:
        """Run all dependency checks and return a SystemHealthReport.

        Returns:
            SystemHealthReport with all component results.
        """
        logger.debug("Running system health checks.")
        return SystemHealthReport(
            python_version=sys.version,
            platform_info=f"{platform.system()} {platform.machine()} "
                          f"{platform.release()}",
            pymupdf=self._check_pymupdf(),
            paddleocr=self._check_paddleocr(),
            docling=self._check_docling(),
            gpu=self._check_gpu(),
            memory=self._check_memory(),
        )

    # ── Individual checks ────────────────────────────────────────────────────

    def _check_pymupdf(self) -> ComponentHealth:
        """Check PyMuPDF (fitz) availability."""
        try:
            import fitz  # type: ignore[import]
            version = getattr(fitz, "version", ("unknown",))[0]
            return ComponentHealth(
                name="PyMuPDF",
                available=True,
                version=str(version),
                detail="Zero-copy PDF loader ready.",
            )
        except ImportError as exc:
            return ComponentHealth(
                name="PyMuPDF",
                available=False,
                error=f"Import failed: {exc}. Install: pip install pymupdf",
            )

    def _check_paddleocr(self) -> ComponentHealth:
        """Check PaddleOCR availability (non-critical — has stub fallback)."""
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]  # noqa: F401
            import paddlepaddle  # type: ignore[import]
            version = getattr(paddlepaddle, "__version__", "unknown")
            return ComponentHealth(
                name="PaddleOCR",
                available=True,
                version=str(version),
                detail="OCR engine ready for scanned pages.",
            )
        except ImportError:
            return ComponentHealth(
                name="PaddleOCR",
                available=False,
                warning="PaddleOCR not installed. Scanned pages use stub (empty text).",
                detail="Install on Linux: pip install paddlepaddle paddleocr",
            )

    def _check_docling(self) -> ComponentHealth:
        """Check Docling availability (non-critical — has heuristic fallback)."""
        try:
            import docling  # type: ignore[import]  # noqa: F401
            version = getattr(docling, "__version__", "unknown")
            return ComponentHealth(
                name="Docling",
                available=True,
                version=str(version),
                detail="Layout analysis ready.",
            )
        except ImportError:
            return ComponentHealth(
                name="Docling",
                available=False,
                warning=(
                    "Docling not installed. "
                    "Layout analysis uses heuristic fallback."
                ),
                detail="Install: pip install docling",
            )

    def _check_gpu(self) -> ComponentHealth:
        """Check GPU and CUDA availability."""
        # Try PyTorch first (used by Docling)
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda or "unknown"
                return ComponentHealth(
                    name="GPU/CUDA",
                    available=True,
                    version=cuda_version,
                    detail=f"Device: {device_name}",
                )
            else:
                return ComponentHealth(
                    name="GPU/CUDA",
                    available=False,
                    warning="CUDA not available. OCR runs on CPU (slower).",
                    detail="CPU fallback will be used.",
                )
        except ImportError:
            pass

        # Try cupy or nvidia-smi as fallback
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split("\n")[0]
                return ComponentHealth(
                    name="GPU/CUDA",
                    available=True,
                    detail=f"nvidia-smi: {gpu_info}",
                )
        except Exception:
            pass

        return ComponentHealth(
            name="GPU/CUDA",
            available=False,
            warning="No GPU detected. Processing runs on CPU.",
        )

    def _check_memory(self) -> ComponentHealth:
        """Check available system memory."""
        try:
            import psutil
            ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
            ram_available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if ram_total_gb >= self.MIN_MEMORY_GB:
                return ComponentHealth(
                    name="Memory",
                    available=True,
                    detail=(
                        f"Total: {ram_total_gb:.1f} GB, "
                        f"Available: {ram_available_gb:.1f} GB"
                    ),
                    warning=(
                        "Available RAM < 1 GB — processing may be slow."
                        if ram_available_gb < 1.0
                        else None
                    ),
                )
            else:
                return ComponentHealth(
                    name="Memory",
                    available=False,
                    error=(
                        f"Total RAM {ram_total_gb:.1f} GB < "
                        f"{self.MIN_MEMORY_GB:.0f} GB minimum."
                    ),
                    detail=f"Total: {ram_total_gb:.1f} GB",
                )
        except ImportError:
            return ComponentHealth(
                name="Memory",
                available=True,
                warning="psutil not installed — memory check skipped.",
            )
