"""PDF Analyzer — Module 1: Document Processing Engine.

Validates PDFs and extracts structural metadata before scheduling begins.
This information feeds the Adaptive Scheduler via PDFMetadata.

Responsibilities:
    - PDF validation (corruption, encryption, page count)
    - Metadata extraction (author, title, creation date)
    - Page dimension extraction
    - Font inventory
    - Image density estimation (scanned vs. digital signal)
    - PDF statistics summary

Uses PyMuPDF (fitz). Falls back gracefully if the file is unreadable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PageDimensions:
    """Width and height of a single page in PDF points.

    Attributes:
        page_number: 1-indexed page number.
        width_pts: Width in PDF points (1 pt = 1/72 inch).
        height_pts: Height in PDF points.
        rotation: Page rotation in degrees (0, 90, 180, 270).
    """

    page_number: int
    width_pts: float
    height_pts: float
    rotation: int = 0

    @property
    def is_landscape(self) -> bool:
        """Return True if the page is wider than it is tall."""
        return self.width_pts > self.height_pts

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "page_number": self.page_number,
            "width_pts": self.width_pts,
            "height_pts": self.height_pts,
            "rotation": self.rotation,
            "is_landscape": self.is_landscape,
        }


@dataclass
class PDFAnalysisResult:
    """Complete analysis result for a single PDF file.

    Attributes:
        file_path: Absolute path to the analysed PDF.
        is_valid: True if the file is a readable, non-corrupt PDF.
        is_encrypted: True if the PDF is encrypted / password-protected.
        page_count: Number of pages (0 if file could not be opened).
        file_size_bytes: Size of the PDF file in bytes.
        title: Document title from PDF metadata. None if not set.
        author: Document author from PDF metadata. None if not set.
        subject: Document subject from PDF metadata. None if not set.
        creator: PDF creator application. None if not set.
        creation_date: Creation date string from PDF metadata. None if not set.
        pdf_version: PDF format version string (e.g. '1.7').
        page_dimensions: Per-page width/height records.
        fonts: Set of font names used in the document.
        estimated_image_density: Mean fraction of page area covered by images.
            Values near 1.0 suggest a scanned document.
        has_text_layer: True if at least one page has a native text layer.
        error_message: Error description if is_valid is False.
        warnings: Non-fatal warnings accumulated during analysis.
    """

    file_path: str
    is_valid: bool
    is_encrypted: bool = False
    page_count: int = 0
    file_size_bytes: int = 0
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    creation_date: str | None = None
    pdf_version: str | None = None
    page_dimensions: list[PageDimensions] = field(default_factory=list)
    fonts: set[str] = field(default_factory=set)
    estimated_image_density: float = 0.0
    has_text_layer: bool = False
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def estimated_size_mb(self) -> float:
        """File size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)

    @property
    def likely_scanned(self) -> bool:
        """Heuristic: True if image density > 0.5 and no text layer."""
        return self.estimated_image_density > 0.5 and not self.has_text_layer

    @property
    def likely_digital(self) -> bool:
        """Heuristic: True if text layer present and image density < 0.3."""
        return self.has_text_layer and self.estimated_image_density < 0.3

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "file_path": self.file_path,
            "is_valid": self.is_valid,
            "is_encrypted": self.is_encrypted,
            "page_count": self.page_count,
            "file_size_bytes": self.file_size_bytes,
            "estimated_size_mb": round(self.estimated_size_mb, 3),
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "creator": self.creator,
            "creation_date": self.creation_date,
            "pdf_version": self.pdf_version,
            "fonts": sorted(self.fonts),
            "estimated_image_density": self.estimated_image_density,
            "has_text_layer": self.has_text_layer,
            "likely_scanned": self.likely_scanned,
            "likely_digital": self.likely_digital,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "page_dimensions": [d.to_dict() for d in self.page_dimensions],
        }

    def __repr__(self) -> str:
        return (
            f"PDFAnalysisResult(path='{Path(self.file_path).name}', "
            f"valid={self.is_valid}, "
            f"pages={self.page_count}, "
            f"encrypted={self.is_encrypted})"
        )


class PDFAnalyzer:
    """Analyses a PDF file and returns a PDFAnalysisResult.

    Requires PyMuPDF (fitz). If PyMuPDF is not installed, all analysis
    returns a PDFAnalysisResult with is_valid=False and an error message.

    Usage:
        >>> analyzer = PDFAnalyzer()
        >>> result = analyzer.analyse("/data/paper.pdf")
        >>> result.page_count
        42

    Args:
        max_pages_for_density: Maximum pages to sample for image density
            estimation. Defaults to 10 (first and last pages).
        min_text_chars_for_text_layer: Minimum characters to consider a
            page as having a text layer. Defaults to 10.
    """

    def __init__(
        self,
        max_pages_for_density: int = 10,
        min_text_chars_for_text_layer: int = 10,
    ) -> None:
        self._max_pages_for_density = max_pages_for_density
        self._min_text_chars = min_text_chars_for_text_layer
        self._fitz_available = self._check_fitz()

    def _check_fitz(self) -> bool:
        """Return True if PyMuPDF is importable."""
        try:
            import fitz  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def analyse(self, file_path: str) -> PDFAnalysisResult:
        """Analyse a PDF file and return a complete analysis result.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            PDFAnalysisResult — is_valid=False if file cannot be opened.
        """
        path = Path(file_path)

        if not self._fitz_available:
            return PDFAnalysisResult(
                file_path=file_path,
                is_valid=False,
                error_message="PyMuPDF not installed. Cannot analyse PDF.",
            )

        if not path.exists():
            return PDFAnalysisResult(
                file_path=file_path,
                is_valid=False,
                error_message=f"File not found: {file_path}",
            )

        if not path.is_file():
            return PDFAnalysisResult(
                file_path=file_path,
                is_valid=False,
                error_message=f"Path is not a file: {file_path}",
            )

        file_size_bytes = path.stat().st_size

        try:
            import fitz  # type: ignore[import]
            return self._analyse_with_fitz(
                file_path, file_size_bytes, fitz
            )
        except Exception as exc:
            logger.warning("PDF analysis failed for '%s': %s", file_path, exc)
            return PDFAnalysisResult(
                file_path=file_path,
                is_valid=False,
                file_size_bytes=file_size_bytes,
                error_message=f"Analysis failed: {exc}",
            )

    def _analyse_with_fitz(
        self,
        file_path: str,
        file_size_bytes: int,
        fitz: Any,
    ) -> PDFAnalysisResult:
        """Internal: perform analysis using PyMuPDF.

        Args:
            file_path: Path to the PDF.
            file_size_bytes: Pre-computed file size.
            fitz: The imported fitz module.

        Returns:
            Fully populated PDFAnalysisResult.
        """
        warnings: list[str] = []
        doc = None

        try:
            doc = fitz.open(file_path)

            # Encryption check
            if doc.needs_pass:
                return PDFAnalysisResult(
                    file_path=file_path,
                    is_valid=False,
                    is_encrypted=True,
                    file_size_bytes=file_size_bytes,
                    page_count=len(doc),
                    error_message="PDF is encrypted. Cannot process without password.",
                )

            page_count = len(doc)
            if page_count == 0:
                warnings.append("PDF has 0 pages.")

            # PDF metadata
            meta = doc.metadata or {}
            pdf_version = getattr(doc, "pdf_version", None)

            # Per-page analysis
            page_dimensions: list[PageDimensions] = []
            fonts: set[str] = set()
            has_text_layer = False
            density_samples: list[float] = []

            sample_indices = self._sample_indices(page_count)

            for i in range(page_count):
                page = doc[i]
                rect = page.rect
                page_dimensions.append(
                    PageDimensions(
                        page_number=i + 1,
                        width_pts=rect.width,
                        height_pts=rect.height,
                        rotation=page.rotation,
                    )
                )

                # Fonts
                font_list = page.get_fonts(full=False)
                for font_info in font_list:
                    if len(font_info) > 3 and font_info[3]:
                        fonts.add(str(font_info[3]))

                # Text layer check
                if not has_text_layer:
                    text = page.get_text("text")
                    if len(text.strip()) >= self._min_text_chars:
                        has_text_layer = True

                # Image density sampling
                if i in sample_indices:
                    density = self._estimate_image_density(page, rect)
                    density_samples.append(density)

            estimated_image_density = (
                sum(density_samples) / len(density_samples)
                if density_samples else 0.0
            )

            if page_count > 500:
                warnings.append(
                    f"Large document ({page_count} pages) may require "
                    "significant processing time."
                )

            return PDFAnalysisResult(
                file_path=file_path,
                is_valid=True,
                is_encrypted=False,
                page_count=page_count,
                file_size_bytes=file_size_bytes,
                title=meta.get("title") or None,
                author=meta.get("author") or None,
                subject=meta.get("subject") or None,
                creator=meta.get("creator") or None,
                creation_date=meta.get("creationDate") or None,
                pdf_version=str(pdf_version) if pdf_version else None,
                page_dimensions=page_dimensions,
                fonts=fonts,
                estimated_image_density=round(estimated_image_density, 4),
                has_text_layer=has_text_layer,
                warnings=warnings,
            )

        finally:
            if doc is not None:
                doc.close()

    def _sample_indices(self, page_count: int) -> set[int]:
        """Compute page indices to sample for image density estimation.

        Samples up to max_pages_for_density pages spread across the document.

        Args:
            page_count: Total number of pages.

        Returns:
            Set of 0-indexed page indices to sample.
        """
        if page_count == 0:
            return set()
        if page_count <= self._max_pages_for_density:
            return set(range(page_count))
        step = max(1, page_count // self._max_pages_for_density)
        return set(range(0, page_count, step))

    @staticmethod
    def _estimate_image_density(page: Any, rect: Any) -> float:
        """Estimate the fraction of page area covered by raster images.

        Args:
            page: PyMuPDF Page object.
            rect: Page rect (bounds).

        Returns:
            Image density in [0.0, 1.0]. 0 = no images, 1 = fully covered.
        """
        if rect.width <= 0 or rect.height <= 0:
            return 0.0

        page_area = rect.width * rect.height
        image_area = 0.0

        image_list = page.get_images(full=True)
        for img_info in image_list:
            try:
                rects = page.get_image_rects(img_info[0])
                for r in rects:
                    image_area += abs(r.width * r.height)
            except Exception:
                # Some images may not have extractable rects
                image_area += page_area * 0.1  # conservative estimate

        return min(1.0, image_area / page_area)
