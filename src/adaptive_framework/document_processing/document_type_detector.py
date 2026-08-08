"""Document Type Detector — Module 2: Per-page page type classification.

Every page is classified independently:
    digital  — native text layer, few/no raster images
    scanned  — rasterised image, no text layer
    mixed    — combination of text layer and significant raster content

Architecture rule:
    Classification is ALWAYS per-page.
    Never classify the entire document as one type.
    A single PDF may contain digital pages, scanned pages, and mixed pages.

The result feeds ProcessingStrategyFactory to select the correct strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_framework.models.page import PageType

logger = logging.getLogger(__name__)

# Thresholds for classification heuristics
_MIN_TEXT_CHARS_DIGITAL: int = 20     # chars needed to call a page "digital"
_IMAGE_DENSITY_SCANNED_THRESHOLD: float = 0.40  # > 40% coverage = scanned signal
_IMAGE_DENSITY_MIXED_THRESHOLD: float = 0.15    # > 15% with text = mixed


@dataclass(frozen=True)
class PageClassification:
    """Classification result for a single page.

    Attributes:
        page_number: 1-indexed page number.
        page_type: Detected type (DIGITAL, SCANNED, MIXED, UNKNOWN).
        char_count: Number of characters found in the text layer.
        image_density: Fraction of page area covered by raster images.
        has_text_layer: True if text was extractable from the page.
        confidence: Classification confidence [0.0, 1.0].
        reason: Human-readable explanation of the classification decision.
    """

    page_number: int
    page_type: PageType
    char_count: int
    image_density: float
    has_text_layer: bool
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "page_number": self.page_number,
            "page_type": self.page_type.value,
            "char_count": self.char_count,
            "image_density": self.image_density,
            "has_text_layer": self.has_text_layer,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return (
            f"PageClassification(page={self.page_number}, "
            f"type={self.page_type.value}, "
            f"confidence={self.confidence:.2f})"
        )


@dataclass
class DocumentClassificationSummary:
    """Classification summary for all pages in a document.

    Attributes:
        document_id: Parent document identifier.
        page_classifications: Per-page classification results.
        digital_count: Number of digital pages.
        scanned_count: Number of scanned pages.
        mixed_count: Number of mixed pages.
        unknown_count: Number of unclassified pages.
        dominant_type: Most common page type.
    """

    document_id: str
    page_classifications: list[PageClassification] = field(default_factory=list)

    @property
    def digital_count(self) -> int:
        """Count of DIGITAL pages."""
        return sum(1 for p in self.page_classifications if p.page_type == PageType.DIGITAL)

    @property
    def scanned_count(self) -> int:
        """Count of SCANNED pages."""
        return sum(1 for p in self.page_classifications if p.page_type == PageType.SCANNED)

    @property
    def mixed_count(self) -> int:
        """Count of MIXED pages."""
        return sum(1 for p in self.page_classifications if p.page_type == PageType.MIXED)

    @property
    def unknown_count(self) -> int:
        """Count of UNKNOWN pages."""
        return sum(1 for p in self.page_classifications if p.page_type == PageType.UNKNOWN)

    @property
    def dominant_type(self) -> PageType:
        """Most common page type across the document."""
        counts = {
            PageType.DIGITAL: self.digital_count,
            PageType.SCANNED: self.scanned_count,
            PageType.MIXED: self.mixed_count,
            PageType.UNKNOWN: self.unknown_count,
        }
        return max(counts, key=lambda k: counts[k])

    def get_page(self, page_number: int) -> PageClassification | None:
        """Return the classification for a specific page number.

        Args:
            page_number: 1-indexed page number.

        Returns:
            PageClassification or None if not found.
        """
        for pc in self.page_classifications:
            if pc.page_number == page_number:
                return pc
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "document_id": self.document_id,
            "total_pages": len(self.page_classifications),
            "digital_count": self.digital_count,
            "scanned_count": self.scanned_count,
            "mixed_count": self.mixed_count,
            "unknown_count": self.unknown_count,
            "dominant_type": self.dominant_type.value,
            "page_classifications": [pc.to_dict() for pc in self.page_classifications],
        }


class DocumentTypeDetector:
    """Classifies every page of a PDF independently.

    Classification algorithm (per page):
        1. Extract text from the native text layer (PyMuPDF get_text).
        2. Count characters in the extracted text.
        3. Estimate image density (raster image area / page area).
        4. Apply thresholds:
            - char_count >= MIN_TEXT_CHARS AND density < MIXED_THRESHOLD → DIGITAL
            - char_count >= MIN_TEXT_CHARS AND density >= MIXED_THRESHOLD → MIXED
            - char_count < MIN_TEXT_CHARS AND density >= SCANNED_THRESHOLD → SCANNED
            - otherwise → UNKNOWN

    Args:
        min_text_chars: Minimum characters to consider a text layer present.
        scanned_density_threshold: Image density above which scanned is assumed.
        mixed_density_threshold: Image density above which mixed is assumed.
    """

    def __init__(
        self,
        min_text_chars: int = _MIN_TEXT_CHARS_DIGITAL,
        scanned_density_threshold: float = _IMAGE_DENSITY_SCANNED_THRESHOLD,
        mixed_density_threshold: float = _IMAGE_DENSITY_MIXED_THRESHOLD,
    ) -> None:
        self._min_text_chars = min_text_chars
        self._scanned_threshold = scanned_density_threshold
        self._mixed_threshold = mixed_density_threshold
        self._fitz_available = self._check_fitz()

    def _check_fitz(self) -> bool:
        """Return True if PyMuPDF is importable."""
        try:
            import fitz  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def classify_document(
        self,
        file_path: str,
        document_id: str,
    ) -> DocumentClassificationSummary:
        """Classify every page in a PDF document.

        Args:
            file_path: Absolute path to the PDF file.
            document_id: Parent document identifier.

        Returns:
            DocumentClassificationSummary with per-page results.
        """
        summary = DocumentClassificationSummary(document_id=document_id)

        if not self._fitz_available:
            logger.warning(
                "PyMuPDF not available. All pages classified as UNKNOWN."
            )
            return summary

        try:
            import fitz  # type: ignore[import]
            doc = fitz.open(file_path)
            try:
                for i in range(len(doc)):
                    classification = self._classify_page(doc[i], i + 1)
                    summary.page_classifications.append(classification)
            finally:
                doc.close()
        except Exception as exc:
            logger.error("Failed to classify document '%s': %s", file_path, exc)

        return summary

    def classify_single_page(
        self,
        file_path: str,
        page_number: int,
    ) -> PageClassification:
        """Classify a single page by number (1-indexed).

        Args:
            file_path: Absolute path to the PDF file.
            page_number: 1-indexed page number to classify.

        Returns:
            PageClassification for the requested page.
        """
        if not self._fitz_available:
            return self._unknown_classification(page_number, "PyMuPDF not available.")

        try:
            import fitz  # type: ignore[import]
            doc = fitz.open(file_path)
            try:
                idx = page_number - 1
                if idx < 0 or idx >= len(doc):
                    return self._unknown_classification(
                        page_number,
                        f"Page {page_number} out of range (doc has {len(doc)} pages).",
                    )
                return self._classify_page(doc[idx], page_number)
            finally:
                doc.close()
        except Exception as exc:
            return self._unknown_classification(page_number, str(exc))

    def classify_page_from_fitz(self, page: Any, page_number: int) -> PageClassification:
        """Classify a PyMuPDF page object directly (zero re-open overhead).

        Args:
            page: PyMuPDF Page object (already opened).
            page_number: 1-indexed page number.

        Returns:
            PageClassification for this page.
        """
        return self._classify_page(page, page_number)

    # ── Internal classification logic ────────────────────────────────────────

    def _classify_page(self, page: Any, page_number: int) -> PageClassification:
        """Apply classification heuristics to a single PyMuPDF page.

        Args:
            page: PyMuPDF Page object.
            page_number: 1-indexed page number.

        Returns:
            PageClassification with type, confidence, and reason.
        """
        # Extract text
        text = page.get_text("text") or ""
        char_count = len(text.strip())
        has_text_layer = char_count >= self._min_text_chars

        # Estimate image density
        image_density = self._compute_image_density(page)

        # Classify
        page_type, confidence, reason = self._apply_rules(
            has_text_layer=has_text_layer,
            char_count=char_count,
            image_density=image_density,
        )

        return PageClassification(
            page_number=page_number,
            page_type=page_type,
            char_count=char_count,
            image_density=round(image_density, 4),
            has_text_layer=has_text_layer,
            confidence=round(confidence, 3),
            reason=reason,
        )

    def _apply_rules(
        self,
        has_text_layer: bool,
        char_count: int,
        image_density: float,
    ) -> tuple[PageType, float, str]:
        """Apply classification rules and return (type, confidence, reason).

        Args:
            has_text_layer: Whether text was found.
            char_count: Number of characters found.
            image_density: Fraction of page covered by images.

        Returns:
            Tuple of (PageType, confidence, reason string).
        """
        if has_text_layer and image_density < self._mixed_threshold:
            confidence = min(1.0, 0.7 + (char_count / 1000) * 0.3)
            return (
                PageType.DIGITAL,
                confidence,
                f"Text layer ({char_count} chars), "
                f"low image density ({image_density:.2f})",
            )

        if has_text_layer and image_density >= self._mixed_threshold:
            return (
                PageType.MIXED,
                0.80,
                f"Text layer ({char_count} chars) with "
                f"significant images ({image_density:.2f})",
            )

        if not has_text_layer and image_density >= self._scanned_threshold:
            confidence = min(1.0, 0.65 + image_density * 0.35)
            return (
                PageType.SCANNED,
                confidence,
                f"No text layer, high image density ({image_density:.2f})",
            )

        # Neither significant text nor significant images
        return (
            PageType.UNKNOWN,
            0.50,
            f"Inconclusive: {char_count} chars, density={image_density:.2f}",
        )

    @staticmethod
    def _compute_image_density(page: Any) -> float:
        """Compute fraction of page area covered by raster images.

        Args:
            page: PyMuPDF Page object.

        Returns:
            Density in [0.0, 1.0].
        """
        rect = page.rect
        if rect.width <= 0 or rect.height <= 0:
            return 0.0

        page_area = rect.width * rect.height
        image_area = 0.0

        try:
            images = page.get_images(full=True)
            for img_info in images:
                try:
                    rects = page.get_image_rects(img_info[0])
                    for r in rects:
                        image_area += abs(r.width * r.height)
                except Exception:
                    image_area += page_area * 0.05
        except Exception:
            pass

        return min(1.0, image_area / page_area)

    @staticmethod
    def _unknown_classification(page_number: int, reason: str) -> PageClassification:
        """Create an UNKNOWN classification record.

        Args:
            page_number: Page number.
            reason: Why the page could not be classified.

        Returns:
            PageClassification with type=UNKNOWN.
        """
        return PageClassification(
            page_number=page_number,
            page_type=PageType.UNKNOWN,
            char_count=0,
            image_density=0.0,
            has_text_layer=False,
            confidence=0.0,
            reason=reason,
        )
