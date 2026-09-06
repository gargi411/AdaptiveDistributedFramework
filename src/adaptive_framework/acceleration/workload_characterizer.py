"""Workload Characterization -- Quantifies document and page processing requirements.

Produces structured WorkloadProfile descriptors consumed by AdaptiveWorkRouter.
Evaluates:
  - Page dimensions and area
  - Native text density vs rasterized image coverage
  - Image region counts
  - Scanned vs digital classification
  - Normalized workload complexity heuristic in [0.0, 1.0]

Strict Design Principles:
  - Systems-only optimization: Zero clinical data, patient identities, or medical
    terms are used as routing features or retained in profiles.
  - Ultra-low overhead: Deterministic heuristic executes in < 0.2 ms per page.
  - Extensible: Supports page-level, batch-level, and document-level characterization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)

# Standard standard letter/A4 area in PDF points (~612 x 792)
_STANDARD_PAGE_AREA_PTS: float = 612.0 * 792.0


@dataclass(frozen=True)
class WorkloadProfile:
    """Structured descriptor of a document or page processing workload.

    Attributes:
        document_id: Parent document identifier.
        page_number: 1-indexed page number (1 if document-level).
        page_count: Total pages in this workload unit.
        width_pts: Page width in PDF points.
        height_pts: Page height in PDF points.
        char_count: Extracted native text character count.
        image_count: Number of raster image objects embedded in the page.
        image_density: Fraction of page area covered by raster images [0.0, 1.0].
        is_scanned: True if the page lacks a native text layer and is rasterized.
        estimated_complexity: Normalized complexity score [0.0, 1.0].
        estimated_work_units: Estimated compute load in abstract work units.
        metadata: Optional non-sensitive systems metadata.
    """

    document_id: str
    page_number: int = 1
    page_count: int = 1
    width_pts: float = 612.0
    height_pts: float = 792.0
    char_count: int = 0
    image_count: int = 0
    image_density: float = 0.0
    is_scanned: bool = True
    estimated_complexity: float = 0.5
    estimated_work_units: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        d = asdict(self)
        d["estimated_complexity"] = round(self.estimated_complexity, 4)
        d["estimated_work_units"] = round(self.estimated_work_units, 2)
        return d

    def format_summary(self) -> str:
        """Format as a clean ASCII summary string."""
        kind = "SCANNED" if self.is_scanned else "DIGITAL"
        return (
            f"WorkloadProfile(doc='{self.document_id}', page={self.page_number}/{self.page_count}, "
            f"type={kind}, density={self.image_density:.2f}, "
            f"complexity={self.estimated_complexity:.2f}, work_units={self.estimated_work_units:.1f})"
        )


class WorkloadCharacterizer:
    """Lightweight deterministic workload characterization engine.

    Computes complexity based on geometry, image density, text layer presence,
    and region count without invoking heavy ML models.

    Usage:
        characterizer = WorkloadCharacterizer()
        profile = characterizer.characterize_page(fitz_page, page_number=1, document_id="doc_01")
        print(profile.estimated_complexity)
    """

    def __init__(
        self,
        min_text_chars: int = 20,
        scanned_density_threshold: float = 0.40,
    ) -> None:
        """Initialize the workload characterizer.

        Args:
            min_text_chars: Minimum characters to confirm a native text layer.
            scanned_density_threshold: Image coverage threshold indicating scanned content.
        """
        self._min_text_chars = min_text_chars
        self._scanned_threshold = scanned_density_threshold

    def characterize_page(
        self,
        fitz_page: Any,
        page_number: int,
        document_id: str,
        page_count: int = 1,
    ) -> WorkloadProfile:
        """Characterize an open PyMuPDF page object.

        Args:
            fitz_page: PyMuPDF Page instance.
            page_number: 1-indexed page number.
            document_id: Parent document identifier.
            page_count: Total pages in the containing document or batch.

        Returns:
            WorkloadProfile populated with measured page characteristics.
        """
        rect = fitz_page.rect
        width = float(rect.width) if rect.width > 0 else 612.0
        height = float(rect.height) if rect.height > 0 else 792.0
        page_area = width * height

        # 1. Native text character count
        try:
            text = fitz_page.get_text("text") or ""
            char_count = len(text.strip())
        except Exception:
            char_count = 0

        # 2. Embedded images and area density
        image_count = 0
        image_area = 0.0
        try:
            images = fitz_page.get_images(full=True)
            image_count = len(images)
            for img_info in images:
                try:
                    rects = fitz_page.get_image_rects(img_info[0])
                    for r in rects:
                        image_area += abs(r.width * r.height)
                except Exception:
                    image_area += page_area * 0.1
        except Exception:
            pass

        image_density = min(1.0, max(0.0, image_area / page_area if page_area > 0 else 0.0))
        is_scanned = (char_count < self._min_text_chars) and (image_density >= self._scanned_threshold or image_count > 0)

        complexity = self._compute_complexity(
            is_scanned=is_scanned,
            char_count=char_count,
            image_density=image_density,
            image_count=image_count,
            page_area=page_area,
        )
        work_units = self._compute_work_units(page_count, complexity)

        return WorkloadProfile(
            document_id=document_id,
            page_number=page_number,
            page_count=page_count,
            width_pts=round(width, 1),
            height_pts=round(height, 1),
            char_count=char_count,
            image_count=image_count,
            image_density=round(image_density, 4),
            is_scanned=is_scanned,
            estimated_complexity=round(complexity, 4),
            estimated_work_units=round(work_units, 2),
        )

    def characterize_synthetic(
        self,
        document_id: str = "doc_sim",
        page_number: int = 1,
        page_count: int = 1,
        width_pts: float = 612.0,
        height_pts: float = 792.0,
        char_count: int = 0,
        image_count: int = 1,
        image_density: float = 0.85,
        is_scanned: bool = True,
        complexity_override: float | None = None,
    ) -> WorkloadProfile:
        """Create a WorkloadProfile from parameters (for simulations and tests)."""
        page_area = width_pts * height_pts
        if complexity_override is not None:
            complexity = max(0.0, min(1.0, complexity_override))
        else:
            complexity = self._compute_complexity(
                is_scanned=is_scanned,
                char_count=char_count,
                image_density=image_density,
                image_count=image_count,
                page_area=page_area,
            )
        work_units = self._compute_work_units(page_count, complexity)

        return WorkloadProfile(
            document_id=document_id,
            page_number=page_number,
            page_count=page_count,
            width_pts=round(width_pts, 1),
            height_pts=round(height_pts, 1),
            char_count=char_count,
            image_count=image_count,
            image_density=round(image_density, 4),
            is_scanned=is_scanned,
            estimated_complexity=round(complexity, 4),
            estimated_work_units=round(work_units, 2),
        )

    # ------------------------------------------------------------------ #
    # Heuristic Formulation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_complexity(
        is_scanned: bool,
        char_count: int,
        image_density: float,
        image_count: int,
        page_area: float,
    ) -> float:
        """Compute normalized complexity heuristic in [0.0, 1.0].

        Formulation:
          - Digital pages: Base 0.05 + modest text volume term.
          - Scanned pages:
              Base: 0.35
              Image coverage: + 0.25 * image_density
              Oversized / high-res scan: + 0.15 * min(1.0, max(0.0, (area - std_area) / std_area))
              Multi-region fragmentation: + 0.15 * min(1.0, (image_count - 1) * 0.1)
        """
        if not is_scanned:
            # Digital text extraction is inherently lightweight
            text_scale = min(0.15, (char_count / 4000.0) * 0.15)
            return round(min(0.20, 0.05 + text_scale), 4)

        # Scanned page OCR requires significant compute
        base = 0.35
        coverage_term = 0.25 * min(1.0, max(0.0, image_density))

        # Oversized scans (e.g. 300+ DPI rasterization)
        area_ratio = max(0.0, (page_area - _STANDARD_PAGE_AREA_PTS) / _STANDARD_PAGE_AREA_PTS)
        size_term = 0.15 * min(1.0, area_ratio)

        # Multiple embedded image fragments / layout fragmentation
        regions_term = 0.15 * min(1.0, max(0, image_count - 1) * 0.1)

        raw = base + coverage_term + size_term + regions_term
        return round(max(0.10, min(1.0, raw)), 4)

    @staticmethod
    def _compute_work_units(page_count: int, complexity: float) -> float:
        """Calculate abstract work units scaling with page count and complexity."""
        # Baseline work unit per page: 1.0 + up to 2.5 additional units for high complexity
        per_page = 1.0 + (complexity * 2.5)
        return round(float(page_count) * per_page, 2)
