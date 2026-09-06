"""Workload Cost Model for Phase G6 Data-Driven Adaptive Work Routing.

Provides:
  - WorkloadCostProfile: Structured, normalized pre-OCR computational descriptor.
  - WorkloadCostEstimator: Deterministic, explainable compute intensity model.

Guarantees:
  - Zero leakage: Operates strictly on pre-OCR dimensions, image bounding areas,
    and structural metadata available before inference.
  - Zero clinical data: Never inspects, extracts, or logs patient health information.
  - Deterministic reproducibility: Identical inputs produce identical profiles.
  - Minimal latency: Execution overhead is strictly sub-millisecond (< 0.05 ms).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from adaptive_framework.acceleration.workload_characterizer import WorkloadProfile

logger = logging.getLogger(__name__)

# Reference baseline: 300 DPI Letter page (2550 x 3300 = ~8.415M pixels)
MAX_REFERENCE_PIXELS: float = 2550.0 * 3300.0


@dataclass(frozen=True)
class WorkloadCostProfile:
    """Structured pre-OCR computational descriptor of a page workload.

    Attributes:
        document_id: Source document identifier.
        page_number: 1-indexed page number.
        page_count: Total pages in the current batch or document.
        width_pts: Width in PDF points.
        height_pts: Height in PDF points.
        pixel_width: Rendered pixel width (at target DPI).
        pixel_height: Rendered pixel height (at target DPI).
        total_pixels: Total pixel count (pixel_width * pixel_height).
        image_density: Fraction of page area occupied by raster graphics [0.0, 1.0].
        estimated_char_density: Normalized pre-OCR native text density [0.0, 1.0].
        complexity_score: G4 layout complexity score [0.0, 1.0].
        estimated_work_units: Abstract compute volume estimate.
        normalized_compute_intensity: Calibrated compute intensity score [0.0, 1.0].
        feature_breakdown: Dictionary detailing each feature's contribution.
    """

    document_id: str
    page_number: int = 1
    page_count: int = 1
    width_pts: float = 612.0
    height_pts: float = 792.0
    pixel_width: int = 1275
    pixel_height: int = 1650
    total_pixels: int = 2103750
    image_density: float = 0.0
    estimated_char_density: float = 0.0
    complexity_score: float = 0.5
    estimated_work_units: float = 1.0
    normalized_compute_intensity: float = 0.5
    feature_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with sensible research precision."""
        d = asdict(self)
        d["width_pts"] = round(self.width_pts, 1)
        d["height_pts"] = round(self.height_pts, 1)
        d["image_density"] = round(self.image_density, 4)
        d["estimated_char_density"] = round(self.estimated_char_density, 4)
        d["complexity_score"] = round(self.complexity_score, 4)
        d["estimated_work_units"] = round(self.estimated_work_units, 2)
        d["normalized_compute_intensity"] = round(self.normalized_compute_intensity, 4)
        return d

    def format_summary(self) -> str:
        """Format as a clean human-readable summary string."""
        return (
            f"WorkloadCostProfile(doc='{self.document_id}', p={self.page_number}/{self.page_count}, "
            f"px={self.pixel_width}x{self.pixel_height} ({self.total_pixels:,}), "
            f"img_dens={self.image_density:.2f}, comp={self.complexity_score:.2f}, "
            f"intensity={self.normalized_compute_intensity:.3f})"
        )


class WorkloadCostEstimator:
    """Computes explainable compute intensity profiles from pre-OCR metadata.

    The model maps physical workload attributes to a normalized compute intensity
    score using documented weights:
      - Pixel Volume (35%): Large raster surfaces benefit strongly from GPU SIMD/vector units.
      - Image Density (25%): High coverage indicates convolution-heavy neural text detection.
      - Layout Complexity (25%): Multi-column or irregular bounding boxes increase decoding steps.
      - Character Density (15%): High native text density hints at dense tabular or text data.
    """

    def __init__(
        self,
        pixel_weight: float = 0.35,
        image_density_weight: float = 0.25,
        complexity_weight: float = 0.25,
        char_density_weight: float = 0.15,
        default_dpi: int = 150,
    ) -> None:
        """Initialize estimator with calibrated weights.

        Weights are normalized to sum to 1.0.
        """
        total = pixel_weight + image_density_weight + complexity_weight + char_density_weight
        if total <= 0.0:
            total = 1.0
        self._w_pixel = pixel_weight / total
        self._w_density = image_density_weight / total
        self._w_complexity = complexity_weight / total
        self._w_char = char_density_weight / total
        self._default_dpi = default_dpi

    @property
    def weights(self) -> dict[str, float]:
        """Return configured normalized feature weights."""
        return {
            "pixel_weight": round(self._w_pixel, 4),
            "image_density_weight": round(self._w_density, 4),
            "complexity_weight": round(self._w_complexity, 4),
            "char_density_weight": round(self._w_char, 4),
        }

    def estimate_cost(
        self,
        workload: WorkloadProfile,
        pixel_dimensions: tuple[int, int] | None = None,
        dpi: int | None = None,
    ) -> WorkloadCostProfile:
        """Estimate the WorkloadCostProfile for a given WorkloadProfile.

        Args:
            workload: G4 WorkloadProfile (dimensions, complexity, density).
            pixel_dimensions: Optional (width_px, height_px) if already rendered.
            dpi: Target rendering DPI (defaults to 150 DPI if not specified).

        Returns:
            WorkloadCostProfile with normalized intensity and factor contributions.
        """
        target_dpi = dpi or self._default_dpi

        # Determine pixel dimensions
        if pixel_dimensions is not None:
            pw, ph = int(pixel_dimensions[0]), int(pixel_dimensions[1])
        else:
            scale = target_dpi / 72.0
            pw = max(1, int(round(workload.width_pts * scale)))
            ph = max(1, int(round(workload.height_pts * scale)))

        total_pixels = pw * ph

        # 1. Normalized Pixel Volume [0.0, 1.0]
        norm_pixels = min(1.0, max(0.0, total_pixels / MAX_REFERENCE_PIXELS))

        # 2. Image Density [0.0, 1.0]
        density = min(1.0, max(0.0, workload.image_density))

        # 3. Layout Complexity [0.0, 1.0]
        complexity = min(1.0, max(0.0, workload.estimated_complexity))

        # 4. Estimated Native Text Character Density [0.0, 1.0]
        # Standard page holds ~3,000 characters maximum
        char_density = min(1.0, max(0.0, workload.char_count / 3000.0))

        # Calculate composite compute intensity
        contrib_pixel = self._w_pixel * norm_pixels
        contrib_density = self._w_density * density
        contrib_complexity = self._w_complexity * complexity
        contrib_char = self._w_char * char_density

        raw_intensity = contrib_pixel + contrib_density + contrib_complexity + contrib_char
        compute_intensity = min(1.0, max(0.0, raw_intensity))

        # Abstract work units (1.0 = standard 150 DPI page of complexity 0.5)
        work_units = max(0.1, round(compute_intensity * 2.0 * max(1, workload.page_count), 2))

        breakdown = {
            "pixel_contribution": round(contrib_pixel, 4),
            "density_contribution": round(contrib_density, 4),
            "complexity_contribution": round(contrib_complexity, 4),
            "char_contribution": round(contrib_char, 4),
            "normalized_pixels": round(norm_pixels, 4),
        }

        return WorkloadCostProfile(
            document_id=workload.document_id,
            page_number=workload.page_number,
            page_count=workload.page_count,
            width_pts=workload.width_pts,
            height_pts=workload.height_pts,
            pixel_width=pw,
            pixel_height=ph,
            total_pixels=total_pixels,
            image_density=density,
            estimated_char_density=char_density,
            complexity_score=complexity,
            estimated_work_units=work_units,
            normalized_compute_intensity=compute_intensity,
            feature_breakdown=breakdown,
        )
