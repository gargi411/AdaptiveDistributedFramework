"""Image & Figure Extractor — Module 8: Figure detection and extraction.

Extracts biomedical images, charts, graphs, figures from PDF pages.
Stores: bounding box, resolution, page number, caption, image path.

Strategy:
    1. PyMuPDF get_images() for embedded raster images
    2. Block-type analysis for vector figures
    3. Caption detection via proximity to image blocks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_framework.models.page import BoundingBox, FigureData

logger = logging.getLogger(__name__)

# Figure type classification keywords
_BIOMEDICAL_KEYWORDS = frozenset({
    "fig", "figure", "chart", "graph", "diagram", "plot",
    "image", "scan", "mri", "ct", "xray", "histology",
})


@dataclass
class FigureExtractionResult:
    """Result of figure/image extraction for a single page.

    Attributes:
        page_number: 1-indexed page number.
        figures: List of extracted FigureData objects.
        extraction_time_seconds: Wall-clock time.
        success: True if extraction completed without error.
        error: Error message if success is False.
    """

    page_number: int
    figures: list[FigureData] = field(default_factory=list)
    extraction_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None

    @property
    def count(self) -> int:
        """Number of figures extracted."""
        return len(self.figures)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "page_number": self.page_number,
            "figure_count": self.count,
            "extraction_time_seconds": self.extraction_time_seconds,
            "success": self.success,
            "error": self.error,
        }


class FigureExtractor:
    """Extracts figures and images from PDF page objects.

    Detects embedded raster images using PyMuPDF get_images().
    Attempts to associate captions using proximity analysis.

    Args:
        output_dir: Directory to save extracted images.
                    If None, images are not saved to disk.
        min_width_px: Minimum image width in pixels to extract.
        min_height_px: Minimum image height in pixels to extract.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        min_width_px: int = 50,
        min_height_px: int = 50,
    ) -> None:
        self._output_dir = Path(output_dir) if output_dir else None
        self._min_width = min_width_px
        self._min_height = min_height_px

    def extract_page(
        self,
        page: Any,
        page_number: int,
        text_blocks: list[Any] | None = None,
    ) -> FigureExtractionResult:
        """Extract all figures from a single PyMuPDF page.

        Args:
            page: fitz.Page object.
            page_number: 1-indexed page number.
            text_blocks: Text blocks for caption association. Optional.

        Returns:
            FigureExtractionResult with zero or more FigureData objects.
        """
        import time
        t0 = time.perf_counter()
        result = FigureExtractionResult(page_number=page_number)

        try:
            figures = self._extract_images(page, page_number, text_blocks or [])
            result.figures = figures
        except Exception as exc:
            logger.warning(
                "Figure extraction failed for page %d: %s", page_number, exc
            )
            result.success = False
            result.error = str(exc)

        result.extraction_time_seconds = time.perf_counter() - t0
        return result

    def _extract_images(
        self,
        page: Any,
        page_number: int,
        text_blocks: list[Any],
    ) -> list[FigureData]:
        """Extract embedded raster images from a page.

        Args:
            page: fitz.Page object.
            page_number: 1-indexed page number.
            text_blocks: Text blocks for caption detection.

        Returns:
            List of FigureData objects.
        """
        figures: list[FigureData] = []
        doc = page.parent

        image_list = page.get_images(full=True)
        for img_info in image_list:
            try:
                xref = img_info[0]
                img_rects = page.get_image_rects(xref)

                for rect in img_rects:
                    width_px = int(abs(rect.width))
                    height_px = int(abs(rect.height))

                    if width_px < self._min_width or height_px < self._min_height:
                        continue

                    bbox = BoundingBox(
                        x0=rect.x0, y0=rect.y0,
                        x1=rect.x1, y1=rect.y1,
                    )

                    # Detect caption
                    caption = self._find_caption(bbox, text_blocks)

                    # Classify figure type
                    figure_type = self._classify_figure(
                        caption, width_px, height_px
                    )

                    # Optionally save image to disk
                    image_path: str | None = None
                    if self._output_dir:
                        image_path = self._save_image(
                            doc, xref, page_number, len(figures)
                        )

                    # Estimate DPI from rect vs pixmap size
                    dpi = self._estimate_dpi(rect, width_px, height_px)

                    fd = FigureData(
                        page_number=page_number,
                        bbox=bbox,
                        figure_type=figure_type,
                        image_path=image_path,
                        width_px=width_px,
                        height_px=height_px,
                        resolution_dpi=dpi,
                        caption=caption,
                        confidence=0.85,
                    )
                    figures.append(fd)

            except Exception as exc:
                logger.debug(
                    "Skipping image xref=%s on page %d: %s",
                    img_info[0], page_number, exc,
                )

        return figures

    def _find_caption(
        self,
        img_bbox: BoundingBox,
        text_blocks: list[Any],
        max_distance_pts: float = 50.0,
    ) -> str | None:
        """Find a caption text block near the image.

        Searches below and above the image for blocks containing
        caption keywords (Fig., Figure, etc.).

        Args:
            img_bbox: Image bounding box.
            text_blocks: Text blocks on the page.
            max_distance_pts: Maximum vertical distance in PDF points.

        Returns:
            Caption string if found, None otherwise.
        """
        best_caption: str | None = None
        best_dist = float("inf")

        for block in text_blocks:
            text = getattr(block, "text", "").strip()
            if not text:
                continue

            text_lower = text.lower()
            is_caption_like = any(kw in text_lower for kw in _BIOMEDICAL_KEYWORDS)
            if not is_caption_like:
                continue

            block_bbox = getattr(block, "bbox", None)
            if block_bbox is None:
                continue

            # Check proximity (below the image)
            dist = abs(block_bbox.y0 - img_bbox.y1)
            if dist < best_dist and dist < max_distance_pts:
                best_dist = dist
                best_caption = text

        return best_caption

    @staticmethod
    def _classify_figure(
        caption: str | None,
        width_px: int,
        height_px: int,
    ) -> str:
        """Classify figure type from caption keywords and dimensions.

        Args:
            caption: Detected caption text. None if no caption.
            width_px: Image width in pixels.
            height_px: Image height in pixels.

        Returns:
            Figure type string: 'chart', 'graph', 'diagram', 'image', 'unknown'.
        """
        if caption:
            cap = caption.lower()
            if any(k in cap for k in ("chart", "bar", "pie", "histogram")):
                return "chart"
            if any(k in cap for k in ("graph", "plot", "curve")):
                return "graph"
            if any(k in cap for k in ("diagram", "flow", "schematic")):
                return "diagram"
            if any(k in cap for k in ("mri", "ct", "scan", "xray", "histol")):
                return "biomedical_image"
        aspect = width_px / height_px if height_px > 0 else 1.0
        if aspect > 2.5:
            return "chart"  # wide = likely chart
        return "image"

    def _save_image(
        self,
        doc: Any,
        xref: int,
        page_number: int,
        index: int,
    ) -> str | None:
        """Save an extracted image to the output directory.

        Args:
            doc: Open fitz.Document.
            xref: Image cross-reference number.
            page_number: 1-indexed page number.
            index: Image index on this page.

        Returns:
            Absolute path to the saved file, or None on failure.
        """
        if self._output_dir is None:
            return None
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            img_data = doc.extract_image(xref)
            ext = img_data.get("ext", "png")
            filename = f"page{page_number:04d}_fig{index:03d}.{ext}"
            out_path = self._output_dir / filename
            out_path.write_bytes(img_data["image"])
            return str(out_path)
        except Exception as exc:
            logger.debug("Could not save image xref=%d: %s", xref, exc)
            return None

    @staticmethod
    def _estimate_dpi(rect: Any, width_px: int, height_px: int) -> int:
        """Estimate image DPI from rect size and pixel dimensions.

        Args:
            rect: fitz.Rect of the image on the page.
            width_px: Image width in pixels.
            height_px: Image height in pixels.

        Returns:
            Estimated DPI (integer), minimum 72.
        """
        try:
            pts_width = rect.width
            if pts_width > 0 and width_px > 0:
                dpi = int((width_px / pts_width) * 72)
                return max(72, dpi)
        except Exception:
            pass
        return 72
