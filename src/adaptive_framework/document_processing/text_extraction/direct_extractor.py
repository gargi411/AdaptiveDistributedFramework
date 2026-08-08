"""Direct text extraction for digital PDF pages — Module 4.

Extracts text, blocks, fonts, and reading order directly from the native
PDF text layer using PyMuPDF. Never invokes OCR.

Called by DirectExtractionStrategy. Also usable standalone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_framework.models.page import BoundingBox, TextBlock

logger = logging.getLogger(__name__)


@dataclass
class DirectExtractionResult:
    """Result of direct text extraction from one PDF page.

    Attributes:
        page_number: 1-indexed page number.
        text: Full extracted text in reading order.
        text_blocks: Ordered list of TextBlock objects.
        font_names: Set of font names used on this page.
        char_count: Total characters extracted.
        word_count: Approximate word count.
        extraction_time_seconds: Wall-clock time for extraction.
        success: True if extraction completed without error.
        error: Error message if success is False.
    """

    page_number: int
    text: str = ""
    text_blocks: list[TextBlock] = field(default_factory=list)
    font_names: set[str] = field(default_factory=set)
    char_count: int = 0
    word_count: int = 0
    extraction_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "page_number": self.page_number,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "font_names": sorted(self.font_names),
            "text_block_count": len(self.text_blocks),
            "extraction_time_seconds": self.extraction_time_seconds,
            "success": self.success,
            "error": self.error,
        }


class DirectTextExtractor:
    """Extracts text directly from digital PDF pages using PyMuPDF.

    Architecture rule:
        Call ONLY on pages classified as DIGITAL.
        Never call on SCANNED pages — use OCRStrategy instead.

    Usage:
        >>> extractor = DirectTextExtractor()
        >>> result = extractor.extract_page(page, page_number=1)
        >>> result.char_count
        1247

    Args:
        min_block_chars: Minimum characters for a text block to be included.
        sort_reading_order: If True, sort blocks top-to-bottom, left-to-right.
    """

    def __init__(
        self,
        min_block_chars: int = 1,
        sort_reading_order: bool = True,
    ) -> None:
        self._min_block_chars = min_block_chars
        self._sort_reading_order = sort_reading_order

    def extract_page(
        self,
        page: Any,
        page_number: int,
    ) -> DirectExtractionResult:
        """Extract text from a single PyMuPDF page object.

        Args:
            page: fitz.Page object (already open, zero-copy).
            page_number: 1-indexed page number.

        Returns:
            DirectExtractionResult with text, blocks, and statistics.
        """
        import time
        t0 = time.perf_counter()

        result = DirectExtractionResult(page_number=page_number)

        try:
            blocks_raw = page.get_text("dict", sort=self._sort_reading_order)
            text_parts: list[str] = []
            font_names: set[str] = set()
            order = 0

            for block in blocks_raw.get("blocks", []):
                if block.get("type") != 0:
                    continue  # skip image blocks

                block_text_parts: list[str] = []
                dominant_font: str | None = None
                dominant_size: float | None = None

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text:
                            block_text_parts.append(span_text)
                            font = span.get("font")
                            if font:
                                font_names.add(font)
                                if dominant_font is None:
                                    dominant_font = font
                                    dominant_size = span.get("size")

                block_text = " ".join(block_text_parts).strip()
                if len(block_text) < self._min_block_chars:
                    continue

                bbox_raw = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
                bbox = BoundingBox(
                    x0=bbox_raw[0], y0=bbox_raw[1],
                    x1=bbox_raw[2], y1=bbox_raw[3],
                )

                # Infer block type from font size heuristic
                block_type = self._infer_block_type(
                    block_text, dominant_size, order
                )

                tb = TextBlock(
                    text=block_text,
                    bbox=bbox,
                    block_type=block_type,
                    confidence=1.0,
                    font_name=dominant_font,
                    font_size=dominant_size,
                    reading_order=order,
                )
                result.text_blocks.append(tb)
                text_parts.append(block_text)
                order += 1

            result.text = "\n".join(text_parts)
            result.font_names = font_names
            result.char_count = len(result.text)
            result.word_count = len(result.text.split()) if result.text else 0
            result.success = True

        except Exception as exc:
            logger.warning(
                "DirectTextExtractor failed for page %d: %s", page_number, exc
            )
            result.success = False
            result.error = str(exc)

        result.extraction_time_seconds = time.perf_counter() - t0
        return result

    @staticmethod
    def _infer_block_type(text: str, font_size: float | None, order: int) -> str:
        """Heuristically infer block type from font size and position.

        Args:
            text: Block text.
            font_size: Dominant font size in points. None if unknown.
            order: Reading order index.

        Returns:
            Block type string: 'heading', 'paragraph', 'footnote', etc.
        """
        if font_size is None:
            return "paragraph"
        if font_size >= 14.0:
            return "heading"
        if font_size <= 8.0:
            return "footnote"
        if text.strip().startswith(("•", "-", "–", "*", "◦")):
            return "list_item"
        return "paragraph"
