"""Immutable Page model — the atomic output of one distributed worker.

Each worker processes a PageWorkUnit and returns exactly one Page object.
Pages are immutable after construction; all fields are frozen.

Architecture:
    Worker → PageObjectBuilder → Page (immutable)
    Coordinator collects all Pages → UnifiedDocument (immutable)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProcessingMethod(str, Enum):
    """How a page was processed.

    Values:
        DIRECT_TEXT: Native PDF text layer extracted directly.
        OCR: Optical character recognition applied (PaddleOCR).
        MIXED: Combination of direct extraction and OCR.
        SKIPPED: Page was skipped (e.g. blank, unreadable).
        FAILED: Processing failed; page contains error information.
    """

    DIRECT_TEXT = "direct_text"
    OCR = "ocr"
    MIXED = "mixed"
    SKIPPED = "skipped"
    FAILED = "failed"


class PageType(str, Enum):
    """Classification of a page's content origin.

    Values:
        DIGITAL: Page has a native text layer (vector PDF).
        SCANNED: Page is a rasterised image with no text layer.
        MIXED: Page has both text layer and embedded raster images.
        UNKNOWN: Classification could not be determined.
    """

    DIGITAL = "digital"
    SCANNED = "scanned"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in PDF points (1 pt = 1/72 inch).

    Attributes:
        x0: Left edge.
        y0: Top edge.
        x1: Right edge.
        y1: Bottom edge.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Width of the bounding box in points."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Height of the bounding box in points."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Area of the bounding box in square points."""
        return self.width * self.height

    def is_valid(self) -> bool:
        """Return True if the bounding box has positive dimensions."""
        return self.x1 > self.x0 and self.y1 > self.y0

    def to_dict(self) -> dict[str, float]:
        """Serialise to plain dictionary."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True)
class TextBlock:
    """A contiguous block of extracted text on a page.

    Attributes:
        text: Raw extracted text.
        bbox: Bounding box of this block.
        block_type: 'paragraph', 'heading', 'caption', 'footnote', 'list_item'.
        confidence: OCR confidence score [0.0, 1.0]. 1.0 for direct extraction.
        font_name: Primary font used. None if unknown.
        font_size: Font size in points. None if unknown.
        reading_order: Position of this block in left-to-right, top-to-bottom order.
    """

    text: str
    bbox: BoundingBox
    block_type: str = "paragraph"
    confidence: float = 1.0
    font_name: str | None = None
    font_size: float | None = None
    reading_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "block_type": self.block_type,
            "confidence": self.confidence,
            "font_name": self.font_name,
            "font_size": self.font_size,
            "reading_order": self.reading_order,
        }


@dataclass(frozen=True)
class TableData:
    """Extracted table from a page.

    Attributes:
        table_id: Unique identifier for this table.
        page_number: 1-indexed page number.
        bbox: Bounding box on the page.
        rows: Number of data rows.
        cols: Number of columns.
        headers: Column headers if detected.
        cells: Row-major list of cell text values.
        markdown: Table rendered as markdown string.
        csv: Table rendered as CSV string.
        confidence: Extraction confidence [0.0, 1.0].
        caption: Table caption if detected.
    """

    page_number: int
    bbox: BoundingBox
    rows: int
    cols: int
    cells: tuple[tuple[str, ...], ...]
    table_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    headers: tuple[str, ...] = field(default_factory=tuple)
    markdown: str = ""
    csv: str = ""
    confidence: float = 1.0
    caption: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "bbox": self.bbox.to_dict(),
            "rows": self.rows,
            "cols": self.cols,
            "headers": list(self.headers),
            "cells": [list(row) for row in self.cells],
            "markdown": self.markdown,
            "csv": self.csv,
            "confidence": self.confidence,
            "caption": self.caption,
        }


@dataclass(frozen=True)
class FigureData:
    """Extracted figure or image from a page.

    Attributes:
        figure_id: Unique identifier.
        page_number: 1-indexed page number.
        bbox: Bounding box on the page.
        figure_type: 'chart', 'graph', 'image', 'diagram', 'table_image', 'unknown'.
        image_path: Saved image file path. None if not saved.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        resolution_dpi: Image resolution.
        caption: Detected caption text. None if not found.
        confidence: Detection confidence [0.0, 1.0].
    """

    page_number: int
    bbox: BoundingBox
    figure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    figure_type: str = "unknown"
    image_path: str | None = None
    width_px: int = 0
    height_px: int = 0
    resolution_dpi: int = 72
    caption: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "figure_id": self.figure_id,
            "page_number": self.page_number,
            "bbox": self.bbox.to_dict(),
            "figure_type": self.figure_type,
            "image_path": self.image_path,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "resolution_dpi": self.resolution_dpi,
            "caption": self.caption,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class LayoutElement:
    """A structural element detected by layout analysis.

    Attributes:
        element_type: 'heading', 'paragraph', 'list', 'table', 'figure',
                      'caption', 'footnote', 'header', 'footer'.
        text: Text content of this element.
        bbox: Bounding box.
        level: Heading level (1–6) for headings. None for other types.
        reading_order: Position in reading order.
    """

    element_type: str
    text: str
    bbox: BoundingBox
    level: int | None = None
    reading_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "element_type": self.element_type,
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "level": self.level,
            "reading_order": self.reading_order,
        }


@dataclass(frozen=True)
class PageStatistics:
    """Processing statistics for a single page.

    Attributes:
        text_block_count: Number of text blocks extracted.
        table_count: Number of tables extracted.
        figure_count: Number of figures extracted.
        char_count: Total character count in extracted text.
        word_count: Approximate word count.
        image_density: Fraction of page area covered by images [0.0, 1.0].
        has_text_layer: True if page had a native text layer.
        ocr_confidence_avg: Average OCR confidence. 1.0 for direct extraction.
    """

    text_block_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    char_count: int = 0
    word_count: int = 0
    image_density: float = 0.0
    has_text_layer: bool = False
    ocr_confidence_avg: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "text_block_count": self.text_block_count,
            "table_count": self.table_count,
            "figure_count": self.figure_count,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "image_density": self.image_density,
            "has_text_layer": self.has_text_layer,
            "ocr_confidence_avg": self.ocr_confidence_avg,
        }


@dataclass(frozen=True)
class Page:
    """Immutable output of one distributed worker processing a PageWorkUnit.

    Created by PageObjectBuilder. Never modified after construction.
    Collected by the coordinator and merged into an UnifiedDocument.

    Attributes:
        document_id: Parent document identifier.
        page_number: 1-indexed page number within the document.
        page_type: Classification of this page (digital/scanned/mixed).
        processing_method: How this page was processed.
        text: Full extracted text, reading-order concatenated.
        text_blocks: Ordered tuple of TextBlock objects.
        tables: Tuple of TableData objects found on this page.
        figures: Tuple of FigureData objects found on this page.
        layout_elements: Tuple of LayoutElement objects (heading, paragraph, etc.).
        worker_id: ID of the worker that produced this page.
        node_id: Hostname of the node that processed this page.
        processing_time_seconds: Total wall-clock time for this page.
        ocr_confidence: Average OCR confidence (1.0 = direct extraction).
        success: True if processing completed without fatal errors.
        error_message: Error description if success is False.
        warnings: Tuple of non-fatal warning messages.
        statistics: Quantitative processing statistics.
        width_pts: Page width in PDF points.
        height_pts: Page height in PDF points.

    Example:
        >>> page = Page(
        ...     document_id="doc-001",
        ...     page_number=1,
        ...     page_type=PageType.DIGITAL,
        ...     processing_method=ProcessingMethod.DIRECT_TEXT,
        ...     text="Introduction ...",
        ...     text_blocks=(),
        ...     tables=(),
        ...     figures=(),
        ...     layout_elements=(),
        ...     worker_id="worker-0",
        ...     node_id="laptop-1",
        ...     processing_time_seconds=0.12,
        ...     ocr_confidence=1.0,
        ...     success=True,
        ... )
        >>> page.page_number
        1
    """

    document_id: str
    page_number: int
    page_type: PageType
    processing_method: ProcessingMethod
    text: str
    text_blocks: tuple[TextBlock, ...]
    tables: tuple[TableData, ...]
    figures: tuple[FigureData, ...]
    layout_elements: tuple[LayoutElement, ...]
    worker_id: str
    node_id: str
    processing_time_seconds: float
    ocr_confidence: float
    success: bool
    error_message: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    statistics: PageStatistics = field(default_factory=PageStatistics)
    width_pts: float = 0.0
    height_pts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary.

        Returns:
            Dictionary representation of this page.
        """
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "page_type": self.page_type.value,
            "processing_method": self.processing_method.value,
            "text": self.text,
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "layout_elements": [e.to_dict() for e in self.layout_elements],
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "processing_time_seconds": self.processing_time_seconds,
            "ocr_confidence": self.ocr_confidence,
            "success": self.success,
            "error_message": self.error_message,
            "warnings": list(self.warnings),
            "statistics": self.statistics.to_dict(),
            "width_pts": self.width_pts,
            "height_pts": self.height_pts,
        }

    @property
    def word_count(self) -> int:
        """Approximate word count from extracted text."""
        return len(self.text.split()) if self.text else 0

    @property
    def is_empty(self) -> bool:
        """Return True if the page has no extracted content."""
        return not self.text and not self.tables and not self.figures

    def __repr__(self) -> str:
        return (
            f"Page(document_id='{self.document_id}', "
            f"page_number={self.page_number}, "
            f"type={self.page_type.value}, "
            f"method={self.processing_method.value}, "
            f"success={self.success})"
        )
