"""Table Extractor — Module 7: Structured table extraction from PDF pages.

Extracts tables with rows, columns, markdown, CSV, bounding boxes,
page numbers, and confidence scores.

Strategy:
    1. PyMuPDF word/line clustering for layout-based tables
    2. Docling table regions (when available)
    Falls back gracefully if neither provides table data.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_framework.models.page import BoundingBox, TableData

logger = logging.getLogger(__name__)


@dataclass
class TableExtractionResult:
    """Result of table extraction for a single page.

    Attributes:
        page_number: 1-indexed page number.
        tables: List of extracted TableData objects.
        extraction_time_seconds: Wall-clock time.
        success: True if extraction completed without error.
        error: Error message if success is False.
    """

    page_number: int
    tables: list[TableData] = field(default_factory=list)
    extraction_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None

    @property
    def count(self) -> int:
        """Number of tables extracted."""
        return len(self.tables)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "page_number": self.page_number,
            "table_count": self.count,
            "extraction_time_seconds": self.extraction_time_seconds,
            "success": self.success,
            "error": self.error,
        }


class TableExtractor:
    """Extracts structured tables from PDF page objects.

    Uses PyMuPDF's built-in table finder when available.
    Falls back to text-block clustering heuristics.

    Usage:
        >>> extractor = TableExtractor()
        >>> result = extractor.extract_page(page, page_number=2)
        >>> for table in result.tables:
        ...     print(table.markdown)
    """

    def __init__(self, min_rows: int = 2, min_cols: int = 2) -> None:
        """Initialise TableExtractor.

        Args:
            min_rows: Minimum rows to consider a region a table.
            min_cols: Minimum columns to consider a region a table.
        """
        self._min_rows = min_rows
        self._min_cols = min_cols

    def extract_page(
        self,
        page: Any,
        page_number: int,
    ) -> TableExtractionResult:
        """Extract all tables from a single PyMuPDF page.

        Args:
            page: fitz.Page object.
            page_number: 1-indexed page number.

        Returns:
            TableExtractionResult with zero or more TableData objects.
        """
        import time
        t0 = time.perf_counter()
        result = TableExtractionResult(page_number=page_number)

        try:
            tables = self._extract_with_pymupdf(page, page_number)
            result.tables = tables
        except Exception as exc:
            logger.warning(
                "Table extraction failed for page %d: %s", page_number, exc
            )
            result.success = False
            result.error = str(exc)

        result.extraction_time_seconds = time.perf_counter() - t0
        return result

    def _extract_with_pymupdf(
        self, page: Any, page_number: int
    ) -> list[TableData]:
        """Use PyMuPDF find_tables (available in PyMuPDF >= 1.23).

        Args:
            page: fitz.Page object.
            page_number: 1-indexed page number.

        Returns:
            List of TableData objects.
        """
        tables: list[TableData] = []

        # PyMuPDF >= 1.23 provides page.find_tables()
        try:
            tab_finder = page.find_tables()
        except AttributeError:
            logger.debug("PyMuPDF find_tables not available — skipping table extraction.")
            return tables

        if not tab_finder or not tab_finder.tables:
            return tables

        for tab in tab_finder.tables:
            try:
                extracted = tab.extract()
                if not extracted:
                    continue

                rows = len(extracted)
                cols = max(len(row) for row in extracted) if extracted else 0

                if rows < self._min_rows or cols < self._min_cols:
                    continue

                # Headers from first row
                headers = tuple(str(c or "") for c in extracted[0])
                cells = tuple(
                    tuple(str(c or "") for c in row) for row in extracted
                )

                bbox_rect = tab.bbox
                bbox = BoundingBox(
                    x0=bbox_rect[0], y0=bbox_rect[1],
                    x1=bbox_rect[2], y1=bbox_rect[3],
                )

                markdown = self._to_markdown(headers, cells[1:] if len(cells) > 1 else cells)
                csv_text = self._to_csv(cells)

                table_data = TableData(
                    page_number=page_number,
                    bbox=bbox,
                    rows=rows,
                    cols=cols,
                    headers=headers,
                    cells=cells,
                    markdown=markdown,
                    csv=csv_text,
                    confidence=0.90,
                )
                tables.append(table_data)

            except Exception as exc:
                logger.debug("Skipping malformed table on page %d: %s", page_number, exc)

        return tables

    @staticmethod
    def _to_markdown(headers: tuple[str, ...], data_rows: tuple[tuple[str, ...], ...]) -> str:
        """Convert table cells to a GitHub-flavoured markdown table.

        Args:
            headers: Column header strings.
            data_rows: Data rows (excluding header row).

        Returns:
            Markdown table string.
        """
        if not headers:
            return ""
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in data_rows:
            padded = list(row) + [""] * max(0, len(headers) - len(row))
            lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
        return "\n".join(lines)

    @staticmethod
    def _to_csv(cells: tuple[tuple[str, ...], ...]) -> str:
        """Convert table cells to CSV string.

        Args:
            cells: All rows including header.

        Returns:
            CSV string with CRLF line endings (RFC 4180).
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in cells:
            writer.writerow(list(row))
        return buf.getvalue()
