"""PDF Metadata Extractor — Module 2 of Phase 2A.

Extracts structured metadata from PDF files using ``pypdf`` (pure-Python,
no external dependencies). Produces ``PDFMetadata`` records that are consumed
by the Adaptive Page Count Partitioner.

Fields extracted (per architecture v2.0 §2.4):
    - document_id       (UUID4, auto-generated)
    - filename          (basename of the PDF file)
    - file_path         (absolute resolved path)
    - pages             (page count — the PRIMARY scheduler input)
    - estimated_size_mb (file size from filesystem)
    - language          (from PDF document info, if available)
    - resolution_dpi    (from page MediaBox dimensions, heuristic)
    - source_type       ('digital' if text layer found, 'scanned' otherwise,
                         'unknown' when detection is ambiguous)
    - creation_timestamp  (from PDF /CreationDate metadata)
    - processing_timestamp (ISO 8601 UTC, set at extraction time)

Design:
    - ``extract(file_path)``  → single PDFMetadata
    - ``extract_batch(paths)`` → list[PDFMetadata] with per-file error handling
    - All fields are populated with safe defaults when extraction fails
    - Never raises on individual file failures; logs warnings instead
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from adaptive_framework.core.constants import (
    SOURCE_TYPE_DIGITAL,
    SOURCE_TYPE_SCANNED,
)
from adaptive_framework.core.exceptions import DatasetError
from adaptive_framework.models.document import PDFMetadata

logger = logging.getLogger("adaptive_framework.metadata_generator")

# Heuristic: A4 page at 72 DPI baseline. Points-per-inch = 72.
_POINTS_PER_INCH: float = 72.0

# Source type strings (architecture §2.4)
_SOURCE_UNKNOWN: str = "unknown"

# Regex for parsing PDF /CreationDate format: D:YYYYMMDDHHmmSS[+|-HH'mm']
_PDF_DATE_PATTERN = re.compile(
    r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})"
)


class MetadataExtractor:
    """Extracts PDFMetadata from PDF files using pypdf.

    Attributes:
        _text_sample_pages: Number of pages to sample for text-layer detection.

    Example:
        >>> extractor = MetadataExtractor()
        >>> meta = extractor.extract(Path("dataset/raw/PMC1234567.pdf"))
        >>> print(meta.pages)
        42
        >>> print(meta.source_type)
        'digital'
    """

    def __init__(self, text_sample_pages: int = 3) -> None:
        """Initialise the MetadataExtractor.

        Args:
            text_sample_pages: Number of pages to sample when determining
                whether a PDF has a native text layer. Higher values are more
                accurate but slower.
        """
        self._text_sample_pages = text_sample_pages

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(self, file_path: Path) -> PDFMetadata:
        """Extract full metadata from a single PDF file.

        Args:
            file_path: Path to the PDF file. Must exist.

        Returns:
            PDFMetadata record with all available fields populated.

        Raises:
            DatasetError: If the file does not exist or cannot be opened.
        """
        if not file_path.exists():
            raise DatasetError(f"PDF file not found: '{file_path}'")
        if not file_path.is_file():
            raise DatasetError(f"'{file_path}' is not a file.")

        resolved = file_path.resolve()
        file_size_mb = resolved.stat().st_size / (1024 * 1024)
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            return self._extract_with_pypdf(
                file_path=resolved,
                file_size_mb=file_size_mb,
                processing_timestamp=now_iso,
            )
        except Exception as exc:  # noqa: BLE001 — never crash on single file
            logger.warning(
                "Metadata extraction failed for '%s': %s — using safe defaults.",
                file_path.name,
                exc,
            )
            return PDFMetadata(
                document_id=str(uuid.uuid4()),
                pages=1,
                estimated_size_mb=round(file_size_mb, 4),
                file_path=str(resolved),
                resolution_dpi=None,
                source_type=None,
                language=None,
                processing_timestamp=now_iso,
            )

    def extract_batch(
        self,
        file_paths: list[Path],
        skip_invalid: bool = True,
    ) -> list[PDFMetadata]:
        """Extract metadata for a list of PDF files.

        Args:
            file_paths: Paths to extract metadata from.
            skip_invalid: If True, silently skip (log warning) files that
                fail extraction. If False, raises DatasetError on first failure.

        Returns:
            List of PDFMetadata records (one per successfully processed file).
            Order matches the input list for successfully extracted files.

        Raises:
            DatasetError: If skip_invalid=False and any file fails.
        """
        results: list[PDFMetadata] = []
        total = len(file_paths)

        for i, path in enumerate(file_paths, start=1):
            logger.debug(
                "Extracting metadata [%d/%d]: %s", i, total, path.name
            )
            try:
                meta = self.extract(path)
                results.append(meta)
            except DatasetError as exc:
                if not skip_invalid:
                    raise
                logger.warning("Skipping '%s': %s", path.name, exc)

        logger.info(
            "Batch extraction complete: %d/%d files processed.",
            len(results),
            total,
        )
        return results

    # ------------------------------------------------------------------ #
    # Private: pypdf extraction                                            #
    # ------------------------------------------------------------------ #

    def _extract_with_pypdf(
        self,
        file_path: Path,
        file_size_mb: float,
        processing_timestamp: str,
    ) -> PDFMetadata:
        """Perform the actual pypdf extraction.

        Args:
            file_path: Resolved absolute path to the PDF.
            file_size_mb: Pre-computed file size in MB.
            processing_timestamp: ISO 8601 UTC timestamp.

        Returns:
            Populated PDFMetadata record.
        """
        from pypdf import PdfReader  # type: ignore[import]

        reader = PdfReader(str(file_path), strict=False)
        page_count = len(reader.pages)

        # Guard: must have at least one page
        if page_count < 1:
            logger.warning(
                "'%s' reports 0 pages — forcing page_count=1.", file_path.name
            )
            page_count = 1

        # Resolution heuristic from first page MediaBox
        resolution_dpi = self._estimate_dpi(reader)

        # Source type detection (digital vs scanned)
        source_type = self._detect_source_type(reader, page_count)

        # Language from document info
        language = self._extract_language(reader)

        # Creation timestamp from PDF metadata
        creation_ts = self._extract_creation_date(reader)

        return PDFMetadata(
            document_id=str(uuid.uuid4()),
            pages=page_count,
            estimated_size_mb=round(file_size_mb, 4),
            file_path=str(file_path),
            resolution_dpi=resolution_dpi,
            source_type=source_type,
            language=language,
            processing_timestamp=processing_timestamp,
        )

    # ------------------------------------------------------------------ #
    # Private: field extractors                                            #
    # ------------------------------------------------------------------ #

    def _estimate_dpi(self, reader: object) -> int | None:
        """Estimate resolution in DPI from the first page MediaBox.

        The PDF standard defines page dimensions in points (1 pt = 1/72 inch).
        By comparing to standard A4 (595 × 842 pt) or US Letter (612 × 792 pt),
        we can estimate the scan resolution if the document was originally
        a scanned image.

        Args:
            reader: pypdf PdfReader instance.

        Returns:
            Estimated DPI (integer), or None if detection fails.
        """
        try:
            # Access via pypdf attribute name
            pages = getattr(reader, "pages", [])
            if not pages:
                return None
            page = pages[0]
            media_box = getattr(page, "mediabox", None)
            if media_box is None:
                return None
            width_pts = float(media_box.width)
            height_pts = float(media_box.height)
            if width_pts <= 0 or height_pts <= 0:
                return None
            # Use the longer dimension to estimate DPI
            longer_pts = max(width_pts, height_pts)
            # Assume A4 paper 11.69 inches
            longer_inches = 11.69
            dpi = int(longer_pts / longer_inches)
            # Clamp to sensible range [72, 1200]
            return max(72, min(1200, dpi))
        except Exception:  # noqa: BLE001
            return None

    def _detect_source_type(self, reader: object, page_count: int) -> str:
        """Detect whether a PDF is digital or scanned.

        Samples up to ``_text_sample_pages`` pages. If any sampled page
        has a non-empty text extraction result, the PDF is classified as
        'digital'. Otherwise, it is classified as 'scanned'.

        Args:
            reader: pypdf PdfReader instance.
            page_count: Total page count of the document.

        Returns:
            'digital', 'scanned', or 'unknown'.
        """
        try:
            pages = getattr(reader, "pages", [])
            sample_count = min(self._text_sample_pages, page_count)
            for i in range(sample_count):
                text = pages[i].extract_text() or ""
                if text.strip():
                    return SOURCE_TYPE_DIGITAL
            return SOURCE_TYPE_SCANNED
        except Exception:  # noqa: BLE001
            return _SOURCE_UNKNOWN

    def _extract_language(self, reader: object) -> str | None:
        """Extract language code from PDF document info.

        Args:
            reader: pypdf PdfReader instance.

        Returns:
            ISO 639-1 language code (e.g. 'en'), or None if unavailable.
        """
        try:
            meta = getattr(reader, "metadata", None) or {}
            # Some PDFs embed /Language in XMP metadata
            lang = meta.get("/Language") or meta.get("Language")
            if lang and isinstance(lang, str):
                # Normalize: 'en-US' → 'en', 'EN' → 'en'
                return lang.split("-")[0].lower()[:2]
            return "en"  # Default to English for biomedical papers
        except Exception:  # noqa: BLE001
            return None

    def _extract_creation_date(self, reader: object) -> str | None:
        """Extract the PDF creation date from document metadata.

        Args:
            reader: pypdf PdfReader instance.

        Returns:
            ISO 8601 date string or None if unavailable.
        """
        try:
            meta = getattr(reader, "metadata", None) or {}
            date_str = meta.get("/CreationDate") or meta.get("CreationDate")
            if not date_str:
                return None
            m = _PDF_DATE_PATTERN.match(str(date_str))
            if m:
                year, month, day, hour, minute, second = m.groups()
                return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
            return None
        except Exception:  # noqa: BLE001
            return None
