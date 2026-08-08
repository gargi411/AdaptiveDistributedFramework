"""Metadata Store — persistence layer for PDFMetadata records.

Provides CSV and JSON serialization / deserialization for the metadata
catalog. The store is append-aware: loading an existing store and
re-saving will not duplicate entries.

File layout::

    dataset/
    ├── raw/                    ← PDFs from DatasetBuilder
    │   ├── PMC1234567.pdf
    │   └── ...
    └── metadata_extracted.csv  ← produced by MetadataStore.save_csv()

The ``metadata_extracted.csv`` differs from ``metadata.csv``
(produced by DatasetBuilder):
    - ``metadata.csv`` has PubMed article-level fields.
    - ``metadata_extracted.csv`` has PDF-level technical fields
      (pages, DPI, source_type, etc.) used by the Scheduler.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from adaptive_framework.models.document import PDFMetadata

logger = logging.getLogger("adaptive_framework.metadata_generator.store")

# Default filenames
_CSV_FILENAME = "metadata_extracted.csv"
_JSON_FILENAME = "metadata_extracted.json"

# Ordered CSV column names (matches PDFMetadata fields)
_CSV_FIELDNAMES: list[str] = [
    "document_id",
    "file_path",
    "pages",
    "estimated_size_mb",
    "resolution_dpi",
    "source_type",
    "language",
    "processing_timestamp",
]


class MetadataStore:
    """Persist and load PDFMetadata records to/from CSV and JSON.

    Attributes:
        _output_dir: Directory where metadata files are written.

    Example:
        >>> store = MetadataStore(output_dir=Path("dataset"))
        >>> store.save_csv(metadata_list)
        >>> loaded = store.load_csv()
        >>> print(len(loaded))
        42
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialise the MetadataStore.

        Args:
            output_dir: Directory for metadata output files.
                        Created automatically if it does not exist.
        """
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # CSV persistence                                                      #
    # ------------------------------------------------------------------ #

    def save_csv(
        self,
        metadata_list: list[PDFMetadata],
        filename: str = _CSV_FILENAME,
    ) -> Path:
        """Write a list of PDFMetadata records to a CSV file.

        Overwrites any existing file with the same name.

        Args:
            metadata_list: Records to serialize.
            filename: Output filename relative to output_dir.

        Returns:
            Path to the written CSV file.

        Raises:
            OSError: If the file cannot be written.
        """
        csv_path = self._output_dir / filename
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            for meta in metadata_list:
                writer.writerow(self._to_csv_row(meta))
        logger.info(
            "Saved %d metadata records to '%s'.", len(metadata_list), csv_path
        )
        return csv_path

    def load_csv(
        self,
        filename: str = _CSV_FILENAME,
    ) -> list[PDFMetadata]:
        """Load PDFMetadata records from a previously saved CSV file.

        Args:
            filename: CSV filename relative to output_dir.

        Returns:
            List of PDFMetadata objects. Empty list if file does not exist.
        """
        csv_path = self._output_dir / filename
        if not csv_path.exists():
            logger.warning("Metadata CSV not found: '%s'", csv_path)
            return []

        records: list[PDFMetadata] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    records.append(self._from_csv_row(row))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipped malformed CSV row: %s", exc)

        logger.info(
            "Loaded %d metadata records from '%s'.", len(records), csv_path
        )
        return records

    # ------------------------------------------------------------------ #
    # JSON persistence                                                     #
    # ------------------------------------------------------------------ #

    def save_json(
        self,
        metadata_list: list[PDFMetadata],
        filename: str = _JSON_FILENAME,
        indent: int = 2,
    ) -> Path:
        """Serialize metadata to a JSON file.

        Args:
            metadata_list: Records to serialize.
            filename: Output filename relative to output_dir.
            indent: JSON indentation level.

        Returns:
            Path to the written JSON file.
        """
        json_path = self._output_dir / filename
        data: list[dict[str, Any]] = [m.to_dict() for m in metadata_list]
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(
            "Saved %d metadata records to '%s'.", len(metadata_list), json_path
        )
        return json_path

    def load_json(
        self,
        filename: str = _JSON_FILENAME,
    ) -> list[PDFMetadata]:
        """Load PDFMetadata from a JSON file.

        Args:
            filename: JSON filename relative to output_dir.

        Returns:
            List of PDFMetadata objects. Empty list if file does not exist.
        """
        json_path = self._output_dir / filename
        if not json_path.exists():
            logger.warning("Metadata JSON not found: '%s'", json_path)
            return []

        with json_path.open("r", encoding="utf-8") as f:
            raw: list[dict[str, Any]] = json.load(f)

        records: list[PDFMetadata] = []
        for item in raw:
            try:
                records.append(
                    PDFMetadata(
                        document_id=item.get("document_id", ""),
                        pages=int(item.get("pages", 1)),
                        estimated_size_mb=float(item.get("estimated_size_mb", 0.0)),
                        file_path=item.get("file_path", ""),
                        resolution_dpi=item.get("resolution_dpi"),
                        source_type=item.get("source_type"),
                        language=item.get("language"),
                        processing_timestamp=item.get("processing_timestamp", ""),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipped malformed JSON record: %s", exc)

        logger.info(
            "Loaded %d metadata records from '%s'.", len(records), json_path
        )
        return records

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def compute_statistics(
        self, metadata_list: list[PDFMetadata]
    ) -> dict[str, Any]:
        """Compute aggregate statistics over a metadata collection.

        Args:
            metadata_list: PDFMetadata records to summarise.

        Returns:
            Dictionary with keys:
                - count, total_pages, avg_pages, min_pages, max_pages,
                - total_size_mb, digital_count, scanned_count, unknown_count.
        """
        if not metadata_list:
            return {
                "count": 0,
                "total_pages": 0,
                "avg_pages": 0.0,
                "min_pages": 0,
                "max_pages": 0,
                "total_size_mb": 0.0,
                "digital_count": 0,
                "scanned_count": 0,
                "unknown_count": 0,
            }

        pages = [m.pages for m in metadata_list]
        sizes = [m.estimated_size_mb for m in metadata_list]

        return {
            "count": len(metadata_list),
            "total_pages": sum(pages),
            "avg_pages": round(sum(pages) / len(pages), 2),
            "min_pages": min(pages),
            "max_pages": max(pages),
            "total_size_mb": round(sum(sizes), 2),
            "digital_count": sum(1 for m in metadata_list if m.is_digital()),
            "scanned_count": sum(1 for m in metadata_list if m.is_scanned()),
            "unknown_count": sum(
                1 for m in metadata_list
                if m.source_type not in ("digital", "scanned")
            ),
        }

    # ------------------------------------------------------------------ #
    # Private serialization helpers                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_csv_row(meta: PDFMetadata) -> dict[str, Any]:
        """Convert a PDFMetadata to a flat CSV row dict.

        Args:
            meta: PDFMetadata record.

        Returns:
            Flat dictionary with CSV-safe string values.
        """
        return {
            "document_id": meta.document_id,
            "file_path": meta.file_path,
            "pages": meta.pages,
            "estimated_size_mb": meta.estimated_size_mb,
            "resolution_dpi": meta.resolution_dpi if meta.resolution_dpi else "",
            "source_type": meta.source_type or "",
            "language": meta.language or "",
            "processing_timestamp": meta.processing_timestamp,
        }

    @staticmethod
    def _from_csv_row(row: dict[str, str]) -> PDFMetadata:
        """Reconstruct a PDFMetadata from a CSV row dict.

        Args:
            row: CSV row as a string dictionary.

        Returns:
            PDFMetadata with fields populated from the row.
        """
        resolution_dpi = None
        if row.get("resolution_dpi"):
            try:
                resolution_dpi = int(row["resolution_dpi"])
            except ValueError:
                pass

        return PDFMetadata(
            document_id=row.get("document_id", ""),
            pages=max(1, int(row.get("pages", 1))),
            estimated_size_mb=float(row.get("estimated_size_mb", 0.0)),
            file_path=row.get("file_path", ""),
            resolution_dpi=resolution_dpi,
            source_type=row.get("source_type") or None,
            language=row.get("language") or None,
            processing_timestamp=row.get("processing_timestamp", ""),
        )
