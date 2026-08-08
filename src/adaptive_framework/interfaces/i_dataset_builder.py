"""IDatasetBuilder — Abstract dataset builder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from adaptive_framework.models.document import PDFMetadata


class IDatasetBuilder(ABC):
    """Abstract interface for building the document dataset.

    A DatasetBuilder scans a directory of PDF files, extracts metadata
    for each document, and returns a list of PDFMetadata records.
    The Adaptive Scheduler uses this metadata (specifically page counts
    and file sizes) to create the partition plan.

    Implementations may support:
        - Local filesystem scans (Phase 2)
        - Remote storage (S3, GCS) in future phases

    Example:
        >>> builder: IDatasetBuilder = ConcreteDatasetBuilder(logger)
        >>> dataset = builder.build(source_dir=Path("/data/raw"))
        >>> print(len(dataset))
        150
    """

    @abstractmethod
    def build(self, source_dir: Path) -> list[PDFMetadata]:
        """Scan source_dir and build a list of PDFMetadata records.

        Args:
            source_dir: Path to the directory containing PDF files.

        Returns:
            List of PDFMetadata, one per discovered document.
                Empty list if no PDF files are found.

        Raises:
            DatasetError: If source_dir does not exist or is not readable.
        """

    @abstractmethod
    def validate(self, source_dir: Path) -> bool:
        """Check whether source_dir is a valid, accessible dataset directory.

        Args:
            source_dir: Path to validate.

        Returns:
            True if the directory exists and contains at least one PDF.
        """

    @abstractmethod
    def get_document_count(self, source_dir: Path) -> int:
        """Return the number of PDF files in source_dir.

        Args:
            source_dir: Path to the dataset directory.

        Returns:
            Count of PDF files (non-recursive). 0 if none found.
        """
