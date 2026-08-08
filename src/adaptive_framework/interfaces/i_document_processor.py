"""IDocumentProcessor — Abstract document processor interface.

Orchestrates the four Document Processing Engine sub-components:
    OCR → Layout Analysis → Table Extraction → Figure Detection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from adaptive_framework.models.document import DocumentResult, PDFMetadata
from adaptive_framework.models.scheduling import PageWorkUnit


class IDocumentProcessor(ABC):
    """Abstract interface for the Document Processing Engine.

    Coordinates the four sub-components of the Document Processing Engine
    as defined in architecture v2.0 §2.1:

        Document Processing Engine
        ├── OCR
        ├── Layout Analysis
        ├── Table Extraction
        └── Figure Detection

    The IDocumentProcessor is what each IWorker uses to process
    its assigned PageWorkUnit. Workers never call IOCREngine directly.

    Example:
        >>> processor: IDocumentProcessor = DocumentProcessingEngineImpl(
        ...     ocr_engine=engine, cfg=cfg, logger=logger)
        >>> processor.initialize()
        >>> result = processor.process_work_unit(work_unit, metadata)
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize all four sub-components.

        Raises:
            ProcessingError: If any sub-component fails to initialize.
        """

    @abstractmethod
    def process_work_unit(
        self,
        work_unit: PageWorkUnit,
        metadata: PDFMetadata,
    ) -> DocumentResult:
        """Process all pages in a PageWorkUnit.

        Runs OCR, Layout Analysis, Table Extraction, and Figure Detection
        on each page in work_unit.start_page .. work_unit.end_page.

        Args:
            work_unit: The work unit specifying the page range.
            metadata: PDFMetadata for the parent document.

        Returns:
            DocumentResult aggregating all page results.

        Raises:
            ProcessingError: If processing fails unrecoverably.
        """

    @abstractmethod
    def extract_metadata(self, file_path: Path) -> PDFMetadata:
        """Extract and return PDFMetadata from a PDF file.

        Populates all fields defined in architecture v2.0 §2.4:
            pages, estimated_size_mb, resolution_dpi, source_type, language.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            Fully populated PDFMetadata record.

        Raises:
            ProcessingError: If the file cannot be opened or read.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully release all sub-component resources."""
