"""IOCREngine — Abstract OCR engine interface.

This is the boundary between the Document Processing Engine and
the OCR backend. The scheduler and coordinator never import any
concrete OCR class — only this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from adaptive_framework.models.document import PageMetadata, PageResult


class IOCREngine(ABC):
    """Abstract interface for an OCR engine backend.

    Concrete implementations may use:
        - PaddleOCR (Phase 2 default)
        - TrOCR
        - Nougat
        - MinerU
        - Docling

    All backends must return the same PageResult model, ensuring that
    swapping backends requires zero changes to any other component.

    Example:
        >>> engine: IOCREngine = PaddleOCREngine(cfg.ocr, logger)
        >>> engine.initialize()
        >>> result = engine.process_page("/data/paper.pdf", page_number=1, metadata=meta)
        >>> print(result.text[:50])
        'Introduction to biomedical information extraction'
    """

    @abstractmethod
    def initialize(self) -> None:
        """Load models and prepare the OCR engine for processing.

        Raises:
            OCRError: If the backend fails to load or initialize.
        """

    @abstractmethod
    def process_page(
        self,
        file_path: Path,
        page_number: int,
        metadata: PageMetadata,
    ) -> PageResult:
        """Run OCR on a single page of a PDF document.

        Args:
            file_path: Absolute path to the source PDF.
            page_number: 1-indexed page number to process.
            metadata: PageMetadata for this page (dimensions, layer info).

        Returns:
            PageResult with extracted text and processing metadata.

        Raises:
            OCRError: If OCR fails on this page.
        """

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the identifier of this OCR backend.

        Returns:
            Backend name string (e.g., 'paddleocr', 'trocr').
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release model weights and free resources held by this engine."""
