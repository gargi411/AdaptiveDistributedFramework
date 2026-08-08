"""PaddleOCR Engine — Module 5: OCR for scanned and mixed pages.

Runs ONLY on pages classified as SCANNED or MIXED.
Never called on DIGITAL pages.

Features:
    - Batch OCR
    - GPU support (auto-detected)
    - Confidence scores per word/block
    - Bounding boxes
    - Rotation correction (angle classifier)
    - Language detection support

Graceful degradation:
    If PaddleOCR is not installed (Windows dev, CI), all methods return
    stub results with empty text and ocr_confidence=0.0.
    All unit tests pass without PaddleOCR installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_framework.models.page import BoundingBox, TextBlock

logger = logging.getLogger(__name__)

# Attempt import — graceful degradation if unavailable
try:
    from paddleocr import PaddleOCR  # type: ignore[import]
    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PADDLEOCR_AVAILABLE = False
    logger.info(
        "PaddleOCR not installed. OCR engine uses stub mode. "
        "Install on Linux: pip install paddlepaddle paddleocr"
    )


@dataclass
class OCRPageResult:
    """OCR result for a single page.

    Attributes:
        page_number: 1-indexed page number.
        text: Full extracted text.
        text_blocks: Ordered list of TextBlock objects.
        ocr_confidence: Average confidence across all blocks [0.0, 1.0].
        ocr_time_seconds: Time spent running OCR.
        word_count: Total words detected.
        char_count: Total characters detected.
        language_detected: ISO 639-1 code if detected. None otherwise.
        success: True if OCR completed without critical error.
        error: Error message if success is False.
        stub_mode: True if results are from stub (PaddleOCR unavailable).
    """

    page_number: int
    text: str = ""
    text_blocks: list[TextBlock] = field(default_factory=list)
    ocr_confidence: float = 0.0
    ocr_time_seconds: float = 0.0
    word_count: int = 0
    char_count: int = 0
    language_detected: str | None = None
    success: bool = True
    error: str | None = None
    stub_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "page_number": self.page_number,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "ocr_confidence": self.ocr_confidence,
            "ocr_time_seconds": self.ocr_time_seconds,
            "language_detected": self.language_detected,
            "success": self.success,
            "error": self.error,
            "stub_mode": self.stub_mode,
        }


class PaddleOCREngine:
    """PaddleOCR wrapper for the Document Processing Engine.

    Usage:
        >>> engine = PaddleOCREngine(lang="en", use_gpu=False)
        >>> result = engine.ocr_page_array(image_array, page_number=3)
        >>> result.ocr_confidence
        0.92

    Args:
        lang: OCR language code (default 'en').
        use_gpu: Attempt GPU acceleration if True. Falls back to CPU.
        use_angle_cls: Enable rotation/angle correction classifier.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
    ) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._use_angle_cls = use_angle_cls
        self._engine: Any = None
        self._available = _PADDLEOCR_AVAILABLE
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialise PaddleOCR engine. Silently falls back on failure."""
        if not _PADDLEOCR_AVAILABLE:
            return
        try:
            self._engine = PaddleOCR(
                use_angle_cls=self._use_angle_cls,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.debug(
                "PaddleOCR engine ready (lang=%s, gpu=%s).",
                self._lang, self._use_gpu,
            )
        except Exception as exc:
            logger.warning("PaddleOCR init failed: %s — using stub mode.", exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        """Return True if the OCR engine is initialised and ready."""
        return self._available and self._engine is not None

    def ocr_page_array(
        self,
        image_array: Any,
        page_number: int,
    ) -> OCRPageResult:
        """Run OCR on a NumPy image array.

        Args:
            image_array: HxWxC NumPy uint8 array (RGB or grayscale).
            page_number: 1-indexed page number (for result labelling).

        Returns:
            OCRPageResult with text, blocks, and confidence.
        """
        import time

        if not self.is_available or image_array is None:
            return OCRPageResult(
                page_number=page_number,
                stub_mode=True,
                success=True,
                ocr_confidence=0.0,
            )

        t0 = time.perf_counter()
        try:
            raw = self._engine.ocr(image_array, cls=self._use_angle_cls)
            elapsed = time.perf_counter() - t0
            return self._parse_result(raw, page_number, elapsed)
        except Exception as exc:
            logger.warning(
                "PaddleOCR failed on page %d: %s", page_number, exc
            )
            return OCRPageResult(
                page_number=page_number,
                success=False,
                error=str(exc),
                ocr_time_seconds=time.perf_counter() - t0,
            )

    def ocr_batch(
        self,
        image_arrays: list[Any],
        start_page: int = 1,
    ) -> list[OCRPageResult]:
        """Run OCR on a batch of page images.

        Args:
            image_arrays: List of HxWxC NumPy uint8 arrays.
            start_page: 1-indexed page number of the first image.

        Returns:
            List of OCRPageResult, one per image.
        """
        return [
            self.ocr_page_array(img, start_page + i)
            for i, img in enumerate(image_arrays)
        ]

    @staticmethod
    def _parse_result(
        raw: Any,
        page_number: int,
        ocr_time_seconds: float,
    ) -> OCRPageResult:
        """Parse raw PaddleOCR output into structured OCRPageResult.

        Args:
            raw: Raw PaddleOCR result (list of lists).
            page_number: 1-indexed page number.
            ocr_time_seconds: Elapsed OCR time.

        Returns:
            Structured OCRPageResult.
        """
        result = OCRPageResult(
            page_number=page_number,
            ocr_time_seconds=ocr_time_seconds,
        )

        if not raw or not raw[0]:
            result.success = True
            result.ocr_confidence = 0.0
            return result

        text_parts: list[str] = []
        text_blocks: list[TextBlock] = []
        confidences: list[float] = []
        order = 0

        for line in raw[0]:
            if not line or len(line) < 2:
                continue

            bbox_pts = line[0]  # [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
            text_conf = line[1]
            text = text_conf[0] if text_conf else ""
            conf = float(text_conf[1]) if len(text_conf) > 1 else 0.0

            if not text.strip():
                continue

            x_coords = [float(pt[0]) for pt in bbox_pts]
            y_coords = [float(pt[1]) for pt in bbox_pts]
            bbox = BoundingBox(
                x0=min(x_coords), y0=min(y_coords),
                x1=max(x_coords), y1=max(y_coords),
            )

            tb = TextBlock(
                text=text,
                bbox=bbox,
                confidence=conf,
                reading_order=order,
            )
            text_blocks.append(tb)
            text_parts.append(text)
            confidences.append(conf)
            order += 1

        result.text = "\n".join(text_parts)
        result.text_blocks = text_blocks
        result.char_count = len(result.text)
        result.word_count = len(result.text.split()) if result.text else 0
        result.ocr_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        result.success = True
        return result
