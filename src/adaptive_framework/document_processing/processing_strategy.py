"""Processing Strategy — Module 5 improvement: Strategy Pattern for page processing.

Replaces if/else branching on page type with a clean strategy hierarchy.

Architecture:
    Worker calls:
        strategy = ProcessingStrategyFactory.get_strategy(page_type)
        result = strategy.process(page, page_number, work_unit)

    Strategies:
        DirectExtractionStrategy  → digital pages (text layer)
        OCRStrategy               → scanned pages (PaddleOCR)
        MixedStrategy             → mixed pages (text + OCR)

This is the Strategy Pattern as approved in the architecture review.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from adaptive_framework.models.page import (
    BoundingBox,
    FigureData,
    LayoutElement,
    PageType,
    TableData,
    TextBlock,
)

logger = logging.getLogger(__name__)


@dataclass
class PageExtractionResult:
    """Raw extraction output from any processing strategy.

    Passed to PageObjectBuilder which assembles the final immutable Page.

    Attributes:
        text: Concatenated full text in reading order.
        text_blocks: List of individual text blocks.
        tables: List of extracted tables.
        figures: List of extracted figures.
        layout_elements: List of layout elements from Docling.
        ocr_confidence: Average OCR confidence (1.0 = direct extraction).
        processing_method: 'direct_text', 'ocr', 'mixed', 'skipped', 'failed'.
        error: Error message if extraction failed. None on success.
        warnings: Non-fatal warnings.
        pdf_load_time_s: Time spent loading the page image.
        text_extraction_time_s: Time spent in direct text extraction.
        ocr_time_s: Time spent in PaddleOCR.
        layout_time_s: Time spent in Docling layout analysis.
    """

    text: str = ""
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    figures: list[FigureData] = field(default_factory=list)
    layout_elements: list[LayoutElement] = field(default_factory=list)
    ocr_confidence: float = 1.0
    processing_method: str = "direct_text"
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    pdf_load_time_s: float = 0.0
    text_extraction_time_s: float = 0.0
    ocr_time_s: float = 0.0
    layout_time_s: float = 0.0
    ocr_engine: str | None = None
    ocr_device: str | None = None
    ocr_model: str | None = None


class IProcessingStrategy(ABC):
    """Abstract base class for all page processing strategies.

    Each strategy encapsulates the extraction logic for one page type.
    Workers call process() without knowing which strategy is active.

    Subclasses implement:
        process(page, page_number, document_id, file_path) → PageExtractionResult
    """

    @abstractmethod
    def process(
        self,
        page: Any,
        page_number: int,
        document_id: str,
        file_path: str,
    ) -> PageExtractionResult:
        """Process a single PDF page and return extraction results.

        Args:
            page: PyMuPDF Page object (already open, zero-copy).
            page_number: 1-indexed page number.
            document_id: Parent document identifier.
            file_path: Absolute path to the source PDF.

        Returns:
            PageExtractionResult with all extracted content.
        """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name of this strategy."""


class DirectExtractionStrategy(IProcessingStrategy):
    """Processing strategy for digital pages with a native text layer.

    Extracts text, blocks, fonts, and reading order directly from the
    PDF text layer using PyMuPDF. Never invokes OCR.

    Used for: PageType.DIGITAL
    """

    @property
    def strategy_name(self) -> str:
        """Name of this strategy."""
        return "direct_extraction"

    def process(
        self,
        page: Any,
        page_number: int,
        document_id: str,
        file_path: str,
    ) -> PageExtractionResult:
        """Extract text directly from the PDF text layer.

        Args:
            page: PyMuPDF Page object.
            page_number: 1-indexed page number.
            document_id: Parent document identifier.
            file_path: Source PDF path.

        Returns:
            PageExtractionResult with processing_method='direct_text'.
        """
        import time

        result = PageExtractionResult(processing_method="direct_text")

        try:
            t0 = time.perf_counter()

            # Extract text in reading order
            blocks_raw = page.get_text("dict", sort=True)
            text_parts: list[str] = []
            text_blocks: list[TextBlock] = []
            order = 0

            for block in blocks_raw.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue
                block_text_parts: list[str] = []
                font_name: str | None = None
                font_size: float | None = None

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            block_text_parts.append(span_text)
                            if font_name is None:
                                font_name = span.get("font")
                                font_size = span.get("size")

                block_text = " ".join(block_text_parts).strip()
                if not block_text:
                    continue

                bbox_raw = block.get("bbox", (0, 0, 0, 0))
                bbox = BoundingBox(
                    x0=bbox_raw[0], y0=bbox_raw[1],
                    x1=bbox_raw[2], y1=bbox_raw[3],
                )

                tb = TextBlock(
                    text=block_text,
                    bbox=bbox,
                    block_type="paragraph",
                    confidence=1.0,
                    font_name=font_name,
                    font_size=font_size,
                    reading_order=order,
                )
                text_blocks.append(tb)
                text_parts.append(block_text)
                order += 1

            result.text = "\n".join(text_parts)
            result.text_blocks = text_blocks
            result.ocr_confidence = 1.0
            result.text_extraction_time_s = time.perf_counter() - t0

        except Exception as exc:
            logger.warning(
                "DirectExtractionStrategy failed for page %d of '%s': %s",
                page_number, file_path, exc,
            )
            result.error = str(exc)
            result.processing_method = "failed"

        return result


class OCRStrategy(IProcessingStrategy):
    """Processing strategy for scanned pages requiring OCR.

    Uses PaddleOCR to extract text from rasterised page images.
    If PaddleOCR is not installed, falls back to StubOCRStrategy.

    Used for: PageType.SCANNED
    """

    def __init__(self, dpi: int = 150, lang: str = "en") -> None:
        """Initialise OCRStrategy.

        Args:
            dpi: Rendering resolution for rasterisation (default 150).
            lang: OCR language code (default 'en').
        """
        self._dpi = dpi
        self._lang = lang
        self._ocr_engine: Any = None
        self._ocr_available = self._init_ocr()

    def _init_ocr(self) -> bool:
        """Attempt to initialise PaddleOCR. Returns True on success."""
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                show_log=False,
            )
            logger.debug("PaddleOCR initialised (lang=%s).", self._lang)
            return True
        except ImportError:
            logger.info(
                "PaddleOCR not installed. OCR strategy uses stub (empty text)."
            )
            return False
        except Exception as exc:
            logger.warning("PaddleOCR initialisation failed: %s", exc)
            return False

    @property
    def strategy_name(self) -> str:
        """Name of this strategy."""
        return "ocr"

    def process(
        self,
        page: Any,
        page_number: int,
        document_id: str,
        file_path: str,
    ) -> PageExtractionResult:
        """Run OCR on a rasterised page image.

        Args:
            page: PyMuPDF Page object.
            page_number: 1-indexed page number.
            document_id: Parent document identifier.
            file_path: Source PDF path.

        Returns:
            PageExtractionResult with processing_method='ocr'.
        """
        import time
        result = PageExtractionResult(processing_method="ocr")

        if not self._ocr_available:
            result.warnings.append(
                "PaddleOCR not available. Returning empty text (stub mode)."
            )
            result.ocr_confidence = 0.0
            return result

        try:
            # Rasterise page to NumPy array (zero copy via pixmap)
            t_load = time.perf_counter()
            matrix = None
            try:
                import fitz  # type: ignore[import]
                scale = self._dpi / 72.0
                matrix = fitz.Matrix(scale, scale)
            except ImportError:
                pass

            if matrix is not None:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                import numpy as np
                img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
            else:
                result.warnings.append("PyMuPDF not available for rasterisation.")
                return result

            result.pdf_load_time_s = time.perf_counter() - t_load

            # Run OCR
            t_ocr = time.perf_counter()
            ocr_result = self._ocr_engine.ocr(img_array, cls=True)
            result.ocr_time_s = time.perf_counter() - t_ocr

            # Parse results
            text_parts: list[str] = []
            text_blocks: list[TextBlock] = []
            confidences: list[float] = []
            order = 0

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if not line or len(line) < 2:
                        continue
                    bbox_pts = line[0]  # [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
                    text_conf = line[1]
                    text = text_conf[0] if text_conf else ""
                    conf = float(text_conf[1]) if len(text_conf) > 1 else 0.0

                    if not text.strip():
                        continue

                    x_coords = [pt[0] for pt in bbox_pts]
                    y_coords = [pt[1] for pt in bbox_pts]
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
            result.ocr_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

        except Exception as exc:
            logger.warning(
                "OCRStrategy failed for page %d of '%s': %s",
                page_number, file_path, exc,
            )
            result.error = str(exc)
            result.processing_method = "failed"

        return result


class MixedStrategy(IProcessingStrategy):
    """Processing strategy for mixed pages (text layer + significant images).

    Combines DirectExtractionStrategy for the text layer and OCRStrategy
    for image regions. Merges results in reading order.

    Used for: PageType.MIXED
    """

    def __init__(self, dpi: int = 150, lang: str = "en") -> None:
        """Initialise MixedStrategy.

        Args:
            dpi: OCR rendering DPI.
            lang: OCR language code.
        """
        self._direct = DirectExtractionStrategy()
        self._ocr = OCRStrategy(dpi=dpi, lang=lang)

    @property
    def strategy_name(self) -> str:
        """Name of this strategy."""
        return "mixed"

    def process(
        self,
        page: Any,
        page_number: int,
        document_id: str,
        file_path: str,
    ) -> PageExtractionResult:
        """Apply both direct extraction and OCR, merging results.

        Args:
            page: PyMuPDF Page object.
            page_number: 1-indexed page number.
            document_id: Parent document identifier.
            file_path: Source PDF path.

        Returns:
            PageExtractionResult with processing_method='mixed'.
        """
        direct_result = self._direct.process(page, page_number, document_id, file_path)
        ocr_result = self._ocr.process(page, page_number, document_id, file_path)

        # Merge text blocks (direct first, then OCR)
        all_blocks = direct_result.text_blocks + ocr_result.text_blocks
        all_text_parts = []
        if direct_result.text:
            all_text_parts.append(direct_result.text)
        if ocr_result.text:
            all_text_parts.append(ocr_result.text)

        # Weight confidence: prefer direct extraction (confidence=1.0)
        n_direct = len(direct_result.text_blocks)
        n_ocr = len(ocr_result.text_blocks)
        if n_direct + n_ocr > 0:
            combined_confidence = (
                (n_direct * 1.0 + n_ocr * ocr_result.ocr_confidence)
                / (n_direct + n_ocr)
            )
        else:
            combined_confidence = 1.0

        warnings = direct_result.warnings + ocr_result.warnings
        error = direct_result.error or ocr_result.error

        return PageExtractionResult(
            text="\n".join(all_text_parts),
            text_blocks=all_blocks,
            tables=direct_result.tables + ocr_result.tables,
            figures=direct_result.figures + ocr_result.figures,
            layout_elements=direct_result.layout_elements,
            ocr_confidence=round(combined_confidence, 3),
            processing_method="mixed",
            error=error,
            warnings=warnings,
            pdf_load_time_s=max(direct_result.pdf_load_time_s, ocr_result.pdf_load_time_s),
            text_extraction_time_s=direct_result.text_extraction_time_s,
            ocr_time_s=ocr_result.ocr_time_s,
            layout_time_s=direct_result.layout_time_s,
        )


class SkippedStrategy(IProcessingStrategy):
    """Strategy used for pages that should be skipped (blank, unreadable).

    Returns an empty PageExtractionResult with method='skipped'.
    """

    @property
    def strategy_name(self) -> str:
        """Name of this strategy."""
        return "skipped"

    def process(
        self,
        page: Any,
        page_number: int,
        document_id: str,
        file_path: str,
    ) -> PageExtractionResult:
        """Return empty result for a skipped page.

        Args:
            page: PyMuPDF Page object (not used).
            page_number: 1-indexed page number.
            document_id: Not used.
            file_path: Not used.

        Returns:
            Empty PageExtractionResult with processing_method='skipped'.
        """
        return PageExtractionResult(
            processing_method="skipped",
            warnings=[f"Page {page_number} was skipped."],
        )


class AdaptiveRoutingStrategyProxy(IProcessingStrategy):
    """Adaptive routing proxy that routes scanned page OCR between CPU and GPU.

    Consumes real-time G3 ResourceSnapshot telemetry and WorkloadProfile
    to decide between GPU and CPU execution. If GPU execution fails at runtime,
    it records the event, logs the error, and automatically falls back to CPU OCR.
    """

    def __init__(
        self,
        gpu_strategy: IProcessingStrategy,
        cpu_strategy: IProcessingStrategy,
        router: Any | None = None,
        resource_monitor: Any | None = None,
    ) -> None:
        """Initialize the adaptive routing proxy.

        Args:
            gpu_strategy: Strategy instance targeting GPU.
            cpu_strategy: Strategy instance targeting CPU.
            router: AdaptiveWorkRouter instance (created if None).
            resource_monitor: ResourceMonitor instance for runtime telemetry.
        """
        self._gpu_strategy = gpu_strategy
        self._cpu_strategy = cpu_strategy
        self._router = router
        self._monitor = resource_monitor
        self._characterizer: Any = None

        if self._router is None:
            try:
                from adaptive_framework.acceleration.adaptive_router import AdaptiveWorkRouter
                self._router = AdaptiveWorkRouter()
            except Exception as exc:
                logger.warning("Could not instantiate AdaptiveWorkRouter: %s", exc)

        try:
            from adaptive_framework.acceleration.workload_characterizer import WorkloadCharacterizer
            self._characterizer = WorkloadCharacterizer()
        except Exception as exc:
            logger.warning("Could not instantiate WorkloadCharacterizer: %s", exc)

    @property
    def strategy_name(self) -> str:
        return "adaptive_routing_proxy"

    @property
    def router(self) -> Any:
        return self._router

    def process(
        self,
        page: Any,
        page_number: int,
        document_id: str,
        file_path: str,
        workload_profile: Any | None = None,
        resource_snapshot: Any | None = None,
    ) -> PageExtractionResult:
        """Process page using dynamically selected device with seamless CPU fallback."""
        # 1. Workload profile
        if workload_profile is None and self._characterizer is not None and page is not None:
            try:
                workload_profile = self._characterizer.characterize_page(
                    page, page_number=page_number, document_id=document_id
                )
            except Exception as exc:
                logger.debug("Workload characterization fallback: %s", exc)

        if workload_profile is None:
            try:
                from adaptive_framework.acceleration.workload_characterizer import WorkloadProfile
                workload_profile = WorkloadProfile(
                    document_id=document_id,
                    page_number=page_number,
                    is_scanned=True,
                    estimated_complexity=0.5,
                )
            except Exception:
                pass

        # 2. Resource snapshot
        if resource_snapshot is None and self._monitor is not None:
            try:
                resource_snapshot = self._monitor.sample()
            except Exception as exc:
                logger.debug("Resource snapshot query fallback: %s", exc)

        # 3. Route decision
        target_device = "CPU"
        decision = None
        if self._router is not None and workload_profile is not None:
            try:
                decision = self._router.route(workload_profile, resource_snapshot)
                target_device = decision.target_device
            except Exception as exc:
                logger.warning("Adaptive router error (%s), defaulting to CPU", exc)
                target_device = "CPU"

        # 4. Execute on selected device
        if target_device == "GPU":
            try:
                if hasattr(self._gpu_strategy, "model_loaded") and not self._gpu_strategy.model_loaded:
                    self._gpu_strategy.initialize()
                result = self._gpu_strategy.process(page, page_number, document_id, file_path)
                result.ocr_device = "GPU"
                return result
            except Exception as exc:
                # Catch GPU execution failure and trigger fallback
                logger.error(
                    "GPU execution failure on page %d of '%s': %s. Executing CPU fallback.",
                    page_number, file_path, exc,
                )
                if self._router is not None and workload_profile is not None:
                    self._router.record_execution_fallback(
                        failed_device="GPU",
                        fallback_device="CPU",
                        error_message=str(exc),
                        workload=workload_profile,
                    )
                if hasattr(self._cpu_strategy, "model_loaded") and not self._cpu_strategy.model_loaded:
                    self._cpu_strategy.initialize()
                fallback_result = self._cpu_strategy.process(page, page_number, document_id, file_path)
                fallback_result.warnings.append(
                    f"GPU execution failed ({exc}); fell back to CPU OCR."
                )
                fallback_result.ocr_device = "CPU_FALLBACK"
                return fallback_result
        else:
            if hasattr(self._cpu_strategy, "model_loaded") and not self._cpu_strategy.model_loaded:
                self._cpu_strategy.initialize()
            result = self._cpu_strategy.process(page, page_number, document_id, file_path)
            result.ocr_device = "CPU"
            return result


class ProcessingStrategyFactory:
    """Factory that selects the correct processing strategy for a page type.

    Usage:
        >>> factory = ProcessingStrategyFactory()
        >>> strategy = factory.get_strategy(PageType.SCANNED)
        >>> result = strategy.process(page, page_number, doc_id, path)

    Workers call get_strategy() — no if/else on page type anywhere else.
    """

    def __init__(
        self,
        ocr_dpi: int = 150,
        ocr_lang: str = "en",
        ocr_backend: str = "paddleocr",
        ocr_device: str = "GPU",
        ocr_strategy: IProcessingStrategy | None = None,
        router: Any | None = None,
        resource_monitor: Any | None = None,
        enable_adaptive_routing: bool = False,
    ) -> None:
        """Initialise factory with shared strategy instances.

        Args:
            ocr_dpi: DPI for OCR rasterisation.
            ocr_lang: Language code for OCR.
            ocr_backend: OCR engine to use ('paddleocr', 'openvino').
            ocr_device: Execution device for OpenVINO ('GPU', 'CPU', 'AUTO').
            ocr_strategy: Explicit strategy instance override.
            router: Optional AdaptiveWorkRouter instance.
            resource_monitor: Optional ResourceMonitor instance.
            enable_adaptive_routing: If True, uses AdaptiveRoutingStrategyProxy for OCR.
        """
        self._router = router
        self._resource_monitor = resource_monitor

        if ocr_strategy is not None:
            scanned_strat = ocr_strategy
        elif enable_adaptive_routing and ocr_backend.lower() == "openvino":
            try:
                from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy
                gpu_strat = OpenVINOOCRStrategy(device="GPU", ocr_dpi=ocr_dpi)
                cpu_strat = OpenVINOOCRStrategy(device="CPU", ocr_dpi=ocr_dpi)
                scanned_strat = AdaptiveRoutingStrategyProxy(
                    gpu_strategy=gpu_strat,
                    cpu_strategy=cpu_strat,
                    router=self._router,
                    resource_monitor=self._resource_monitor,
                )
                self._router = scanned_strat.router
            except Exception as exc:
                logger.warning(
                    "Could not configure AdaptiveRoutingStrategyProxy (%s), falling back to single OCR strategy",
                    exc,
                )
                scanned_strat = OCRStrategy(dpi=ocr_dpi, lang=ocr_lang)
        elif ocr_backend.lower() == "openvino":
            try:
                from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy
                scanned_strat = OpenVINOOCRStrategy(device=ocr_device, ocr_dpi=ocr_dpi)
            except Exception as exc:
                logger.warning("Could not instantiate OpenVINOOCRStrategy (%s), falling back to OCRStrategy", exc)
                scanned_strat = OCRStrategy(dpi=ocr_dpi, lang=ocr_lang)
        else:
            scanned_strat = OCRStrategy(dpi=ocr_dpi, lang=ocr_lang)

        self._strategies: dict[PageType, IProcessingStrategy] = {
            PageType.DIGITAL: DirectExtractionStrategy(),
            PageType.SCANNED: scanned_strat,
            PageType.MIXED: MixedStrategy(dpi=ocr_dpi, lang=ocr_lang),
            PageType.UNKNOWN: DirectExtractionStrategy(),  # best-effort
        }

    @property
    def router(self) -> Any:
        """Return the active AdaptiveWorkRouter if adaptive routing is enabled."""
        scanned = self._strategies.get(PageType.SCANNED)
        if isinstance(scanned, AdaptiveRoutingStrategyProxy):
            return scanned.router
        return self._router

    def get_strategy(self, page_type: PageType) -> IProcessingStrategy:
        """Return the strategy for the given page type.

        Args:
            page_type: Classification of the page.

        Returns:
            Appropriate IProcessingStrategy instance.
        """
        strategy = self._strategies.get(page_type)
        if strategy is None:
            logger.warning(
                "No strategy for page type %s. Using DirectExtractionStrategy.",
                page_type.value,
            )
            return self._strategies[PageType.DIGITAL]
        return strategy

    def available_strategies(self) -> list[str]:
        """Return names of all registered strategies.

        Returns:
            List of strategy name strings.
        """
        return [s.strategy_name for s in self._strategies.values()]
