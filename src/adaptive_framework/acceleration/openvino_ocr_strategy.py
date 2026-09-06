"""OpenVINOOCRStrategy -- OpenVINO-accelerated OCR processing strategy.

Phase G2: Real OpenVINO OCR Integration
========================================
This module provides the architecture, model compilation, device-selection,
and executable inference pipeline for GPU-accelerated OCR via Intel OpenVINO Runtime.

Features:
  - HardwareCapabilityProbe integration (device selection at init time)
  - OpenVINO Core initialisation with GPU > CPU > AUTO device preference
  - Two-stage OCR execution using official Intel Open Model Zoo FP16 models:
      1. Text Detection:   horizontal-text-detection-0001 (FCOS-based, 704x704 input)
      2. Text Recognition: text-recognition-0012 (VGG+BiLSTM, CTC greedy decoding)
  - Pure Pillow + NumPy image preprocessing (no OpenCV dependency required)
  - Batched GPU inference for recognized text regions
  - Full provenance preservation (document_id, page_number, ocr_engine, ocr_device, ocr_model)
  - Preserves Phase G1 backward compatibility (stub mode when model not loaded)
  - Never crashes worker processes; all exceptions caught and reflected in PageExtractionResult

Interface compatibility:
  - Implements IProcessingStrategy exactly as defined in processing_strategy.py.
  - Accepts PyMuPDF fitz.Page or NumPy images.
  - Returns PageExtractionResult with full provenance in all code paths.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adaptive_framework.acceleration.hardware_probe import (
    HardwareCapabilityProbe,
    HardwareProfile,
)
from adaptive_framework.acceleration.model_setup import ensure_openvino_ocr_models
from adaptive_framework.document_processing.processing_strategy import (
    IProcessingStrategy,
    PageExtractionResult,
)
from adaptive_framework.models.page import BoundingBox, TextBlock

logger = logging.getLogger(__name__)

# OpenVINO Runtime -- optional dependency.
try:
    import openvino as ov  # type: ignore[import]
    _OPENVINO_AVAILABLE = True
except ImportError:
    _OPENVINO_AVAILABLE = False
    ov = None  # type: ignore[assignment]

# CTC Alphabet for text-recognition-0012 (36 alphanumeric symbols + blank '#')
ALPHABET_0012 = "0123456789abcdefghijklmnopqrstuvwxyz#"
BLANK_INDEX_0012 = len(ALPHABET_0012) - 1


class OpenVINOOCRStrategy(IProcessingStrategy):
    """OpenVINO-accelerated OCR strategy with real GPU inference.

    Implements IProcessingStrategy with the same interface as OCRStrategy.
    Accepts a PyMuPDF fitz.Page or raw NumPy array, performs two-stage
    text detection and recognition on Intel GPU or CPU, and outputs
    structured TextBlocks and plain text.

    Args:
        device: OpenVINO execution device: 'AUTO', 'GPU', 'CPU'.
        ocr_dpi: DPI for rasterising scanned pages (default 150).
        probe: HardwareCapabilityProbe to use. Auto-created if None.
        det_model_path: Optional path to horizontal-text-detection-0001 XML.
        rec_model_path: Optional path to text-recognition-0012 XML.
        det_threshold: Minimum confidence score for detected text boxes (default 0.3).
        batch_size: Batch size for text recognition inference (default 16).
        auto_initialize: If True, immediately loads and compiles models on init.
    """

    def __init__(
        self,
        device: str = "AUTO",
        ocr_dpi: int = 150,
        probe: HardwareCapabilityProbe | None = None,
        det_model_path: str | Path | None = None,
        rec_model_path: str | Path | None = None,
        det_threshold: float = 0.3,
        batch_size: int = 16,
        auto_initialize: bool = False,
    ) -> None:
        self._requested_device = device.upper()
        self._dpi = ocr_dpi
        self._probe = probe or HardwareCapabilityProbe()
        self._det_threshold = det_threshold
        self._batch_size = batch_size

        self._det_model_path: str | None = str(det_model_path) if det_model_path else None
        self._rec_model_path: str | None = str(rec_model_path) if rec_model_path else None

        # Runtime state
        self._profile: HardwareProfile | None = None
        self._core: Any = None
        self._model: Any = None        # Alias for compiled model (backward-compatibility)
        self._det_model: Any = None    # Compiled detection model
        self._rec_model: Any = None    # Compiled recognition model
        self._active_device: str = "cpu_stub"
        self._compiled_device: str = "none"
        self._initialized: bool = False

        self._init_openvino()

        if auto_initialize:
            self.initialize()

    # ------------------------------------------------------------------ #
    # Initialisation & Device Selection                                  #
    # ------------------------------------------------------------------ #

    def _init_openvino(self) -> None:
        """Initialise OpenVINO Core and select target execution device."""
        self._profile = self._probe.probe()

        if not _OPENVINO_AVAILABLE:
            logger.info("OpenVINO not installed. OpenVINOOCRStrategy in cpu_stub mode.")
            self._active_device = "cpu_stub"
            self._compiled_device = "cpu_stub"
            return

        try:
            self._core = ov.Core()  # type: ignore[union-attr]

            if self._requested_device == "AUTO":
                if self._profile.openvino.gpu_device_available:
                    self._active_device = "GPU"
                else:
                    self._active_device = "CPU"
            else:
                self._active_device = self._requested_device

            self._initialized = True
            logger.info(
                "OpenVINO Core initialised. Selected device: %s.",
                self._active_device,
            )

        except Exception as exc:
            logger.warning(
                "OpenVINO Core() initialisation failed: %s. Falling back to cpu_stub mode.",
                exc,
            )
            self._core = None
            self._active_device = "cpu_stub"
            self._compiled_device = "cpu_stub"
            self._initialized = False

    def initialize(
        self,
        det_model_path: str | Path | None = None,
        rec_model_path: str | Path | None = None,
    ) -> bool:
        """Load and compile both detection and recognition models.

        If model paths are not specified, ensures models are present via
        the Open Model Zoo downloader utility.

        Returns:
            True if models were successfully compiled, False otherwise.
        """
        if not _OPENVINO_AVAILABLE or self._core is None:
            logger.warning("initialize() called but OpenVINO runtime is not available.")
            return False

        det_path = det_model_path or self._det_model_path
        rec_path = rec_model_path or self._rec_model_path

        if not det_path or not rec_path:
            try:
                auto_det, auto_rec = ensure_openvino_ocr_models()
                det_path = det_path or str(auto_det)
                rec_path = rec_path or str(auto_rec)
            except Exception as exc:
                logger.error("Failed to automatically acquire OpenVINO OCR models: %s", exc)
                return False

        try:
            logger.info("Compiling detection model on %s: %s", self._active_device, det_path)
            det_ov = self._core.read_model(str(det_path))
            self._det_model = self._core.compile_model(det_ov, self._active_device)

            logger.info("Compiling recognition model on %s: %s", self._active_device, rec_path)
            rec_ov = self._core.read_model(str(rec_path))
            # Shape for batch recognition: dynamic batch size [B, 32, 120, 1]
            rec_ov.reshape([-1, 32, 120, 1])
            self._rec_model = self._core.compile_model(rec_ov, self._active_device)
            self._model = self._rec_model  # alias for backward-compatibility

            self._det_model_path = str(det_path)
            self._rec_model_path = str(rec_path)

            # Query actual compiled execution device
            devices = self._rec_model.get_property("EXECUTION_DEVICES")
            self._compiled_device = devices[0] if devices else self._active_device
            logger.info(
                "OpenVINO OCR models successfully compiled on device '%s' (requested: '%s').",
                self._compiled_device,
                self._active_device,
            )
            return True

        except Exception as exc:
            logger.error("Failed to compile OpenVINO OCR models on device '%s': %s", self._active_device, exc)
            return False

    def load_model(
        self,
        model_path: str | Path | None = None,
        det_model_path: str | Path | None = None,
        rec_model_path: str | Path | None = None,
    ) -> bool:
        """Load and compile an OpenVINO model (preserves Phase G1 signature).

        If a single model_path is passed:
            - If it represents detection, sets detection model.
            - Otherwise compiles and sets self._model and self._rec_model.
        If both det and rec paths are passed, compiles both via initialize().
        """
        if not _OPENVINO_AVAILABLE or self._core is None:
            logger.warning("load_model() called but OpenVINO is not available.")
            return False

        if det_model_path is not None or rec_model_path is not None:
            return self.initialize(det_model_path=det_model_path, rec_model_path=rec_model_path)

        if model_path is None:
            return self.initialize()

        try:
            model = self._core.read_model(str(model_path))
            compiled = self._core.compile_model(model, self._active_device)
            self._model = compiled
            self._rec_model = compiled

            devices = compiled.get_property("EXECUTION_DEVICES")
            self._compiled_device = devices[0] if devices else self._active_device
            logger.info("OpenVINO OCR model loaded: path='%s', device='%s'.", model_path, self._compiled_device)
            return True

        except Exception as exc:
            logger.error("Failed to load/compile OpenVINO model from '%s': %s", model_path, exc)
            return False

    # ------------------------------------------------------------------ #
    # Properties & Status                                                #
    # ------------------------------------------------------------------ #

    @property
    def strategy_name(self) -> str:
        """Return the strategy name including selected device."""
        return f"openvino_ocr_{self._active_device.lower()}"

    @property
    def is_gpu_available(self) -> bool:
        """Return True if GPU device is usable through this strategy."""
        if not self._initialized or self._profile is None:
            return False
        return self._profile.openvino.gpu_device_available and self._active_device == "GPU"

    @property
    def active_device(self) -> str:
        """Return active execution device string ('GPU', 'CPU', 'cpu_stub')."""
        return self._active_device

    @property
    def compiled_device(self) -> str:
        """Return actual compiled device string (e.g. 'GPU.0', 'CPU', 'none')."""
        return self._compiled_device

    @property
    def model_loaded(self) -> bool:
        """Return True if an OCR model is compiled and ready for inference."""
        return self._model is not None or self._rec_model is not None

    @property
    def openvino_initialized(self) -> bool:
        """Return True if OpenVINO Core was successfully initialised."""
        return self._initialized

    def is_available(self) -> bool:
        """Return True if OpenVINO is available and functional."""
        return self._initialized and _OPENVINO_AVAILABLE

    def get_device_info(self) -> dict[str, Any]:
        """Return structured diagnostic dictionary of device state."""
        info: dict[str, Any] = {
            "requested_device": self._requested_device,
            "active_device": self._active_device,
            "compiled_device": self._compiled_device,
            "openvino_installed": _OPENVINO_AVAILABLE,
            "available_devices": [],
            "full_device_name": "Unknown",
            "is_gpu": self._active_device == "GPU" or "GPU" in self._compiled_device,
        }

        if self._core is not None:
            try:
                info["available_devices"] = list(self._core.available_devices)
                if self._active_device in self._core.available_devices:
                    info["full_device_name"] = self._core.get_property(self._active_device, "FULL_DEVICE_NAME")
            except Exception as exc:
                info["device_query_error"] = str(exc)

        return info

    # ------------------------------------------------------------------ #
    # Two-Stage Inference Execution                                      #
    # ------------------------------------------------------------------ #

    def recognize(self, image: np.ndarray) -> tuple[str, list[TextBlock], float]:
        """Execute text detection and recognition on a NumPy image array.

        Args:
            image: NumPy uint8 array of shape (H, W, C) or (H, W).

        Returns:
            tuple[str, list[TextBlock], float]:
                (full_extracted_text, list_of_textblocks, average_confidence)
        """
        if not self.model_loaded:
            success = self.initialize()
            if not success:
                logger.warning("recognize() called but models could not be compiled.")
                return "", [], 0.0

        if image is None or image.size == 0:
            return "", [], 0.0

        # Ensure 3 channels RGB
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = Image.fromarray(image)

        orig_w, orig_h = pil_img.size

        # ── Step 1: Text Detection ───────────────────────────────────────
        boxes: list[list[float]] = []
        if self._det_model is not None:
            boxes = self._detect_text_regions(pil_img)
        else:
            # If only recognition model is loaded (mock/single-model test), treat full image as box
            boxes = [[0.0, 0.0, float(orig_w), float(orig_h), 1.0]]

        if not boxes:
            logger.debug("No text regions detected by detection model.")
            return "", [], 0.0

        # ── Step 2: Text Recognition ─────────────────────────────────────
        text_blocks, confidences = self._recognize_text_regions(pil_img, boxes)

        # Sort text blocks in natural reading order (top-to-bottom, left-to-right)
        text_blocks.sort(key=lambda b: (round(b.bbox.y0 / 15.0) * 15.0, b.bbox.x0))

        full_text = "\n".join(b.text for b in text_blocks if b.text.strip())
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return full_text, text_blocks, avg_confidence

    def recognize_batch(self, images: list[np.ndarray]) -> list[tuple[str, list[TextBlock], float]]:
        """Batch process multiple images through OCR recognition."""
        return [self.recognize(img) for img in images]

    def _detect_text_regions(self, pil_img: Image.Image) -> list[list[float]]:
        """Run text detection model on input image and return filtered bounding boxes."""
        orig_w, orig_h = pil_img.size

        # Resize to 704x704 for horizontal-text-detection-0001
        det_pil = pil_img.resize((704, 704), Image.Resampling.BILINEAR)
        det_arr = np.array(det_pil, dtype=np.float32)

        # Convert RGB to BGR and format to NCHW [1, 3, 704, 704]
        det_bgr = det_arr[:, :, ::-1]
        inp_det = np.expand_dims(np.transpose(det_bgr, (2, 0, 1)), 0)

        # Infer
        det_out = self._det_model([inp_det])
        raw_boxes = det_out[0]  # shape [100, 5]: [x_min, y_min, x_max, y_max, conf]

        scale_x = orig_w / 704.0
        scale_y = orig_h / 704.0

        valid_boxes: list[list[float]] = []
        for b in raw_boxes:
            conf = float(b[4])
            if conf >= self._det_threshold:
                x0 = max(0.0, float(b[0] * scale_x))
                y0 = max(0.0, float(b[1] * scale_y))
                x1 = min(float(orig_w), float(b[2] * scale_x))
                y1 = min(float(orig_h), float(b[3] * scale_y))
                if (x1 - x0) >= 3.0 and (y1 - y0) >= 3.0:
                    valid_boxes.append([x0, y0, x1, y1, conf])

        return valid_boxes

    def _recognize_text_regions(
        self,
        pil_img: Image.Image,
        boxes: list[list[float]],
    ) -> tuple[list[TextBlock], list[float]]:
        """Crop text regions, run batched text recognition, and decode characters."""
        scale_pt = 72.0 / float(self._dpi)
        gray_pil = pil_img.convert("L")

        crops: list[np.ndarray] = []
        valid_coords: list[tuple[float, float, float, float, float]] = []

        for b in boxes:
            x0, y0, x1, y1, conf = b
            crop_pil = gray_pil.crop((int(x0), int(y0), int(x1), int(y1))).resize((120, 32), Image.Resampling.BILINEAR)
            crop_arr = (np.array(crop_pil, dtype=np.float32) / 255.0)[..., np.newaxis]  # [32, 120, 1]
            crops.append(crop_arr)
            valid_coords.append((x0, y0, x1, y1, conf))

        if not crops:
            return [], []

        batch_arr = np.array(crops, dtype=np.float32)  # [B, 32, 120, 1]
        rec_out = self._rec_model([batch_arr])[0]       # [30, B, 37]

        text_blocks: list[TextBlock] = []
        confidences: list[float] = []

        for idx, (x0, y0, x1, y1, det_conf) in enumerate(valid_coords):
            preds = rec_out[:, idx, :]  # [30, 37]
            decoded_text, char_conf = self._decode_ctc(preds)

            if decoded_text.strip():
                # Scale box to PDF points
                pt_bbox = BoundingBox(
                    x0=x0 * scale_pt,
                    y0=y0 * scale_pt,
                    x1=x1 * scale_pt,
                    y1=y1 * scale_pt,
                )
                combined_conf = round(float(det_conf * 0.4 + char_conf * 0.6), 4)
                tb = TextBlock(
                    text=decoded_text,
                    bbox=pt_bbox,
                    confidence=combined_conf,
                    reading_order=idx,
                    block_type="paragraph",
                )
                text_blocks.append(tb)
                confidences.append(combined_conf)

        return text_blocks, confidences

    @staticmethod
    def _decode_ctc(preds: np.ndarray) -> tuple[str, float]:
        """CTC greedy decoding for text-recognition-0012 logits."""
        # Softmax over last dimension
        exp_preds = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)

        best_indices = np.argmax(probs, axis=-1)
        chars: list[str] = []
        token_probs: list[float] = []
        prev_idx = -1

        for t, idx in enumerate(best_indices):
            if idx != prev_idx:
                if idx != BLANK_INDEX_0012 and idx < len(ALPHABET_0012):
                    chars.append(ALPHABET_0012[idx])
                    token_probs.append(float(probs[t, idx]))
                prev_idx = idx

        text = "".join(chars)
        avg_prob = float(np.mean(token_probs)) if token_probs else 1.0
        return text, avg_prob

    # ------------------------------------------------------------------ #
    # IProcessingStrategy.process()                                      #
    # ------------------------------------------------------------------ #

    def process(
        self,
        page: Any,
        page_number: int,
        document_id: str,
        file_path: str,
    ) -> PageExtractionResult:
        """Process a scanned PDF page through OpenVINO OCR inference.

        Rasterises the PyMuPDF page to a NumPy array, executes text
        detection and recognition on the compiled OpenVINO device, and
        returns a structured PageExtractionResult.

        This method NEVER raises. All exceptions are caught and reflected
        in the returned PageExtractionResult.
        """
        t_start = time.perf_counter()
        result = PageExtractionResult(
            processing_method=f"openvino_stub_{self._active_device.lower()}",
        )

        try:
            # 1. Rasterise page (zero-copy PyMuPDF path)
            img_array = self._rasterise_page(page, page_number, file_path, result)
            result.pdf_load_time_s = time.perf_counter() - t_start

            if img_array is None:
                return result

            # 2. Stub mode fallback when model is not loaded (Phase G1 behavior)
            if not self.model_loaded:
                result.warnings.append(
                    f"OpenVINO OCR model not loaded. Selected device: {self._active_device}. "
                    f"Image rasterised successfully: shape={img_array.shape}, dtype={img_array.dtype}. "
                    f"Returning empty text until model is provided."
                )
                result.text = ""
                result.text_blocks = []
                result.ocr_confidence = 0.0
                result.ocr_time_s = time.perf_counter() - t_start - result.pdf_load_time_s
                return result

            # 3. Real inference path (Phase G2)
            t_ocr_start = time.perf_counter()
            full_text, text_blocks, avg_confidence = self.recognize(img_array)
            ocr_elapsed = time.perf_counter() - t_ocr_start

            result.text = full_text
            result.text_blocks = text_blocks
            result.ocr_confidence = avg_confidence
            result.ocr_time_s = ocr_elapsed
            result.processing_method = f"openvino_ocr_{self._active_device.lower()}"

            # Provenance metadata
            result.ocr_engine = "OpenVINO"
            result.ocr_device = self._compiled_device if self._compiled_device != "none" else self._active_device
            result.ocr_model = "horizontal-text-detection-0001+text-recognition-0012"

            logger.debug(
                "OpenVINO OCR processed page %d of '%s': %d blocks, %d chars, dev=%s, time=%.2f ms",
                page_number,
                file_path,
                len(text_blocks),
                len(full_text),
                result.ocr_device,
                ocr_elapsed * 1000.0,
            )

        except Exception as exc:
            logger.error(
                "OpenVINOOCRStrategy.process() error on page %d of '%s': %s",
                page_number,
                file_path,
                exc,
            )
            result.error = str(exc)
            result.processing_method = "failed"
            result.warnings.append(f"Unexpected OpenVINO OCR error: {exc}")

        return result

    # ------------------------------------------------------------------ #
    # Rasterisation                                                      #
    # ------------------------------------------------------------------ #

    def _rasterise_page(
        self,
        page: Any,
        page_number: int,
        file_path: str,
        result: PageExtractionResult,
    ) -> Any:
        """Rasterise a PyMuPDF page to a NumPy HxWxC uint8 array, or return ndarray directly."""
        if isinstance(page, np.ndarray):
            return page

        try:
            import fitz  # type: ignore[import]

            scale = self._dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            return img_array

        except ImportError as exc:
            msg = f"Rasterisation dependency missing: {exc}"
            logger.warning("OpenVINOOCRStrategy: %s", msg)
            result.warnings.append(msg)
            result.error = msg
            result.processing_method = "failed"
            return None

        except Exception as exc:
            msg = f"Rasterisation failed for page {page_number} of '{file_path}': {exc}"
            logger.warning("OpenVINOOCRStrategy: %s", msg)
            result.warnings.append(msg)
            result.error = str(exc)
            result.processing_method = "failed"
            return None

    # ------------------------------------------------------------------ #
    # Repr                                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"OpenVINOOCRStrategy("
            f"device='{self._active_device}', "
            f"compiled_device='{self._compiled_device}', "
            f"model_loaded={self.model_loaded}, "
            f"ov_initialized={self._initialized})"
        )
