"""Unit tests for Phase G2: OpenVINO OCR Integration.

Test coverage (all 11 mandatory G2 test cases):
  1. test_openvino_available
  2. test_device_enumeration
  3. test_cpu_compilation
  4. test_gpu_compilation (skips cleanly when Intel GPU absent)
  5. test_ocr_initialization
  6. test_single_image_inference
  7. test_batch_inference
  8. test_pdf_page_to_ocr
  9. test_ocr_provenance
 10. test_unavailable_device_handling
 11. test_missing_model_handling
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import numpy as np
from PIL import Image
import pytest

from adaptive_framework.acceleration.hardware_probe import HardwareCapabilityProbe
from adaptive_framework.acceleration.model_setup import ensure_openvino_ocr_models
from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy
from adaptive_framework.document_processing.processing_strategy import (
    PageExtractionResult,
    ProcessingStrategyFactory,
)
from adaptive_framework.models.page import PageType, TextBlock

try:
    import openvino as ov
    _OPENVINO_INSTALLED = True
    _AVAILABLE_DEVICES = list(ov.Core().available_devices)
    _GPU_AVAILABLE = "GPU" in _AVAILABLE_DEVICES
except Exception:
    _OPENVINO_INSTALLED = False
    _AVAILABLE_DEVICES = []
    _GPU_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ocr_models() -> tuple[Path, Path]:
    """Ensure OpenVINO OCR models exist for tests."""
    return ensure_openvino_ocr_models()


@pytest.fixture
def synthetic_page_image() -> np.ndarray:
    """Create a synthetic text image for deterministic OCR testing."""
    # Create an image with black text on white background
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    # Fill with a dummy shape/pattern
    arr = np.array(img, dtype=np.uint8)
    arr[30:70, 50:150] = 0  # Black block
    return arr


@pytest.fixture
def synthetic_fitz_page(tmp_path: Path) -> tuple[fitz.Page, fitz.Document, Path]:
    """Create an in-memory PyMuPDF page containing medical text."""
    pdf_path = tmp_path / "test_medical.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "DIAGNOSIS: HYPERTENSION AND DIABETES MELLITUS", fontsize=14)
    page.insert_text((50, 130), "PATIENT PROTOCOL: METFORMIN 500MG DAILY", fontsize=11)
    doc.save(str(pdf_path))
    return page, doc, pdf_path


# ---------------------------------------------------------------------------
# Test Cases 1 - 4: OpenVINO Environment & Device Compilation
# ---------------------------------------------------------------------------

def test_openvino_available() -> None:
    """Case 1: Verify OpenVINO runtime is installed and importable."""
    assert _OPENVINO_INSTALLED, "OpenVINO Runtime must be installed for Phase G2"
    assert len(_AVAILABLE_DEVICES) > 0, "At least one OpenVINO device must be available"


def test_device_enumeration() -> None:
    """Case 2: Verify device enumeration reports CPU and any available GPUs."""
    probe = HardwareCapabilityProbe()
    profile = probe.probe()
    assert profile.openvino.installed is True
    assert "CPU" in profile.openvino.available_devices
    assert isinstance(profile.openvino.available_devices, list)


def test_cpu_compilation(ocr_models: tuple[Path, Path]) -> None:
    """Case 3: Verify OpenVINO OCR models compile successfully on CPU."""
    det_path, rec_path = ocr_models
    strategy = OpenVINOOCRStrategy(
        device="CPU",
        det_model_path=det_path,
        rec_model_path=rec_path,
    )
    success = strategy.initialize()
    assert success is True
    assert strategy.model_loaded is True
    assert "CPU" in strategy.compiled_device.upper()


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="Physical Intel GPU not available on this system")
def test_gpu_compilation(ocr_models: tuple[Path, Path]) -> None:
    """Case 4: Verify OpenVINO OCR models compile on the Intel GPU."""
    det_path, rec_path = ocr_models
    strategy = OpenVINOOCRStrategy(
        device="GPU",
        det_model_path=det_path,
        rec_model_path=rec_path,
    )
    success = strategy.initialize()
    assert success is True
    assert strategy.model_loaded is True
    assert "GPU" in strategy.compiled_device.upper()
    dev_info = strategy.get_device_info()
    assert dev_info["is_gpu"] is True


# ---------------------------------------------------------------------------
# Test Cases 5 - 7: Strategy API & Inference Execution
# ---------------------------------------------------------------------------

def test_ocr_initialization(ocr_models: tuple[Path, Path]) -> None:
    """Case 5: Verify lazy initialization and strategy properties."""
    det_path, rec_path = ocr_models
    strategy = OpenVINOOCRStrategy(
        device="AUTO",
        det_model_path=det_path,
        rec_model_path=rec_path,
        auto_initialize=False,
    )
    # Model not loaded yet
    assert strategy.model_loaded is False
    assert strategy.openvino_initialized is True

    # Lazy compilation
    init_success = strategy.initialize()
    assert init_success is True
    assert strategy.model_loaded is True
    assert strategy.is_available() is True


def test_single_image_inference(ocr_models: tuple[Path, Path], synthetic_page_image: np.ndarray) -> None:
    """Case 6: Verify recognize() handles single image inference."""
    det_path, rec_path = ocr_models
    device = "GPU" if _GPU_AVAILABLE else "CPU"
    strategy = OpenVINOOCRStrategy(
        device=device,
        det_model_path=det_path,
        rec_model_path=rec_path,
    )
    strategy.initialize()
    text, blocks, conf = strategy.recognize(synthetic_page_image)
    assert isinstance(text, str)
    assert isinstance(blocks, list)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_batch_inference(ocr_models: tuple[Path, Path], synthetic_page_image: np.ndarray) -> None:
    """Case 7: Verify recognize_batch() handles multiple images."""
    det_path, rec_path = ocr_models
    device = "GPU" if _GPU_AVAILABLE else "CPU"
    strategy = OpenVINOOCRStrategy(
        device=device,
        det_model_path=det_path,
        rec_model_path=rec_path,
    )
    strategy.initialize()
    batch = [synthetic_page_image, synthetic_page_image]
    results = strategy.recognize_batch(batch)
    assert len(results) == 2
    for text, blocks, conf in results:
        assert isinstance(text, str)
        assert isinstance(blocks, list)
        assert isinstance(conf, float)


# ---------------------------------------------------------------------------
# Test Cases 8 - 9: End-to-End PDF to OCR & Provenance Preservation
# ---------------------------------------------------------------------------

def test_pdf_page_to_ocr(ocr_models: tuple[Path, Path], synthetic_fitz_page: tuple[fitz.Page, fitz.Document, Path]) -> None:
    """Case 8: Verify full path: PyMuPDF Page -> NumPy rasterisation -> OpenVINO OCR -> PageExtractionResult."""
    page, doc, pdf_path = synthetic_fitz_page
    det_path, rec_path = ocr_models
    device = "GPU" if _GPU_AVAILABLE else "CPU"

    strategy = OpenVINOOCRStrategy(
        device=device,
        det_model_path=det_path,
        rec_model_path=rec_path,
    )
    strategy.initialize()

    result = strategy.process(
        page=page,
        page_number=1,
        document_id="doc-test-001",
        file_path=str(pdf_path),
    )

    doc.close()
    assert isinstance(result, PageExtractionResult)
    assert result.error is None
    assert result.ocr_time_s > 0.0
    assert result.pdf_load_time_s > 0.0
    assert len(result.text) > 0
    assert len(result.text_blocks) > 0


def test_ocr_provenance(ocr_models: tuple[Path, Path], synthetic_fitz_page: tuple[fitz.Page, fitz.Document, Path]) -> None:
    """Case 9: Verify OCR provenance fields survive in PageExtractionResult and TextBlocks."""
    page, doc, pdf_path = synthetic_fitz_page
    det_path, rec_path = ocr_models
    device = "GPU" if _GPU_AVAILABLE else "CPU"

    strategy = OpenVINOOCRStrategy(
        device=device,
        det_model_path=det_path,
        rec_model_path=rec_path,
    )
    strategy.initialize()

    result = strategy.process(
        page=page,
        page_number=1,
        document_id="doc-provenance-001",
        file_path=str(pdf_path),
    )
    doc.close()

    assert result.ocr_engine == "OpenVINO"
    assert result.ocr_device is not None
    assert device in result.ocr_device.upper()
    assert "horizontal-text-detection" in str(result.ocr_model)
    assert "text-recognition" in str(result.ocr_model)

    # Check TextBlocks have valid coordinates and confidence
    for b in result.text_blocks:
        assert isinstance(b, TextBlock)
        assert b.bbox.is_valid()
        assert 0.0 <= b.confidence <= 1.0


# ---------------------------------------------------------------------------
# Test Cases 10 - 11: Error Recovery & Unavailable Devices
# ---------------------------------------------------------------------------

def test_unavailable_device_handling() -> None:
    """Case 10: Verify graceful handling when an invalid device is requested."""
    strategy = OpenVINOOCRStrategy(device="INVALID_DEVICE_XYZ")
    assert strategy.active_device == "INVALID_DEVICE_XYZ"
    # initialize with invalid device should fail gracefully without crashing
    success = strategy.initialize()
    assert success is False
    assert strategy.model_loaded is False

    # process() must return a valid PageExtractionResult with error warning
    dummy_page = MagicMock()
    with patch.object(strategy, "_rasterise_page", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        result = strategy.process(dummy_page, 1, "doc", "test.pdf")
    assert isinstance(result, PageExtractionResult)
    assert "stub" in result.processing_method


def test_missing_model_handling() -> None:
    """Case 11: Verify error reporting when model paths do not exist."""
    strategy = OpenVINOOCRStrategy(
        device="CPU",
        det_model_path="nonexistent_det_path_12345.xml",
        rec_model_path="nonexistent_rec_path_12345.xml",
    )
    success = strategy.initialize()
    assert success is False
    assert strategy.model_loaded is False

    # Calling recognize on uninitialized strategy should return empty safely
    text, blocks, conf = strategy.recognize(np.zeros((50, 50, 3), dtype=np.uint8))
    assert text == ""
    assert blocks == []
    assert conf == 0.0


# ---------------------------------------------------------------------------
# Factory Integration Test
# ---------------------------------------------------------------------------

def test_factory_dispatches_openvino_strategy() -> None:
    """Verify ProcessingStrategyFactory creates OpenVINOOCRStrategy when configured."""
    factory = ProcessingStrategyFactory(ocr_backend="openvino", ocr_device="CPU")
    strategy = factory.get_strategy(PageType.SCANNED)
    assert isinstance(strategy, OpenVINOOCRStrategy)
    assert "openvino_ocr" in strategy.strategy_name
