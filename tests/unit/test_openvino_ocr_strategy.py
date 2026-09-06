"""Unit tests for OpenVINOOCRStrategy.

All OpenVINO and fitz/numpy internals are mocked so these tests run on any
machine, regardless of whether openvino is installed.

Test coverage:
  - Initialisation: no OV, OV+GPU, OV+CPU-only, OV Core() failure
  - strategy_name reflects active device
  - is_gpu_available / active_device / model_loaded properties
  - Explicit device override (device='CPU' when GPU available)
  - process(): always returns PageExtractionResult, never raises
  - process() stub mode (no model) -- warning + empty text
  - process() rasterisation failure handling
  - load_model(): no OV, success, failure
  - __repr__
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from adaptive_framework.acceleration.hardware_probe import (
    CPUInfo,
    GPUDeviceInfo,
    HardwareCapabilityProbe,
    HardwareProfile,
    OpenVINOInfo,
)
from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy
from adaptive_framework.document_processing.processing_strategy import PageExtractionResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _cpu() -> CPUInfo:
    return CPUInfo(
        brand="Intel64 Family 6 Model 140",
        architecture="AMD64",
        logical_cores=8,
        physical_cores=4,
    )


def _ov_not_installed() -> OpenVINOInfo:
    return OpenVINOInfo(installed=False, version="")


def _ov_cpu_only() -> OpenVINOInfo:
    return OpenVINOInfo(
        installed=True,
        version="2024.1.0",
        available_devices=["CPU"],
        gpu_devices=[],
        cpu_device_available=True,
        gpu_device_available=False,
        auto_device_available=True,
    )


def _ov_with_gpu() -> OpenVINOInfo:
    return OpenVINOInfo(
        installed=True,
        version="2024.1.0",
        available_devices=["CPU", "GPU"],
        gpu_devices=["GPU"],
        cpu_device_available=True,
        gpu_device_available=True,
        auto_device_available=True,
    )


def _iris_xe() -> GPUDeviceInfo:
    return GPUDeviceInfo(
        device_id="GPU",
        full_name="Intel(R) Iris(R) Xe Graphics",
        vendor="Intel",
        is_intel=True,
        is_integrated=True,
    )


def _make_profile(ov_info: OpenVINOInfo, gpu_available: bool = False) -> HardwareProfile:
    gpu_devices = [_iris_xe()] if gpu_available else []
    backend = (
        "openvino_gpu" if gpu_available
        else ("openvino_cpu" if ov_info.installed else "cpu_stub")
    )
    return HardwareProfile(
        cpu=_cpu(),
        openvino=ov_info,
        gpu_devices=gpu_devices,
        gpu_execution_available=gpu_available,
        recommended_ocr_backend=backend,
    )


def _mock_probe(ov_info: OpenVINOInfo, gpu_available: bool = False) -> MagicMock:
    probe = MagicMock(spec=HardwareCapabilityProbe)
    probe.probe.return_value = _make_profile(ov_info, gpu_available)
    return probe


def _fake_image() -> np.ndarray:
    """Return a small fake page image array (100 x 80 x 3)."""
    return np.zeros((100, 80, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestOpenVINOOCRStrategyInit:
    def test_no_openvino_active_device_is_cpu_stub(self) -> None:
        probe = _mock_probe(_ov_not_installed(), gpu_available=False)
        with patch(
            "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
            False,
        ):
            strategy = OpenVINOOCRStrategy(probe=probe)

        assert strategy.active_device == "cpu_stub"
        assert not strategy.is_gpu_available
        assert not strategy.model_loaded
        assert not strategy.openvino_initialized

    def test_openvino_gpu_available_selects_gpu(self) -> None:
        probe = _mock_probe(_ov_with_gpu(), gpu_available=True)
        mock_core = MagicMock()

        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            strategy = OpenVINOOCRStrategy(device="AUTO", probe=probe)

        assert strategy.active_device == "GPU"
        assert strategy.is_gpu_available
        assert strategy.openvino_initialized
        assert not strategy.model_loaded

    def test_openvino_cpu_only_selects_cpu(self) -> None:
        probe = _mock_probe(_ov_cpu_only(), gpu_available=False)
        mock_core = MagicMock()

        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            strategy = OpenVINOOCRStrategy(device="AUTO", probe=probe)

        assert strategy.active_device == "CPU"
        assert not strategy.is_gpu_available
        assert strategy.openvino_initialized

    def test_explicit_cpu_device_overrides_gpu_preference(self) -> None:
        """When GPU is available but device='CPU', CPU must be selected."""
        probe = _mock_probe(_ov_with_gpu(), gpu_available=True)
        mock_core = MagicMock()

        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            strategy = OpenVINOOCRStrategy(device="CPU", probe=probe)

        assert strategy.active_device == "CPU"

    def test_core_init_failure_falls_back_to_cpu_stub(self) -> None:
        """If ov.Core() raises, active_device must be cpu_stub."""
        probe = _mock_probe(_ov_with_gpu(), gpu_available=True)

        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.side_effect = RuntimeError("GPU driver not found")
            strategy = OpenVINOOCRStrategy(probe=probe)

        assert strategy.active_device == "cpu_stub"
        assert not strategy.is_gpu_available
        assert not strategy.openvino_initialized


# ---------------------------------------------------------------------------
# strategy_name property
# ---------------------------------------------------------------------------

class TestStrategyName:
    def test_strategy_name_cpu_stub(self) -> None:
        probe = _mock_probe(_ov_not_installed())
        with patch(
            "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
            False,
        ):
            strategy = OpenVINOOCRStrategy(probe=probe)
        assert strategy.strategy_name == "openvino_ocr_cpu_stub"

    def test_strategy_name_gpu(self) -> None:
        probe = _mock_probe(_ov_with_gpu(), gpu_available=True)
        mock_core = MagicMock()
        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            strategy = OpenVINOOCRStrategy(probe=probe)
        assert strategy.strategy_name == "openvino_ocr_gpu"

    def test_strategy_name_cpu(self) -> None:
        probe = _mock_probe(_ov_cpu_only())
        mock_core = MagicMock()
        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            strategy = OpenVINOOCRStrategy(probe=probe)
        assert strategy.strategy_name == "openvino_ocr_cpu"


# ---------------------------------------------------------------------------
# process() -- stub mode (no model loaded)
# ---------------------------------------------------------------------------

class TestProcessStubMode:
    """Tests for process() when model is NOT loaded (Phase 4.1)."""

    def _make_strategy_no_ov(self) -> OpenVINOOCRStrategy:
        probe = _mock_probe(_ov_not_installed())
        with patch(
            "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
            False,
        ):
            return OpenVINOOCRStrategy(probe=probe)

    def test_process_returns_page_extraction_result(self) -> None:
        strategy = self._make_strategy_no_ov()
        with patch.object(strategy, "_rasterise_page", return_value=_fake_image()):
            result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
        assert isinstance(result, PageExtractionResult)

    def test_process_stub_text_is_empty(self) -> None:
        strategy = self._make_strategy_no_ov()
        with patch.object(strategy, "_rasterise_page", return_value=_fake_image()):
            result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
        assert result.text == ""

    def test_process_stub_has_warning(self) -> None:
        strategy = self._make_strategy_no_ov()
        with patch.object(strategy, "_rasterise_page", return_value=_fake_image()):
            result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
        assert len(result.warnings) > 0
        combined = " ".join(result.warnings).lower()
        assert "stub" in combined or "phase 4" in combined or "not loaded" in combined

    def test_process_stub_confidence_is_zero(self) -> None:
        strategy = self._make_strategy_no_ov()
        with patch.object(strategy, "_rasterise_page", return_value=_fake_image()):
            result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
        assert result.ocr_confidence == 0.0

    def test_process_stub_processing_method_contains_stub(self) -> None:
        strategy = self._make_strategy_no_ov()
        with patch.object(strategy, "_rasterise_page", return_value=_fake_image()):
            result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
        assert "stub" in result.processing_method

    def test_process_rasterisation_failure_returns_result(self) -> None:
        """If rasterisation returns None, process() must return a valid result."""
        strategy = self._make_strategy_no_ov()
        with patch.object(strategy, "_rasterise_page", return_value=None):
            result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
        assert isinstance(result, PageExtractionResult)

    def test_process_never_raises_on_rasterise_exception(self) -> None:
        """process() must catch all exceptions and never raise."""
        strategy = self._make_strategy_no_ov()
        with patch.object(
            strategy,
            "_rasterise_page",
            side_effect=RuntimeError("Unexpected GPU crash"),
        ):
            try:
                result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
            except Exception as exc:  # pragma: no cover
                pytest.fail(f"process() raised unexpectedly: {exc}")
        assert isinstance(result, PageExtractionResult)

    def test_process_error_reflected_in_result(self) -> None:
        """Unexpected exceptions must be stored in result.error."""
        strategy = self._make_strategy_no_ov()
        with patch.object(
            strategy,
            "_rasterise_page",
            side_effect=RuntimeError("driver error"),
        ):
            result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
        assert result.error is not None
        assert result.processing_method == "failed"

    def test_process_pdf_load_time_populated(self) -> None:
        """pdf_load_time_s must be set when rasterisation succeeds."""
        strategy = self._make_strategy_no_ov()
        with patch.object(strategy, "_rasterise_page", return_value=_fake_image()):
            result = strategy.process(MagicMock(), 1, "doc1", "test.pdf")
        assert result.pdf_load_time_s >= 0.0


# ---------------------------------------------------------------------------
# load_model()
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_load_model_no_openvino_returns_false(self) -> None:
        probe = _mock_probe(_ov_not_installed())
        with patch(
            "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
            False,
        ):
            strategy = OpenVINOOCRStrategy(probe=probe)
        assert strategy.load_model("model.xml") is False
        assert not strategy.model_loaded

    def test_load_model_success(self) -> None:
        probe = _mock_probe(_ov_cpu_only())
        mock_core = MagicMock()
        mock_model = MagicMock()
        mock_compiled = MagicMock()
        mock_core.read_model.return_value = mock_model
        mock_core.compile_model.return_value = mock_compiled

        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            strategy = OpenVINOOCRStrategy(probe=probe)
            result = strategy.load_model("ocr_model.xml")

        assert result is True
        assert strategy.model_loaded

    def test_load_model_file_not_found(self) -> None:
        probe = _mock_probe(_ov_cpu_only())
        mock_core = MagicMock()
        mock_core.read_model.side_effect = RuntimeError("File not found")

        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            strategy = OpenVINOOCRStrategy(probe=probe)
            result = strategy.load_model("nonexistent.xml")

        assert result is False
        assert not strategy.model_loaded

    def test_load_model_compile_failure(self) -> None:
        probe = _mock_probe(_ov_cpu_only())
        mock_core = MagicMock()
        mock_core.read_model.return_value = MagicMock()
        mock_core.compile_model.side_effect = RuntimeError("Compile error")

        with (
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.openvino_ocr_strategy.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            strategy = OpenVINOOCRStrategy(probe=probe)
            result = strategy.load_model("model.xml")

        assert result is False
        assert not strategy.model_loaded


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_device(self) -> None:
        probe = _mock_probe(_ov_not_installed())
        with patch(
            "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
            False,
        ):
            strategy = OpenVINOOCRStrategy(probe=probe)
        r = repr(strategy)
        assert "cpu_stub" in r
        assert "model_loaded=False" in r

    def test_repr_no_emojis(self) -> None:
        probe = _mock_probe(_ov_not_installed())
        with patch(
            "adaptive_framework.acceleration.openvino_ocr_strategy._OPENVINO_AVAILABLE",
            False,
        ):
            strategy = OpenVINOOCRStrategy(probe=probe)
        r = repr(strategy)
        for ch in r:
            cp = ord(ch)
            assert not (0x1F300 <= cp <= 0x1FFFF), f"Emoji in repr: U+{cp:04X}"
