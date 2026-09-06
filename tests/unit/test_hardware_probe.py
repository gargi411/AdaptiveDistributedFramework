"""Unit tests for HardwareCapabilityProbe and related data classes.

All tests mock OpenVINO so they run on any machine regardless of whether
openvino is installed.  No CUDA/NVIDIA assumptions.

Test matrix:
  - OpenVINO not installed              -> cpu_stub, no GPU devices
  - OpenVINO installed, CPU only        -> openvino_cpu, no GPU
  - OpenVINO installed, Intel GPU       -> openvino_gpu, GPU detected
  - OpenVINO Core() raises at runtime   -> installed=True, error set, no GPU
  - GPU device unavailable in devices   -> gpu_execution_available=False
  - Probe caching behaviour
  - force=True bypasses cache
  - format_report() plain ASCII, no emojis
  - No nvidia/cuda strings in output
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adaptive_framework.acceleration.hardware_probe import (
    CPUInfo,
    GPUDeviceInfo,
    HardwareCapabilityProbe,
    HardwareProfile,
    OpenVINOInfo,
)


# ---------------------------------------------------------------------------
# CPUInfo
# ---------------------------------------------------------------------------

class TestCPUInfo:
    def test_all_fields(self) -> None:
        cpu = CPUInfo(
            brand="Intel64 Family 6 Model 140",
            architecture="AMD64",
            logical_cores=8,
            physical_cores=4,
            frequency_mhz=2800.0,
        )
        assert cpu.brand == "Intel64 Family 6 Model 140"
        assert cpu.logical_cores == 8
        assert cpu.physical_cores == 4
        assert cpu.frequency_mhz == 2800.0

    def test_no_frequency(self) -> None:
        cpu = CPUInfo(
            brand="Intel processor",
            architecture="AMD64",
            logical_cores=4,
            physical_cores=2,
        )
        assert cpu.frequency_mhz is None

    def test_logical_ge_physical(self) -> None:
        cpu = CPUInfo(
            brand="Intel Core i7",
            architecture="AMD64",
            logical_cores=8,
            physical_cores=4,
        )
        assert cpu.logical_cores >= cpu.physical_cores


# ---------------------------------------------------------------------------
# OpenVINOInfo
# ---------------------------------------------------------------------------

class TestOpenVINOInfo:
    def test_not_installed_defaults(self) -> None:
        info = OpenVINOInfo(installed=False, version="")
        assert not info.installed
        assert not info.gpu_device_available
        assert not info.cpu_device_available
        assert not info.auto_device_available
        assert info.available_devices == []
        assert info.gpu_devices == []

    def test_installed_cpu_only(self) -> None:
        info = OpenVINOInfo(
            installed=True,
            version="2024.1.0",
            available_devices=["CPU"],
            gpu_devices=[],
            cpu_device_available=True,
            gpu_device_available=False,
            auto_device_available=True,
        )
        assert info.installed
        assert info.cpu_device_available
        assert not info.gpu_device_available
        assert info.gpu_devices == []

    def test_installed_with_gpu(self) -> None:
        info = OpenVINOInfo(
            installed=True,
            version="2024.1.0",
            available_devices=["CPU", "GPU"],
            gpu_devices=["GPU"],
            cpu_device_available=True,
            gpu_device_available=True,
            auto_device_available=True,
        )
        assert info.gpu_device_available
        assert "GPU" in info.gpu_devices
        assert "CPU" in info.available_devices


# ---------------------------------------------------------------------------
# GPUDeviceInfo
# ---------------------------------------------------------------------------

class TestGPUDeviceInfo:
    def test_intel_iris_xe_integrated(self) -> None:
        gpu = GPUDeviceInfo(
            device_id="GPU",
            full_name="Intel(R) Iris(R) Xe Graphics",
            vendor="Intel",
            is_intel=True,
            is_integrated=True,
        )
        assert gpu.is_intel
        assert gpu.is_integrated
        assert gpu.supports_inference is True
        assert gpu.device_id == "GPU"

    def test_discrete_gpu_not_integrated(self) -> None:
        gpu = GPUDeviceInfo(
            device_id="GPU.0",
            full_name="Intel(R) Arc(TM) A770 Graphics",
            vendor="Intel",
            is_intel=True,
            is_integrated=False,
        )
        assert gpu.is_intel
        assert not gpu.is_integrated

    def test_non_intel_vendor(self) -> None:
        gpu = GPUDeviceInfo(
            device_id="GPU",
            full_name="AMD Radeon RX 6600",
            vendor="AMD",
            is_intel=False,
            is_integrated=False,
        )
        assert not gpu.is_intel
        assert gpu.vendor == "AMD"


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------

class TestHardwareProfile:
    # --- helpers ---

    @staticmethod
    def _cpu() -> CPUInfo:
        return CPUInfo(
            brand="Intel64 Family 6 Model 140",
            architecture="AMD64",
            logical_cores=8,
            physical_cores=4,
            frequency_mhz=1300.0,
        )

    @staticmethod
    def _ov_not_installed() -> OpenVINOInfo:
        return OpenVINOInfo(
            installed=False,
            version="",
            error="openvino package not installed.",
        )

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _iris_xe() -> GPUDeviceInfo:
        return GPUDeviceInfo(
            device_id="GPU",
            full_name="Intel(R) Iris(R) Xe Graphics",
            vendor="Intel",
            is_intel=True,
            is_integrated=True,
        )

    # --- to_dict ---

    def test_to_dict_contains_required_keys(self) -> None:
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_not_installed(),
        )
        d = profile.to_dict()
        assert "cpu" in d
        assert "openvino" in d
        assert "gpu_devices" in d
        assert "gpu_execution_available" in d
        assert "recommended_ocr_backend" in d
        assert "platform_info" in d
        assert "probed_at" in d

    def test_to_dict_gpu_list_empty_when_no_gpu(self) -> None:
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_not_installed(),
            gpu_devices=[],
        )
        assert profile.to_dict()["gpu_devices"] == []

    def test_to_dict_gpu_list_populated(self) -> None:
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_with_gpu(),
            gpu_devices=[self._iris_xe()],
            gpu_execution_available=True,
            recommended_ocr_backend="openvino_gpu",
        )
        gpu_list = profile.to_dict()["gpu_devices"]
        assert len(gpu_list) == 1
        assert gpu_list[0]["full_name"] == "Intel(R) Iris(R) Xe Graphics"
        assert gpu_list[0]["is_intel"] is True

    # --- format_report ---

    def test_format_report_has_header(self) -> None:
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_not_installed(),
        )
        report = profile.format_report()
        assert "ADF Hardware Capability Report" in report

    def test_format_report_no_openvino_says_not_installed(self) -> None:
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_not_installed(),
        )
        report = profile.format_report()
        assert "not installed" in report
        assert "cpu_stub" in report

    def test_format_report_with_gpu(self) -> None:
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_with_gpu(),
            gpu_devices=[self._iris_xe()],
            gpu_execution_available=True,
            recommended_ocr_backend="openvino_gpu",
        )
        report = profile.format_report()
        assert "GPU" in report
        assert "openvino_gpu" in report
        assert "Intel" in report
        assert "Iris" in report

    def test_format_report_cpu_only_backend(self) -> None:
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_cpu_only(),
            gpu_devices=[],
            gpu_execution_available=False,
            recommended_ocr_backend="openvino_cpu",
        )
        report = profile.format_report()
        assert "openvino_cpu" in report

    def test_format_report_contains_cpu_info(self) -> None:
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_not_installed(),
        )
        report = profile.format_report()
        assert "Intel64 Family 6 Model 140" in report or "AMD64" in report
        assert "8" in report  # logical cores

    def test_format_report_no_emojis(self) -> None:
        """Report must contain only plain ASCII -- no emojis."""
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_with_gpu(),
            gpu_devices=[self._iris_xe()],
            gpu_execution_available=True,
            recommended_ocr_backend="openvino_gpu",
        )
        report = profile.format_report()
        for char in report:
            code_point = ord(char)
            # Emoji range check (broad range covering all emoji blocks)
            assert not (0x1F300 <= code_point <= 0x1FFFF), (
                f"Emoji found in report: U+{code_point:04X} '{char}'"
            )

    def test_format_report_no_cuda_no_nvidia(self) -> None:
        """Report must not reference CUDA or NVIDIA."""
        profile = HardwareProfile(
            cpu=self._cpu(),
            openvino=self._ov_with_gpu(),
            gpu_devices=[self._iris_xe()],
            gpu_execution_available=True,
            recommended_ocr_backend="openvino_gpu",
        )
        report = profile.format_report().lower()
        assert "cuda" not in report
        assert "nvidia" not in report


# ---------------------------------------------------------------------------
# HardwareCapabilityProbe -- unit tests (all OpenVINO mocked)
# ---------------------------------------------------------------------------

class TestHardwareCapabilityProbe:
    """All OpenVINO calls are mocked so tests run without openvino installed."""

    # --- helpers ---

    @staticmethod
    def _patch_ov_absent() -> tuple[Any, Any]:
        """Return context manager patches for 'OpenVINO not installed'."""
        p1 = patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        )
        return (p1,)

    # --- core scenarios ---

    def test_probe_openvino_not_installed(self) -> None:
        """Probe must succeed and return cpu_stub when openvino absent."""
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert profile is not None
        assert not profile.openvino.installed
        assert not profile.gpu_execution_available
        assert profile.recommended_ocr_backend == "cpu_stub"
        assert profile.openvino.error is not None

    def test_probe_cpu_info_always_populated(self) -> None:
        """CPU info must be populated even when OpenVINO is absent."""
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert profile.cpu.logical_cores >= 1
        assert profile.cpu.physical_cores >= 1
        assert isinstance(profile.cpu.brand, str)
        assert len(profile.cpu.brand) > 0
        assert isinstance(profile.cpu.architecture, str)

    def test_probe_openvino_cpu_only(self) -> None:
        """CPU-only OpenVINO installation => openvino_cpu backend."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]

        with (
            patch(
                "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.hardware_probe._OPENVINO_VERSION",
                "2024.1.0",
            ),
            patch(
                "adaptive_framework.acceleration.hardware_probe.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert profile.openvino.installed
        assert profile.openvino.version == "2024.1.0"
        assert profile.openvino.cpu_device_available
        assert not profile.openvino.gpu_device_available
        assert profile.recommended_ocr_backend == "openvino_cpu"
        assert not profile.gpu_execution_available
        assert profile.gpu_devices == []

    def test_probe_openvino_with_intel_gpu(self) -> None:
        """OpenVINO + GPU device detected => openvino_gpu, gpu_execution_available=True."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU"]
        mock_core.get_property.return_value = "Intel(R) Iris(R) Xe Graphics"

        with (
            patch(
                "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.hardware_probe._OPENVINO_VERSION",
                "2024.1.0",
            ),
            patch(
                "adaptive_framework.acceleration.hardware_probe.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert profile.openvino.installed
        assert profile.openvino.gpu_device_available
        assert "GPU" in profile.openvino.gpu_devices
        assert profile.gpu_execution_available
        assert profile.recommended_ocr_backend == "openvino_gpu"

        assert len(profile.gpu_devices) == 1
        gpu = profile.gpu_devices[0]
        assert gpu.is_intel
        assert gpu.is_integrated  # Iris Xe is integrated
        assert "Iris" in gpu.full_name or "Intel" in gpu.full_name
        assert gpu.vendor == "Intel"

    def test_probe_gpu_device_absent_from_devices_list(self) -> None:
        """If 'GPU' not in available_devices, gpu_execution_available is False."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]  # no GPU entry

        with (
            patch(
                "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.hardware_probe.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert not profile.gpu_execution_available
        assert profile.gpu_devices == []
        assert profile.openvino.gpu_devices == []

    def test_probe_openvino_core_init_failure(self) -> None:
        """If ov.Core() raises, probe must still return a valid profile."""
        with (
            patch(
                "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.hardware_probe._OPENVINO_VERSION",
                "2024.1.0",
            ),
            patch(
                "adaptive_framework.acceleration.hardware_probe.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.side_effect = RuntimeError("GPU driver not found")
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert profile is not None
        assert profile.openvino.installed        # package is present
        assert profile.openvino.error is not None  # but Core() failed
        assert not profile.gpu_execution_available
        assert profile.recommended_ocr_backend == "cpu_stub"  # no OV devices usable

    def test_probe_graceful_cpu_fallback(self) -> None:
        """No OpenVINO -> cpu_stub, no GPU, safe to use."""
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert profile.recommended_ocr_backend == "cpu_stub"
        assert not profile.gpu_execution_available
        assert profile.gpu_devices == []

    # --- no NVIDIA/CUDA assumptions ---

    def test_no_nvidia_cuda_in_report(self) -> None:
        """format_report() must not mention CUDA or NVIDIA."""
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        report = profile.format_report().lower()
        assert "cuda" not in report, "Report must not mention CUDA"
        assert "nvidia" not in report, "Report must not mention NVIDIA"
        assert "nvidia-smi" not in report

    def test_to_dict_no_nvidia_keys(self) -> None:
        """to_dict() must not contain CUDA or NVIDIA keys."""
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        d = profile.to_dict()
        d_str = str(d).lower()
        assert "cuda" not in d_str
        assert "nvidia" not in d_str

    # --- caching ---

    def test_probe_result_is_cached(self) -> None:
        """Second call without force=True must return the same object."""
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            p1 = probe.probe()
            p2 = probe.probe()

        assert p1 is p2

    def test_probe_force_returns_new_profile(self) -> None:
        """force=True must re-probe and return a freshly created object."""
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            p1 = probe.probe()
            p2 = probe.probe(force=True)

        # Different objects (re-probed)
        assert p1 is not p2
        # But same logical content
        assert p1.openvino.installed == p2.openvino.installed
        assert p1.recommended_ocr_backend == p2.recommended_ocr_backend

    # --- multiple GPU devices ---

    def test_probe_multiple_gpu_devices(self) -> None:
        """System with two GPU devices should list both."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU.0", "GPU.1"]

        def get_prop(device_id: str, prop: str) -> str:
            if device_id == "GPU.0":
                return "Intel(R) Iris(R) Xe Graphics"
            return "Intel(R) Arc(TM) A770 Graphics"

        mock_core.get_property.side_effect = get_prop

        with (
            patch(
                "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
                True,
            ),
            patch(
                "adaptive_framework.acceleration.hardware_probe.ov",
            ) as mock_ov,
        ):
            mock_ov.Core.return_value = mock_core
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert len(profile.gpu_devices) == 2
        device_ids = {g.device_id for g in profile.gpu_devices}
        assert "GPU.0" in device_ids
        assert "GPU.1" in device_ids

        # Iris Xe should be identified as integrated
        iris = next(g for g in profile.gpu_devices if "Iris" in g.full_name)
        assert iris.is_integrated
        # Arc should be identified as discrete
        arc = next(g for g in profile.gpu_devices if "Arc" in g.full_name)
        assert not arc.is_integrated

    # --- platform info ---

    def test_probe_platform_info_populated(self) -> None:
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        assert isinstance(profile.platform_info, str)
        assert len(profile.platform_info) > 0

    def test_probe_probed_at_is_iso8601(self) -> None:
        from datetime import datetime
        with patch(
            "adaptive_framework.acceleration.hardware_probe._OPENVINO_AVAILABLE",
            False,
        ):
            probe = HardwareCapabilityProbe()
            profile = probe.probe(force=True)

        # Should parse without raising
        datetime.fromisoformat(profile.probed_at)
