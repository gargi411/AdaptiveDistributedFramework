"""HardwareCapabilityProbe -- Detects CPU, Intel GPU, and OpenVINO availability.

Inspects the current machine and produces a HardwareProfile describing:
  - CPU brand, architecture, logical/physical core counts, clock frequency
  - OpenVINO Runtime installation status and version
  - OpenVINO execution devices (CPU, GPU, AUTO, ...)
  - Intel GPU presence (detected via OpenVINO device enumeration -- not GPUtil/nvidia-smi)
  - Whether GPU inference execution is actually usable right now
  - Recommended OCR backend string for downstream strategy selection

Design principles:
  - No NVIDIA/CUDA assumptions.  Intel Iris Xe is targeted via OpenVINO GPU plugin.
  - Graceful degradation: every check succeeds even when psutil or openvino absent.
  - Results are cached after first call; use probe(force=True) to refresh.
  - All output is plain ASCII text -- no emojis.

Typical usage:
    probe = HardwareCapabilityProbe()
    profile = probe.probe()
    print(profile.format_report())
    print(profile.gpu_execution_available)   # True if Intel GPU + OpenVINO ready
    print(profile.recommended_ocr_backend)  # 'openvino_gpu' | 'openvino_cpu' | 'cpu_stub'
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# psutil: optional but highly recommended
try:
    import psutil  # type: ignore[import]
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# OpenVINO Runtime: optional dependency (install with: pip install openvino>=2024.0)
try:
    import openvino as ov  # type: ignore[import]
    _OPENVINO_AVAILABLE = True
    _OPENVINO_VERSION: str = getattr(ov, "__version__", "unknown")
except ImportError:
    _OPENVINO_AVAILABLE = False
    _OPENVINO_VERSION = ""
    ov = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CPUInfo:
    """CPU hardware identification.

    Attributes:
        brand: Processor brand string (e.g. 'Intel64 Family 6 Model 140').
        architecture: Machine architecture (e.g. 'AMD64', 'x86_64').
        logical_cores: Logical CPU count including hyper-threading.
        physical_cores: Physical core count.
        frequency_mhz: Current base frequency in MHz. None if undetectable.
    """

    brand: str
    architecture: str
    logical_cores: int
    physical_cores: int
    frequency_mhz: float | None = None


@dataclass
class GPUDeviceInfo:
    """Metadata for a single GPU device visible to OpenVINO.

    Attributes:
        device_id: OpenVINO device string (e.g. 'GPU', 'GPU.0', 'GPU.1').
        full_name: Human-readable device name returned by OpenVINO
                   (e.g. 'Intel(R) Iris(R) Xe Graphics').
        vendor: Vendor string derived from full_name ('Intel', 'AMD', 'Unknown').
        is_intel: True if the device name contains 'INTEL'.
        is_integrated: True for Iris Xe, UHD, or similar integrated GPUs.
        supports_inference: True if OpenVINO confirmed device usable for inference.
    """

    device_id: str
    full_name: str
    vendor: str
    is_intel: bool
    is_integrated: bool
    supports_inference: bool = True


@dataclass
class OpenVINOInfo:
    """OpenVINO Runtime installation and device details.

    Attributes:
        installed: True if openvino package is importable.
        version: Version string (e.g. '2024.1.0'). Empty when not installed.
        available_devices: All devices reported by ov.Core().available_devices.
        gpu_devices: Subset of available_devices whose names start with 'GPU'.
        cpu_device_available: True if 'CPU' in available_devices.
        gpu_device_available: True if at least one GPU device is present.
        auto_device_available: True if AUTO plugin is usable (always True when
                               OpenVINO is installed).
        error: Error message if Core() initialisation failed. None on success.
    """

    installed: bool
    version: str
    available_devices: list[str] = field(default_factory=list)
    gpu_devices: list[str] = field(default_factory=list)
    cpu_device_available: bool = False
    gpu_device_available: bool = False
    auto_device_available: bool = False
    error: str | None = None


@dataclass
class HardwareProfile:
    """Complete hardware capability profile for this machine.

    Produced by HardwareCapabilityProbe.probe().
    Consumed by ProcessingStrategyFactory (Phase 4.2), dashboard, and
    diagnostic scripts.

    Attributes:
        cpu: CPU identification and core counts.
        openvino: OpenVINO installation and device details.
        gpu_devices: List of GPU devices visible to OpenVINO (may be empty).
        gpu_execution_available: True when at least one GPU device is present
                                 AND OpenVINO is installed and initialised.
        recommended_ocr_backend: Suggested backend for OCR inference.
            'openvino_gpu'  -- Intel GPU via OpenVINO GPU plugin (best)
            'openvino_cpu'  -- OpenVINO CPU plugin (no GPU, but OV optimized)
            'cpu_stub'      -- OpenVINO not installed; falls back to PaddleOCR
                               or empty-text stub.
        platform_info: OS + machine + release string.
        probed_at: ISO 8601 UTC timestamp of this probe run.
    """

    cpu: CPUInfo
    openvino: OpenVINOInfo
    gpu_devices: list[GPUDeviceInfo] = field(default_factory=list)
    gpu_execution_available: bool = False
    recommended_ocr_backend: str = "cpu_stub"
    platform_info: str = field(
        default_factory=lambda: (
            f"{platform.system()} {platform.machine()} {platform.release()}"
        )
    )
    probed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------ #
    # Serialisation                                                         #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary suitable for JSON / dashboard."""
        return {
            "cpu": {
                "brand": self.cpu.brand,
                "architecture": self.cpu.architecture,
                "logical_cores": self.cpu.logical_cores,
                "physical_cores": self.cpu.physical_cores,
                "frequency_mhz": self.cpu.frequency_mhz,
            },
            "openvino": {
                "installed": self.openvino.installed,
                "version": self.openvino.version,
                "available_devices": self.openvino.available_devices,
                "gpu_devices": self.openvino.gpu_devices,
                "cpu_device_available": self.openvino.cpu_device_available,
                "gpu_device_available": self.openvino.gpu_device_available,
                "auto_device_available": self.openvino.auto_device_available,
                "error": self.openvino.error,
            },
            "gpu_devices": [
                {
                    "device_id": g.device_id,
                    "full_name": g.full_name,
                    "vendor": g.vendor,
                    "is_intel": g.is_intel,
                    "is_integrated": g.is_integrated,
                    "supports_inference": g.supports_inference,
                }
                for g in self.gpu_devices
            ],
            "gpu_execution_available": self.gpu_execution_available,
            "recommended_ocr_backend": self.recommended_ocr_backend,
            "platform_info": self.platform_info,
            "probed_at": self.probed_at,
        }

    def format_report(self) -> str:
        """Return a plain ASCII hardware capability report.

        No emojis. No colour codes. Safe for terminal output and log files.

        Returns:
            Multi-line ASCII string.
        """
        lines: list[str] = [
            "ADF Hardware Capability Report",
            "=" * 44,
            "",
            f"CPU:              {self.cpu.brand}",
            f"  Architecture:   {self.cpu.architecture}",
            f"  Logical cores:  {self.cpu.logical_cores}",
            f"  Physical cores: {self.cpu.physical_cores}",
        ]
        if self.cpu.frequency_mhz is not None:
            lines.append(f"  Frequency:      {self.cpu.frequency_mhz:.0f} MHz")

        # --- OpenVINO section -------------------------------------------
        lines.append("")
        if self.openvino.installed:
            lines.append(f"OpenVINO:         installed  v{self.openvino.version}")
            devices_str = (
                ", ".join(self.openvino.available_devices)
                if self.openvino.available_devices
                else "none"
            )
            gpu_str = (
                ", ".join(self.openvino.gpu_devices)
                if self.openvino.gpu_devices
                else "none"
            )
            lines += [
                f"  All devices:    {devices_str}",
                f"  GPU devices:    {gpu_str}",
                f"  CPU plugin:     {self.openvino.cpu_device_available}",
                f"  GPU plugin:     {self.openvino.gpu_device_available}",
                f"  AUTO plugin:    {self.openvino.auto_device_available}",
            ]
            if self.openvino.error:
                lines.append(f"  Init error:     {self.openvino.error}")
        else:
            lines.append("OpenVINO:         not installed")
            lines.append(
                "  Install:        pip install openvino>=2024.0"
                "  (or: uv add openvino --optional accelerate)"
            )
            if self.openvino.error:
                lines.append(f"  Note:           {self.openvino.error}")

        # --- GPU device section -----------------------------------------
        lines.append("")
        if self.gpu_devices:
            lines.append(f"GPU devices:      {len(self.gpu_devices)} detected")
            for g in self.gpu_devices:
                kind = "integrated" if g.is_integrated else "discrete"
                lines.append(
                    f"  [{g.device_id}]  {g.full_name}  ({g.vendor}, {kind})"
                )
        else:
            lines.append("GPU devices:      none detected via OpenVINO")

        # --- Summary section --------------------------------------------
        lines += [
            "",
            f"GPU backend:      {'available' if self.gpu_execution_available else 'not available'}",
            f"OCR acceleration: {self.recommended_ocr_backend}",
        ]

        if self.openvino.installed and self.openvino.gpu_device_available:
            fallback_desc = "Intel GPU via OpenVINO GPU plugin"
        elif self.openvino.installed:
            fallback_desc = "CPU inference via OpenVINO CPU plugin"
        else:
            fallback_desc = "PaddleOCR (CPU) or empty-text stub if not installed"
        lines.append(f"Fallback:         {fallback_desc}")

        lines += [
            "",
            f"Platform:         {self.platform_info}",
            f"Probed at:        {self.probed_at}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class HardwareCapabilityProbe:
    """Probes the current machine for CPU, GPU, and OpenVINO capabilities.

    Designed to be instantiated once and cached. All individual checks
    degrade gracefully when optional dependencies are absent.

    No NVIDIA/CUDA-specific probes.  GPU detection uses OpenVINO device
    enumeration (ov.Core().available_devices), which surfaces Intel Iris Xe
    as 'GPU' via the Intel GPU plugin.

    Usage:
        probe = HardwareCapabilityProbe()
        profile = probe.probe()          # cached after first call
        profile = probe.probe(force=True)  # bypass cache
    """

    def __init__(self) -> None:
        self._cached: HardwareProfile | None = None

    def probe(self, *, force: bool = False) -> HardwareProfile:
        """Run all hardware checks and return a HardwareProfile.

        Results are cached after first call.

        Args:
            force: If True, bypass the cache and re-probe.

        Returns:
            HardwareProfile populated with detected capabilities.
        """
        if self._cached is not None and not force:
            return self._cached

        cpu = self._probe_cpu()
        ov_info = self._probe_openvino()
        gpu_devices = self._build_gpu_device_list(ov_info)
        gpu_available = ov_info.gpu_device_available
        backend = self._select_backend(ov_info)

        self._cached = HardwareProfile(
            cpu=cpu,
            openvino=ov_info,
            gpu_devices=gpu_devices,
            gpu_execution_available=gpu_available,
            recommended_ocr_backend=backend,
        )
        logger.debug(
            "HardwareCapabilityProbe: backend=%s, gpu_available=%s, ov=%s",
            backend,
            gpu_available,
            ov_info.installed,
        )
        return self._cached

    # ------------------------------------------------------------------ #
    # CPU                                                                   #
    # ------------------------------------------------------------------ #

    def _probe_cpu(self) -> CPUInfo:
        """Probe CPU brand, architecture, and core counts."""
        arch = platform.machine()
        logical = 1
        physical = 1
        freq_mhz: float | None = None

        if _PSUTIL_AVAILABLE:
            logical = psutil.cpu_count(logical=True) or 1
            physical = psutil.cpu_count(logical=False) or 1
            try:
                freq = psutil.cpu_freq()
                if freq is not None:
                    freq_mhz = round(freq.current, 1)
            except Exception:
                pass  # cpu_freq() may fail on some VMs

        brand = platform.processor()
        if not brand:
            brand = f"{arch} processor"

        return CPUInfo(
            brand=brand,
            architecture=arch,
            logical_cores=logical,
            physical_cores=physical,
            frequency_mhz=freq_mhz,
        )

    # ------------------------------------------------------------------ #
    # OpenVINO                                                              #
    # ------------------------------------------------------------------ #

    def _probe_openvino(self) -> OpenVINOInfo:
        """Probe OpenVINO Runtime and enumerate execution devices."""
        if not _OPENVINO_AVAILABLE:
            return OpenVINOInfo(
                installed=False,
                version="",
                error=(
                    "openvino package not installed. "
                    "To enable GPU acceleration: pip install openvino>=2024.0"
                ),
            )

        # OpenVINO is importable -- attempt Core initialisation
        try:
            core = ov.Core()  # type: ignore[union-attr]
            devices: list[str] = list(core.available_devices)
        except Exception as exc:
            logger.warning("OpenVINO Core() failed: %s", exc)
            return OpenVINOInfo(
                installed=True,
                version=_OPENVINO_VERSION,
                error=f"OpenVINO Core init failed: {exc}",
            )

        gpu_devices = [d for d in devices if d.upper().startswith("GPU")]
        cpu_available = "CPU" in devices
        gpu_available = len(gpu_devices) > 0

        logger.info(
            "OpenVINO v%s: devices=%s, GPU devices=%s",
            _OPENVINO_VERSION,
            devices,
            gpu_devices,
        )

        return OpenVINOInfo(
            installed=True,
            version=_OPENVINO_VERSION,
            available_devices=devices,
            gpu_devices=gpu_devices,
            cpu_device_available=cpu_available,
            gpu_device_available=gpu_available,
            auto_device_available=True,  # AUTO is always a virtual device when OV is installed
        )

    # ------------------------------------------------------------------ #
    # GPU device details                                                    #
    # ------------------------------------------------------------------ #

    def _build_gpu_device_list(self, ov_info: OpenVINOInfo) -> list[GPUDeviceInfo]:
        """Build detailed GPUDeviceInfo list from OpenVINO device enumeration.

        Queries ov.Core().get_property(device_id, 'FULL_DEVICE_NAME') for each
        GPU device found.  Falls back to using the raw device_id string if the
        property query fails.

        Args:
            ov_info: Previously probed OpenVINOInfo.

        Returns:
            List of GPUDeviceInfo (empty if no GPU devices or OV unavailable).
        """
        if not ov_info.installed or not ov_info.gpu_devices:
            return []

        result: list[GPUDeviceInfo] = []
        try:
            core = ov.Core()  # type: ignore[union-attr]
            for device_id in ov_info.gpu_devices:
                try:
                    full_name: str = core.get_property(device_id, "FULL_DEVICE_NAME")
                except Exception:
                    full_name = device_id  # fallback to raw device string

                name_upper = full_name.upper()
                is_intel = "INTEL" in name_upper

                # Integrated GPU indicators: Iris Xe, UHD Graphics, HD Graphics
                is_integrated = any(
                    kw in name_upper
                    for kw in ("IRIS", "UHD", "HD GRAPHICS", "INTEGRATED")
                )
                # Intel Arc is discrete
                if "ARC" in name_upper:
                    is_integrated = False

                vendor = "Intel" if is_intel else ("AMD" if "AMD" in name_upper else "Unknown")

                result.append(
                    GPUDeviceInfo(
                        device_id=device_id,
                        full_name=full_name,
                        vendor=vendor,
                        is_intel=is_intel,
                        is_integrated=is_integrated,
                        supports_inference=True,
                    )
                )
        except Exception as exc:
            logger.warning("Failed to enumerate GPU device details: %s", exc)

        return result

    # ------------------------------------------------------------------ #
    # Backend recommendation                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_backend(ov_info: OpenVINOInfo) -> str:
        """Return the recommended OCR backend string.

        Returns:
            'openvino_gpu'  -- OpenVINO installed AND GPU device detected
            'openvino_cpu'  -- OpenVINO installed, CPU plugin confirmed usable
            'cpu_stub'      -- OpenVINO not installed, or Core() failed at init
        """
        if not ov_info.installed:
            return "cpu_stub"
        # Core() raised during init — no devices are actually usable
        if ov_info.error is not None:
            return "cpu_stub"
        if ov_info.gpu_device_available:
            return "openvino_gpu"
        if ov_info.cpu_device_available:
            return "openvino_cpu"
        # Installed but no recognised devices at all
        return "cpu_stub"
