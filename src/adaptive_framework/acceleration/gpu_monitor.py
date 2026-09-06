"""GPUMonitor -- Dedicated GPU utilization and resource monitoring abstraction.

Designed for Intel Iris Xe Graphics (iGPU) and OpenVINO runtime environments.
Provides runtime telemetry collection for:
  - GPU device identification and architecture
  - OpenVINO device memory (total, allocated, free)
  - Windows GPU Engine utilization percentage (via PDH)
  - Graceful handling of unsupported metrics (temperature, power)

Strict Design Principles:
  - Read-only observability: Never routes work or makes scheduling decisions.
  - Never fabricate numbers: Reports None / UNAVAILABLE when telemetry is absent.
  - Plain ASCII formatting: No emojis anywhere in code, docstrings, or logs.
  - Clean resource management: Closes PDH query handles properly.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.acceleration.hardware_probe import (
    HardwareCapabilityProbe,
    GPUDeviceInfo,
)

logger = logging.getLogger(__name__)

# OpenVINO Runtime: optional dependency
try:
    import openvino as ov  # type: ignore[import]
    _OPENVINO_AVAILABLE = True
except ImportError:
    _OPENVINO_AVAILABLE = False
    ov = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Windows PDH Telemetry structures
# ---------------------------------------------------------------------------

class _PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
    _fields_ = [
        ("szName", wintypes.LPWSTR),
        ("CStatus", wintypes.DWORD),
        ("padding", wintypes.DWORD),
        ("doubleValue", ctypes.c_double),
    ]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class GPUMetrics:
    """Snapshot of GPU runtime telemetry.

    Attributes:
        device_index: Integer index of this GPU (0 for primary).
        device_name: Human-readable name (e.g. 'Intel(R) Iris(R) Xe Graphics (iGPU)').
        vendor: Vendor string (e.g. 'Intel').
        architecture: Hardware architecture string (e.g. 'GPU: vendor=0x8086 arch=v12.0.0').
        openvino_device_id: OpenVINO device identifier (e.g. 'GPU', 'GPU.0').
        utilization_percent: Current engine utilization in [0.0, 100.0], or None.
        memory_total_mb: Total available GPU memory pool in MB, or None.
        memory_used_mb: GPU memory currently allocated by runtime in MB, or None.
        memory_free_mb: Available GPU memory in MB, or None.
        temperature_c: GPU temperature in Celsius, or None (UNAVAILABLE on iGPU).
        power_w: GPU power consumption in Watts, or None (UNAVAILABLE on iGPU).
        supported_metrics: Dictionary detailing which metrics are supported.
        timestamp: ISO 8601 UTC timestamp of the measurement.
    """

    device_index: int = 0
    device_name: str = "Unknown GPU"
    vendor: str = "Unknown"
    architecture: str = "Unknown"
    openvino_device_id: str = "GPU"
    utilization_percent: float | None = None
    memory_total_mb: float | None = None
    memory_used_mb: float | None = None
    memory_free_mb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    supported_metrics: dict[str, bool] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device_index": self.device_index,
            "device_name": self.device_name,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "openvino_device_id": self.openvino_device_id,
            "utilization_percent": self.utilization_percent,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
            "supported_metrics": self.supported_metrics,
            "timestamp": self.timestamp,
        }

    def format_summary(self) -> str:
        """Format as a clean ASCII report block."""
        util_str = (
            f"{self.utilization_percent:.1f}%"
            if self.utilization_percent is not None
            else "UNAVAILABLE"
        )
        mem_str = (
            f"{self.memory_used_mb:.1f} / {self.memory_total_mb:.1f} MB"
            if (self.memory_used_mb is not None and self.memory_total_mb is not None)
            else (
                f"Total: {self.memory_total_mb:.1f} MB"
                if self.memory_total_mb is not None
                else "UNAVAILABLE"
            )
        )
        temp_str = (
            f"{self.temperature_c:.1f} C"
            if self.temperature_c is not None
            else "UNAVAILABLE"
        )
        pwr_str = (
            f"{self.power_w:.1f} W"
            if self.power_w is not None
            else "UNAVAILABLE"
        )

        lines = [
            f"GPU [{self.device_index}] {self.device_name}",
            f"  Vendor:         {self.vendor}",
            f"  Architecture:   {self.architecture}",
            f"  OpenVINO ID:    {self.openvino_device_id}",
            f"  Utilization:    {util_str}",
            f"  Memory:         {mem_str}",
            f"  Temperature:    {temp_str}",
            f"  Power:          {pwr_str}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPUMonitor Class
# ---------------------------------------------------------------------------

class GPUMonitor:
    """Dedicated GPU runtime resource monitor.

    Inspects Intel Iris Xe and OpenVINO GPU devices and queries runtime telemetry.
    All telemetry methods degrade gracefully and report None / UNAVAILABLE when
    a specific sensor or API is absent.

    Usage:
        monitor = GPUMonitor()
        if monitor.is_available():
            info = monitor.get_device_info()
            metrics = monitor.sample()
            print(metrics.format_summary())
    """

    def __init__(
        self,
        device_id: str = "GPU",
        enable_pdh: bool = True,
    ) -> None:
        """Initialize the GPU monitor.

        Args:
            device_id: OpenVINO target device string (e.g. 'GPU', 'GPU.0').
            enable_pdh: Whether to enable Windows PDH counter for GPU utilization.
        """
        self._target_device = device_id
        self._enable_pdh = enable_pdh and (platform.system() == "Windows")

        self._ov_core: Any | None = None
        self._device_info: dict[str, Any] = {}
        self._is_available: bool = False
        self._device_index: int = 0

        # PDH handles for Windows GPU engine utilization
        self._pdh: Any | None = None
        self._pdh_query: wintypes.HANDLE | None = None
        self._pdh_counter: wintypes.HANDLE | None = None
        self._pdh_initialized: bool = False
        self._pdh_last_sample_time: float = 0.0

        self._init_hardware()
        if self._enable_pdh and self._is_available:
            self._init_pdh()

    # ------------------------------------------------------------------ #
    # Hardware Initialization & Discovery
    # ------------------------------------------------------------------ #

    def _init_hardware(self) -> None:
        """Probe hardware and OpenVINO devices."""
        if not _OPENVINO_AVAILABLE:
            self._is_available = False
            return

        try:
            self._ov_core = ov.Core()
            available_devices = list(self._ov_core.available_devices)
            gpu_devices = [d for d in available_devices if d.upper().startswith("GPU")]
            if not gpu_devices:
                self._is_available = False
                return

            self._is_available = True
            # Use specific target or first GPU found
            chosen_device = self._target_device if self._target_device in available_devices else gpu_devices[0]
            self._target_device = chosen_device

            # Retrieve OpenVINO properties safely
            supported = set(self._ov_core.get_property(chosen_device, "SUPPORTED_PROPERTIES"))

            def _get_prop(prop_name: str, default: Any = None) -> Any:
                if prop_name in supported:
                    try:
                        return self._ov_core.get_property(chosen_device, prop_name)
                    except Exception:
                        return default
                return default

            full_name = _get_prop("FULL_DEVICE_NAME", "Intel Graphics")
            arch = _get_prop("DEVICE_ARCHITECTURE", "Intel Gen12")
            device_type = str(_get_prop("DEVICE_TYPE", "INTEGRATED"))
            total_mem_bytes = _get_prop("GPU_DEVICE_TOTAL_MEM_SIZE", None)
            total_mem_mb = (
                round(float(total_mem_bytes) / (1024 * 1024), 2)
                if total_mem_bytes is not None
                else None
            )

            is_intel = "INTEL" in full_name.upper()
            vendor = "Intel" if is_intel else "Unknown"

            self._device_info = {
                "device_index": self._device_index,
                "device_name": full_name,
                "vendor": vendor,
                "architecture": str(arch),
                "openvino_device_id": chosen_device,
                "device_type": device_type,
                "total_memory_mb": total_mem_mb,
                "supported_properties": list(supported),
            }
        except Exception as exc:
            logger.warning("GPUMonitor initialization failed: %s", exc)
            self._is_available = False

    def _init_pdh(self) -> None:
        """Initialize Windows PDH query for GPU Engine utilization."""
        try:
            self._pdh = ctypes.windll.pdh
            h_query = wintypes.HANDLE()
            status = self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(h_query))
            if status != 0:
                logger.debug("PdhOpenQueryW failed with status: 0x%08x", status)
                return

            self._pdh_query = h_query
            h_counter = wintypes.HANDLE()
            status = self._pdh.PdhAddEnglishCounterW(
                self._pdh_query,
                "\\GPU Engine(*)\\Utilization Percentage",
                0,
                ctypes.byref(h_counter),
            )
            if status != 0:
                logger.debug("PdhAddEnglishCounterW failed with status: 0x%08x", status)
                self._close_pdh()
                return

            self._pdh_counter = h_counter
            # Prime the counter with an initial query
            self._pdh.PdhCollectQueryData(self._pdh_query)
            self._pdh_last_sample_time = time.monotonic()
            self._pdh_initialized = True
            logger.debug("Windows PDH GPU Engine counter initialized successfully.")
        except Exception as exc:
            logger.debug("Failed to initialize Windows PDH GPU counters: %s", exc)
            self._close_pdh()

    def _close_pdh(self) -> None:
        """Safely close PDH query handle."""
        if self._pdh and self._pdh_query:
            try:
                self._pdh.PdhCloseQuery(self._pdh_query)
            except Exception:
                pass
        self._pdh_query = None
        self._pdh_counter = None
        self._pdh_initialized = False

    # ------------------------------------------------------------------ #
    # Public Query Interface
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Return True if a GPU device is present and accessible via OpenVINO."""
        return self._is_available

    def get_device_info(self) -> dict[str, Any]:
        """Return static metadata for the monitored GPU device."""
        return dict(self._device_info)

    def get_utilization(self) -> float | None:
        """Return current GPU engine utilization percentage in [0.0, 100.0].

        Returns:
            Float utilization percentage, or None if unavailable.
        """
        if not self._is_available or not self._pdh_initialized or not self._pdh_query or not self._pdh_counter:
            return None

        try:
            # Need a minimum interval between samples for meaningful PDH delta
            now = time.monotonic()
            if now - self._pdh_last_sample_time < 0.05:
                time.sleep(0.05)

            status = self._pdh.PdhCollectQueryData(self._pdh_query)
            if status != 0:
                return None
            self._pdh_last_sample_time = time.monotonic()

            item_count = wintypes.DWORD(0)
            buffer_size = wintypes.DWORD(0)

            # First call retrieves required buffer size
            status = self._pdh.PdhGetFormattedCounterArrayW(
                self._pdh_counter,
                0x00000200,  # PDH_FMT_DOUBLE
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                None,
            )
            status_u = status & 0xFFFFFFFF
            # PDH_MORE_DATA is 0x800007D2
            if status_u not in (0, 0x800007D2) or buffer_size.value == 0:
                return None

            buf = ctypes.create_string_buffer(buffer_size.value)
            status = self._pdh.PdhGetFormattedCounterArrayW(
                self._pdh_counter,
                0x00000200,  # PDH_FMT_DOUBLE
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                buf,
            )
            if (status & 0xFFFFFFFF) != 0:
                return None

            items = ctypes.cast(buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM))
            total_util = 0.0
            for i in range(item_count.value):
                val = items[i].doubleValue
                if val > 0.0:
                    total_util += val

            # Clamp between 0.0 and 100.0
            clamped = max(0.0, min(100.0, round(total_util, 2)))
            return clamped
        except Exception as exc:
            logger.debug("Error querying PDH GPU utilization: %s", exc)
            return None

    def get_memory(self) -> dict[str, float | None]:
        """Return GPU memory metrics in megabytes.

        Returns:
            Dictionary with keys:
                'total_mb': Total device memory pool (MB), or None.
                'used_mb': Memory allocated by OpenVINO runtime (MB), or None.
                'free_mb': Available memory (MB), or None.
        """
        if not self._is_available or self._ov_core is None:
            return {"total_mb": None, "used_mb": None, "free_mb": None}

        total_mb = self._device_info.get("total_memory_mb")
        used_mb: float | None = None

        try:
            supported = self._device_info.get("supported_properties", [])
            if "GPU_MEMORY_STATISTICS" in supported:
                stats = self._ov_core.get_property(self._target_device, "GPU_MEMORY_STATISTICS")
                if isinstance(stats, dict):
                    total_alloc_bytes = sum(
                        v for v in stats.values() if isinstance(v, (int, float)) and v > 0
                    )
                    used_mb = round(float(total_alloc_bytes) / (1024 * 1024), 2)
        except Exception as exc:
            logger.debug("Failed to query OpenVINO GPU memory statistics: %s", exc)

        free_mb: float | None = None
        if total_mb is not None and used_mb is not None:
            free_mb = max(0.0, round(total_mb - used_mb, 2))
        elif total_mb is not None:
            free_mb = total_mb

        return {
            "total_mb": total_mb,
            "used_mb": used_mb,
            "free_mb": free_mb,
        }

    def get_temperature(self) -> float | None:
        """Return GPU temperature in Celsius.

        Note: Intel Iris Xe is integrated directly into the CPU package.
        A standalone discrete GPU temperature sensor is not exposed on Windows.
        Returns None to indicate UNAVAILABLE rather than fabricating values.
        """
        return None

    def get_power(self) -> float | None:
        """Return GPU power consumption in Watts.

        Note: Intel Iris Xe shares the SoC power package with the CPU cores.
        A dedicated discrete GPU power sensor is not exposed on Windows.
        Returns None to indicate UNAVAILABLE rather than fabricating values.
        """
        return None

    def sample(self) -> GPUMetrics:
        """Capture and return a point-in-time snapshot of all GPU telemetry.

        Returns:
            GPUMetrics dataclass populated with current telemetry.
        """
        if not self._is_available:
            return GPUMetrics(
                device_index=self._device_index,
                device_name="No GPU Available",
                supported_metrics={"utilization": False, "memory": False, "temperature": False, "power": False},
            )

        util = self.get_utilization()
        mem = self.get_memory()
        temp = self.get_temperature()
        power = self.get_power()

        supported = {
            "utilization": util is not None,
            "memory": mem["total_mb"] is not None,
            "temperature": False,  # Not supported on Intel Iris Xe iGPU on Windows
            "power": False,        # Not supported on Intel Iris Xe iGPU on Windows
        }

        return GPUMetrics(
            device_index=self._device_index,
            device_name=self._device_info.get("device_name", "Intel Graphics"),
            vendor=self._device_info.get("vendor", "Intel"),
            architecture=self._device_info.get("architecture", "Unknown"),
            openvino_device_id=self._target_device,
            utilization_percent=util,
            memory_total_mb=mem["total_mb"],
            memory_used_mb=mem["used_mb"],
            memory_free_mb=mem["free_mb"],
            temperature_c=temp,
            power_w=power,
            supported_metrics=supported,
        )

    # ------------------------------------------------------------------ #
    # Context Manager & Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close any open system query handles."""
        self._close_pdh()

    def __enter__(self) -> GPUMonitor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
