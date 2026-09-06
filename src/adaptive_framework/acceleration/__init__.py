"""Adaptive Framework — Hardware Acceleration Package (Phase 4.1).

Provides hardware detection and OpenVINO integration foundation.
All components degrade gracefully when OpenVINO is not installed.

Public API:
    HardwareCapabilityProbe    -- detects CPU, Intel GPU, OpenVINO
    HardwareProfile            -- structured result of hardware detection
    CPUInfo                    -- CPU brand/core/frequency details
    OpenVINOInfo               -- OpenVINO installation and device details
    GPUDeviceInfo              -- per-GPU device metadata
    OpenVINOOCRStrategy        -- OpenVINO OCR strategy (Phase 4.1 foundation)

Usage:
    from adaptive_framework.acceleration import HardwareCapabilityProbe
    probe = HardwareCapabilityProbe()
    profile = probe.probe()
    print(profile.format_report())
"""

from __future__ import annotations

from adaptive_framework.acceleration.hardware_probe import (
    CPUInfo,
    GPUDeviceInfo,
    HardwareCapabilityProbe,
    HardwareProfile,
    OpenVINOInfo,
)
from adaptive_framework.acceleration.gpu_monitor import GPUMetrics, GPUMonitor
from adaptive_framework.acceleration.model_setup import ensure_openvino_ocr_models
from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy
from adaptive_framework.acceleration.resource_monitor import ResourceMonitor

__all__ = [
    "CPUInfo",
    "GPUDeviceInfo",
    "GPUMetrics",
    "GPUMonitor",
    "HardwareCapabilityProbe",
    "HardwareProfile",
    "OpenVINOInfo",
    "OpenVINOOCRStrategy",
    "ResourceMonitor",
    "ensure_openvino_ocr_models",
]
