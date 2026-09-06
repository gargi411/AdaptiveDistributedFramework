"""System utility functions for the Adaptive Distributed Framework.

Wraps psutil for CPU, memory, GPU, and network metrics used by
the ResourceSnapshot and RuntimeMetrics models.
"""

from __future__ import annotations

import os
import platform
import socket
from typing import Any

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def get_cpu_percent(interval: float = 0.1) -> float:
    """Return CPU utilization across all cores as a percentage.

    Args:
        interval: Sampling interval in seconds. A shorter interval is less
            accurate but blocks for less time.

    Returns:
        CPU utilization in [0.0, 100.0]. Returns -1.0 if psutil unavailable.

    Example:
        >>> cpu = get_cpu_percent()
        >>> print(f"CPU: {cpu:.1f}%")
        CPU: 42.3%
    """
    if not _PSUTIL_AVAILABLE:
        return -1.0
    return psutil.cpu_percent(interval=interval)


def get_memory_percent() -> float:
    """Return RAM utilization as a percentage.

    Returns:
        Memory utilization in [0.0, 100.0]. Returns -1.0 if unavailable.

    Example:
        >>> mem = get_memory_percent()
        >>> print(f"Memory: {mem:.1f}%")
        Memory: 65.4%
    """
    if not _PSUTIL_AVAILABLE:
        return -1.0
    return psutil.virtual_memory().percent


def get_cpu_count(logical: bool = True) -> int:
    """Return the number of CPUs available.

    Args:
        logical: If True, return logical CPU count (hyperthreading included).
            If False, return physical core count.

    Returns:
        CPU count. Returns 1 if psutil unavailable or detection fails.
    """
    if not _PSUTIL_AVAILABLE:
        return os.cpu_count() or 1
    count = psutil.cpu_count(logical=logical)
    return count if count is not None else 1


def get_cpu_physical_count() -> int:
    """Return the number of physical CPU cores.

    Returns:
        Physical core count, or logical count / 1 if undetectable.
    """
    return get_cpu_count(logical=False)


def get_available_memory_gb() -> float:
    """Return available system memory in gigabytes.

    Returns:
        Available RAM in GB. Returns -1.0 if unavailable.
    """
    if not _PSUTIL_AVAILABLE:
        return -1.0
    return psutil.virtual_memory().available / (1024 ** 3)


def get_memory_breakdown_mb() -> tuple[float, float, float]:
    """Return total, used, and available RAM in megabytes.

    Returns:
        Tuple of (total_mb, used_mb, available_mb).
        Returns (-1.0, -1.0, -1.0) if psutil is unavailable.
    """
    if not _PSUTIL_AVAILABLE:
        return (-1.0, -1.0, -1.0)
    vm = psutil.virtual_memory()
    total_mb = round(vm.total / (1024 * 1024), 2)
    used_mb = round(vm.used / (1024 * 1024), 2)
    available_mb = round(vm.available / (1024 * 1024), 2)
    return (total_mb, used_mb, available_mb)


def get_hostname() -> str:
    """Return the hostname of the current machine.

    Returns:
        Hostname string.

    Example:
        >>> print(get_hostname())
        'laptop-01'
    """
    return socket.gethostname()


def get_platform_info() -> dict[str, str]:
    """Return a dictionary of platform information.

    Returns:
        Dictionary with keys: os, os_version, python_version, machine,
        processor, hostname.

    Example:
        >>> info = get_platform_info()
        >>> print(info["os"])
        'Windows'
    """
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": get_hostname(),
    }


def is_psutil_available() -> bool:
    """Return True if psutil is installed and importable.

    Returns:
        True if psutil is available.
    """
    return _PSUTIL_AVAILABLE
