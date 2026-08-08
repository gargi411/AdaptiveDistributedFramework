"""NodeInfo — Physical node metadata for the Ray cluster.

Captures hardware identity and capabilities for each node in the cluster.
Used by the ClusterManager for node discovery and by the Engineering Dashboard
for cluster topology display.
"""

from __future__ import annotations

import platform
import socket
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

try:
    import psutil  # type: ignore[import]
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PSUTIL_AVAILABLE = False


@dataclass
class NodeInfo:
    """Hardware and identity metadata for a single cluster node.

    Attributes:
        node_id: Unique identifier for this node (auto-generated UUID4).
        hostname: Network hostname of this machine.
        ip_address: Primary IP address of this machine.
        os_platform: Operating system platform string.
        cpu_count_logical: Number of logical CPU cores (with HT/SMT).
        cpu_count_physical: Number of physical CPU cores.
        ram_total_gb: Total installed RAM in gigabytes.
        gpu_count: Number of available GPUs (0 if none detected).
        gpu_names: List of GPU device names.
        ray_node_id: Ray's internal node identifier (set after Ray init).
        is_head: True if this node is the Ray head node.
        registered_at: ISO 8601 UTC timestamp of when this record was created.

    Example:
        >>> node = NodeInfo.from_current_host()
        >>> print(node.hostname)
        'my-laptop'
        >>> print(node.cpu_count_logical)
        8
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hostname: str = field(default_factory=socket.gethostname)
    ip_address: str = ""
    os_platform: str = field(default_factory=platform.system)
    cpu_count_logical: int = 0
    cpu_count_physical: int = 0
    ram_total_gb: float = 0.0
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    ray_node_id: str = ""
    is_head: bool = False
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_current_host(cls, is_head: bool = False) -> "NodeInfo":
        """Build a NodeInfo by probing the current machine's hardware.

        Uses ``psutil`` for memory/CPU and socket for network info.
        Gracefully degrades when psutil is unavailable.

        Args:
            is_head: True if this node is the cluster head node.

        Returns:
            NodeInfo populated with the current machine's details.
        """
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except socket.gaierror:
            ip_address = "127.0.0.1"

        cpu_logical = 0
        cpu_physical = 0
        ram_gb = 0.0
        gpu_count = 0
        gpu_names: list[str] = []

        if _PSUTIL_AVAILABLE:
            cpu_logical = psutil.cpu_count(logical=True) or 0
            cpu_physical = psutil.cpu_count(logical=False) or 0
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

        # Optional GPU detection via gputil (not a hard dependency)
        try:
            import GPUtil  # type: ignore[import]
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            gpu_names = [g.name for g in gpus]
        except ImportError:
            pass

        return cls(
            hostname=hostname,
            ip_address=ip_address,
            os_platform=platform.system(),
            cpu_count_logical=cpu_logical,
            cpu_count_physical=cpu_physical,
            ram_total_gb=ram_gb,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            is_head=is_head,
        )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def display_name(self) -> str:
        """Return a human-readable node label for the dashboard.

        Returns:
            'Head: hostname' or 'Worker: hostname'.
        """
        role = "Head" if self.is_head else "Worker"
        return f"{role}: {self.hostname}"

    @property
    def has_gpu(self) -> bool:
        """Return True if this node has at least one GPU.

        Returns:
            True when gpu_count > 0.
        """
        return self.gpu_count > 0

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary representation of this NodeInfo.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"NodeInfo(node_id='{self.node_id[:8]}', "
            f"hostname='{self.hostname}', "
            f"ip='{self.ip_address}', "
            f"cpus={self.cpu_count_logical}, "
            f"ram={self.ram_total_gb:.1f}GB, "
            f"gpus={self.gpu_count}, "
            f"head={self.is_head})"
        )
