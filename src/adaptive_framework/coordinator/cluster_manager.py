"""RayClusterManager — Ray cluster lifecycle management.

Responsibilities:
    - Initialize a Ray cluster (head node) or connect to a running one.
    - Discover all connected nodes after startup.
    - Monitor cluster health via Ray's internal API.
    - Expose ClusterStatus for the Coordinator and Dashboard.
    - Shut down the cluster gracefully.

Supports two deployment modes (controlled by cluster.yaml):
    dev          — Single laptop: ray.init() starts a local cluster.
    presentation — Three laptops: ray.init(address="<head_ip>:6379") connects.

The mode selection requires ZERO source code changes — only config.yaml changes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.core.constants import ROOT_LOGGER_NAME
from adaptive_framework.core.exceptions import FrameworkError

logger = logging.getLogger(ROOT_LOGGER_NAME + ".cluster_manager")

# Guard Ray import — unit tests run without a live cluster
try:
    import ray  # type: ignore[import]
    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAY_AVAILABLE = False
    ray = None  # type: ignore[assignment]


class ClusterManager:
    """Manages the Ray cluster lifecycle.

    This class wraps Ray initialization, node discovery, and shutdown.
    It is designed to be mode-agnostic: pass a ``ClusterConfig`` dict
    derived from ``cluster.yaml`` and the correct behavior is selected
    automatically.

    Attributes:
        _config: Cluster configuration dictionary.
        _mode: Deployment mode ('dev' or 'presentation').
        _initialized: True after successful ray.init().
        _head_node: NodeInfo for the head node.
        _worker_nodes: NodeInfo list for all connected workers.
        _init_time: Monotonic time at cluster initialization.

    Example:
        >>> manager = ClusterManager(config={"mode": "dev", "head": {"address": "auto"}})
        >>> manager.start()
        >>> status = manager.get_status()
        >>> manager.shutdown()
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the ClusterManager with a configuration dict.

        Args:
            config: Cluster configuration (from cluster.yaml → cluster section).
        """
        self._config = config
        self._mode: str = config.get("mode", "dev")
        self._initialized: bool = False
        self._head_node: NodeInfo | None = None
        self._worker_nodes: list[NodeInfo] = []
        self._init_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Startup                                                              #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Initialize the Ray cluster.

        In **dev mode**: calls ``ray.init()`` to start a local cluster.
        In **presentation mode**: calls ``ray.init(address=...)`` to connect
        to the head node started on Laptop 1.

        Raises:
            FrameworkError: If Ray is not available or init fails.
        """
        if not _RAY_AVAILABLE:
            raise FrameworkError(
                "Ray is not installed. Install with: pip install 'adaptive-distributed-framework[dev]'"
            )

        if self._initialized:
            logger.warning("ClusterManager.start() called on already-initialized cluster.")
            return

        head_cfg = self._config.get("head", {})
        address = head_cfg.get("address", "auto")
        dashboard_port = head_cfg.get("dashboard_port", 8265)
        namespace = self._config.get("workers", {}).get("namespace", "adaptive_framework")

        logger.info("Starting Ray cluster (mode=%s, address=%s).", self._mode, address)

        try:
            if address == "auto":
                # Dev mode — start a local Ray cluster
                ray.init(
                    ignore_reinit_error=True,
                    dashboard_port=dashboard_port,
                    namespace=namespace,
                    logging_level=logging.WARNING,
                )
            else:
                # Presentation mode — connect to running head node
                ray.init(
                    address=address,
                    namespace=namespace,
                    logging_level=logging.WARNING,
                )
        except Exception as exc:
            raise FrameworkError(f"Ray initialization failed: {exc}") from exc

        self._initialized = True
        self._init_time = time.monotonic()
        self._head_node = NodeInfo.from_current_host(is_head=(address == "auto"))
        self._discover_nodes()
        logger.info(
            "Ray cluster started. Nodes discovered: %d.", len(self._worker_nodes) + 1
        )

    def _discover_nodes(self) -> None:
        """Query Ray for all connected nodes and build NodeInfo records.

        Populates ``self._worker_nodes`` from Ray's node list.
        """
        if not _RAY_AVAILABLE or not self._initialized:
            return

        try:
            nodes = ray.nodes()
        except Exception:
            nodes = []

        self._worker_nodes = []
        for node in nodes:
            if not node.get("Alive", False):
                continue
            node_ip = node.get("NodeManagerAddress", "unknown")
            resources = node.get("Resources", {})
            cpu_count = int(resources.get("CPU", 0))
            hostname = node.get("NodeManagerHostname", node_ip)

            info = NodeInfo(
                hostname=hostname,
                ip_address=node_ip,
                cpu_count_logical=cpu_count,
                cpu_count_physical=max(1, cpu_count // 2),
                ray_node_id=node.get("NodeID", ""),
                is_head=False,
            )
            self._worker_nodes.append(info)

    # ------------------------------------------------------------------ #
    # Health & Status                                                      #
    # ------------------------------------------------------------------ #

    def is_alive(self) -> bool:
        """Return True if the cluster is running and Ray is initialized.

        Returns:
            True when initialized and ray.is_initialized() is True.
        """
        if not _RAY_AVAILABLE or not self._initialized:
            return False
        try:
            return ray.is_initialized()
        except Exception:
            return False

    def get_node_count(self) -> int:
        """Return the number of alive Ray nodes (including head).

        Returns:
            Total alive node count, or 0 if Ray is not running.
        """
        if not self.is_alive():
            return 0
        try:
            return sum(1 for n in ray.nodes() if n.get("Alive", False))
        except Exception:
            return 0

    def get_cluster_resources(self) -> dict[str, float]:
        """Return Ray's view of available cluster resources.

        Returns:
            Dictionary of resource_name → available_quantity (e.g. 'CPU': 16.0).
        """
        if not self.is_alive():
            return {}
        try:
            return ray.available_resources()
        except Exception:
            return {}

    def get_all_nodes(self) -> list[NodeInfo]:
        """Return all known NodeInfo records (head + workers).

        Returns:
            List of NodeInfo objects.
        """
        nodes = []
        if self._head_node is not None:
            nodes.append(self._head_node)
        nodes.extend(self._worker_nodes)
        return nodes

    def get_head_node(self) -> NodeInfo | None:
        """Return the head NodeInfo.

        Returns:
            NodeInfo for the head node, or None if not initialized.
        """
        return self._head_node

    def refresh_nodes(self) -> None:
        """Re-discover connected nodes from Ray.

        Call periodically to detect new nodes joining the cluster.
        """
        self._discover_nodes()

    @property
    def uptime_seconds(self) -> float:
        """Return cluster uptime in seconds.

        Returns:
            Seconds since ray.init() completed, or 0 if not started.
        """
        if not self._initialized:
            return 0.0
        return time.monotonic() - self._init_time

    # ------------------------------------------------------------------ #
    # Shutdown                                                             #
    # ------------------------------------------------------------------ #

    def shutdown(self, graceful: bool = True) -> None:
        """Shut down the Ray cluster.

        Args:
            graceful: If True (default), wait for in-flight tasks to finish
                before calling ray.shutdown(). In the current implementation
                this is a hint — actual graceful drain is handled by the
                DistributedCoordinator.

        Raises:
            FrameworkError: If Ray is not available.
        """
        if not _RAY_AVAILABLE:
            return
        if not self._initialized:
            logger.debug("ClusterManager.shutdown() called on uninitialized cluster.")
            return

        logger.info("Shutting down Ray cluster (graceful=%s).", graceful)
        try:
            ray.shutdown()
        except Exception as exc:
            logger.warning("Error during ray.shutdown(): %s", exc)
        finally:
            self._initialized = False
            self._head_node = None
            self._worker_nodes = []
            logger.info("Ray cluster shut down.")

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize current cluster state to a plain dictionary.

        Returns:
            Dictionary with mode, alive status, node count, and uptime.
        """
        return {
            "mode": self._mode,
            "initialized": self._initialized,
            "alive": self.is_alive(),
            "node_count": self.get_node_count(),
            "uptime_seconds": self.uptime_seconds,
            "head_node": self._head_node.to_dict() if self._head_node else None,
            "worker_node_count": len(self._worker_nodes),
        }

    def __repr__(self) -> str:
        return (
            f"ClusterManager(mode={self._mode!r}, "
            f"initialized={self._initialized}, "
            f"nodes={self.get_node_count()}, "
            f"uptime={self.uptime_seconds:.1f}s)"
        )
