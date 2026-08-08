"""start_presentation_worker.py — Launch a worker node for three-laptop presentation.

Run this script on Laptop 2 and Laptop 3.

Usage:
    python scripts/start_presentation_worker.py [--config configs/simulation_presentation.yaml]

Prerequisites:
    1. Laptop 1 must be running start_presentation_head.py.
    2. The head IP must be set in configs/simulation_presentation.yaml:
           head.address: "<laptop1_ip>:6379"
    3. All laptops must be on the same LAN (or VPN).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import socket
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml

from adaptive_framework.coordinator.heartbeat_monitor import HeartbeatPayload
from adaptive_framework.coordinator.node_info import NodeInfo
from adaptive_framework.models.runtime import WorkerState

_SHUTDOWN_REQUESTED = False

try:
    import ray  # type: ignore[import]
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False


def _handle_sigint(signum: int, frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    print("\n\n[WORKER] Shutdown requested...")
    _SHUTDOWN_REQUESTED = True


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("cluster", raw)


def main() -> None:
    """Connect to the head node and start processing as a worker."""
    parser = argparse.ArgumentParser(description="ADF Presentation Worker Node")
    parser.add_argument("--config", default="configs/simulation_presentation.yaml")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of local worker processes (default from config)",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _handle_sigint)

    config = _load_config(args.config)
    head_address = config.get("head", {}).get("address", "auto")
    namespace = config.get("workers", {}).get("namespace", "adaptive_framework")
    num_workers = args.num_workers or config.get("presentation", {}).get("workers_per_laptop", 2)

    print(f"[WORKER] Connecting to head node: {head_address}")
    print(f"[WORKER] Local workers: {num_workers}")

    if _RAY_AVAILABLE:
        try:
            ray.init(address=head_address, namespace=namespace)
            print("[WORKER] Connected to Ray head node.")
        except Exception as exc:
            print(f"[WORKER] Failed to connect to Ray: {exc}")
            print("[WORKER] Running in standalone simulation mode.")

    hostname = socket.gethostname()
    node_info = NodeInfo.from_current_host(is_head=False)
    print(f"[WORKER] Host: {hostname} ({node_info.ip_address})")
    print(f"[WORKER] CPUs: {node_info.cpu_count_logical} | RAM: {node_info.ram_total_gb:.1f} GB")
    print(f"[WORKER] Press Ctrl+C to stop.\n")

    heartbeat_interval = config.get("health", {}).get("heartbeat_interval_seconds", 5.0)

    import psutil
    worker_ids = [f"{hostname}_w{i:02d}" for i in range(num_workers)]

    while not _SHUTDOWN_REQUESTED:
        for i, wid in enumerate(worker_ids):
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            payload = HeartbeatPayload(
                worker_id=wid,
                cpu_percent=cpu,
                ram_percent=ram,
                gpu_percent=None,
                queue_depth=0,
                current_task_id=None,
                state=WorkerState.IDLE.value,
            )
            print(
                f"\r[WORKER] {wid}: CPU={cpu:.1f}% RAM={ram:.1f}%    ",
                end="", flush=True,
            )

        time.sleep(heartbeat_interval)

    if _RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()

    print("\n[WORKER] Worker node stopped.")


if __name__ == "__main__":
    main()
