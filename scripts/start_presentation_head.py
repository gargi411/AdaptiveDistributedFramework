"""start_presentation_head.py — Launch the head node for three-laptop presentation.

Run this script on Laptop 1 (the head/coordinator machine).

Usage:
    python scripts/start_presentation_head.py [--config configs/simulation_presentation.yaml]

After starting:
    1. Note the IP address printed to console.
    2. Update configs/simulation_presentation.yaml:
           head.address: "<this_IP>:6379"
           presentation.head_ip: "<this_IP>"
    3. Run start_presentation_worker.py on Laptops 2 and 3.
    4. Open dashboard: streamlit run dashboard/app.py
"""

from __future__ import annotations

import signal
import sys
import time
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml

from adaptive_framework.coordinator.cluster_manager import ClusterManager
from adaptive_framework.coordinator.distributed_coordinator import DistributedCoordinator
from adaptive_framework.coordinator.node_info import NodeInfo
from dashboard.state.dashboard_state import DashboardStateStore

_SHUTDOWN_REQUESTED = False


def _handle_sigint(signum: int, frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    print("\n\n[HEAD] Shutdown requested...")
    _SHUTDOWN_REQUESTED = True


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("cluster", raw)


def main() -> None:
    """Start the head node and coordinator for presentation mode."""
    parser = argparse.ArgumentParser(description="ADF Presentation Head Node")
    parser.add_argument("--config", default="configs/simulation_presentation.yaml")
    parser.add_argument("--state-file", default="outputs/dashboard_state.json")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    config = _load_config(args.config)
    # Force head to auto-start in presentation mode (head machine)
    config_for_head = dict(config)
    config_for_head["head"] = dict(config.get("head", {}))
    config_for_head["head"]["address"] = "auto"

    print("[HEAD] Starting Ray head node...")
    state_store = DashboardStateStore(file_path=args.state_file)
    coordinator = DistributedCoordinator(
        cluster_config=config_for_head,
        run_id=f"presentation_{int(time.time())}",
    )
    coordinator.start(init_ray=True)
    head_info = NodeInfo.from_current_host(is_head=True)
    print(f"[HEAD] Head node started: {head_info.hostname} ({head_info.ip_address})")
    print(f"[HEAD] Update worker configs with: head.address = {head_info.ip_address}:6379")
    print(f"[HEAD] Dashboard: streamlit run dashboard/app.py")
    print(f"[HEAD] Waiting for workers... Ctrl+C to stop.\n")

    interval = config.get("health", {}).get("heartbeat_interval_seconds", 5.0)
    while not _SHUTDOWN_REQUESTED:
        status = coordinator.get_status_dict()
        registry = coordinator.registry
        status["workers"] = [rec.to_dict() for rec in registry.get_all()]
        state_store.update(status)
        reg = status.get("registry", {})
        print(
            f"\r[HEAD] Workers: {reg.get('total_workers', 0)} | "
            f"Active: {reg.get('active_workers', 0)} | "
            f"Queue: {status.get('queue_size', 0)}",
            end="", flush=True,
        )
        time.sleep(interval)

    print("\n[HEAD] Stopping head node...")
    coordinator.stop(shutdown_ray=True)
    print("[HEAD] Head node stopped.")


if __name__ == "__main__":
    main()
