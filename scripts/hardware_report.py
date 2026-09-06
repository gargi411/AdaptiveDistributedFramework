#!/usr/bin/env python3
"""ADF Hardware Capability Diagnostic Script.

Probes the current machine for CPU, Intel GPU, and OpenVINO support.
Prints a plain ASCII report -- no emojis, no colour codes.

Usage:
    uv run python scripts/hardware_report.py

Output sections:
    CPU:              brand, architecture, cores, frequency
    OpenVINO:         installed / not installed, version, devices
    GPU devices:      list of GPU devices visible to OpenVINO
    GPU backend:      available / not available
    OCR acceleration: recommended backend string
    Fallback:         description of fallback behaviour
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- allow running directly from project root or scripts/ dir
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> None:
    """Run hardware probe and print report."""
    try:
        from adaptive_framework.acceleration.hardware_probe import HardwareCapabilityProbe
    except ImportError as exc:
        print(f"[ERROR] Cannot import adaptive_framework: {exc}")
        print(
            "Make sure you are running from the project root with uv run, "
            "or that src/ is on your PYTHONPATH."
        )
        sys.exit(1)

    print("Probing hardware...")
    probe = HardwareCapabilityProbe()
    profile = probe.probe()
    print()
    print(profile.format_report())


if __name__ == "__main__":
    main()
