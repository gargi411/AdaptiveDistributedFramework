#!/usr/bin/env python3
"""run_framework.py — Framework execution entry point script.

Delegates to main.py. Provides a script-based entry point with
argument support for future phases.

Usage:
    python scripts/run_framework.py
    python scripts/run_framework.py --config-dir path/to/configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    """Parse arguments and launch the framework."""
    parser = argparse.ArgumentParser(
        description="Adaptive Distributed Parallel Processing Framework"
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Path to the configs directory (default: configs/)",
    )
    args = parser.parse_args()

    # Import here to ensure sys.path is set first
    from adaptive_framework.config import ConfigManager
    from adaptive_framework.core.constants import FRAMEWORK_NAME, FRAMEWORK_VERSION
    from adaptive_framework.core.exceptions import ConfigurationError

    print(f"{FRAMEWORK_NAME} v{FRAMEWORK_VERSION}")
    print("-" * 60)

    try:
        cfg = ConfigManager.get_instance()
        cfg.load(config_dir=Path(args.config_dir))
        print(f"  Configuration loaded from '{args.config_dir}/'")
    except ConfigurationError as exc:
        print(f"  [FAIL] Configuration error: {exc}")
        return 1

    print("  [OK] Framework ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
