#!/usr/bin/env python3
"""setup_environment.py — Development environment setup script.

Guides the developer through installing dependencies and setting up
pre-commit hooks. Infrastructure only — no framework logic.

Usage:
    python scripts/setup_environment.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a shell command and report success/failure.

    Args:
        cmd: Command list to execute.
        description: Human-readable description of the command.

    Returns:
        True on success, False on failure.
    """
    print(f"  ▶ {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"    ✅ Done.")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"    ❌ Failed: {exc.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"    ❌ Command not found: {cmd[0]}")
        return False


def main() -> int:
    """Run setup steps and return exit code."""
    print("=" * 60)
    print("  Adaptive Distributed Framework — Environment Setup")
    print("=" * 60)
    print()

    failures: list[str] = []

    # Step 1: Install in editable mode with dev extras
    if not run_command(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        "Installing package in editable mode with dev dependencies"
    ):
        failures.append("pip install -e .[dev]")

    # Step 2: Install pre-commit hooks
    if not run_command(
        [sys.executable, "-m", "pre_commit", "install"],
        "Installing pre-commit hooks"
    ):
        print("    ⚠️  pre-commit not installed — run 'pip install pre-commit' first.")

    print()
    print("=" * 60)
    if failures:
        print(f"  ❌ Setup incomplete — {len(failures)} step(s) failed.")
        return 1
    else:
        print("  ✅ Setup complete.")
        print()
        print("  Next steps:")
        print("    1. python scripts/check_environment.py")
        print("    2. python scripts/validate_configuration.py")
        print("    3. python main.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
