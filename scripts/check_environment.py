#!/usr/bin/env python3
"""check_environment.py — Environment verification script.

Checks that all required dependencies, Python version, and
directory structure are in place before running the framework.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on the path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import importlib
import platform


def check_python_version() -> bool:
    """Verify Python >= 3.12."""
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = (major, minor) >= (3, 12)
    status = "[OK]" if ok else "[FAIL]"
    print(f"  {status} Python {major}.{minor} (required: 3.12+)")
    return ok


def check_package(package: str, required: bool = True) -> bool:
    """Check whether a Python package is importable."""
    try:
        importlib.import_module(package)
        print(f"  [OK] {package}")
        return True
    except ImportError:
        symbol = "[FAIL]" if required else "[WARN] "
        note = "(required)" if required else "(optional)"
        print(f"  {symbol} {package} — NOT FOUND {note}")
        return not required  # optional packages don't fail the check


def check_directory(path: Path, name: str) -> bool:
    """Check whether a required directory exists."""
    ok = path.is_dir()
    status = "[OK]" if ok else "[FAIL]"
    print(f"  {status} {name}: {path}")
    return ok


def check_config_files() -> bool:
    """Check that all 7 YAML config files are present."""
    config_dir = Path("configs")
    required = [
        "framework.yaml", "logging.yaml", "ray_cluster.yaml",
        "scheduler.yaml", "ocr.yaml", "evaluation.yaml", "rag.yaml",
    ]
    all_ok = True
    for filename in required:
        path = config_dir / filename
        ok = path.exists()
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} configs/{filename}")
        if not ok:
            all_ok = False
    return all_ok


def main() -> int:
    """Run all environment checks. Returns 0 on success, 1 on failure."""
    print("=" * 60)
    print("  Adaptive Distributed Framework — Environment Check")
    print("=" * 60)
    print()

    failures: list[str] = []

    # Python version
    print("> Python Version")
    if not check_python_version():
        failures.append("Python version")
    print()

    # Required packages
    print("> Required Packages")
    required_pkgs = ["yaml", "psutil", "rich"]
    for pkg in required_pkgs:
        if not check_package(pkg):
            failures.append(pkg)
    print()

    # Optional packages (Phase 2+)
    print("> Optional Packages (Phase 2+)")
    optional_pkgs = ["ray", "paddleocr", "langchain", "chromadb"]
    for pkg in optional_pkgs:
        check_package(pkg, required=False)
    print()

    # Directory structure
    print("> Directory Structure")
    dirs = {
        "configs/": Path("configs"),
        "src/adaptive_framework/": Path("src/adaptive_framework"),
        "tests/": Path("tests"),
        "scripts/": Path("scripts"),
    }
    for name, path in dirs.items():
        if not check_directory(path, name):
            failures.append(f"directory: {name}")
    print()

    # Config files
    print("> Configuration Files")
    if not check_config_files():
        failures.append("config files")
    print()

    # Summary
    print("=" * 60)
    if failures:
        print(f"  [FAIL] FAILED — {len(failures)} issue(s) found:")
        for f in failures:
            print(f"     • {f}")
        print()
        print("  Fix the above issues before running the framework.")
        return 1
    else:
        print("  [OK] All checks passed. Environment is ready.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
