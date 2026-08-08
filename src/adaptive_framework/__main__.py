"""Entry point for the adaptive_framework package when invoked as a module.

Usage:
    python -m adaptive_framework

Prints the framework version and status. Full initialization is done
via the project root's main.py (the composition root).
"""

from __future__ import annotations

from adaptive_framework.core.constants import FRAMEWORK_NAME, FRAMEWORK_VERSION


def main() -> None:
    """Print the framework identity banner."""
    print(f"{FRAMEWORK_NAME} v{FRAMEWORK_VERSION}")
    print("Run `python main.py` from the project root to fully initialize the framework.")


if __name__ == "__main__":
    main()
