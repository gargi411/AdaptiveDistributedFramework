"""Path utility functions for the Adaptive Distributed Framework."""

from __future__ import annotations

import os
from pathlib import Path

from adaptive_framework.core.constants import DEFAULT_CONFIG_DIR, DEFAULT_OUTPUT_DIR


def get_project_root() -> Path:
    """Return the project root directory.

    Resolved as the directory containing the ``src`` folder,
    traversed up from this file's location.

    Returns:
        Absolute Path to the project root.

    Example:
        >>> root = get_project_root()
        >>> print(root.name)
        'AdaptiveDistributedFramework'
    """
    # This file is at src/adaptive_framework/utils/path_utils.py
    # Root is 3 levels up
    return Path(__file__).resolve().parents[3]


def get_configs_dir() -> Path:
    """Return the default configs directory path.

    Returns:
        Absolute Path to the configs directory (project_root/configs).
    """
    return get_project_root() / DEFAULT_CONFIG_DIR


def get_output_dir() -> Path:
    """Return the default output directory path.

    Returns:
        Absolute Path to the outputs directory (project_root/outputs).
    """
    return get_project_root() / DEFAULT_OUTPUT_DIR


def get_logs_dir(output_dir: Path | None = None) -> Path:
    """Return the logs directory path within output_dir.

    Args:
        output_dir: Override the default output directory. Defaults to get_output_dir().

    Returns:
        Absolute Path to the logs directory.
    """
    base = output_dir or get_output_dir()
    return base / "logs"


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path relative to base (or CWD if base is None).

    Args:
        path: The path to resolve (absolute or relative).
        base: Base directory for relative paths. Defaults to CWD.

    Returns:
        Resolved absolute Path.

    Example:
        >>> resolve_path("logs/framework.log", base=Path("/app/outputs"))
        PosixPath('/app/outputs/logs/framework.log')
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base_dir = base or Path.cwd()
    return (base_dir / p).resolve()


def is_subpath(child: Path, parent: Path) -> bool:
    """Check whether child is a subpath of parent.

    Args:
        child: Path to check.
        parent: Expected parent directory.

    Returns:
        True if child is inside parent (non-strictly).

    Example:
        >>> is_subpath(Path("/data/raw/paper.pdf"), Path("/data"))
        True
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
