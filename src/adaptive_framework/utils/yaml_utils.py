"""YAML utility functions for the Adaptive Distributed Framework.

Wraps PyYAML with safe loading and clear error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from adaptive_framework.core.exceptions import ConfigurationError


def load_yaml_file(file_path: Path) -> dict[str, Any]:
    """Load and parse a YAML file using safe_load.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary. Returns empty dict if file
        is empty or contains only null/whitespace.

    Raises:
        ConfigurationError: If the file does not exist, cannot be read,
            or contains invalid YAML syntax.

    Example:
        >>> data = load_yaml_file(Path("configs/framework.yaml"))
        >>> print(data["framework"]["name"])
        'Adaptive Distributed Parallel Processing Framework'
    """
    if not file_path.exists():
        raise ConfigurationError(f"YAML file not found: '{file_path}'")
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        return content if isinstance(content, dict) else {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML file '{file_path}': {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read YAML file '{file_path}': {exc}"
        ) from exc


def dump_yaml(data: dict[str, Any], file_path: Path) -> None:
    """Serialize a dictionary to a YAML file.

    Args:
        data: Dictionary to serialize.
        file_path: Destination file path (parent directories created if needed).

    Raises:
        ConfigurationError: If the file cannot be written.

    Example:
        >>> dump_yaml({"key": "value"}, Path("outputs/config_snapshot.yaml"))
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"Failed to write YAML to '{file_path}': {exc}"
        ) from exc


def load_yaml_string(yaml_string: str) -> dict[str, Any]:
    """Parse a YAML string and return a dictionary.

    Useful for tests that pass inline YAML strings.

    Args:
        yaml_string: Raw YAML content as a string.

    Returns:
        Parsed dictionary. Returns empty dict for empty strings.

    Raises:
        ConfigurationError: If the string contains invalid YAML.

    Example:
        >>> data = load_yaml_string("framework:\\n  name: ADF")
        >>> data["framework"]["name"]
        'ADF'
    """
    try:
        content = yaml.safe_load(yaml_string)
        return content if isinstance(content, dict) else {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML string: {exc}") from exc


def merge_yaml_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge multiple YAML dictionaries (later wins on conflict).

    Args:
        *dicts: Variable number of dictionaries to merge.

    Returns:
        A new dictionary with all keys merged. Nested dicts are merged
        recursively; non-dict values are overwritten by later dicts.

    Example:
        >>> base = {"a": {"x": 1, "y": 2}}
        >>> override = {"a": {"y": 99}}
        >>> result = merge_yaml_dicts(base, override)
        >>> result["a"]
        {'x': 1, 'y': 99}
    """
    result: dict[str, Any] = {}
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_yaml_dicts(result[key], value)
            else:
                result[key] = value
    return result
