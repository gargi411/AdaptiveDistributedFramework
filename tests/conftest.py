"""Root-level pytest configuration and shared fixtures.

All fixtures defined here are available to all test modules in every
sub-directory (unit/, integration/, performance/).

Fixtures:
    tmp_config_dir: A tmp_path-based directory pre-populated with a
        minimal framework.yaml for testing ConfigManager.
    minimal_framework_yaml: The raw YAML string for a minimal
        framework config.
    reset_config_singleton: Auto-use fixture that resets the
        ConfigManager singleton before and after each test.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from adaptive_framework.config.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# Minimal valid framework.yaml content
# ---------------------------------------------------------------------------

MINIMAL_FRAMEWORK_YAML: str = (
    "framework:\n"
    "  name: 'Test Framework'\n"
    "  version: '2.0.0'\n"
    "  run_id_prefix: 'test_run'\n"
    "  output_dir: 'test_outputs'\n"
    "  debug: false\n"
    "  max_concurrent_jobs: 2\n"
    "  stage_timeout_seconds: 60\n"
    "  shutdown_timeout_seconds: 10\n"
)


@pytest.fixture()
def minimal_framework_yaml() -> str:
    """Return a minimal valid framework.yaml content string.

    Returns:
        YAML string suitable for writing to a test config directory.
    """
    return MINIMAL_FRAMEWORK_YAML


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with a minimal framework.yaml.

    Args:
        tmp_path: pytest built-in temporary directory.

    Returns:
        Path to the temporary config directory.
    """
    (tmp_path / "framework.yaml").write_text(MINIMAL_FRAMEWORK_YAML, encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def reset_config_singleton() -> None:
    """Reset ConfigManager singleton state before and after every test.

    This prevents test pollution when ConfigManager is loaded in one test
    and another test expects an unloaded state.
    """
    ConfigManager._instance = None  # type: ignore[attr-defined]
    yield
    ConfigManager._instance = None  # type: ignore[attr-defined]
