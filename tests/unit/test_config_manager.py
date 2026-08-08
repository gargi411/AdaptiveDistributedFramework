"""Unit tests for ConfigManager.

Tests:
    - Singleton behavior
    - Successful YAML load
    - Missing directory raises ConfigurationError
    - Typed accessor returns correct model
    - Hot reload updates config
"""

from __future__ import annotations

import pytest
from pathlib import Path

from adaptive_framework.config.config_manager import ConfigManager
from adaptive_framework.core.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset the ConfigManager singleton before each test."""
    ConfigManager._instance = None
    yield
    ConfigManager._instance = None


class TestConfigManagerSingleton:
    """Tests for singleton behavior."""

    def test_same_instance_returned(self) -> None:
        """get_instance() must return the same object every time."""
        cfg1 = ConfigManager.get_instance()
        cfg2 = ConfigManager.get_instance()
        assert cfg1 is cfg2, "ConfigManager must be a singleton."

    def test_instance_is_config_manager(self) -> None:
        """get_instance() must return a ConfigManager."""
        cfg = ConfigManager.get_instance()
        assert isinstance(cfg, ConfigManager)


class TestConfigManagerLoad:
    """Tests for load() behavior."""

    def test_load_from_valid_configs_dir(self, tmp_path: Path) -> None:
        """Loading from a valid directory with one YAML should not raise."""
        # Create a minimal framework.yaml
        yaml_content = (
            "framework:\n"
            "  name: 'Test Framework'\n"
            "  version: '2.0.0'\n"
            "  run_id_prefix: 'test_run'\n"
            "  output_dir: 'outputs'\n"
            "  debug: false\n"
            "  max_concurrent_jobs: 4\n"
            "  stage_timeout_seconds: 3600\n"
            "  shutdown_timeout_seconds: 30\n"
        )
        (tmp_path / "framework.yaml").write_text(yaml_content, encoding="utf-8")

        cfg = ConfigManager.get_instance()
        cfg.load(config_dir=tmp_path)  # should not raise

        fw = cfg.get_framework_config()
        assert fw.name == "Test Framework"
        assert fw.max_concurrent_jobs == 4

    def test_load_missing_directory_raises(self) -> None:
        """Loading from a non-existent directory raises ConfigurationError."""
        cfg = ConfigManager.get_instance()
        with pytest.raises(ConfigurationError, match="not found"):
            cfg.load(config_dir=Path("/nonexistent/path/xyz"))

    def test_require_loaded_raises_before_load(self) -> None:
        """Accessing config before load() raises ConfigurationError."""
        cfg = ConfigManager.get_instance()
        with pytest.raises(ConfigurationError, match="not been loaded"):
            cfg.get_framework_config()

    def test_empty_directory_loads_without_error(self, tmp_path: Path) -> None:
        """An empty configs directory should load successfully (all files optional)."""
        cfg = ConfigManager.get_instance()
        cfg.load(config_dir=tmp_path)
        raw = cfg.get_raw()
        assert raw == {}
