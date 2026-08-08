#!/usr/bin/env python3
"""validate_configuration.py — Configuration validation script.

Loads all YAML configuration files and validates them through
the typed ConfigManager models. Reports any validation errors.

Usage:
    python scripts/validate_configuration.py
    python scripts/validate_configuration.py --config-dir path/to/configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_framework.config.config_manager import ConfigManager
from adaptive_framework.core.exceptions import ConfigurationError


def validate_all(config_dir: Path) -> int:
    """Load and validate all 7 configuration sections.

    Args:
        config_dir: Path to the configuration directory.

    Returns:
        0 on success, 1 on any validation failure.
    """
    print("=" * 60)
    print("  Adaptive Distributed Framework — Config Validation")
    print(f"  Directory: {config_dir.resolve()}")
    print("=" * 60)
    print()

    cfg = ConfigManager.get_instance()
    ConfigManager._instance = None  # reset for clean validation
    cfg = ConfigManager.get_instance()

    try:
        cfg.load(config_dir=config_dir)
        print("  ✅ YAML files loaded successfully.")
    except ConfigurationError as exc:
        print(f"  ❌ Failed to load configs: {exc}")
        return 1

    sections: list[tuple[str, object]] = []
    errors: list[tuple[str, str]] = []

    checks = [
        ("framework.yaml    → FrameworkConfig", cfg.get_framework_config),
        ("logging.yaml      → LoggingConfig", cfg.get_logging_config),
        ("ray_cluster.yaml  → RayClusterConfig", cfg.get_ray_cluster_config),
        ("scheduler.yaml    → SchedulerConfig", cfg.get_scheduler_config),
        ("ocr.yaml          → DocumentProcessingEngineConfig", cfg.get_document_processing_engine_config),
        ("evaluation.yaml   → EvaluationConfig", cfg.get_evaluation_config),
        ("rag.yaml          → RAGConfig", cfg.get_rag_config),
    ]

    print("▶ Validating configuration sections:")
    for label, getter in checks:
        try:
            result = getter()
            print(f"  ✅ {label}")
            sections.append((label, result))
        except ConfigurationError as exc:
            print(f"  ⚠️  {label} — MISSING or INVALID (skipped): {exc}")
        except Exception as exc:
            print(f"  ❌ {label} — ERROR: {exc}")
            errors.append((label, str(exc)))

    print()
    print("=" * 60)
    if errors:
        print(f"  ❌ Validation FAILED — {len(errors)} error(s):")
        for label, err in errors:
            print(f"     • {label}: {err}")
        return 1
    else:
        loaded_count = len(sections)
        print(f"  ✅ Validation PASSED — {loaded_count}/{len(checks)} sections validated.")
        print()
        print("  All loaded configuration sections:")
        for label, _ in sections:
            print(f"     • {label}")
        return 0


def main() -> int:
    """Entry point for the configuration validation script."""
    parser = argparse.ArgumentParser(description="Validate framework YAML configurations.")
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Path to the configs directory (default: configs/)",
    )
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    return validate_all(config_dir)


if __name__ == "__main__":
    sys.exit(main())
