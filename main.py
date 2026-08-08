"""main.py — Framework entry point and composition root.

Responsibilities (Phase 1):
    1. Load configuration from configs/ directory.
    2. Initialize the centralized FrameworkLogger.
    3. Register infrastructure components in the DI container.
    4. Validate the runtime environment.
    5. Print "Framework Initialized Successfully".

This file is the ONLY place where:
    - ConfigManager is called directly.
    - FrameworkLogger is constructed.
    - DIContainer is populated.

All other components receive their dependencies via constructor injection.

Phase 2+ will add: Ray cluster init, Dataset Builder, Scheduler, Coordinator.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Initialize the Adaptive Distributed Framework.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    # ------------------------------------------------------------------ #
    # Step 1: Load Configuration                                           #
    # ------------------------------------------------------------------ #
    from adaptive_framework.config import ConfigManager
    from adaptive_framework.core.exceptions import ConfigurationError

    config_dir = Path("configs")

    try:
        cfg = ConfigManager.get_instance()
        cfg.load(config_dir=config_dir)
    except ConfigurationError as exc:
        print(f"[FATAL] Configuration error: {exc}", file=sys.stderr)
        return 1

    framework_cfg = cfg.get_framework_config()
    logging_cfg = cfg.get_logging_config()

    # ------------------------------------------------------------------ #
    # Step 2: Initialize Logger                                            #
    # ------------------------------------------------------------------ #
    from adaptive_framework.logging import FrameworkLogger
    from adaptive_framework.utils import now_utc_iso, get_hostname
    import uuid

    run_id = f"{framework_cfg.run_id_prefix}_{now_utc_iso()[:19].replace(':', '-')}"
    output_dir = Path(framework_cfg.output_dir)

    logger = FrameworkLogger.from_config(
        log_cfg=logging_cfg,
        output_dir=output_dir,
        run_id=run_id,
        worker_id="none",
        node_id=get_hostname(),
    )

    logger.info(
        "Logger initialized.",
        run_id=run_id,
        framework_name=framework_cfg.name,
        framework_version=framework_cfg.version,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Register components in the DI Container                     #
    # ------------------------------------------------------------------ #
    from adaptive_framework.di import DIContainer
    from adaptive_framework.interfaces import ILogger, IConfigProvider

    container = DIContainer()
    container.register(ILogger, logger)
    container.register(IConfigProvider, cfg)  # type: ignore[arg-type]

    logger.info(
        "DI container initialized.",
        registered=container.registered_interfaces(),
    )

    # ------------------------------------------------------------------ #
    # Step 4: Validate Environment                                         #
    # ------------------------------------------------------------------ #
    from adaptive_framework.utils import get_platform_info, is_psutil_available

    platform_info = get_platform_info()
    logger.info(
        "Environment validated.",
        os=platform_info["os"],
        python_version=platform_info["python_version"],
        hostname=platform_info["hostname"],
        psutil_available=is_psutil_available(),
    )

    if framework_cfg.debug:
        logger.debug(
            "Debug mode active.",
            platform=platform_info,
        )

    # ------------------------------------------------------------------ #
    # Step 5: Framework Ready                                              #
    # ------------------------------------------------------------------ #
    logger.info(
        "Framework Initialized Successfully.",
        run_id=run_id,
        config_dir=str(config_dir.resolve()),
        output_dir=str(output_dir.resolve()),
    )

    print()
    print("=" * 60)
    print(f"  {framework_cfg.name}")
    print(f"  Version : {framework_cfg.version}")
    print(f"  Run ID  : {run_id}")
    print(f"  Node    : {platform_info['hostname']}")
    print()
    print("  ✅ Framework Initialized Successfully")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
