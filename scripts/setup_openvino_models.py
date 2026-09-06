#!/usr/bin/env python3
"""CLI script to download and prepare OpenVINO OCR models.

Usage:
    python scripts/setup_openvino_models.py [--target-dir models/openvino] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from adaptive_framework.acceleration.model_setup import ensure_openvino_ocr_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup OpenVINO OCR models from Open Model Zoo.")
    parser.add_argument("--target-dir", default="models/openvino", help="Directory to save models (default: models/openvino)")
    parser.add_argument("--force", action="store_true", help="Force redownload even if files exist")
    args = parser.parse_args()

    print("==================================================")
    print("SETTING UP OPENVINO OCR MODELS")
    print("==================================================")
    print(f"Target directory: {args.target_dir}")
    print(f"Force download:   {args.force}")
    print()

    det_xml, rec_xml = ensure_openvino_ocr_models(base_dir=args.target_dir, force_download=args.force)

    print("Model Setup Complete:")
    print(f"  Detection Model:   {det_xml}")
    print(f"  Recognition Model: {rec_xml}")
    print("==================================================")


if __name__ == "__main__":
    main()
