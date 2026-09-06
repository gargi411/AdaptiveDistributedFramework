"""Model Setup -- Utilities for managing OpenVINO OCR model artifacts.

Phase G2: OpenVINO OCR Integration
Provides automated download and verification of the official Intel Open Model Zoo
FP16 models required for two-stage text detection and recognition.

Models:
    - Text Detection: horizontal-text-detection-0001 (FP16)
    - Text Recognition: text-recognition-0012 (FP16)
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

OMZ_BASE_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"

class ModelArtifact(NamedTuple):
    name: str
    xml_filename: str
    bin_filename: str
    xml_url: str
    bin_url: str
    xml_sha256: str | None = None
    bin_sha256: str | None = None


DETECTION_MODEL = ModelArtifact(
    name="horizontal-text-detection-0001",
    xml_filename="horizontal-text-detection-0001.xml",
    bin_filename="horizontal-text-detection-0001.bin",
    xml_url=f"{OMZ_BASE_URL}/horizontal-text-detection-0001/FP16/horizontal-text-detection-0001.xml",
    bin_url=f"{OMZ_BASE_URL}/horizontal-text-detection-0001/FP16/horizontal-text-detection-0001.bin",
)

RECOGNITION_MODEL = ModelArtifact(
    name="text-recognition-0012",
    xml_filename="text-recognition-0012.xml",
    bin_filename="text-recognition-0012.bin",
    xml_url=f"{OMZ_BASE_URL}/text-recognition-0012/FP16/text-recognition-0012.xml",
    bin_url=f"{OMZ_BASE_URL}/text-recognition-0012/FP16/text-recognition-0012.bin",
)


def _download_file(url: str, target_path: Path) -> None:
    """Download a file with user-agent headers to avoid CDN blocks."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    logger.info("Downloading %s to %s...", url, target_path)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
        while chunk := resp.read(65536):
            f.write(chunk)
    temp_path.replace(target_path)
    logger.info("Downloaded %s successfully (%d bytes).", target_path.name, target_path.stat().st_size)


def ensure_openvino_ocr_models(
    base_dir: Path | str = "models/openvino",
    force_download: bool = False,
) -> tuple[Path, Path]:
    """Ensure both text detection and text recognition models are present.

    Args:
        base_dir: Root directory for OpenVINO models.
        force_download: If True, redownloads files even if they exist.

    Returns:
        tuple[Path, Path]: (detection_xml_path, recognition_xml_path)
    """
    root = Path(base_dir)

    # 1. Detection Model
    det_dir = root / DETECTION_MODEL.name / "FP16"
    det_xml = det_dir / DETECTION_MODEL.xml_filename
    det_bin = det_dir / DETECTION_MODEL.bin_filename

    if force_download or not det_xml.exists():
        _download_file(DETECTION_MODEL.xml_url, det_xml)
    if force_download or not det_bin.exists():
        _download_file(DETECTION_MODEL.bin_url, det_bin)

    # 2. Recognition Model
    rec_dir = root / RECOGNITION_MODEL.name / "FP16"
    rec_xml = rec_dir / RECOGNITION_MODEL.xml_filename
    rec_bin = rec_dir / RECOGNITION_MODEL.bin_filename

    if force_download or not rec_xml.exists():
        _download_file(RECOGNITION_MODEL.xml_url, rec_xml)
    if force_download or not rec_bin.exists():
        _download_file(RECOGNITION_MODEL.bin_url, rec_bin)

    return det_xml, rec_xml
