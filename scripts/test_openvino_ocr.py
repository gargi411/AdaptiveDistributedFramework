#!/usr/bin/env python3
"""Diagnostic script for OpenVINO OCR GPU Execution Verification.

Fulfills Phase G2 specification:
1. Detects available OpenVINO devices.
2. Loads detection and recognition models.
3. Compiles models on GPU (or specified device).
4. Renders real PDF page through PyMuPDF.
5. Executes OpenVINO GPU OCR inference.
6. Measures cold-start vs warm inference latency and throughput.
7. Produces a structured report with SUCCESS / FAILURE.

Usage:
    python scripts/test_openvino_ocr.py [--device GPU|CPU|AUTO] [--pdf path/to/document.pdf] [--pages N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from adaptive_framework.acceleration.hardware_probe import HardwareCapabilityProbe
from adaptive_framework.acceleration.model_setup import ensure_openvino_ocr_models
from adaptive_framework.acceleration.openvino_ocr_strategy import OpenVINOOCRStrategy


def find_test_pdf() -> Path:
    """Find a non-sensitive test PDF in the repository."""
    candidates = [
        _PROJECT_ROOT / "dataset" / "raw" / "pmc_pdfs" / "13643_2019_Article_976.pdf",
        _PROJECT_ROOT / "dataset" / "PDF_Original" / "PDF_Deid_Deidentification_0.pdf",
        _PROJECT_ROOT / "dataset" / "PDF_Original" / "Medium" / "PDF_Deid_Deidentification_Medium_0.pdf",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Search for any PDF in dataset/
    found = list((_PROJECT_ROOT / "dataset").glob("**/*.pdf"))
    if found:
        return found[0]

    # Generate small synthetic test PDF if none exist
    synthetic_path = _PROJECT_ROOT / "outputs" / "test_ocr_synthetic.pdf"
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "CLINICAL DISCHARGE SUMMARY", fontsize=18)
    page.insert_text((50, 140), "Patient: Doe, John. Age: 58. Diagnosis: Acute Coronary Syndrome.", fontsize=12)
    page.insert_text((50, 170), "Medications: Aspirin 81mg oral daily, Atorvastatin 80mg oral daily.", fontsize=12)
    page.insert_text((50, 200), "Discharge Plan: Cardiology follow-up in 2 weeks. Low sodium diet.", fontsize=12)
    doc.save(str(synthetic_path))
    doc.close()
    return synthetic_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenVINO OCR inference on Intel GPU/CPU.")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU", "AUTO"], help="Requested device (default: GPU)")
    parser.add_argument("--pdf", default=None, help="Path to test PDF file")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to test (default: 1)")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterisation DPI (default: 150)")
    args = parser.parse_args()

    print("==================================================")
    print("OPENVINO OCR GPU TEST")
    print("==================================================")

    # 1. Hardware and OpenVINO probe
    probe = HardwareCapabilityProbe()
    profile = probe.probe()
    ov_info = profile.openvino

    print()
    print("Available devices:")
    if ov_info.installed:
        for dev in ov_info.available_devices:
            print(f"  {dev}")
    else:
        print("  None (OpenVINO not installed)")

    print()
    print(f"Requested device:\n  {args.device}")

    if not ov_info.installed:
        print("\nStatus:\n  FAILURE - OpenVINO is not installed.")
        print("==================================================")
        sys.exit(1)

    # 2. Check and ensure models
    try:
        det_xml, rec_xml = ensure_openvino_ocr_models()
    except Exception as exc:
        print(f"\nStatus:\n  FAILURE - Could not obtain OCR models: {exc}")
        print("==================================================")
        sys.exit(1)

    print()
    print("OCR model:")
    print(f"  Detection:   {det_xml.name} (FP16)")
    print(f"  Recognition: {rec_xml.name} (FP16)")

    # 3. Strategy Initialization (Measuring Cold Start)
    t_cold_start = time.perf_counter()
    strategy = OpenVINOOCRStrategy(
        device=args.device,
        ocr_dpi=args.dpi,
        probe=probe,
        det_model_path=det_xml,
        rec_model_path=rec_xml,
    )
    init_success = strategy.initialize()
    t_cold_compile = (time.perf_counter() - t_cold_start) * 1000.0

    if not init_success:
        print(f"\nStatus:\n  FAILURE - Failed to compile model for requested device '{args.device}'.")
        print("==================================================")
        sys.exit(1)

    dev_info = strategy.get_device_info()
    compiled_dev = strategy.compiled_device

    print()
    print(f"Compiled device:\n  {compiled_dev}")
    if dev_info.get("full_device_name"):
        print(f"  Full Name:   {dev_info['full_device_name']}")

    # 4. Render real PDF page
    pdf_path = Path(args.pdf) if args.pdf else find_test_pdf()
    print()
    print(f"Input Document:\n  {pdf_path.name}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"\nStatus:\n  FAILURE - Could not open PDF '{pdf_path}': {exc}")
        print("==================================================")
        sys.exit(1)

    num_pages_to_test = min(args.pages, len(doc))
    print(f"  Pages to process: {num_pages_to_test} of {len(doc)}")

    # 5. Execute OCR
    warm_latencies: list[float] = []
    total_chars = 0
    total_blocks = 0
    sample_text = ""
    img_dims = ""

    for p_idx in range(num_pages_to_test):
        fitz_page = doc[p_idx]
        t_infer_start = time.perf_counter()
        result = strategy.process(
            page=fitz_page,
            page_number=p_idx + 1,
            document_id=pdf_path.stem,
            file_path=str(pdf_path),
        )
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        warm_latencies.append(t_infer_ms)
        total_chars += len(result.text)
        total_blocks += len(result.text_blocks)

        if p_idx == 0:
            scale = args.dpi / 72.0
            img_dims = f"{int(fitz_page.rect.width * scale)} x {int(fitz_page.rect.height * scale)}"
            sample_text = result.text

    doc.close()

    print(f"  Dimensions:       {img_dims}")

    print()
    print("OCR output preview (first page):")
    lines = [line.strip() for line in sample_text.splitlines() if line.strip()]
    if lines:
        for line in lines[:8]:
            print(f"  {line}")
        if len(lines) > 8:
            print(f"  ... [{len(lines) - 8} more lines]")
    else:
        print("  [No text recognized or blank page]")

    print()
    print("Inference metrics:")
    print(f"  Cold-start (model compile):   {t_cold_compile:.2f} ms")
    if warm_latencies:
        first_infer = warm_latencies[0]
        warm_mean = float(np.mean(warm_latencies))
        throughput = 1000.0 / warm_mean if warm_mean > 0 else 0.0
        print(f"  First page inference:         {first_infer:.2f} ms")
        print(f"  Warm mean latency:            {warm_mean:.2f} ms")
        print(f"  Total text blocks extracted:  {total_blocks}")
        print(f"  Total characters extracted:   {total_chars}")
        print(f"  OCR throughput:               {throughput:.2f} pages/sec")

    # 6. Verification Assessment
    is_gpu_test = args.device == "GPU"
    is_compiled_gpu = "GPU" in compiled_dev.upper()

    print()
    if is_gpu_test:
        if is_compiled_gpu and len(sample_text) > 0:
            status_msg = f"SUCCESS - OCR executed through OpenVINO GPU ({compiled_dev})"
        else:
            status_msg = f"FAILURE - Expected GPU execution, got {compiled_dev} with {len(sample_text)} characters."
    else:
        if len(sample_text) > 0:
            status_msg = f"SUCCESS - OCR executed through OpenVINO ({compiled_dev})"
        else:
            status_msg = f"FAILURE - No OCR text produced on {compiled_dev}."

    print("Status:")
    print(f"  {status_msg}")
    print("==================================================")

    if "FAILURE" in status_msg:
        sys.exit(1)


if __name__ == "__main__":
    main()
