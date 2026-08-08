# document_processing/ocr

## Purpose

Placeholder package for the **OCR** sub-component of the Document Processing
Engine (Architecture v2.0, §2.1).

OCR (Optical Character Recognition) extracts raw text from PDF page images.
The backend is swappable via `IOCREngine`; the default candidate is PaddleOCR.

## Supported Backends (Phase 2 target)

| Backend | Notes |
|---------|-------|
| PaddleOCR | Current default candidate |
| TrOCR | Transformer-based, high accuracy |
| Nougat | Scientific PDF specialized |
| MinerU | Structured document extraction |
| Docling | IBM's document understanding |

## Architecture Role

```
Document Processing Engine
├── OCR               ← This package
├── Layout Analysis
├── Table Extraction
└── Figure Detection
```

## Implementation Target

**Phase 2** — Document Processing Engine

## Contents (Phase 2)

| Module | Description |
|--------|-------------|
| `paddleocr_engine.py` | Implements `IOCREngine` using PaddleOCR |
| `engine_factory.py` | Factory that creates the configured OCR backend |

## Dependencies (Phase 2)

- `adaptive_framework.interfaces.i_ocr_engine` (IOCREngine)
- `adaptive_framework.models.document` (PageMetadata, PageResult)
- `adaptive_framework.config.models` (OCRConfig)
- `adaptive_framework.core.constants` (SUPPORTED_OCR_BACKENDS)
- PaddleOCR (external, optional at runtime)
