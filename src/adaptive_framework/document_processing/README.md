# Document Processing Engine Package

## Architecture (v2.0 §2.1)

```
Document Processing Engine
├── OCR                 ← ocr/
├── Layout Analysis     ← layout_analysis/
├── Table Extraction    ← table_extraction/
└── Figure Detection    ← figure_detection/
```

## Status: Phase 2 Placeholder

This package is structurally complete. Implementations are added in Phase 2.

## Interface

The public contract is defined in:
- `interfaces/i_document_processor.py` → `IDocumentProcessor`
- `interfaces/i_ocr_engine.py` → `IOCREngine`

## Backend Swap Table

| Backend | Class | Phase |
|---------|-------|-------|
| PaddleOCR | `PaddleOCREngine` | 2 |
| TrOCR | `TrOCREngine` | future |
| Nougat | `NougatEngine` | future |
| MinerU | `MinerUEngine` | future |
| Docling | `DoclingEngine` | future |