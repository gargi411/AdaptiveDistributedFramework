# Dataset Builder Package

## Purpose

Scans a directory of PDF files and produces a list of `PDFMetadata` records.

## Status: Phase 2 Placeholder

## Interface

`interfaces/i_dataset_builder.py` → `IDatasetBuilder`

## Metadata Extracted (Architecture §2.4)

| Field | Required | Source |
|-------|----------|--------|
| `pages` | ✅ | PDF page count |
| `estimated_size_mb` | ✅ | File size |
| `resolution_dpi` | Optional | PDF metadata or image analysis |
| `source_type` | Optional | Text layer detection |
| `language` | Optional | Language detection |