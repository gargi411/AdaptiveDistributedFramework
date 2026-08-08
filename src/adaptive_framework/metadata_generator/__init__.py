"""Metadata Generator package.

Architecture v2.0 §2.4 — Metadata Generator.

Public API:
    MetadataExtractor  — Extracts PDFMetadata from PDF files using pypdf.
    MetadataStore      — Persists and loads PDFMetadata to CSV / JSON.

Usage::

    from adaptive_framework.metadata_generator import MetadataExtractor, MetadataStore
    from pathlib import Path

    extractor = MetadataExtractor()
    pdf_paths = list(Path("dataset/raw").glob("*.pdf"))
    metadata_list = extractor.extract_batch(pdf_paths)

    store = MetadataStore(output_dir=Path("dataset"))
    store.save_csv(metadata_list)
    stats = store.compute_statistics(metadata_list)
    print(stats)
"""

from adaptive_framework.metadata_generator.metadata_extractor import MetadataExtractor
from adaptive_framework.metadata_generator.metadata_store import MetadataStore

__all__ = [
    "MetadataExtractor",
    "MetadataStore",
]
