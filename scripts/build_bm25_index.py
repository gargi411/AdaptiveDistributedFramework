"""Build BM25 Index Runner Script (Phase 4.8).

Builds and persists a BM25Okapi lexical index over the exact same 50 synthetic
clinical discharge summaries used by FAISS.

Usage:
    .venv/Scripts/python scripts/build_bm25_index.py
    .venv/Scripts/python scripts/build_bm25_index.py --output-dir outputs/rag/index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.retrieval.bm25_retriever import (
    DEFAULT_B,
    DEFAULT_K1,
    BM25Retriever,
)


def _build_chunk(
    chunk_id: str,
    document_id: str,
    source_file: str,
    text: str,
    section_heading: str | None,
    document_type: str | None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_file=source_file,
        chunk_index=0,
        page_numbers=(1,),
        text=text,
        section_heading=section_heading,
        document_type=document_type,
        character_count=len(text),
        word_count=len(text.split()),
        start_char_in_page=0,
        end_char_in_page=len(text),
    )


def load_corpus(dataset_path: Path, max_docs: int = 50) -> list[Chunk]:
    """Load evaluation corpus from synthetic clinical notes dataset."""
    raw_notes: list[dict[str, Any]] = []
    if dataset_path.exists():
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_notes = json.load(f)[:max_docs]
        except Exception:
            pass

    chunks: list[Chunk] = []
    for i, item in enumerate(raw_notes):
        note_id = item.get("id", i + 1)
        text = item.get("note", "")[:1000]
        chunks.append(
            _build_chunk(
                chunk_id=f"chk_synth_{note_id:04d}",
                document_id=f"clinical_note_{note_id:04d}",
                source_file="synthetic_notes_medium.json",
                text=text,
                section_heading=f"Specialty: {item.get('specialty', 'general')}",
                document_type="discharge_summary",
            )
        )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BM25 Index for Clinical Notes")
    parser.add_argument(
        "--dataset",
        default="dataset/Synthetic Indian Clinical Notes for Natural Langua/synthetic_notes_medium.json",
        help="Path to synthetic clinical notes dataset",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50,
        help="Number of documents to index from corpus (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rag/index",
        help="Directory where BM25 index and metadata are saved",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=DEFAULT_K1,
        help=f"BM25 k1 parameter (default: {DEFAULT_K1})",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=DEFAULT_B,
        help=f"BM25 b parameter (default: {DEFAULT_B})",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    print("============================================================")
    print("BUILD BM25 INDEX")
    print("Adaptive Distributed Framework v2.0 - Phase 4.8")
    print("============================================================")

    chunks = load_corpus(dataset_path, max_docs=args.max_docs)
    if not chunks:
        print(f"Error: Unable to load clinical corpus from {dataset_path}")
        return 1

    retriever = BM25Retriever(k1=args.k1, b=args.b)
    retriever.index_chunks(chunks)

    index_file = output_dir / "bm25_index.json"
    retriever.save(index_file)

    metrics = retriever.get_metrics()
    doc_ids = {c.document_id for c in chunks}

    print(f"Chunks Indexed     : {len(chunks)}")
    print(f"Documents Indexed  : {len(doc_ids)}")
    print(f"Vocabulary Size    : {metrics['vocab_size']} unique terms")
    print(f"Average Doc Length : {metrics['avgdl']} tokens")
    print(f"Tokenizer          : Deterministic alphanumeric (case-folded)")
    print(f"Parameters         : k1 = {args.k1}, b = {args.b}")
    print(f"Index Location     : {index_file}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
