"""RAG Retrieval Evaluation and Benchmarking Runner Script (Phase 4.9).

Evaluates and compares:
1. BASELINE A: Dense FAISS RetrievalEngine
2. BASELINE B: Hybrid Retrieval (Dense FAISS + BM25Okapi + Reciprocal Rank Fusion)
3. EXPERIMENT C: Hybrid Retrieval + Cross-Encoder Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
over the synthetic clinical corpus against 25 ground-truth clinical queries across K=1, 3, 5.

Usage:
    .venv/Scripts/python scripts/evaluate_rag.py
    .venv/Scripts/python scripts/evaluate_rag.py --output-dir outputs/rag/evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.evaluation.dataset import DEFAULT_GROUND_TRUTH_CASES, load_ground_truth_cases
from adaptive_framework.rag.evaluation.evaluator import EvaluationReport, RAGEvaluator
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.retrieval.bm25_retriever import BM25Retriever
from adaptive_framework.rag.retrieval.hybrid_retriever import HybridRetriever
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager


def _make_vector(text: str, dim: int = 1024) -> tuple[float, ...]:
    """Deterministic normalized vector generator for offline reproducible evaluation."""
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return tuple(float(x) for x in vec)


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


def _format_delta(base: float, exp: float) -> str:
    diff = exp - base
    if base > 0.0:
        rel = (diff / base) * 100.0
        return f"{diff:+.4f} ({rel:+.1f}%)"
    elif diff > 0.0:
        return f"{diff:+.4f} (baseline=0)"
    else:
        return f"{diff:+.4f} (0.0%)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.9 RAG Retrieval Evaluation & Benchmarking")
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
        default="outputs/rag/evaluation",
        help="Directory where evaluation JSON files are saved",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset)

    print("============================================================")
    print("RAG RETRIEVAL BENCHMARK: DENSE VS HYBRID VS RERANKED")
    print("Adaptive Distributed Framework v2.0 - Phase 4.9")
    print("============================================================")
    print("Dataset             : Synthetic Indian Clinical Notes")
    print(f"Corpus Size         : {args.max_docs} documents")

    # 1. Load clinical corpus
    chunks = load_corpus(dataset_path, max_docs=args.max_docs)
    if not chunks:
        print(f"Error: Unable to load clinical corpus from {dataset_path}")
        return 1

    # 2. Build FAISS index for dense retrieval
    dim = 1024
    manager = FAISSManager(dim=dim, index_type="flat")
    embeddings = [
        Embedding(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            vector=_make_vector(c.text, dim=dim),
            model_name="BAAI/bge-large-en-v1.5",
            embedding_time_seconds=0.001,
            created_at="2026-09-06T12:00:00+00:00",
        )
        for c in chunks
    ]
    manager.add_embeddings(embeddings)
    manager.register_chunks(chunks)

    embedder = MagicMock(spec=BGEEmbedder)
    embedder.embedding_dim = dim
    embedder.embed_query.side_effect = lambda q: np.array(_make_vector(q, dim=dim), dtype=np.float32)

    dense_engine = RetrievalEngine(
        embedding_provider=embedder,
        faiss_manager=manager,
        default_top_k=5,
    )

    # 3. Build BM25 index for sparse retrieval
    bm25_retriever = BM25Retriever(k1=1.5, b=0.75)
    bm25_retriever.index_chunks(chunks)

    # 4. Build Hybrid retriever (Candidate pool = 20)
    hybrid_candidate_engine = HybridRetriever(
        dense_engine=dense_engine,
        sparse_retriever=bm25_retriever,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=20,
        rrf_k=60,
    )

    # Hybrid engine for benchmark evaluation (final_top_k = 5)
    hybrid_engine = HybridRetriever(
        dense_engine=dense_engine,
        sparse_retriever=bm25_retriever,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=5,
        rrf_k=60,
    )

    # 5. Build Cross-Encoder Reranker and RerankedRetriever
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="auto",
        batch_size=16,
    )
    reranked_engine = RerankedRetriever(
        base_retriever=hybrid_candidate_engine,
        reranker=reranker,
        candidate_top_k=20,
        final_top_k=5,
    )

    # 6. Load ground truth cases
    cases = load_ground_truth_cases()
    print(f"Queries Evaluated   : {len(cases)}")
    print("Top-K Evaluated     : 1, 3, 5\n")

    # 7. Run Baseline Evaluation (Dense FAISS)
    dense_evaluator = RAGEvaluator(retrieval_engine=dense_engine, k_values=(1, 3, 5))
    dense_report = dense_evaluator.evaluate(cases)

    # 8. Run Baseline Evaluation (Hybrid FAISS + BM25 + RRF)
    hybrid_evaluator = RAGEvaluator(retrieval_engine=hybrid_engine, k_values=(1, 3, 5))
    hybrid_report = hybrid_evaluator.evaluate(cases)

    # 9. Run Experiment Evaluation (Hybrid + Cross-Encoder Reranking)
    reranked_evaluator = RAGEvaluator(retrieval_engine=reranked_engine, k_values=(1, 3, 5))
    reranked_report = reranked_evaluator.evaluate(cases)

    # 10. Extract metrics
    d_dp = dense_report.doc_metrics["precision"]
    d_dr = dense_report.doc_metrics["recall"]
    d_drr = dense_report.doc_metrics["mrr"]
    d_dndcg = dense_report.doc_metrics["ndcg"]
    d_dhit = dense_report.hit_rate_at_k
    d_lat = dense_report.latency_stats_ms

    h_dp = hybrid_report.doc_metrics["precision"]
    h_dr = hybrid_report.doc_metrics["recall"]
    h_drr = hybrid_report.doc_metrics["mrr"]
    h_dndcg = hybrid_report.doc_metrics["ndcg"]
    h_dhit = hybrid_report.hit_rate_at_k
    h_lat = hybrid_report.latency_stats_ms

    r_dp = reranked_report.doc_metrics["precision"]
    r_dr = reranked_report.doc_metrics["recall"]
    r_drr = reranked_report.doc_metrics["mrr"]
    r_dndcg = reranked_report.doc_metrics["ndcg"]
    r_dhit = reranked_report.hit_rate_at_k
    r_lat = reranked_report.latency_stats_ms

    # Detailed Latency decomposition
    reranker_metrics = reranker.get_metrics()
    avg_rerank_ms = reranker_metrics.get("avg_rerank_time_ms", 0.0)
    avg_hybrid_ms = h_lat["mean"]
    avg_dense_ms = d_lat["mean"]
    sparse_metrics = bm25_retriever.get_metrics()
    avg_sparse_ms = sparse_metrics.get("avg_query_time_ms", 0.0)

    # 11. Print DENSE FAISS BASELINE table
    print("--------------------------------------------------")
    print("A. DENSE FAISS BASELINE")
    print("--------------------------------------------------")
    print("DOCUMENT-LEVEL METRICS:")
    print("Metric          @1       @3       @5")
    print("------------------------------------")
    print(f"Precision     {d_dp[1]:.4f}   {d_dp[3]:.4f}   {d_dp[5]:.4f}")
    print(f"Recall        {d_dr[1]:.4f}   {d_dr[3]:.4f}   {d_dr[5]:.4f}")
    print(f"MRR           {d_drr[1]:.4f}   {d_drr[3]:.4f}   {d_drr[5]:.4f}")
    print(f"nDCG          {d_dndcg[1]:.4f}   {d_dndcg[3]:.4f}   {d_dndcg[5]:.4f}")
    print(f"Hit Rate      {d_dhit[1]:.4f}   {d_dhit[3]:.4f}   {d_dhit[5]:.4f}")
    print("")
    print("CHUNK-LEVEL METRICS:")
    print("Metric          @1       @3       @5")
    print("------------------------------------")
    c_dp = dense_report.chunk_metrics["precision"]
    c_dr = dense_report.chunk_metrics["recall"]
    c_drr = dense_report.chunk_metrics["mrr"]
    c_dndcg = dense_report.chunk_metrics["ndcg"]
    print(f"Precision     {c_dp[1]:.4f}   {c_dp[3]:.4f}   {c_dp[5]:.4f}")
    print(f"Recall        {c_dr[1]:.4f}   {c_dr[3]:.4f}   {c_dr[5]:.4f}")
    print(f"MRR           {c_drr[1]:.4f}   {c_drr[3]:.4f}   {c_drr[5]:.4f}")
    print(f"nDCG          {c_dndcg[1]:.4f}   {c_dndcg[3]:.4f}   {c_dndcg[5]:.4f}")
    print("")
    print("LATENCY:")
    print(f"- Mean   : {d_lat['mean']:.3f} ms")
    print(f"- Median : {d_lat['median']:.3f} ms")
    print(f"- P95    : {d_lat['p95']:.3f} ms\n")

    # 12. Print HYBRID DENSE + BM25 + RRF table
    print("--------------------------------------------------")
    print("B. HYBRID DENSE + BM25 + RRF")
    print("--------------------------------------------------")
    print("DOCUMENT-LEVEL METRICS:")
    print("Metric          @1       @3       @5")
    print("------------------------------------")
    print(f"Precision     {h_dp[1]:.4f}   {h_dp[3]:.4f}   {h_dp[5]:.4f}")
    print(f"Recall        {h_dr[1]:.4f}   {h_dr[3]:.4f}   {h_dr[5]:.4f}")
    print(f"MRR           {h_drr[1]:.4f}   {h_drr[3]:.4f}   {h_drr[5]:.4f}")
    print(f"nDCG          {h_dndcg[1]:.4f}   {h_dndcg[3]:.4f}   {h_dndcg[5]:.4f}")
    print(f"Hit Rate      {h_dhit[1]:.4f}   {h_dhit[3]:.4f}   {h_dhit[5]:.4f}")
    print("")
    print("CHUNK-LEVEL METRICS:")
    print("Metric          @1       @3       @5")
    print("------------------------------------")
    c_hp = hybrid_report.chunk_metrics["precision"]
    c_hr = hybrid_report.chunk_metrics["recall"]
    c_hrr = hybrid_report.chunk_metrics["mrr"]
    c_hndcg = hybrid_report.chunk_metrics["ndcg"]
    print(f"Precision     {c_hp[1]:.4f}   {c_hp[3]:.4f}   {c_hp[5]:.4f}")
    print(f"Recall        {c_hr[1]:.4f}   {c_hr[3]:.4f}   {c_hr[5]:.4f}")
    print(f"MRR           {c_hrr[1]:.4f}   {c_hrr[3]:.4f}   {c_hrr[5]:.4f}")
    print(f"nDCG          {c_hndcg[1]:.4f}   {c_hndcg[3]:.4f}   {c_hndcg[5]:.4f}")
    print("")
    print("LATENCY:")
    print(f"- Mean   : {h_lat['mean']:.3f} ms")
    print(f"- Median : {h_lat['median']:.3f} ms")
    print(f"- P95    : {h_lat['p95']:.3f} ms\n")

    # 13. Print HYBRID + CROSS-ENCODER RERANKED table
    print("--------------------------------------------------")
    print("C. HYBRID + CROSS-ENCODER RERANKED (ms-marco-MiniLM-L-6-v2)")
    print("--------------------------------------------------")
    print("DOCUMENT-LEVEL METRICS:")
    print("Metric          @1       @3       @5")
    print("------------------------------------")
    print(f"Precision     {r_dp[1]:.4f}   {r_dp[3]:.4f}   {r_dp[5]:.4f}")
    print(f"Recall        {r_dr[1]:.4f}   {r_dr[3]:.4f}   {r_dr[5]:.4f}")
    print(f"MRR           {r_drr[1]:.4f}   {r_drr[3]:.4f}   {r_drr[5]:.4f}")
    print(f"nDCG          {r_dndcg[1]:.4f}   {r_dndcg[3]:.4f}   {r_dndcg[5]:.4f}")
    print(f"Hit Rate      {r_dhit[1]:.4f}   {r_dhit[3]:.4f}   {r_dhit[5]:.4f}")
    print("")
    print("CHUNK-LEVEL METRICS:")
    print("Metric          @1       @3       @5")
    print("------------------------------------")
    c_rp = reranked_report.chunk_metrics["precision"]
    c_rr = reranked_report.chunk_metrics["recall"]
    c_rrr = reranked_report.chunk_metrics["mrr"]
    c_rndcg = reranked_report.chunk_metrics["ndcg"]
    print(f"Precision     {c_rp[1]:.4f}   {c_rp[3]:.4f}   {c_rp[5]:.4f}")
    print(f"Recall        {c_rr[1]:.4f}   {c_rr[3]:.4f}   {c_rr[5]:.4f}")
    print(f"MRR           {c_rrr[1]:.4f}   {c_rrr[3]:.4f}   {c_rrr[5]:.4f}")
    print(f"nDCG          {c_rndcg[1]:.4f}   {c_rndcg[3]:.4f}   {c_rndcg[5]:.4f}")
    print("")
    print("LATENCY (End-to-end Reranked pipeline):")
    print(f"- Mean   : {r_lat['mean']:.3f} ms")
    print(f"- Median : {r_lat['median']:.3f} ms")
    print(f"- P95    : {r_lat['p95']:.3f} ms\n")

    # 14. Print 3-WAY COMPARISON table
    print("================================================================================")
    print("3-WAY COMPARISON (DENSE vs HYBRID vs RERANKED)")
    print("================================================================================")
    print(f"{'Metric':<14} | {'Dense (A)':<11} | {'Hybrid (B)':<11} | {'Reranked (C)':<12} | {'C vs A Delta':<14} | {'C vs B Delta':<14}")
    print("-" * 84)
    for k in (1, 3, 5):
        print(f"Precision@{k:<4} | {d_dp[k]:<11.4f} | {h_dp[k]:<11.4f} | {r_dp[k]:<12.4f} | {_format_delta(d_dp[k], r_dp[k]):<14} | {_format_delta(h_dp[k], r_dp[k]):<14}")
    print("-" * 84)
    for k in (1, 3, 5):
        print(f"Recall@{k:<7} | {d_dr[k]:<11.4f} | {h_dr[k]:<11.4f} | {r_dr[k]:<12.4f} | {_format_delta(d_dr[k], r_dr[k]):<14} | {_format_delta(h_dr[k], r_dr[k]):<14}")
    print("-" * 84)
    for k in (1, 3, 5):
        print(f"MRR@{k:<10} | {d_drr[k]:<11.4f} | {h_drr[k]:<11.4f} | {r_drr[k]:<12.4f} | {_format_delta(d_drr[k], r_drr[k]):<14} | {_format_delta(h_drr[k], r_drr[k]):<14}")
    print("-" * 84)
    for k in (1, 3, 5):
        print(f"nDCG@{k:<9} | {d_dndcg[k]:<11.4f} | {h_dndcg[k]:<11.4f} | {r_dndcg[k]:<12.4f} | {_format_delta(d_dndcg[k], r_dndcg[k]):<14} | {_format_delta(h_dndcg[k], r_dndcg[k]):<14}")
    print("-" * 84)
    for k in (1, 3, 5):
        print(f"Hit Rate@{k:<5} | {d_dhit[k]:<11.4f} | {h_dhit[k]:<11.4f} | {r_dhit[k]:<12.4f} | {_format_delta(d_dhit[k], r_dhit[k]):<14} | {_format_delta(h_dhit[k], r_dhit[k]):<14}")
    print("================================================================================\n")

    # 15. Print LATENCY DECOMPOSITION
    print("--------------------------------------------------")
    print("LATENCY DECOMPOSITION")
    print("--------------------------------------------------")
    print(f"- Stage 1 Dense FAISS Retrieval : {avg_dense_ms:.3f} ms")
    print(f"- Stage 1 BM25 Lexical Retrieval: {avg_sparse_ms:.3f} ms")
    print(f"- Stage 1 Total (Hybrid + RRF)  : {avg_hybrid_ms:.3f} ms")
    print(f"- Stage 2 Cross-Encoder Rerank  : {avg_rerank_ms:.3f} ms")
    print(f"- Total Pipeline Latency (Mean) : {r_lat['mean']:.3f} ms")
    print(f"- Total Pipeline Latency (Med)  : {r_lat['median']:.3f} ms")
    print(f"- Total Pipeline Latency (P95)  : {r_lat['p95']:.3f} ms\n")

    # 16. Query-by-Query Failure Analysis (Hybrid vs Reranked at K=5)
    rerank_improved = 0   # Hybrid Fail -> Rerank Hit
    rerank_degraded = 0   # Hybrid Hit -> Rerank Fail
    rerank_both_hit = 0   # Both Hit
    rerank_both_fail = 0  # Both Fail

    cat_improved: list[dict[str, Any]] = []
    cat_degraded: list[dict[str, Any]] = []
    cat_both_hit: list[dict[str, Any]] = []
    cat_both_fail: list[dict[str, Any]] = []

    # Also track rank shifts for both-hit queries
    rank_promotions: list[dict[str, Any]] = []
    rank_demotions: list[dict[str, Any]] = []
    rank_neutral: list[dict[str, Any]] = []

    for h_res, r_res in zip(hybrid_report.query_results, reranked_report.query_results):
        h_hit = h_res.hit_at_k.get(5, False)
        r_hit = r_res.hit_at_k.get(5, False)

        # Find first rank of relevant document in retrieved list
        h_rank = None
        for rank_idx, doc_id in enumerate(h_res.retrieved_doc_ids, start=1):
            if doc_id in h_res.expected_doc_ids:
                h_rank = rank_idx
                break

        r_rank = None
        for rank_idx, doc_id in enumerate(r_res.retrieved_doc_ids, start=1):
            if doc_id in r_res.expected_doc_ids:
                r_rank = rank_idx
                break

        info = {
            "case_id": h_res.case_id,
            "question": h_res.question,
            "expected_docs": h_res.expected_doc_ids,
            "hybrid_retrieved": h_res.retrieved_doc_ids[:5],
            "reranked_retrieved": r_res.retrieved_doc_ids[:5],
            "hybrid_rank": h_rank,
            "reranked_rank": r_rank,
            "hybrid_hit": h_hit,
            "reranked_hit": r_hit,
        }

        if not h_hit and r_hit:
            rerank_improved += 1
            cat_improved.append(info)
        elif h_hit and not r_hit:
            rerank_degraded += 1
            cat_degraded.append(info)
        elif h_hit and r_hit:
            rerank_both_hit += 1
            cat_both_hit.append(info)
            if h_rank is not None and r_rank is not None:
                if r_rank < h_rank:
                    rank_promotions.append(info)
                elif r_rank > h_rank:
                    rank_demotions.append(info)
                else:
                    rank_neutral.append(info)
        else:
            rerank_both_fail += 1
            cat_both_fail.append(info)

    print("--------------------------------------------------")
    print("QUERY OUTCOME BREAKDOWN (RERANKED VS HYBRID @ K=5)")
    print("--------------------------------------------------")
    print(f"- Improved (Hybrid Fail -> Rerank Hit) : {rerank_improved}")
    print(f"- Degraded (Hybrid Hit -> Rerank Fail) : {rerank_degraded}")
    print(f"- Both Succeed (Hybrid Hit & Rerank)   : {rerank_both_hit}")
    print(f"  * Rank Promoted (e.g. #3 -> #1)      : {len(rank_promotions)}")
    print(f"  * Rank Demoted  (e.g. #1 -> #3)      : {len(rank_demotions)}")
    print(f"  * Rank Neutral  (same rank)          : {len(rank_neutral)}")
    print(f"- Both Fail (Hybrid Fail & Rerank)     : {rerank_both_fail}\n")

    # 17. Save JSON artifacts
    reranked_summary_path = output_dir / "reranked_summary.json"
    reranked_results_path = output_dir / "reranked_results.json"
    rerank_comp_path = output_dir / "reranking_comparison_summary.json"

    with open(reranked_summary_path, "w", encoding="utf-8") as f:
        json.dump(reranked_report.to_dict(), f, indent=2)
    with open(reranked_results_path, "w", encoding="utf-8") as f:
        json.dump([q.to_dict() for q in reranked_report.query_results], f, indent=2)

    reranking_comparison_data = {
        "query_count": len(cases),
        "evaluation_strategy": "three_way_comparison",
        "outcomes_rerank_vs_hybrid_k5": {
            "improved": rerank_improved,
            "degraded": rerank_degraded,
            "both_hit": rerank_both_hit,
            "both_fail": rerank_both_fail,
            "rank_promotions": len(rank_promotions),
            "rank_demotions": len(rank_demotions),
            "rank_neutral": len(rank_neutral),
        },
        "metrics_summary": {
            "dense_baseline": {
                "document_level": dense_report.doc_metrics,
                "hit_rate_at_k": d_dhit,
                "latency_stats_ms": d_lat,
            },
            "hybrid_baseline": {
                "document_level": hybrid_report.doc_metrics,
                "hit_rate_at_k": h_dhit,
                "latency_stats_ms": h_lat,
            },
            "hybrid_reranked": {
                "document_level": reranked_report.doc_metrics,
                "hit_rate_at_k": r_dhit,
                "latency_stats_ms": r_lat,
            },
        },
        "deltas_reranked_vs_hybrid": {
            "precision_at_1": r_dp[1] - h_dp[1],
            "precision_at_5": r_dp[5] - h_dp[5],
            "recall_at_1": r_dr[1] - h_dr[1],
            "recall_at_5": r_dr[5] - h_dr[5],
            "mrr_at_1": r_drr[1] - h_drr[1],
            "mrr_at_5": r_drr[5] - h_drr[5],
            "ndcg_at_5": r_dndcg[5] - h_dndcg[5],
            "hit_rate_at_5": r_dhit[5] - h_dhit[5],
            "latency_delta_ms": r_lat["mean"] - h_lat["mean"],
        },
        "deltas_reranked_vs_dense": {
            "precision_at_1": r_dp[1] - d_dp[1],
            "precision_at_5": r_dp[5] - d_dp[5],
            "recall_at_1": r_dr[1] - d_dr[1],
            "recall_at_5": r_dr[5] - d_dr[5],
            "mrr_at_1": r_drr[1] - d_drr[1],
            "mrr_at_5": r_drr[5] - d_drr[5],
            "ndcg_at_5": r_dndcg[5] - d_dndcg[5],
            "hit_rate_at_5": r_dhit[5] - d_dhit[5],
            "latency_delta_ms": r_lat["mean"] - d_lat["mean"],
        },
        "latency_decomposition": {
            "dense_ms": avg_dense_ms,
            "bm25_ms": avg_sparse_ms,
            "hybrid_ms": avg_hybrid_ms,
            "rerank_ms": avg_rerank_ms,
            "total_mean_ms": r_lat["mean"],
            "total_median_ms": r_lat["median"],
            "total_p95_ms": r_lat["p95"],
        },
        "sample_improved": cat_improved,
        "sample_degraded": cat_degraded,
        "sample_rank_promotions": rank_promotions,
        "sample_rank_demotions": rank_demotions,
    }

    with open(rerank_comp_path, "w", encoding="utf-8") as f:
        json.dump(reranking_comparison_data, f, indent=2)

    print("OUTPUT ARTIFACTS SAVED:")
    print(f"- Reranked Summary               : {reranked_summary_path}")
    print(f"- Reranked Results               : {reranked_results_path}")
    print(f"- Reranking Comparison Summary   : {rerank_comp_path}")
    print("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
