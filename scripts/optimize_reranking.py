"""Phase 4.10: Adaptive Reranking Optimization & Latency Reduction Runner Script.

Executes controlled empirical benchmarking across:
1. Experiment 1 (Primary): Fixed candidate depths (candidate_top_k in {5, 10, 15, 20})
   - Evaluates Candidate Recall@K
   - Evaluates final retrieval metrics across K=1, 3, 5
   - Measures warm latency across 3 repeated runs with explicit cold-start separation
2. Experiment 2: Batch size optimization (batch_size in {4, 8, 16, 32}) on optimal candidate depth
3. Adaptive Candidate-Depth Policy evaluation vs Fixed configurations
4. Pareto efficiency analysis and optimal configuration selection

Usage:
    .venv/Scripts/python scripts/optimize_reranking.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from adaptive_framework.models.chunk import Chunk
from adaptive_framework.rag.embedding.bge_embedder import BGEEmbedder
from adaptive_framework.rag.evaluation.dataset import load_ground_truth_cases
from adaptive_framework.rag.evaluation.evaluator import (
    EvaluationReport,
    QueryEvaluationResult,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from adaptive_framework.rag.models.embedding import Embedding
from adaptive_framework.rag.reranking.adaptive_policy import AdaptiveCandidateDepthPolicy
from adaptive_framework.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from adaptive_framework.rag.retrieval.bm25_retriever import BM25Retriever
from adaptive_framework.rag.retrieval.hybrid_retriever import HybridRetriever
from adaptive_framework.rag.retrieval.reranked_retriever import RerankedRetriever
from adaptive_framework.rag.retrieval.retrieval_engine import RetrievalEngine
from adaptive_framework.rag.vector_store.faiss_manager import FAISSManager
from scripts.evaluate_rag import load_corpus, _make_vector


def _compute_metrics(
    query_results: list[dict[str, Any]],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    """Compute doc-level metrics and hit rates from a list of query result dicts."""
    n = len(query_results)
    doc_metrics: dict[str, dict[int, float]] = {
        "precision": {},
        "recall": {},
        "mrr": {},
        "ndcg": {},
    }
    hit_rate_at_k: dict[int, float] = {}

    for k in k_values:
        precisions: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        hits: list[bool] = []

        for q in query_results:
            retrieved = q["retrieved_doc_ids"]
            expected = q["expected_doc_ids"]

            p = precision_at_k(retrieved, expected, k=k)
            r = recall_at_k(retrieved, expected, k=k)
            rr = reciprocal_rank(retrieved, expected, k=k)
            nd = ndcg_at_k(retrieved, expected, k=k)
            hit = bool(set(retrieved[:k]) & set(expected))

            precisions.append(p)
            recalls.append(r)
            mrrs.append(rr)
            ndcgs.append(nd)
            hits.append(hit)

        doc_metrics["precision"][k] = sum(precisions) / n if n > 0 else 0.0
        doc_metrics["recall"][k] = sum(recalls) / n if n > 0 else 0.0
        doc_metrics["mrr"][k] = sum(mrrs) / n if n > 0 else 0.0
        doc_metrics["ndcg"][k] = sum(ndcgs) / n if n > 0 else 0.0
        hit_rate_at_k[k] = sum(hits) / n if n > 0 else 0.0

    return doc_metrics, hit_rate_at_k


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.10 Reranking Latency Optimization Benchmark")
    parser.add_argument(
        "--dataset",
        default="dataset/Synthetic Indian Clinical Notes for Natural Langua/synthetic_notes_medium.json",
        help="Path to synthetic clinical notes dataset",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50,
        help="Number of documents to index (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rag/evaluation",
        help="Directory to save evaluation JSON artifacts",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of repeated warm timing runs per configuration (default: 3)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset)

    print("================================================================================")
    print("PHASE 4.10 -- ADAPTIVE RERANKING OPTIMIZATION & LATENCY BENCHMARK")
    print("Adaptive Distributed Framework v2.0")
    print("================================================================================")
    print(f"Dataset             : Synthetic Indian Clinical Notes")
    print(f"Corpus Size         : {args.max_docs} documents")
    print(f"Repeated Runs       : {args.num_runs} warm passes per configuration")

    # 1. Load corpus and ground truth
    chunks = load_corpus(dataset_path, max_docs=args.max_docs)
    if not chunks:
        print(f"Error: Unable to load corpus from {dataset_path}")
        return 1

    cases = load_ground_truth_cases()
    print(f"Evaluation Queries  : {len(cases)}")
    print(f"Evaluation Top-K    : 1, 3, 5\n")

    # 2. Build Dense Index + Engine
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
        default_top_k=20,
    )

    # 3. Build BM25 Index
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index_chunks(chunks)

    # 4. Build Hybrid Engine for Candidate Generation (pool = 20)
    hybrid_candidate_engine = HybridRetriever(
        dense_engine=dense_engine,
        sparse_retriever=bm25,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=20,
        rrf_k=60,
    )

    # Pre-fetch candidate pool for all queries to ensure 100% identical candidate sets
    print("Pre-fetching Stage-1 Hybrid Top-20 candidate pool for all queries...")
    stage1_latencies_ms: list[float] = []
    query_candidates: dict[str, list[Any]] = {}

    for case in cases:
        t0 = time.perf_counter()
        cands = hybrid_candidate_engine.retrieve(case.question, top_k=20)
        elapsed = (time.perf_counter() - t0) * 1000.0
        stage1_latencies_ms.append(elapsed)
        query_candidates[case.case_id] = cands

    mean_stage1_ms = sum(stage1_latencies_ms) / len(stage1_latencies_ms)
    print(f"Stage-1 Hybrid candidate retrieval mean latency: {mean_stage1_ms:.3f} ms\n")

    # 5. Initialize Cross-Encoder & Execute Explicit Warm-Up
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="auto",
        batch_size=16,
    )
    print("Warming up cross-encoder model (cold-start initialization)...")
    cold_start_ms = reranker.warmup()
    print(f"Cold-start model load time: {cold_start_ms:.2f} ms (excluded from warm metrics)\n")

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: FIXED CANDIDATE DEPTH (candidate_top_k in {5, 10, 15, 20})
    # -------------------------------------------------------------------------
    print("================================================================================")
    print("EXPERIMENT 1: FIXED CANDIDATE DEPTH BENCHMARK (K_cand in {5, 10, 15, 20})")
    print("================================================================================")

    candidate_depths = [5, 10, 15, 20]
    depth_results: dict[int, dict[str, Any]] = {}

    # Check Candidate Recall@K
    candidate_recall_summary: dict[int, dict[str, Any]] = {}
    for k_depth in candidate_depths:
        hits = 0
        recalls = []
        for case in cases:
            top_k_cands = query_candidates[case.case_id][:k_depth]
            cand_docs = set(c.document_id for c in top_k_cands)
            expected = set(case.relevant_document_ids)
            if cand_docs & expected:
                hits += 1
            matched = len(cand_docs & expected)
            rec = matched / len(expected) if expected else 0.0
            recalls.append(rec)

        mean_rec = sum(recalls) / len(recalls)
        candidate_recall_summary[k_depth] = {
            "depth": k_depth,
            "queries_with_relevant_in_candidate": hits,
            "total_queries": len(cases),
            "hit_rate_in_candidate": hits / len(cases),
            "candidate_recall": mean_rec,
        }

    print("CANDIDATE RECALL BEFORE RERANKING:")
    print("Depth | Queries with Rel | Candidate Recall@K")
    print("---------------------------------------------")
    for k_depth, info in candidate_recall_summary.items():
        print(f"{k_depth:<5} | {info['queries_with_relevant_in_candidate']:>2}/{info['total_queries']} ({info['hit_rate_in_candidate']*100:.1f}%) | {info['candidate_recall']:.4f}")
    print("")

    # Run retrieval evaluation across candidate depths
    for k_depth in candidate_depths:
        print(f"Evaluating candidate_top_k = {k_depth} ({args.num_runs} warm runs)...")
        run_latencies: list[list[float]] = []
        run_rerank_latencies: list[list[float]] = []
        last_query_results: list[dict[str, Any]] = []

        for run_idx in range(args.num_runs):
            run_total_lats: list[float] = []
            run_ce_lats: list[float] = []
            current_q_results: list[dict[str, Any]] = []

            for case in cases:
                cands = query_candidates[case.case_id][:k_depth]

                t_rerank = time.perf_counter()
                reranked = reranker.rerank(
                    query=case.question,
                    candidates=cands,
                    top_k=5,
                    batch_size=16,
                )
                ce_lat_ms = (time.perf_counter() - t_rerank) * 1000.0
                total_lat_ms = mean_stage1_ms + ce_lat_ms

                run_total_lats.append(total_lat_ms)
                run_ce_lats.append(ce_lat_ms)

                if run_idx == args.num_runs - 1:
                    current_q_results.append({
                        "case_id": case.case_id,
                        "question": case.question,
                        "expected_doc_ids": list(case.relevant_document_ids),
                        "retrieved_doc_ids": [r.document_id for r in reranked],
                        "retrieved_chunk_ids": [r.chunk_id for r in reranked],
                        "scores": [r.score for r in reranked],
                        "ranks": [r.rank for r in reranked],
                        "warm_latency_ms": total_lat_ms,
                        "ce_latency_ms": ce_lat_ms,
                    })

            run_latencies.append(run_total_lats)
            run_rerank_latencies.append(run_ce_lats)
            if run_idx == args.num_runs - 1:
                last_query_results = current_q_results

        # Aggregate latency over all runs
        all_warm_lats = [lat for run in run_latencies for lat in run]
        all_ce_lats = [lat for run in run_rerank_latencies for lat in run]

        mean_warm_lat = float(np.mean(all_warm_lats))
        median_warm_lat = float(np.median(all_warm_lats))
        p95_warm_lat = float(np.percentile(all_warm_lats, 95))
        min_warm_lat = float(np.min(all_warm_lats))
        max_warm_lat = float(np.max(all_warm_lats))
        mean_ce_lat = float(np.mean(all_ce_lats))

        doc_metrics, hit_rate_at_k = _compute_metrics(last_query_results, k_values=(1, 3, 5))

        depth_results[k_depth] = {
            "candidate_depth": k_depth,
            "batch_size": 16,
            "final_top_k": 5,
            "candidate_recall": candidate_recall_summary[k_depth]["candidate_recall"],
            "doc_metrics": doc_metrics,
            "hit_rate_at_k": hit_rate_at_k,
            "warm_latency_ms": {
                "mean": round(mean_warm_lat, 3),
                "median": round(median_warm_lat, 3),
                "p95": round(p95_warm_lat, 3),
                "min": round(min_warm_lat, 3),
                "max": round(max_warm_lat, 3),
                "ce_mean": round(mean_ce_lat, 3),
                "stage1_mean": round(mean_stage1_ms, 3),
            },
            "query_results": last_query_results,
        }

    # Print Candidate Depth Table
    print("\n" + "=" * 95)
    print("EXPERIMENT 1 RESULTS: ACCURACY & WARM LATENCY BY CANDIDATE DEPTH")
    print("=" * 95)
    print(f"{'Depth':<5} | {'CandRec':<8} | {'Rec@1':<7} | {'Rec@5':<7} | {'Hit@1':<7} | {'Hit@5':<7} | {'MRR@5':<7} | {'nDCG@5':<7} | {'Mean(ms)':<9} | {'Med(ms)':<8} | {'P95(ms)':<8}")
    print("-" * 95)
    for k_depth in candidate_depths:
        r = depth_results[k_depth]
        dm = r["doc_metrics"]
        hr = r["hit_rate_at_k"]
        lat = r["warm_latency_ms"]
        print(
            f"{k_depth:<5} | "
            f"{r['candidate_recall']:<8.4f} | "
            f"{dm['recall'][1]:<7.4f} | "
            f"{dm['recall'][5]:<7.4f} | "
            f"{hr[1]:<7.4f} | "
            f"{hr[5]:<7.4f} | "
            f"{dm['mrr'][5]:<7.4f} | "
            f"{dm['ndcg'][5]:<7.4f} | "
            f"{lat['mean']:<9.2f} | "
            f"{lat['median']:<8.2f} | "
            f"{lat['p95']:<8.2f}"
        )
    print("=" * 95)

    # Selection rule: Hit Rate@5 = 100% AND Recall@5 >= 95% of Phase 4.9 baseline (0.9533 * 0.95 = 0.9056)
    p49_recall5 = depth_results[20]["doc_metrics"]["recall"][5]
    recall_threshold = p49_recall5 * 0.95
    print(f"\nOptimal Quality Rule: Hit Rate@5 == 1.0000 AND Recall@5 >= {recall_threshold:.4f}")

    qualifying_depths = [
        k for k in candidate_depths
        if depth_results[k]["hit_rate_at_k"][5] == 1.0
        and depth_results[k]["doc_metrics"]["recall"][5] >= recall_threshold
    ]

    if qualifying_depths:
        # Choose the one with lowest mean latency
        best_depth = min(qualifying_depths, key=lambda k: depth_results[k]["warm_latency_ms"]["mean"])
        print(f"Qualifying Candidate Depths : {qualifying_depths}")
        print(f"Selected Optimal Candidate Depth : {best_depth} (Lowest latency satisfying quality rule)")
    else:
        best_depth = 20
        print(f"No smaller candidate depth satisfied rule. Retaining baseline depth: {best_depth}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: BATCH SIZE BENCHMARK ON OPTIMAL CANDIDATE DEPTH
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print(f"EXPERIMENT 2: BATCH SIZE BENCHMARK ON OPTIMAL DEPTH (candidate_top_k={best_depth})")
    print("================================================================================")

    batch_sizes = [4, 8, 16, 32]
    batch_results: dict[int, dict[str, Any]] = {}

    for bs in batch_sizes:
        print(f"Evaluating batch_size = {bs} on candidate_top_k = {best_depth} ({args.num_runs} warm runs)...")
        run_latencies = []
        run_ce_lats = []
        last_query_results = []

        for run_idx in range(args.num_runs):
            run_total = []
            run_ce = []
            cur_q_res = []

            for case in cases:
                cands = query_candidates[case.case_id][:best_depth]

                t_rerank = time.perf_counter()
                reranked = reranker.rerank(
                    query=case.question,
                    candidates=cands,
                    top_k=5,
                    batch_size=bs,
                )
                ce_lat_ms = (time.perf_counter() - t_rerank) * 1000.0
                total_lat_ms = mean_stage1_ms + ce_lat_ms

                run_total.append(total_lat_ms)
                run_ce.append(ce_lat_ms)

                if run_idx == args.num_runs - 1:
                    cur_q_res.append({
                        "case_id": case.case_id,
                        "question": case.question,
                        "expected_doc_ids": list(case.relevant_document_ids),
                        "retrieved_doc_ids": [r.document_id for r in reranked],
                        "retrieved_chunk_ids": [r.chunk_id for r in reranked],
                        "scores": [r.score for r in reranked],
                        "ranks": [r.rank for r in reranked],
                        "warm_latency_ms": total_lat_ms,
                    })

            run_latencies.append(run_total)
            run_ce_lats.append(run_ce)
            if run_idx == args.num_runs - 1:
                last_query_results = cur_q_res

        all_warm = [lat for run in run_latencies for lat in run]
        all_ce = [lat for run in run_ce_lats for lat in run]

        doc_metrics, hit_rate_at_k = _compute_metrics(last_query_results, k_values=(1, 3, 5))

        batch_results[bs] = {
            "batch_size": bs,
            "candidate_depth": best_depth,
            "final_top_k": 5,
            "doc_metrics": doc_metrics,
            "hit_rate_at_k": hit_rate_at_k,
            "warm_latency_ms": {
                "mean": round(float(np.mean(all_warm)), 3),
                "median": round(float(np.median(all_warm)), 3),
                "p95": round(float(np.percentile(all_warm, 95)), 3),
                "min": round(float(np.min(all_warm)), 3),
                "max": round(float(np.max(all_warm)), 3),
                "ce_mean": round(float(np.mean(all_ce)), 3),
                "stage1_mean": round(mean_stage1_ms, 3),
            },
        }

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2 RESULTS: BATCH SIZE VS LATENCY (at candidate_top_k={best_depth})")
    print("=" * 80)
    print(f"{'Batch':<6} | {'Hit@5':<7} | {'Rec@5':<7} | {'MRR@5':<7} | {'Mean(ms)':<9} | {'Med(ms)':<8} | {'P95(ms)':<8} | {'Speedup':<8}")
    print("-" * 80)
    base_lat = depth_results[20]["warm_latency_ms"]["mean"]
    for bs in batch_sizes:
        b = batch_results[bs]
        dm = b["doc_metrics"]
        hr = b["hit_rate_at_k"]
        lat = b["warm_latency_ms"]
        speedup = f"{base_lat / lat['mean']:.2f}x"
        print(
            f"{bs:<6} | "
            f"{hr[5]:<7.4f} | "
            f"{dm['recall'][5]:<7.4f} | "
            f"{dm['mrr'][5]:<7.4f} | "
            f"{lat['mean']:<9.2f} | "
            f"{lat['median']:<8.2f} | "
            f"{lat['p95']:<8.2f} | "
            f"{speedup:<8}"
        )
    print("=" * 80)

    best_batch = min(batch_sizes, key=lambda b: batch_results[b]["warm_latency_ms"]["mean"])
    print(f"Optimal Batch Size : {best_batch} (Lowest warm latency on CPU)")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: ADAPTIVE CANDIDATE DEPTH POLICY EVALUATION
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("EXPERIMENT 3: ADAPTIVE CANDIDATE DEPTH POLICY EVALUATION")
    print("================================================================================")

    adaptive_policy = AdaptiveCandidateDepthPolicy(
        min_depth=10,
        default_depth=15,
        max_depth=20,
        high_confidence_gap_threshold=0.005,
    )

    adaptive_decisions: list[dict[str, Any]] = []
    adaptive_latencies: list[float] = []
    adaptive_query_results: list[dict[str, Any]] = []

    for case in cases:
        cands_all = query_candidates[case.case_id]
        dynamic_depth = adaptive_policy.determine_depth(cands_all)
        selected_cands = cands_all[:dynamic_depth]

        t0 = time.perf_counter()
        reranked = reranker.rerank(
            query=case.question,
            candidates=selected_cands,
            top_k=5,
            batch_size=best_batch,
        )
        ce_lat_ms = (time.perf_counter() - t0) * 1000.0
        total_lat_ms = mean_stage1_ms + ce_lat_ms

        adaptive_latencies.append(total_lat_ms)
        adaptive_decisions.append({
            "case_id": case.case_id,
            "chosen_depth": dynamic_depth,
            "signals": adaptive_policy.compute_signals(cands_all),
        })

        adaptive_query_results.append({
            "case_id": case.case_id,
            "question": case.question,
            "expected_doc_ids": list(case.relevant_document_ids),
            "retrieved_doc_ids": [r.document_id for r in reranked],
            "scores": [r.score for r in reranked],
            "ranks": [r.rank for r in reranked],
            "warm_latency_ms": total_lat_ms,
        })

    adapt_doc_metrics, adapt_hit_rate = _compute_metrics(adaptive_query_results, k_values=(1, 3, 5))
    adapt_mean_lat = float(np.mean(adaptive_latencies))
    adapt_med_lat = float(np.median(adaptive_latencies))
    adapt_p95_lat = float(np.percentile(adaptive_latencies, 95))

    depth_counts: dict[int, int] = {}
    for d in adaptive_decisions:
        cd = d["chosen_depth"]
        depth_counts[cd] = depth_counts.get(cd, 0) + 1

    print("Adaptive Policy Query Decisions:")
    for d, count in sorted(depth_counts.items()):
        print(f"- Depth {d}: {count} queries ({count / len(cases) * 100:.1f}%)")

    # -------------------------------------------------------------------------
    # FINAL 3-WAY COMPARISON: FIXED BASELINE vs FIXED OPTIMAL vs ADAPTIVE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL COMPARISON: FIXED BASELINE (K=20) vs FIXED OPTIMAL vs ADAPTIVE POLICY")
    print("=" * 90)
    print(f"{'Strategy':<22} | {'Hit@5':<7} | {'Rec@5':<7} | {'MRR@5':<7} | {'Mean(ms)':<9} | {'Med(ms)':<8} | {'P95(ms)':<8} | {'Latency Red.'}")
    print("-" * 90)

    # 1. Fixed Baseline (K=20, batch=16)
    p49_lat = depth_results[20]["warm_latency_ms"]["mean"]
    print(
        f"{'Fixed Baseline (K=20)':<22} | "
        f"{depth_results[20]['hit_rate_at_k'][5]:<7.4f} | "
        f"{depth_results[20]['doc_metrics']['recall'][5]:<7.4f} | "
        f"{depth_results[20]['doc_metrics']['mrr'][5]:<7.4f} | "
        f"{depth_results[20]['warm_latency_ms']['mean']:<9.2f} | "
        f"{depth_results[20]['warm_latency_ms']['median']:<8.2f} | "
        f"{depth_results[20]['warm_latency_ms']['p95']:<8.2f} | "
        f"Baseline"
    )

    # 2. Fixed Optimal (best_depth, best_batch)
    opt_lat = batch_results[best_batch]["warm_latency_ms"]["mean"]
    opt_saving = (p49_lat - opt_lat) / p49_lat * 100.0
    print(
        f"{f'Fixed Optimal (K={best_depth})':<22} | "
        f"{batch_results[best_batch]['hit_rate_at_k'][5]:<7.4f} | "
        f"{batch_results[best_batch]['doc_metrics']['recall'][5]:<7.4f} | "
        f"{batch_results[best_batch]['doc_metrics']['mrr'][5]:<7.4f} | "
        f"{batch_results[best_batch]['warm_latency_ms']['mean']:<9.2f} | "
        f"{batch_results[best_batch]['warm_latency_ms']['median']:<8.2f} | "
        f"{batch_results[best_batch]['warm_latency_ms']['p95']:<8.2f} | "
        f"-{opt_saving:.1f}%"
    )

    # 3. Adaptive Policy
    adapt_saving = (p49_lat - adapt_mean_lat) / p49_lat * 100.0
    print(
        f"{'Adaptive Depth Policy':<22} | "
        f"{adapt_hit_rate[5]:<7.4f} | "
        f"{adapt_doc_metrics['recall'][5]:<7.4f} | "
        f"{adapt_doc_metrics['mrr'][5]:<7.4f} | "
        f"{adapt_mean_lat:<9.2f} | "
        f"{adapt_med_lat:<8.2f} | "
        f"{adapt_p95_lat:<8.2f} | "
        f"-{adapt_saving:.1f}%"
    )
    print("=" * 90)

    # -------------------------------------------------------------------------
    # SAVE PHASE 4.10 ARTIFACTS
    # -------------------------------------------------------------------------
    cd_artifact_path = output_dir / "phase4_10_candidate_depth_results.json"
    bs_artifact_path = output_dir / "phase4_10_batch_size_results.json"
    comp_artifact_path = output_dir / "phase4_10_comparison_summary.json"

    # Save candidate depth results (strip query_results for summary clarity)
    cd_export = {
        str(k): {key: val for key, val in data.items() if key != "query_results"}
        for k, data in depth_results.items()
    }
    with open(cd_artifact_path, "w", encoding="utf-8") as f:
        json.dump(cd_export, f, indent=2)

    with open(bs_artifact_path, "w", encoding="utf-8") as f:
        json.dump({str(k): val for k, val in batch_results.items()}, f, indent=2)

    comparison_summary_data = {
        "phase": "4.10",
        "description": "Adaptive Reranking Optimization & Latency Reduction Benchmark",
        "cold_start_loading_ms": round(cold_start_ms, 2),
        "stage1_mean_latency_ms": round(mean_stage1_ms, 3),
        "quality_rule": {
            "hit_rate_5_target": 1.0000,
            "recall_5_threshold": round(recall_threshold, 4),
            "qualifying_depths": qualifying_depths,
            "selected_depth": best_depth,
            "selected_batch_size": best_batch,
        },
        "candidate_depth_comparison": {
            str(k): {
                "candidate_recall": depth_results[k]["candidate_recall"],
                "recall_at_5": depth_results[k]["doc_metrics"]["recall"][5],
                "hit_rate_at_5": depth_results[k]["hit_rate_at_k"][5],
                "mrr_at_5": depth_results[k]["doc_metrics"]["mrr"][5],
                "ndcg_at_5": depth_results[k]["doc_metrics"]["ndcg"][5],
                "warm_mean_latency_ms": depth_results[k]["warm_latency_ms"]["mean"],
                "warm_median_latency_ms": depth_results[k]["warm_latency_ms"]["median"],
                "warm_p95_latency_ms": depth_results[k]["warm_latency_ms"]["p95"],
            }
            for k in candidate_depths
        },
        "batch_size_comparison": {
            str(b): {
                "recall_at_5": batch_results[b]["doc_metrics"]["recall"][5],
                "hit_rate_at_5": batch_results[b]["hit_rate_at_k"][5],
                "warm_mean_latency_ms": batch_results[b]["warm_latency_ms"]["mean"],
                "warm_median_latency_ms": batch_results[b]["warm_latency_ms"]["median"],
                "warm_p95_latency_ms": batch_results[b]["warm_latency_ms"]["p95"],
            }
            for b in batch_sizes
        },
        "adaptive_policy_evaluation": {
            "depth_distribution": depth_counts,
            "hit_rate_at_5": adapt_hit_rate[5],
            "recall_at_5": adapt_doc_metrics["recall"][5],
            "mrr_at_5": adapt_doc_metrics["mrr"][5],
            "warm_mean_latency_ms": round(adapt_mean_lat, 3),
            "warm_median_latency_ms": round(adapt_med_lat, 3),
            "warm_p95_latency_ms": round(adapt_p95_lat, 3),
            "latency_reduction_vs_baseline_percent": round(adapt_saving, 2),
        },
        "final_recommendation": {
            "optimal_fixed_depth": best_depth,
            "optimal_batch_size": best_batch,
            "baseline_warm_mean_ms": depth_results[20]["warm_latency_ms"]["mean"],
            "optimal_warm_mean_ms": batch_results[best_batch]["warm_latency_ms"]["mean"],
            "absolute_savings_ms": round(p49_lat - opt_lat, 2),
            "relative_savings_percent": round(opt_saving, 2),
            "quality_preserved": (
                batch_results[best_batch]["hit_rate_at_k"][5] == 1.0
                and batch_results[best_batch]["doc_metrics"]["recall"][5] >= recall_threshold
            ),
        },
    }

    with open(comp_artifact_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary_data, f, indent=2)

    print("\nOUTPUT ARTIFACTS SAVED:")
    print(f"- Candidate Depth Results : {cd_artifact_path}")
    print(f"- Batch Size Results      : {bs_artifact_path}")
    print(f"- Comparison Summary      : {comp_artifact_path}")
    print("================================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
