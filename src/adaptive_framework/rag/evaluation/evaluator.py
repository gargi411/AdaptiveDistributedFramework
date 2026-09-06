"""RAG retrieval evaluation orchestrator and reporting."""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from adaptive_framework.rag.evaluation.evaluation_case import RAGEvaluationCase
from adaptive_framework.rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class QueryEvaluationResult:
    """Per-query retrieval evaluation result."""

    case_id: str
    question: str
    category: str
    expected_doc_ids: list[str]
    expected_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_chunk_ids: list[str]
    scores: list[float]
    ranks: list[int]
    latency_ms: float
    doc_precision_at_k: dict[int, float] = field(default_factory=dict)
    doc_recall_at_k: dict[int, float] = field(default_factory=dict)
    doc_rr_at_k: dict[int, float] = field(default_factory=dict)
    doc_ndcg_at_k: dict[int, float] = field(default_factory=dict)
    chunk_precision_at_k: dict[int, float] = field(default_factory=dict)
    chunk_recall_at_k: dict[int, float] = field(default_factory=dict)
    chunk_rr_at_k: dict[int, float] = field(default_factory=dict)
    chunk_ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_at_k: dict[int, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert query evaluation result to serializable dictionary."""
        return {
            "case_id": self.case_id,
            "question": self.question,
            "category": self.category,
            "expected_doc_ids": self.expected_doc_ids,
            "expected_chunk_ids": self.expected_chunk_ids,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "scores": self.scores,
            "ranks": self.ranks,
            "latency_ms": round(self.latency_ms, 3),
            "doc_precision_at_k": {str(k): round(v, 4) for k, v in self.doc_precision_at_k.items()},
            "doc_recall_at_k": {str(k): round(v, 4) for k, v in self.doc_recall_at_k.items()},
            "doc_rr_at_k": {str(k): round(v, 4) for k, v in self.doc_rr_at_k.items()},
            "doc_ndcg_at_k": {str(k): round(v, 4) for k, v in self.doc_ndcg_at_k.items()},
            "chunk_precision_at_k": {str(k): round(v, 4) for k, v in self.chunk_precision_at_k.items()},
            "chunk_recall_at_k": {str(k): round(v, 4) for k, v in self.chunk_recall_at_k.items()},
            "chunk_rr_at_k": {str(k): round(v, 4) for k, v in self.chunk_rr_at_k.items()},
            "chunk_ndcg_at_k": {str(k): round(v, 4) for k, v in self.chunk_ndcg_at_k.items()},
            "hit_at_k": {str(k): v for k, v in self.hit_at_k.items()},
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report over an evaluation dataset."""

    query_count: int
    k_values: tuple[int, ...]
    doc_metrics: dict[str, dict[int, float]]
    chunk_metrics: dict[str, dict[int, float]]
    latency_stats_ms: dict[str, float]
    hit_rate_at_k: dict[int, float]
    query_results: list[QueryEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate report to serializable dictionary."""
        return {
            "query_count": self.query_count,
            "k_values": list(self.k_values),
            "document_level": {
                metric: {str(k): round(val, 4) for k, val in k_map.items()}
                for metric, k_map in self.doc_metrics.items()
            },
            "chunk_level": {
                metric: {str(k): round(val, 4) for k, val in k_map.items()}
                for metric, k_map in self.chunk_metrics.items()
            },
            "latency_stats_ms": {k: round(v, 3) for k, v in self.latency_stats_ms.items()},
            "hit_rate_at_k": {str(k): round(v, 4) for k, v in self.hit_rate_at_k.items()},
        }


class RAGEvaluator:
    """Evaluates a RetrievalEngine against a set of ground-truth test cases."""

    def __init__(
        self,
        retrieval_engine: IRetrievalEngine,
        k_values: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        """Initialize RAGEvaluator.

        Args:
            retrieval_engine: IRetrievalEngine instance under test.
            k_values: Tuple of K cutoffs to evaluate (e.g. 1, 3, 5).
        """
        self.retrieval_engine = retrieval_engine
        self.k_values = tuple(sorted(k_values))
        self.max_k = max(self.k_values) if self.k_values else 5

    def evaluate(
        self,
        cases: Sequence[RAGEvaluationCase],
        top_k: int | None = None,
    ) -> EvaluationReport:
        """Run retrieval evaluation over all cases and compute aggregate metrics.

        Args:
            cases: Sequence of RAGEvaluationCase instances.
            top_k: Optional maximum retrieval limit (defaults to max_k).

        Returns:
            EvaluationReport containing detailed per-query and aggregate metrics.
        """
        eval_top_k = top_k or self.max_k
        query_results: list[QueryEvaluationResult] = []
        latencies: list[float] = []

        for case in cases:
            t0 = time.perf_counter()
            results: list[RetrievalResult] = self.retrieval_engine.retrieve(
                query=case.question,
                top_k=eval_top_k,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            retrieved_doc_ids = [r.document_id for r in results]
            retrieved_chunk_ids = [r.chunk_id for r in results]
            scores = [r.score for r in results]
            ranks = [r.rank for r in results]

            q_res = QueryEvaluationResult(
                case_id=case.case_id,
                question=case.question,
                category=case.category,
                expected_doc_ids=list(case.relevant_document_ids),
                expected_chunk_ids=list(case.relevant_chunk_ids),
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                scores=scores,
                ranks=ranks,
                latency_ms=latency_ms,
            )

            # Evaluate metrics at each K
            for k in self.k_values:
                # Document-level
                p_doc = precision_at_k(retrieved_doc_ids, case.relevant_document_ids, k=k)
                r_doc = recall_at_k(retrieved_doc_ids, case.relevant_document_ids, k=k)
                rr_doc = reciprocal_rank(retrieved_doc_ids, case.relevant_document_ids, k=k)
                ndcg_doc = ndcg_at_k(retrieved_doc_ids, case.relevant_document_ids, k=k)

                q_res.doc_precision_at_k[k] = p_doc
                q_res.doc_recall_at_k[k] = r_doc
                q_res.doc_rr_at_k[k] = rr_doc
                q_res.doc_ndcg_at_k[k] = ndcg_doc

                # Hit at K (at least one relevant document in top K)
                hit = any(doc_id in case.relevant_document_ids for doc_id in retrieved_doc_ids[:k])
                q_res.hit_at_k[k] = hit

                # Chunk-level
                p_chk = precision_at_k(retrieved_chunk_ids, case.relevant_chunk_ids, k=k)
                r_chk = recall_at_k(retrieved_chunk_ids, case.relevant_chunk_ids, k=k)
                rr_chk = reciprocal_rank(retrieved_chunk_ids, case.relevant_chunk_ids, k=k)
                ndcg_chk = ndcg_at_k(retrieved_chunk_ids, case.relevant_chunk_ids, k=k)

                q_res.chunk_precision_at_k[k] = p_chk
                q_res.chunk_recall_at_k[k] = r_chk
                q_res.chunk_rr_at_k[k] = rr_chk
                q_res.chunk_ndcg_at_k[k] = ndcg_chk

            query_results.append(q_res)

        # Aggregate across queries
        n_queries = len(cases)
        doc_metrics: dict[str, dict[int, float]] = {
            "precision": {},
            "recall": {},
            "mrr": {},
            "ndcg": {},
        }
        chunk_metrics: dict[str, dict[int, float]] = {
            "precision": {},
            "recall": {},
            "mrr": {},
            "ndcg": {},
        }
        hit_rate_at_k: dict[int, float] = {}

        for k in self.k_values:
            if n_queries > 0:
                doc_metrics["precision"][k] = sum(q.doc_precision_at_k[k] for q in query_results) / n_queries
                doc_metrics["recall"][k] = sum(q.doc_recall_at_k[k] for q in query_results) / n_queries
                doc_metrics["mrr"][k] = sum(q.doc_rr_at_k[k] for q in query_results) / n_queries
                doc_metrics["ndcg"][k] = sum(q.doc_ndcg_at_k[k] for q in query_results) / n_queries
                hit_rate_at_k[k] = sum(1 for q in query_results if q.hit_at_k[k]) / n_queries

                chunk_metrics["precision"][k] = sum(q.chunk_precision_at_k[k] for q in query_results) / n_queries
                chunk_metrics["recall"][k] = sum(q.chunk_recall_at_k[k] for q in query_results) / n_queries
                chunk_metrics["mrr"][k] = sum(q.chunk_rr_at_k[k] for q in query_results) / n_queries
                chunk_metrics["ndcg"][k] = sum(q.chunk_ndcg_at_k[k] for q in query_results) / n_queries
            else:
                doc_metrics["precision"][k] = 0.0
                doc_metrics["recall"][k] = 0.0
                doc_metrics["mrr"][k] = 0.0
                doc_metrics["ndcg"][k] = 0.0
                hit_rate_at_k[k] = 0.0
                chunk_metrics["precision"][k] = 0.0
                chunk_metrics["recall"][k] = 0.0
                chunk_metrics["mrr"][k] = 0.0
                chunk_metrics["ndcg"][k] = 0.0

        # Latency statistics
        if latencies:
            sorted_lat = sorted(latencies)
            p95_idx = int(math.ceil(0.95 * len(sorted_lat))) - 1
            p95_val = sorted_lat[max(0, p95_idx)]
            latency_stats = {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "p95": p95_val,
            }
        else:
            latency_stats = {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}

        return EvaluationReport(
            query_count=n_queries,
            k_values=self.k_values,
            doc_metrics=doc_metrics,
            chunk_metrics=chunk_metrics,
            latency_stats_ms=latency_stats,
            hit_rate_at_k=hit_rate_at_k,
            query_results=query_results,
        )

    def save_report(
        self,
        report: EvaluationReport,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Save detailed and summary evaluation results to disk.

        Args:
            report: EvaluationReport to persist.
            output_dir: Directory where JSON artifacts will be written.

        Returns:
            Tuple of (summary_path, details_path).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "evaluation_summary.json"
        details_path = output_dir / "evaluation_results.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        with open(details_path, "w", encoding="utf-8") as f:
            details_data = {
                "summary": report.to_dict(),
                "queries": [q.to_dict() for q in report.query_results],
            }
            json.dump(details_data, f, indent=2)

        return summary_path, details_path
