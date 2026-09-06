"""RAG Retrieval Evaluation Package (Phase 4.7).

Provides mathematical retrieval metrics, ground-truth dataset models,
and evaluation orchestrators for measuring RAG retrieval effectiveness.
"""

from adaptive_framework.rag.evaluation.dataset import (
    DEFAULT_GROUND_TRUTH_CASES,
    load_ground_truth_cases,
    save_ground_truth_cases,
)
from adaptive_framework.rag.evaluation.evaluation_case import RAGEvaluationCase
from adaptive_framework.rag.evaluation.evaluator import (
    EvaluationReport,
    QueryEvaluationResult,
    RAGEvaluator,
)
from adaptive_framework.rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "RAGEvaluationCase",
    "QueryEvaluationResult",
    "EvaluationReport",
    "RAGEvaluator",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "DEFAULT_GROUND_TRUTH_CASES",
    "load_ground_truth_cases",
    "save_ground_truth_cases",
]
