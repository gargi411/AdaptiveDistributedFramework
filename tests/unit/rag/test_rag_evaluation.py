"""Unit tests for Phase 4.7 RAG retrieval evaluation metrics and evaluator."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from adaptive_framework.rag.evaluation.dataset import (
    DEFAULT_GROUND_TRUTH_CASES,
    load_ground_truth_cases,
)
from adaptive_framework.rag.evaluation.evaluation_case import RAGEvaluationCase
from adaptive_framework.rag.evaluation.evaluator import (
    EvaluationReport,
    RAGEvaluator,
)
from adaptive_framework.rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from adaptive_framework.rag.interfaces.i_retrieval_engine import IRetrievalEngine
from adaptive_framework.rag.retrieval.retrieval_result import RetrievalResult


class TestPrecisionAtK:
    """Test suite for Precision@K calculation."""

    def test_perfect_precision(self) -> None:
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_2", "doc_3"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_hit_precision(self) -> None:
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_9"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_precision(self) -> None:
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = {"doc_2", "doc_4"}
        # Top 2 has 1 hit ("doc_2") -> 1/2 = 0.5
        assert precision_at_k(retrieved, relevant, k=2) == 0.5
        # Top 4 has 2 hits -> 2/4 = 0.5
        assert precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_fewer_than_k_results(self) -> None:
        """When fewer than K results exist, divisor remains K per standard IR convention."""
        retrieved = ["doc_1"]
        relevant = {"doc_1"}
        assert precision_at_k(retrieved, relevant, k=5) == 0.2

    def test_empty_retrieved_or_relevant(self) -> None:
        assert precision_at_k([], {"doc_1"}, k=3) == 0.0
        assert precision_at_k(["doc_1"], set(), k=3) == 0.0

    def test_invalid_k_raises_error(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k(["doc_1"], {"doc_1"}, k=0)


class TestRecallAtK:
    """Test suite for Recall@K calculation."""

    def test_perfect_recall(self) -> None:
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_2"}
        # Top 2 retrieves all relevant items -> 2/2 = 1.0
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_partial_recall(self) -> None:
        retrieved = ["doc_1", "doc_9"]
        relevant = {"doc_1", "doc_2"}
        # Top 2 retrieves 1 of 2 relevant items -> 1/2 = 0.5
        assert recall_at_k(retrieved, relevant, k=2) == 0.5

    def test_zero_recall(self) -> None:
        retrieved = ["doc_9", "doc_8"]
        relevant = {"doc_1"}
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k(["doc_1"], set(), k=3) == 0.0

    def test_invalid_k_raises_error(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k(["doc_1"], {"doc_1"}, k=-1)


class TestReciprocalRank:
    """Test suite for Reciprocal Rank (MRR component)."""

    def test_first_rank_hit(self) -> None:
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1"}
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_rank_hit(self) -> None:
        retrieved = ["doc_9", "doc_1", "doc_3"]
        relevant = {"doc_1"}
        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_rank_hit(self) -> None:
        retrieved = ["doc_9", "doc_8", "doc_1"]
        relevant = {"doc_1"}
        assert pytest.approx(reciprocal_rank(retrieved, relevant), 0.001) == 1.0 / 3.0

    def test_no_hit_returns_zero(self) -> None:
        retrieved = ["doc_9", "doc_8"]
        relevant = {"doc_1"}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_cutoff_k_excludes_later_hit(self) -> None:
        retrieved = ["doc_9", "doc_8", "doc_1"]
        relevant = {"doc_1"}
        # Cutoff at K=2 excludes rank 3 hit
        assert reciprocal_rank(retrieved, relevant, k=2) == 0.0
        # Cutoff at K=3 includes rank 3 hit
        assert pytest.approx(reciprocal_rank(retrieved, relevant, k=3), 0.001) == 1.0 / 3.0


class TestNDCGAtK:
    """Test suite for nDCG@K calculation."""

    def test_perfect_ranking(self) -> None:
        retrieved = ["doc_1", "doc_2", "doc_9"]
        relevant = {"doc_1", "doc_2"}
        assert pytest.approx(ndcg_at_k(retrieved, relevant, k=3), 0.0001) == 1.0

    def test_reversed_ranking(self) -> None:
        retrieved = ["doc_9", "doc_1"]
        relevant = {"doc_1"}
        # Ideal DCG = 1 / log2(2) = 1.0
        # Actual DCG = 1 / log2(3) = 1 / 1.58496 = 0.6309
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert pytest.approx(ndcg_at_k(retrieved, relevant, k=2), 0.001) == expected

    def test_zero_hits(self) -> None:
        retrieved = ["doc_9", "doc_8"]
        relevant = {"doc_1"}
        assert ndcg_at_k(retrieved, relevant, k=2) == 0.0

    def test_empty_inputs(self) -> None:
        assert ndcg_at_k([], {"doc_1"}, k=3) == 0.0
        assert ndcg_at_k(["doc_1"], set(), k=3) == 0.0


class TestDatasetAndCases:
    """Test suite for evaluation case models and ground truth defaults."""

    def test_default_ground_truth_cases_validity(self) -> None:
        cases = DEFAULT_GROUND_TRUTH_CASES
        assert len(cases) == 25
        for case in cases:
            assert case.case_id.startswith("eval_")
            assert len(case.question) > 10
            assert len(case.relevant_document_ids) >= 1
            assert case.category in ("diagnosis", "symptoms", "investigations", "treatment", "multi_evidence")

    def test_case_to_dict_roundtrip(self) -> None:
        case = RAGEvaluationCase(
            case_id="case_test",
            question="What was the MRI result?",
            relevant_document_ids=("doc_01",),
            relevant_chunk_ids=("chk_01",),
            category="investigations",
            description="Test description",
        )
        d = case.to_dict()
        assert d["case_id"] == "case_test"
        assert d["relevant_document_ids"] == ["doc_01"]
        assert d["relevant_chunk_ids"] == ["chk_01"]


class TestRAGEvaluator:
    """Test suite for RAGEvaluator orchestration."""

    def test_evaluator_runs_and_aggregates(self) -> None:
        mock_engine = MagicMock(spec=IRetrievalEngine)

        def mock_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
            if "Soumya" in query:
                return [
                    RetrievalResult(
                        chunk_id="chk_synth_0001",
                        document_id="clinical_note_0001",
                        source_file="file1.pdf",
                        page_numbers=(1,),
                        text="EEG abnormal spikes",
                        section_heading="Inv",
                        document_type="note",
                        chunk_index=0,
                        score=0.9,
                        rank=1,
                    )
                ]
            return [
                RetrievalResult(
                    chunk_id="chk_synth_9999",
                    document_id="clinical_note_9999",
                    source_file="file9.pdf",
                    page_numbers=(1,),
                    text="Unrelated note",
                    section_heading=None,
                    document_type="note",
                    chunk_index=0,
                    score=0.1,
                    rank=1,
                )
            ]

        mock_engine.retrieve.side_effect = mock_retrieve

        cases = [
            RAGEvaluationCase(
                case_id="c1",
                question="What was documented for Soumya Das?",
                relevant_document_ids=("clinical_note_0001",),
                relevant_chunk_ids=("chk_synth_0001",),
            ),
            RAGEvaluationCase(
                case_id="c2",
                question="Unknown query?",
                relevant_document_ids=("clinical_note_0002",),
                relevant_chunk_ids=("chk_synth_0002",),
            ),
        ]

        evaluator = RAGEvaluator(retrieval_engine=mock_engine, k_values=(1, 3, 5))
        report = evaluator.evaluate(cases)

        assert isinstance(report, EvaluationReport)
        assert report.query_count == 2
        # For c1: hit at 1 (p=1.0). For c2: miss (p=0.0). Mean doc P@1 = 0.5
        assert report.doc_metrics["precision"][1] == 0.5
        assert report.doc_metrics["recall"][1] == 0.5
        assert report.doc_metrics["mrr"][1] == 0.5
        assert report.chunk_metrics["precision"][1] == 0.5
        assert report.hit_rate_at_k[1] == 0.5
        assert report.latency_stats_ms["mean"] >= 0.0
