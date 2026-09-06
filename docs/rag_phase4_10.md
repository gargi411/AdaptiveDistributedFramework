# Phase 4.10 -- Adaptive Reranking Optimization and Latency Reduction

This document details the architectural implementation, experimental methodology, empirical findings, batch inference optimization, Pareto accuracy-latency analysis, and runtime signal-driven adaptive policy developed in Phase 4.10 of the Adaptive Distributed Framework (v2.0).

---

## 1. Motivation and Experimental Objective

Phase 4.9 integrated cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) downstream of hybrid retrieval (Dense FAISS + BM25Okapi + Reciprocal Rank Fusion, k=60). This elevated retrieval effectiveness from Hit Rate@5 = 0.3200 and Recall@5 = 0.2933 (Hybrid) to Hit Rate@5 = 1.0000 and Recall@5 = 0.9533.

However, cross-encoder scoring introduces a substantial computational cost. In Phase 4.9, reranking 20 candidate passages required ~1340 ms on CPU per query, consuming over 99.9% of end-to-end retrieval latency. In distributed biomedical document workflows, sustaining 1.3+ seconds per retrieval stage limits system throughput.

Phase 4.10 addresses the primary research question:
**"Can we reduce cross-encoder reranking latency by pruning the candidate pool and optimizing inference configuration while strictly preserving retrieval quality?"**

Specific research goals:
1. Benchmark candidate depths `candidate_top_k` in {5, 10, 15, 20} as the primary experiment to identify candidate pre-recall and downstream ranking degradation thresholds.
2. Select the optimal candidate depth using a predefined quality constraint: Hit Rate@5 = 1.0000 AND Recall@5 >= 95% of the Phase 4.9 baseline (Recall@5 >= 0.9057).
3. Evaluate batch size configurations {4, 8, 16, 32} on CPU inference for the selected depth.
4. Formulate and evaluate a ground-truth-independent `AdaptiveCandidateDepthPolicy` utilizing pre-reranking signals (RRF score distribution and dense/sparse ranking consensus).
5. Establish rigorous latency measurement protocols: separating cold-start model weight loading from warm steady-state inference across 3 repeated runs.

---

## 2. Synthetic Dataset Disclaimer

The evaluation benchmark is executed over:
`dataset/Synthetic Indian Clinical Notes for Natural Langua/synthetic_notes_medium.json`

DISCLAIMER:
1. All clinical records in this dataset are purely synthetic and artificially generated for computational research and benchmarking purposes.
2. The records do NOT represent real human patients, confidential personal health information (PHI), or actual clinical encounters.
3. No clinical validity, diagnostic efficacy, medical treatment guidance, or therapeutic safety is claimed or implied by these benchmarks or this framework.

---

## 3. Experimental Controls and Baseline Freezing

To ensure absolute scientific reproducibility and experimental isolation:
- Baseline models and artifacts from Phase 4.7 (Dense), Phase 4.8 (Hybrid), and Phase 4.9 (Cross-Encoder) remain unmodified in `outputs/rag/evaluation/`.
- The evaluation dataset comprises 50 synthetic clinical notes chunked via the Phase 4.2 semantic chunker into 207 discrete passages.
- The 25 canonical evaluation queries and ground-truth relevant document IDs are identical across all phases.
- Stage-1 retrieval parameters are frozen: Dense BGE-large (1024-dim, IndexFlatIP), BM25Okapi (k1=1.5, b=0.75), RRF fusion parameter k=60, dense_top_k=20, sparse_top_k=20.
- Cross-encoder architecture is frozen: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Final retrieval depth is fixed at `final_top_k = 5`.
- No query expansion, SPLADE, alternative reranker models, or distributed scheduler modifications were introduced.

---

## 4. Latency Accounting Methodology

### 4.1 Cold-Start Isolation
In Python and PyTorch runtimes, initial model instantiation involves disk I/O, weights deserialization, and graph compilation. In Phase 4.9, this cold-start overhead (~9-10 seconds) was conflated with initial query latency if not warmed.

In Phase 4.10:
- Explicit model warmup (`reranker.warmup()`) is performed before benchmark timing.
- Cold-start load time is recorded separately: **9316.23 ms**.
- Cold-start overhead is excluded from all warm inference statistics.

### 4.2 Warm Steady-State Accounting
- All configurations are benchmarked across **3 independent warm passes** over all 25 queries (75 query evaluations per configuration).
- Per-query latency is decomposed into Stage-1 retrieval (0.580 ms mean) and Stage-2 cross-encoder scoring.
- Reported latency metrics include Mean, Median, P95, Min, and Max.

---

## 5. Experiment 1: Fixed Candidate Depth Benchmark (Primary)

### 5.1 Candidate Pre-Recall Analysis
Before cross-encoder reranking, candidates must be present in the Stage-1 pool. We measure Candidate Recall@K (the proportion of ground-truth relevant documents captured in the top-K hybrid candidates prior to reranking):

| Candidate Depth (K_cand) | Queries with Relevant Document in Pool | Hit Rate in Candidate Pool | Candidate Recall@K |
|---|---|---|---|
| 5 | 9 / 25 | 36.0% | 0.3333 |
| 10 | 22 / 25 | 88.0% | 0.8200 |
| 15 | 25 / 25 | 100.0% | 0.9800 |
| 20 | 25 / 25 | 100.0% | 0.9800 |

Observations:
- At K_cand = 5, 64% of queries do not contain the target document in the candidate pool. Because reranking cannot recover documents absent from candidate generation, downstream retrieval collapses.
- At K_cand = 10, 3 queries miss the target document, capping Hit Rate at 88.0%.
- At K_cand = 15, all 25 queries (100.0%) contain the relevant clinical document, achieving Candidate Recall = 0.9800.
- Expanding candidate depth from 15 to 20 yields 0.0% additional candidate recall.

### 5.2 Retrieval Quality and Latency by Candidate Depth

| Depth | CandRec | Rec@1 | Rec@5 | Hit@1 | Hit@5 | MRR@5 | nDCG@5 | Mean (ms) | Med (ms) | P95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.3333 | 0.2800 | 0.3333 | 0.2800 | 0.3600 | 0.3200 | 0.3171 | 248.75 | 241.01 | 318.79 |
| 10 | 0.8200 | 0.7667 | 0.8200 | 0.8400 | 0.8800 | 0.8600 | 0.8154 | 756.95 | 739.05 | 1002.43 |
| 15 | 0.9800 | 0.7667 | 0.9533 | 0.8400 | 1.0000 | 0.9200 | 0.9067 | 1028.49 | 1023.41 | 1247.25 |
| 20 (Baseline) | 0.9800 | 0.7667 | 0.9533 | 0.8400 | 1.0000 | 0.9200 | 0.9042 | 1340.07 | 1276.69 | 1671.88 |

Key Findings:
1. **Candidate Depth 5 Fails**: Truncating candidates to 5 causes severe quality collapse (Hit Rate@5 drops from 100% to 36.0%, Recall@5 drops to 0.3333). Despite being fastest (248.75 ms), it is clinically unviable.
2. **Candidate Depth 10 Underperforms**: Candidate Recall caps at 0.8200, resulting in an unacceptable 12% drop in Hit Rate@5 (88.0%).
3. **Candidate Depth 15 Matches Baseline Quality**: At K_cand = 15, Hit Rate@5 (1.0000), Recall@5 (0.9533), MRR@5 (0.9200), and Rec@1 (0.7667) are 100% preserved relative to the Phase 4.9 baseline. nDCG@5 slightly improves from 0.9042 to 0.9067 due to reduced distractor noise in the candidate list.
4. **Latency Reduction**: Moving from K_cand = 20 to K_cand = 15 yields an immediate 311.58 ms reduction in mean latency (1340.07 ms -> 1028.49 ms, a 23.3% reduction) before batch optimization.

### 5.3 Optimal Candidate Depth Selection
The predefined quality rule requires:
- Hit Rate@5 == 1.0000
- Recall@5 >= 0.9057 (95% of baseline 0.9533)

Qualifying Depths: `[15, 20]`.
Depth 15 is selected as the optimal candidate depth, providing the minimum candidate pool that satisfies the quality guarantee.

---

## 6. Experiment 2: Batch Size Optimization on Optimal Depth (K=15)

Using `candidate_top_k = 15`, we benchmarked cross-encoder batch sizes in {4, 8, 16, 32} across 3 warm passes:

| Batch Size | Hit Rate@5 | Recall@5 | MRR@5 | Mean Latency (ms) | Median (ms) | P95 (ms) | Speedup vs Baseline (K=20, B=16) |
|---|---|---|---|---|---|---|---|
| 4 | 1.0000 | 0.9533 | 0.9200 | 1050.74 | 1025.87 | 1253.74 | 1.28x |
| 8 | 1.0000 | 0.9533 | 0.9200 | 967.44 | 954.12 | 1117.38 | 1.39x |
| 16 | 1.0000 | 0.9533 | 0.9200 | 978.34 | 932.93 | 1185.54 | 1.37x |
| 32 | 1.0000 | 0.9533 | 0.9200 | 883.23 | 860.99 | 1040.02 | 1.52x |

Analysis:
1. **Mathematical Invariance**: Batch size changes only tensor grouping during forward passes; ranking scores and retrieval metrics remain identical across all batch sizes.
2. **CPU Parallelism Efficiency**: Batch size 32 minimizes PyTorch Python loop overhead and vectorizes matrix multiplications for the 15 candidate pairs in a single forward pass batch.
3. **Latency Improvement**: Batch size 32 achieves a mean warm latency of **883.23 ms** (median 860.99 ms, P95 1040.02 ms), achieving a **1.52x speedup** (34.09% latency reduction) compared to the Phase 4.9 baseline (1340.07 ms).

---

## 7. Experiment 3: Ground-Truth-Independent Adaptive Candidate Depth Policy

### 7.1 Architecture and Policy Formulation
To evaluate whether retrieval latency can be dynamically modulated without human intervention, we implemented `AdaptiveCandidateDepthPolicy` in `src/adaptive_framework/rag/reranking/adaptive_policy.py`.

CRITICAL SCIENTIFIC INTEGRITY CONSTRAINT:
The policy has zero access to ground truth, evaluation labels, or case IDs. It relies exclusively on pre-reranking runtime signals extracted from the Stage-1 hybrid candidate pool:
1. **RRF Score Gap (Rank 1 vs Rank 5)**: If the top candidate demonstrates significant score separation (`rrf_score[0] - rrf_score[4] >= 0.0050`), retrieval confidence is high.
2. **Dual-Engine Agreement Ratio**: The proportion of top-5 candidates that were independently retrieved by both Dense FAISS and Sparse BM25.
3. **Dense-Sparse Top Rank Agreement**: Whether both dense and sparse retrieval placed the same document at rank 1.

Policy Logic:
- High confidence (gap >= 0.0050 AND consensus >= 0.40): Select `min_depth = 10`.
- Moderate confidence (gap >= 0.0025 OR consensus >= 0.20): Select `default_depth = 15`.
- Low confidence / ambiguity: Select `max_depth = 20`.

### 7.2 Empirical Decision Distribution and Performance

Across the 25 evaluation queries:
- **Depth 10**: Selected for 2 queries (8.0%)
- **Depth 15**: Selected for 8 queries (32.0%)
- **Depth 20**: Selected for 15 queries (60.0%)

Retrieval Results:
- Hit Rate@5: **1.0000**
- Recall@5: **0.9533**
- MRR@5: **0.9200**
- Mean Latency: **1135.43 ms** (Median: 1180.27 ms, P95: 1510.34 ms)
- Latency Reduction vs Baseline: **-15.27%**

---

## 8. Final Comparison and Pareto Frontier

| Strategy | Candidate Depth | Batch Size | Hit Rate@5 | Recall@5 | MRR@5 | Mean Latency (ms) | P95 Latency (ms) | Latency Reduction | Quality Retention |
|---|---|---|---|---|---|---|---|---|---|
| Phase 4.9 Baseline | Fixed 20 | 16 | 1.0000 | 0.9533 | 0.9200 | 1340.07 ms | 1671.88 ms | Reference | 100.0% |
| Adaptive Depth Policy | Dynamic (10/15/20) | 32 | 1.0000 | 0.9533 | 0.9200 | 1135.43 ms | 1510.34 ms | -15.27% | 100.0% |
| Fixed Optimal (Recommended) | Fixed 15 | 32 | 1.0000 | 0.9533 | 0.9200 | 883.23 ms | 1040.02 ms | -34.09% | 100.0% |

### Pareto Analysis
1. **Quality Floor**: Depths below 15 (K=5, K=10) breach the quality constraint (Hit Rate drops by 12% to 64%). They lie outside the admissible Pareto frontier.
2. **Fixed 15 Dominance**: The Fixed K=15 (batch=32) configuration strictly dominates both the Phase 4.9 baseline and the Adaptive Policy on latency (883.23 ms vs 1135.43 ms and 1340.07 ms) while delivering identical precision and recall.
3. **Adaptive Policy Behavior**: The Adaptive Policy provides a conservative safety mechanism (defaulting 60% of queries to K=20 when dual-engine signals indicate uncertainty), which guarantees zero quality loss but yields lower latency savings (15.27% vs 34.09%) than the deterministic K=15 configuration.

---

## 9. Architectural Code Modifications

1. `src/adaptive_framework/rag/reranking/cross_encoder_reranker.py`:
   - Added `warmup() -> float`: Instantiates model weights on dummy inputs and returns cold-start latency in ms.
   - Added optional `batch_size: Optional[int] = None` override to `rerank()`.
2. `src/adaptive_framework/rag/retrieval/reranked_retriever.py`:
   - Added `candidate_top_k: Optional[int] = None` and `batch_size: Optional[int] = None` dynamic parameter overrides to `retrieve()` and `retrieve_reranked()`.
3. `src/adaptive_framework/rag/reranking/adaptive_policy.py`:
   - Implemented `AdaptiveCandidateDepthPolicy` with `compute_signals()` and `determine_depth()`.
   - Guaranteed ground-truth isolation (inputs restricted strictly to `RetrievalResult` candidate lists).
4. `src/adaptive_framework/rag/reranking/__init__.py` & `src/adaptive_framework/rag/__init__.py`:
   - Exported `AdaptiveCandidateDepthPolicy`.
5. `scripts/optimize_reranking.py`:
   - Automated benchmark runner with cold-start separation, 3 repeated warm runs, candidate recall analysis, and JSON serialization.

---

## 10. Regression Testing and Verification

- Unit Tests (`tests/unit/rag/test_reranking_optimization.py`):
  - 11 tests covering candidate depth overrides, batch size overrides, model warmup, pre-reranking signal calculation, confidence routing, and ground-truth isolation.
  - All 11 tests passed in 7.01s.
- RAG Test Suite:
  - `pytest tests/unit/rag/ tests/integration/rag/`: **324 passed**, 0 failed.
- Full Unit Test Suite:
  - `pytest tests/unit/`: **996 passed**, 0 failed.
- Baseline Preservation:
  - Phase 4.7 (`dense_baseline_summary.json`), Phase 4.8 (`hybrid_summary.json`), and Phase 4.9 (`reranked_summary.json`) baseline files verified untouched.

---

## 11. Artifacts Generated

All evaluation artifacts are preserved in `outputs/rag/evaluation/`:
- `phase4_10_candidate_depth_results.json`: Full metric breakdown across depths {5, 10, 15, 20}.
- `phase4_10_batch_size_results.json`: Latency and accuracy across batch sizes {4, 8, 16, 32}.
- `phase4_10_comparison_summary.json`: End-to-end benchmark comparison and recommendation.
