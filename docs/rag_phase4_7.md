# Phase 4.7 -- RAG Retrieval Evaluation and Benchmarking

This document details the quantitative evaluation framework, metric definitions, dataset methodology, baseline benchmark results, and performance analysis for the Retrieval-Augmented Generation (RAG) subsystem of the Adaptive Distributed Framework (v2.0).

---

## 1. System Overview and Evaluation Purpose

Following the integration and live validation of the real LLM provider in Phase 4.6, the objective of Phase 4.7 is to quantitatively evaluate the retrieval subsystem in isolation before any generation takes place. 

In a RAG architecture:
```
Doctor/User Question
        |
        v
  [Query Embedding]  <-- BAAI/bge-small-en-v1.5
        |
        v
 [FAISS Index Search] <-- Cosine / Inner Product Search
        |
        v
  [Top-K Evidence]   <-- EVALUATED IN PHASE 4.7
        |
        v
 [Context & Prompt]
        |
        v
     [Real LLM]      <-- Evaluated separately / Phase 4.6
```

If the retrieval engine returns irrelevant or misranked evidence, the downstream LLM either hallucinates or fails to produce grounded answers. Phase 4.7 establishes an objective, mathematically rigorous baseline for retrieval quality and latency.

---

## 2. Dataset Methodology and Synthetic Data Disclaimer

### 2.1 Synthetic Dataset Disclaimer
The evaluation corpus uses clinical records sourced from:
`dataset/Synthetic Indian Clinical Notes for Natural Langua/synthetic_notes_medium.json`

**DISCLAIMER**:
1. All clinical records in this dataset are purely synthetic and artificially generated for computational research and benchmarking purposes.
2. The records do NOT represent real human patients, confidential personal health information (PHI), or actual clinical encounters.
3. No clinical validity, diagnostic efficacy, medical treatment guidance, or therapeutic safety is claimed or implied by these benchmarks or this framework.

### 2.2 Ground-Truth Evaluation Dataset
The evaluation dataset is implemented in `adaptive_framework.rag.evaluation.dataset` and contains 25 distinct clinical query cases. Each case specifies:
- `query_id`: Unique identifier (e.g., `synth_eval_001`).
- `query`: Clinical information retrieval question (e.g., patient discharge medications, presenting symptoms, lab values).
- `expected_document_ids`: Canonical document ID(s) containing the relevant facts.
- `expected_chunk_ids`: Canonical chunk ID(s) containing the relevant facts.
- `relevant_metadata`: Clinical metadata including `synthetic_case_id`, `specialty`, and `ground_truth_excerpt`.

All 25 ground-truth cases are directly grounded in the first 50 synthetic discharge summaries of the corpus (`clinical_note_0001` through `clinical_note_0050`). No ground-truth queries leak test answers into the retrieval engine during indexing.

---

## 3. Mathematical Metric Definitions

The evaluation engine computes standard Information Retrieval (IR) metrics at cutoffs K = 1, 3, 5:

### 3.1 Precision@K
The fraction of retrieved items in top-K that are relevant:
$$\text{Precision}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$

In accordance with standard IR convention, the denominator is strictly K (not the number of returned results if fewer than K).

### 3.2 Recall@K
The fraction of all known relevant items that are retrieved in top-K:
$$\text{Recall}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$
If the ground truth set of relevant items is empty, Recall@K is defined as 0.0.

### 3.3 Mean Reciprocal Rank (MRR@K)
Evaluates the position of the first relevant item in the retrieved top-K ranking:
$$\text{RR}@K = \begin{cases} \frac{1}{\text{rank}_i} & \text{if first relevant item is at rank } 1 \le \text{rank}_i \le K \\ 0 & \text{otherwise} \end{cases}$$
$$\text{MRR}@K = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}_q@K$$

### 3.4 Normalized Discounted Cumulative Gain (nDCG@K)
Evaluates ranking quality, giving higher credit when relevant items appear near the top of the result list:
$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
where $\text{rel}_i \in \{0, 1\}$ for binary relevance.

$$\text{IDCG}@K = \sum_{i=1}^{\min(K, |\text{Relevant}|)} \frac{1}{\log_2(i + 1)}$$
$$\text{nDCG}@K = \begin{cases} \frac{\text{DCG}@K}{\text{IDCG}@K} & \text{if } \text{IDCG}@K > 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## 4. Latency Measurement Methodology

Retrieval latency measures exclusively the wall-clock execution time of the retrieval engine:
1. Generation of query embedding via BGE (`bge-small-en-v1.5` or deterministic mock in test harness).
2. FAISS vector similarity search across indexed chunk vectors.
3. Candidate chunk resolution and metadata enrichment.

LLM prompt generation, API network requests, and text synthesis are strictly excluded from this measurement.

The engine collects high-resolution timings using `time.perf_counter()` across all 25 queries and calculates:
- Mean latency
- Median latency
- Minimum latency
- Maximum latency
- 95th Percentile (P95) latency

---

## 5. Baseline Benchmark Results

The baseline evaluation was run using `scripts/evaluate_rag.py` on the 50-document synthetic corpus with 25 evaluation queries.

### 5.1 Document-Level Metrics

| Metric | @1 | @3 | @5 |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.0000 | 0.0000 | 0.0160 |
| **Recall** | 0.0000 | 0.0000 | 0.0800 |
| **MRR** | 0.0000 | 0.0000 | 0.0160 |
| **nDCG** | 0.0000 | 0.0000 | 0.0309 |
| **Hit Rate** | 0.0000 | 0.0000 | 0.0800 |

### 5.2 Chunk-Level Metrics

| Metric | @1 | @3 | @5 |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.0000 | 0.0000 | 0.0160 |
| **Recall** | 0.0000 | 0.0000 | 0.0800 |
| **MRR** | 0.0000 | 0.0000 | 0.0160 |
| **nDCG** | 0.0000 | 0.0000 | 0.0309 |

### 5.3 Retrieval Latency (FAISS Engine Only, excluding LLM)

| Latency Metric | Measured Duration (ms) |
| :--- | :---: |
| **Mean** | 0.257 ms |
| **Median** | 0.219 ms |
| **Min** | 0.197 ms |
| **Max** | 0.551 ms |
| **P95** | 0.392 ms |

All benchmark results are automatically exported to:
- Summary: `outputs/rag/evaluation/evaluation_summary.json`
- Full Details: `outputs/rag/evaluation/evaluation_results.json`

---

## 6. Diagnostic Analysis of Baseline Results

The baseline retrieval results demonstrate sub-millisecond retrieval execution speed (P95 = 0.392 ms) but low ranking precision and recall at K=1 and K=3, with signal appearing at K=5 (Recall@5 = 0.0800, Hit Rate@5 = 0.0800).

Key diagnostic factors:
1. **Pure Dense Retrieval Limitation**: The baseline retrieval engine relies entirely on dense inner-product cosine similarity over 384-dimensional dense vectors without any sparse keyword matching (e.g., BM25). Clinical queries frequently depend on exact domain tokens, specific drug names, dosage amounts, and diagnostic acronyms that general semantic embeddings disperse across high-dimensional semantic space.
2. **Chunk Granularity and Lexical Dilution**: Whole discharge summaries or large semantic chunks contain substantial auxiliary clinical text (admission histories, family histories, review of systems). When vectorized as a single embedding, specific isolated facts (e.g., a single medication dosage) suffer from representation dilution.
3. **Absence of Reranking**: Current Top-K candidates are selected directly from FAISS approximate cosine similarity without a cross-encoder reranker to perform query-chunk token-level cross-attention.

---

## 7. Future Optimization Recommendations

To elevate retrieval precision and recall in subsequent iterations while retaining sub-millisecond throughput:
1. **Hybrid Dense-Sparse Search**: Combine dense vector retrieval (BGE) with sparse lexical retrieval (BM25 or SPLADE) using Reciprocal Rank Fusion (RRF).
2. **Cross-Encoder Reranking**: Retrieve top 20 candidates via fast FAISS, followed by a lightweight cross-encoder (e.g., `bge-reranker-base`) to rescore the top 5 candidates.
3. **Hierarchical / Sub-Chunk Indexing**: Split clinical discharge summaries into section-level sub-chunks (e.g., "Discharge Medications", "Hospital Course", "Laboratory Findings") linked to parent documents.
4. **Domain-Specific Query Expansion**: Automatically expand clinical queries with synonymous medical terminology (e.g., UMLS / MeSH concepts) prior to vectorization.
