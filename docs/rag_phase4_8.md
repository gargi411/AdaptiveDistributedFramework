# Phase 4.8 -- Hybrid Dense + Sparse Retrieval with Reciprocal Rank Fusion

This document details the architecture, implementation, mathematical foundation, configuration, and empirical evaluation of the Hybrid Retrieval subsystem in the Adaptive Distributed Framework (v2.0).

---

## 1. Motivation

Phase 4.7 established the baseline evaluation of the pure dense FAISS retrieval engine over 50 synthetic Indian clinical discharge summaries against 25 ground-truth clinical queries. That evaluation revealed:
- Extremely fast execution: Mean latency = 0.261 ms.
- Low ranking effectiveness: Precision@5 = 0.0080, Recall@5 = 0.0400, MRR@5 = 0.0080, nDCG@5 = 0.0155, Hit Rate@5 = 0.0400.

Clinical queries frequently hinge on exact lexical tokens (e.g. drug brand names, specific dosages, diagnostic codes, patient identifiers, abbreviations). In contrast, dense semantic embeddings project text into continuous semantic manifolds where narrative clinical discharge summaries overlap densely, diluting exact keyword signals.

The objective of Phase 4.8 is to implement **Hybrid Retrieval** by combining:
1. Dense semantic search (FAISS + BAAI/bge-large-en-v1.5)
2. Sparse lexical search (BM25Okapi with clinical tokenization)
3. Reciprocal Rank Fusion (RRF) for non-parametric rank combination

---

## 2. Synthetic Dataset Disclaimer

The evaluation benchmark is executed over:
`dataset/Synthetic Indian Clinical Notes for Natural Langua/synthetic_notes_medium.json`

**DISCLAIMER**:
1. All clinical records in this dataset are purely synthetic and artificially generated for computational research and benchmarking purposes.
2. The records do NOT represent real human patients, confidential personal health information (PHI), or actual clinical encounters.
3. No clinical validity, diagnostic efficacy, medical treatment guidance, or therapeutic safety is claimed or implied by these benchmarks or this framework.

---

## 3. Architecture Overview

Hybrid retrieval operates as an experimental, decoupled retrieval engine implementing `IHybridRetriever`, which directly extends `IRetrievalEngine`:

```
                             Natural Language Query
                                       |
                +----------------------+----------------------+
                |                                             |
                v                                             v
       Dense Retrieval Layer                         Sparse Retrieval Layer
     (Existing RetrievalEngine)                          (BM25Retriever)
     BAAI/bge-large-en-v1.5                                BM25Okapi
                |                                             |
                v                                             v
       Dense Top-K Candidates                        Sparse Top-K Candidates
     (scores: cosine similarity)                       (scores: BM25 score)
                |                                             |
                +----------------------+----------------------+
                                       |
                                       v
                           Reciprocal Rank Fusion
                          RRF(d) = sum(1 / (k + r))
                                       |
                                       v
                              Chunk Deduplication
                                       |
                                       v
                            HybridRetrievalResult[]
                           (fused scores & rankings)
                                       |
                                       v
                           Downstream RAG Pipeline
                    (ContextBuilder -> PromptBuilder -> LLM)
```

The dense `RetrievalEngine` is NOT modified or replaced. The baseline remains 100% untouched and reproducible.

---

## 4. Sparse Retrieval (BM25Okapi) Implementation

### 4.1 Deterministic Clinical Tokenizer
Implemented in `adaptive_framework.rag.retrieval.bm25_retriever.tokenize_clinical_text`:
- Converts text to lowercase.
- Normalizes punctuation and extracts alphanumeric tokens (`re.findall(r"[a-z0-9]+", text.lower())`).
- Preserves clinically meaningful numeric values, dosages (e.g. `500mg`, `20mg`), diagnostic abbreviations (`t2dm`, `cad`, `stemi`), and lab values.
- Avoids aggressive stopword filtering, preventing distortion of medical phrases (e.g. "no evidence of infarction").

### 4.2 BM25Okapi Scoring Function
For a query $Q = \{q_1, \dots, q_m\}$ and document chunk $D$:
$$\text{BM25}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

To avoid negative IDF for ubiquitous terms, the Robertson-Spärck Jones IDF is computed with +1 smoothing:
$$\text{IDF}(q) = \ln\left(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5}\right)$$

Default baseline parameters:
- $k_1 = 1.5$ (term frequency saturation)
- $b = 0.75$ (document length normalization)

### 4.3 BM25 Index Persistence
The index and full chunk metadata are serialized to JSON in `outputs/rag/index/bm25_index.json` using `BM25Retriever.save()` and `BM25Retriever.load()`, preserving zero external database dependencies.

---

## 5. Reciprocal Rank Fusion (RRF)

Dense cosine similarity and BM25 lexical scores reside in fundamentally different metric spaces with incomparable dynamic ranges. Reciprocal Rank Fusion combines results strictly using positional ranks:

For each retrieved chunk $d$:
$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{\text{rrf\_k} + r_m(d)}$$
where $r_m(d)$ is the 1-based rank in retriever $m$, and $\text{rrf\_k} = 60$ (default smoothing constant).

If chunk $d$ is retrieved by only one retriever, only its single rank contributes to $\text{RRF}(d)$. If $d$ appears in both, both reciprocal ranks are summed. Unique chunks are deduplicated by `chunk_id`, sorted descending by RRF score, and assigned new 1-based ranks.

---

## 6. Configuration Management

Configured in `configs/rag.yaml`:
```yaml
rag:
  retrieval:
    strategy: "dense"  # "dense" (default) | "hybrid"
    top_k: 5
    max_top_k: 50
    similarity_metric: "cosine"
    min_score_threshold: 0.0

    hybrid:
      enabled: false
      dense_top_k: 20
      sparse_top_k: 20
      final_top_k: 5
      rrf_k: 60

    bm25:
      k1: 1.5
      b: 0.75
      index_path: "outputs/rag/index"
```

The default strategy remains `dense`, preserving the Phase 4.7 baseline. Hybrid retrieval is activated dynamically via `create_retrieval_engine(strategy="hybrid", ...)` or `configs/rag.yaml`.

---

## 7. Empirical Evaluation and Benchmark Results

The benchmark was executed using `scripts/evaluate_rag.py` over 50 synthetic discharge summaries and 25 evaluation queries.

### 7.1 Dense FAISS Baseline (Phase 4.7 Baseline)

```
DOCUMENT-LEVEL METRICS:
Metric          @1       @3       @5
------------------------------------
Precision     0.0000   0.0000   0.0080
Recall        0.0000   0.0000   0.0400
MRR           0.0000   0.0000   0.0080
nDCG          0.0000   0.0000   0.0155
Hit Rate      0.0000   0.0000   0.0400

CHUNK-LEVEL METRICS:
Metric          @1       @3       @5
------------------------------------
Precision     0.0000   0.0000   0.0080
Recall        0.0000   0.0000   0.0400
MRR           0.0000   0.0000   0.0080
nDCG          0.0000   0.0000   0.0155

LATENCY (FAISS engine only, excluding LLM):
- Mean   : 0.261 ms
- Median : 0.212 ms
- P95    : 0.638 ms
```

### 7.2 Hybrid Retrieval (Dense FAISS + BM25 + RRF)

```
DOCUMENT-LEVEL METRICS:
Metric          @1       @3       @5
------------------------------------
Precision     0.1200   0.0800   0.0560
Recall        0.1000   0.1933   0.2333
MRR           0.1200   0.1667   0.1747
nDCG          0.1200   0.1564   0.1718
Hit Rate      0.1200   0.2400   0.2800

CHUNK-LEVEL METRICS:
Metric          @1       @3       @5
------------------------------------
Precision     0.1200   0.0800   0.0560
Recall        0.1200   0.2133   0.2533
MRR           0.1200   0.1667   0.1747
nDCG          0.1200   0.1718   0.1873

LATENCY (Hybrid pipeline: FAISS + BM25 + RRF):
- Mean   : 0.853 ms
- Median : 0.406 ms
- P95    : 0.617 ms
```

### 7.3 Quantitative Comparison (Hybrid vs Dense Baseline)

```
Metric                Dense Baseline   Hybrid (BM25+RRF)   Delta (Absolute / Relative)
--------------------------------------------------------------------------------------
Precision@1               0.0000            0.1200          +0.1200 (baseline=0)
Precision@3               0.0000            0.0800          +0.0800 (baseline=0)
Precision@5               0.0080            0.0560          +0.0480 (+600.0%)
Recall@1                  0.0000            0.1000          +0.1000 (baseline=0)
Recall@3                  0.0000            0.1933          +0.1933 (baseline=0)
Recall@5                  0.0400            0.2333          +0.1933 (+483.3%)
MRR@5                     0.0080            0.1747          +0.1667 (+2083.3%)
nDCG@5                    0.0155            0.1718          +0.1564 (+1010.5%)
Hit Rate@5                0.0400            0.2800          +0.2400 (+600.0%)
Mean Latency              0.261 ms          0.853 ms        +0.592 ms (2.27x overhead)
```

Query Outcomes across 25 queries (at K=5):
- **Improved (Dense Fail -> Hybrid Hit)**: 6 queries
- **Degraded (Dense Hit -> Hybrid Fail)**: 0 queries
- **Unchanged**: 19 queries (18 both fail, 1 both succeed)

---

## 8. Failure Analysis

Representative query case analysis from `outputs/rag/evaluation/comparison_summary.json`:

### 8.1 Dense Fails -> Hybrid Hits (BM25 Disambiguation)
1. **Case `eval_006`**:
   - Query: *"What diagnosis and CRP investigation results were documented for 45-year-old Ananya Banerjee?"*
   - Expected Doc: `clinical_note_0004`
   - Dense Top-5: `[clinical_note_0025, 0024, 0039, 0011, 0040]` (Miss)
   - Hybrid Top-5: `[clinical_note_0011, 0006, clinical_note_0004, 0047, 0016]` (Hit at rank 3)
   - Reason: Exact patient name ("Ananya Banerjee") and investigation token ("CRP") matched strongly in BM25, pulling the relevant document into the top 3 through RRF.

2. **Case `eval_012`**:
   - Query: *"What PET scan and CBC findings were documented for Suresh Iyer with breast carcinoma?"*
   - Expected Doc: `clinical_note_0009`
   - Dense Top-5: `[clinical_note_0046, 0001, 0032, 0024, 0002]` (Miss)
   - Hybrid Top-5: `[clinical_note_0009, 0046, 0024, 0040, 0010]` (Hit at rank 1)
   - Reason: The combination of "Suresh Iyer" and "breast carcinoma" produced high BM25 term weighting that dense vector cosine similarity had averaged out.

### 8.2 Both Fail (Remaining Semantic Gaps)
1. **Case `eval_001`**:
   - Query: *"What diagnosis is documented for patient Soumya Das?"*
   - Expected Doc: `clinical_note_0001`
   - Dense Top-5: `[clinical_note_0044, 0036, 0046, 0030, 0004]` (Miss)
   - Hybrid Top-5: `[clinical_note_0030, 0046, 0044, 0047, 0010]` (Miss)
   - Reason: The query lacks specific clinical terms beyond the generic word "diagnosis", causing both dense and lexical retrievers to return documents with similar generic headers.

---

## 9. Latency and Throughput Trade-off

Hybrid retrieval requires executing both FAISS vector search and BM25 token scoring, followed by rank fusion:
- Dense latency: 0.261 ms mean
- Hybrid latency: 0.853 ms mean
- Delta: +0.592 ms

While hybrid retrieval increases latency by ~0.59 ms, the overall execution duration remains well under 1 millisecond (P95 = 0.617 ms). This minor overhead provides a nearly 5x gain in Recall@5 and a 21x gain in MRR@5.

---

## 10. Limitations & Future Directions

1. **Vocabulary Mismatch**: Synonyms (e.g. "myocardial infarction" vs "heart attack") cannot be resolved by BM25 alone.
2. **Whole-Chunk Lexical Dilution**: Whole discharge summaries still disperse keyword weights across long documents.
3. **No Cross-Attention**: RRF relies purely on rank combination without cross-attention between query and chunk tokens.
4. **Future Work**: Subsequent phases will evaluate cross-encoder reranking (e.g. `bge-reranker-base`) on the top 20 hybrid candidates to further sharpen top-1 and top-3 accuracy.
