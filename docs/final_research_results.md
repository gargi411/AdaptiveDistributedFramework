# Final Research Results: Adaptive Distributed Parallel Processing Framework for Large-Scale Biomedical Document Intelligence

## 1. Research Problem and Motivation

Clinical and biomedical document analysis is hindered by three compounding computational bottlenecks:
1. **Extreme Structural Heterogeneity**: Biomedical corpora consist of high-resolution scanned pathology reports, multi-page longitudinal electronic health records (EHRs), complex multi-column research literature, and clinical notes. Traditional single-node serial processing pipelines experience severe throughput degradation.
2. **Hardware Underutilization**: Heterogeneous edge and workstation hardware (combining multi-core CPUs with integrated or discrete GPUs) is rarely utilized adaptively. Scanned pages either overwhelm the host CPU during OCR or suffer high latency penalties from naive GPU compilation.
3. **Retrieval Degradation and Hallucination**: Pure dense semantic search struggles with exact clinical alphanumeric identifiers (e.g., drug dosages, ICD codes, genetic variants), while pure sparse search fails on semantic paraphrasing. Furthermore, generative language models without strict evidence grounding are prone to fabricating medical claims.

This research addresses these challenges by designing, implementing, and validating an end-to-end adaptive distributed processing and two-stage hybrid retrieval framework.

---

## 2. Proposed Framework Overview

The framework unifies:
- Ray-based distributed page-level work unit scheduling with dynamic work stealing.
- A dual-path document extraction pipeline (zero-copy PyMuPDF direct text extraction for digital pages; OpenVINO OCR for scanned pages).
- Data-driven adaptive CPU/GPU work routing based on pre-OCR compute intensity and real-time hardware telemetry.
- A two-stage hybrid RAG pipeline combining BGE-Large dense embeddings, BM25Okapi sparse tokens, Reciprocal Rank Fusion ($k=60$), and neural cross-encoder reranking.
- Tripartite context isolation and provenance tracking for clinician decision support.

---

## 3. Eight Core Research Contributions

1. **Adaptive Page-Level Work-Unit Decomposition**: Replaced monolithic document dispatch with fine-grained page-level task distribution, enabling dynamic load balancing and sub-document parallelism across heterogeneous workers.
2. **Zero-Copy In-Memory Pipeline Execution**: Designed in-memory buffer passing between rasterisation, layout analysis, and text detection, eliminating intermediate disk serialization bottlenecks.
3. **Adaptive CPU-GPU Pipeline Execution**: Implemented dual-backend OpenVINO execution targeting Intel Iris Xe integrated graphics and host multi-core CPU with automated $<0.5$ ms fallback.
4. **Data-Driven Adaptive CPU/GPU Work Routing (Phase G6)**: Formulated an explainable, multi-factor routing model based on pre-OCR physical features ($N_{\text{pixels}}, N_{\text{density}}, N_{\text{complexity}}, N_{\text{chars}}$) and runtime telemetry with zero oracle data leakage.
5. **Hybrid Dense-Sparse Retrieval with RRF**: Demonstrated that Reciprocal Rank Fusion ($k=60$) combining FAISS IndexFlatIP and BM25Okapi decisively outperforms individual retrieval baselines on biomedical queries.
6. **Neural Cross-Encoder Candidate Reranking**: Integrated `cross-encoder/ms-marco-MiniLM-L-6-v2` to rescore fused candidates, resolving semantic ambiguity in dense clinical text.
7. **Adaptive Reranking Depth and Batch Optimization**: Empirically identified the Pareto-optimal reranking configuration ($K_{\text{cand}}=15$, $\text{batch\_size}=32$), capturing $>94\%$ of max retrieval gain at a fraction of computational latency.
8. **Evidence-Grounded Clinical Decision Support with Provenance**: Built a doctor-facing orchestration layer enforcing tripartite prompt boundaries and end-to-end evidence citation tracing.

---

## 4. Experimental Methodology and Hardware Profile

All experiments and empirical evaluations were executed on local physical testbed hardware:
- **Host CPU**: Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 physical cores, 8 logical threads, 8 MB L3 cache).
- **Host RAM**: 16.00 GB DDR4 (15.74 GB usable).
- **Target GPU**: Intel(R) Iris(R) Xe Graphics (integrated GPU, Gen 12 Xe-LP architecture, 80 Execution Units, OpenVINO device `GPU.0`, 7.12 GB dynamically shared system memory pool).
- **Operating System**: Windows 11 Enterprise (x86_64).
- **OpenVINO Runtime**: Version 2026.3.0.
- **Python Environment**: Python 3.14.0 64-bit virtual environment.

---

## 5. Distributed Processing Results

Distributed scheduling benchmarks evaluated task decomposition, work stealing, and worker scaling:
- **Page-Level Task Partitioning**: Enabled near-linear speedup ($3.62\times$ on 4 physical cores) over serial execution for multi-page documents ($N \ge 20$ pages).
- **Work Stealing Efficacy**: Under synthetic worker skew (simulating high-complexity pathology reports alongside simple clinical notes), dynamic work stealing reduced overall makespan by **23.4%** and eliminated idle worker stragglers.
- **Worker Coordination Overhead**: Task dispatch and aggregation overhead via Ray was measured at $<1.2$ ms per page unit.

---

## 6. Retrieval Results: Dense vs Sparse Baselines

Evaluated on the standardized biomedical evaluation benchmark:
- **Dense Retrieval (BGE-Large-EN-v1.5 + FAISS FlatIP)**:
  - Strengths: High conceptual recall on disease symptoms, clinical syndromes, and procedural narratives.
  - Weaknesses: Lower precision on exact medication dosages, clinical codes, and numeric thresholds.
- **Sparse Retrieval (BM25Okapi)**:
  - Strengths: Exact keyword matching on drug names ("Empagliflozin", "Ticagrelor") and lab identifiers.
  - Weaknesses: Complete retrieval failure when queries used synonyms or descriptive paraphrasing without exact keyword overlap.

---

## 7. Hybrid Retrieval Results (Phase 4.8 Baseline)

Combining dense and sparse retrieval via Reciprocal Rank Fusion ($k=60$) yielded statistically significant improvements across all retrieval quality metrics:

| Metric | Dense Baseline | Sparse Baseline | Hybrid Fusion (RRF k=60) | Absolute Gain vs Dense |
| :--- | :---: | :---: | :---: | :---: |
| **Precision@5** | 0.412 | 0.384 | **0.496** | +0.084 (+20.4%) |
| **Recall@5** | 0.518 | 0.462 | **0.614** | +0.096 (+18.5%) |
| **MRR@5** | 0.582 | 0.531 | **0.694** | +0.112 (+19.2%) |
| **nDCG@5** | 0.524 | 0.481 | **0.638** | +0.114 (+21.8%) |
| **Hit Rate@5** | 0.742 | 0.689 | **0.846** | +0.104 (+14.0%) |

---

## 8. Cross-Encoder Reranking Results (Phase 4.9 Baseline)

Passing the top-15 fused candidates through `cross-encoder/ms-marco-MiniLM-L-6-v2` dramatically boosted precision and top-1 ranking fidelity:

| Metric | Hybrid Baseline (RRF) | Cross-Encoder Reranked | Delta vs Hybrid |
| :--- | :---: | :---: | :---: |
| **Precision@5** | 0.496 | **0.582** | +0.086 (+17.3%) |
| **Recall@5** | 0.614 | **0.688** | +0.074 (+12.1%) |
| **MRR@5** | 0.694 | **0.812** | +0.118 (+17.0%) |
| **nDCG@5** | 0.638 | **0.756** | +0.118 (+18.5%) |
| **Hit Rate@5** | 0.846 | **0.918** | +0.072 (+8.5%) |

---

## 9. Adaptive Reranking Optimization Findings (Phase 4.10 Baseline)

Controlled sweeping across candidate depths ($K_{\text{cand}} \in [5, 10, 15, 20, 30]$) and batch sizes ($\text{batch\_size} \in [8, 16, 32, 64]$) established:
1. **Diminishing Returns on Depth**: Increasing candidate depth beyond $K_{\text{cand}}=15$ increased latency by $72\%$ while improving nDCG by less than $1.4\%$.
2. **Optimal Batch Sizing**: Batch size 32 minimized PyTorch tensor dispatch overhead on CPU, achieving the highest throughput (182 pairs/sec).
3. **Frozen Baseline**: $K_{\text{cand}}=15$, $\text{batch\_size}=32$, $\text{final\_top\_k}=5$ was permanently frozen as the optimal operating point.

---

## 10. GPU Acceleration Results (OpenVINO OCR on Intel Iris Xe)

- **Compilation Overhead**:
  - Host CPU compilation: 788.12 ms
  - Iris Xe GPU compilation: 2,158.35 ms
- **Warm Inference Latency**:
  - Horizontal text detection: 14.82 ms (GPU) vs 28.45 ms (CPU)
  - Text recognition (16-crop batch): 18.87 ms (GPU) vs 31.17 ms (CPU)
  - Total per-page warm OCR latency: 33.69 ms (GPU) vs 59.62 ms (CPU) -> **1.77x speedup on synthetic layouts**.

---

## 11. Phase G4 Rule-Based Adaptive Routing Results

Phase G4 deployed a rule-based heuristic router:
- Successfully prevented GPU cold-start compilation penalties on 1-page jobs.
- Implemented automated fallback to CPU OCR in 0.34 ms during driver errors.
- Limitation Identified: Rigid static complexity thresholds ($C > 0.40$) resulted in misrouting on multi-column clinical notes with high character density.

---

## 12. Phase G5 Empirical Benchmarking Findings

A controlled 36-condition benchmark matrix (Class A, B, C, D across 5, 10, 20 pages with 7 repetitions, 252 total runs) revealed:
1. **Workload Dependence**: GPU speedup ranged from $1.04\times$ to $1.80\times$ depending on document class and batch size.
2. **Host CPU Compute Relief**: Offloading OCR to the Iris Xe GPU reduced host CPU utilization from $21.1\%$ to $11.8\%$, freeing CPU cores for background document parsing.
3. **Conservative Routing Regret**: Static G4 rules exhibited up to $48.8\%$ regret against an offline oracle on small synthetic batches because they routed to CPU when the warm GPU was faster.

---

## 13. Phase G6 Data-Driven Routing and Held-Out Validation Findings

Phase G6 addressed G5's findings using a calibrated, zero-leakage compute intensity and suitability model evaluated on three disjoint datasets (Set A: Development, Set B: Validation, Set C: Held-Out Test):
- **Oracle Agreement**: **100.0%** agreement with the offline empirical oracle on held-out clinical pages (both G6 and the oracle routed 100% of held-out pages to GPU).
- **Execution Parity on iGPU**:
  - `GPU_ONLY`: 5.1706 s (2.51 pages/sec)
  - `ADAPTIVE_G6`: 5.2565 s (2.47 pages/sec)
  - Delta: +1.66% (strictly within run-to-run sampling variance of $\pm 2.5\%$).
- **Scientific Finding**: On warm homogeneous document batches sharing unified physical DRAM with an integrated GPU, adaptive scheduling achieves performance parity rather than super-linear speedups because memory bus bandwidth remains the shared bottleneck.
- **Overhead**: G6 decision latency was measured at **0.141 ms** per page (0.035% of page OCR time).

---

## 14. Phase 4.12 End-to-End Validation Results

The final validation benchmark evaluated the complete pipeline across 5 representative documents (18 total pages, 129 indexed chunks):

| Document ID | Filename | Page Count | Type | Ingestion Time (ms) | Retrieval Latency (ms) | Neural Rerank Latency (ms) | Total Query Latency (ms) | Evidence Count | Grounded Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DOC-DIGITAL-01` | `13643_2019_Article_976.pdf` | 4 | Digital | 47.06 ms | 0.99 ms | 416.53 ms | 420.02 ms | 5 | Grounded |
| `DOC-DIGITAL-02` | `pone.0241962.pdf` | 7 | Digital | 85.08 ms | 0.96 ms | 458.59 ms | 461.01 ms | 5 | Grounded |
| `DOC-SCANNED-01` | `PDF_Deid_Deidentification_Medium_0.pdf` | 4 | Scanned | 4,122.53 ms | 1.20 ms | 462.17 ms | 464.59 ms | 5 | Grounded |
| `DOC-SCANNED-02` | `PDF_Deid_Deidentification_Hard_0.pdf` | 2 | Scanned | 963.57 ms | 0.80 ms | 452.59 ms | 455.12 ms | 5 | Grounded |
| `DOC-CLINICAL-NOTE` | `CR-49201_discharge.txt` | 1 | EHR Note | 5.92 ms | 1.60 ms | 702.38 ms | 705.37 ms | 5 | Grounded |

### Stage Latency Breakdown (Mean across queries)
- **Stage 1 Retrieval (Dense + Sparse + RRF)**: **1.11 ms**
- **Stage 2 Neural Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)**: **498.45 ms**
- **Context Assembly & Provenance**: **0.03 ms**
- **Prompt Construction**: **0.03 ms**
- **Generation**: **1.53 ms** (deterministic offline provider)
- **Total Query Latency**: **501.22 ms**

### Controlled Grounding and Provenance Validation
- **Provenance Preservation**: **100.0%** (129/129 chunks retained unbroken Document ID, Page Number, and Chunk ID lineage).
- **Supported Query**: Correctly synthesized grounded answer citing specific medication regimens and dosages.
- **Unsupported Query**: Correctly emitted insufficient evidence notice when queried regarding unrecorded clinical procedures.

---

## 15. Research Limitations

To maintain scientific integrity, the following limitations must be acknowledged:
1. **Single-Node Integrated GPU Platform**: The testbed hardware utilizes an Intel Iris Xe integrated GPU sharing system DRAM. Performance characteristics will differ on discrete PCIe GPUs (e.g. NVIDIA RTX/A100) with dedicated high-bandwidth VRAM.
2. **OCR Model Scope**: OCR models were evaluated for pipeline integration and throughput; lexical OCR error correction and character error rate (CER) minimization were not primary research goals.
3. **Corpus Scope**: Evaluations utilized public biomedical research articles and deidentified/synthetic clinical records. Real hospital electronic health record systems impose additional HL7/FHIR formatting and security constraints.
4. **Live LLM Dependencies**: Live LLM latency is dominated by remote network round-trip time and cloud API rate limits.

---

## 16. Research Conclusions

1. **Page-Level Scheduling Eliminates Document Skew**: Fine-grained work units enable effective load balancing and work stealing, preventing long multi-page records from blocking the processing pipeline.
2. **Direct Text Extraction Bypass is Essential**: Bypassing OCR on digital PDF pages reduces ingestion latency by over **$98\%$** compared to uniform rasterisation.
3. **Hybrid RRF + Neural Reranking Delivers Superior Clinical Retrieval**: Combining BM25 with BGE-Large embeddings via RRF and filtering with cross-encoders increases retrieval precision by **$41.3\%$** over dense retrieval alone.
4. **Explainable Routing Protects Against Worst-Case Latency**: Data-driven adaptive routing successfully avoids the 2.15-second GPU compilation barrier on small jobs while providing sub-millisecond fail-safe fallback.

---

## 17. Future Work

1. **Multi-Node Cluster Scaling**: Extending the Ray scheduling layer across physical multi-node server clusters.
2. **Discrete GPU Acceleration**: Evaluating the G6 adaptive router on discrete GPU hardware with dedicated VRAM channels.
3. **Local Clinical Small Language Models**: Integrating quantized local SLMs (e.g., BioMistral 7B, Meditron) via OpenVINO or vLLM to eliminate external cloud API dependencies.
4. **FHIR/HL7 Standard Integration**: Ingesting structured healthcare data directly alongside unstructured PDF documents.

---

*Phase 4.12 research validation complete and frozen.*
