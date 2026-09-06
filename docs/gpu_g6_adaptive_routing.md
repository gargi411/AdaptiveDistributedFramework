# Phase G6: Data-Driven Adaptive CPU/GPU Work Routing

**Project**: Adaptive Distributed Parallel Processing Framework for Large-Scale Biomedical Document Intelligence  
**Phase**: G6 — Data-Driven Adaptive CPU/GPU Work Routing  
**Status**: **IMPLEMENTATION & EVALUATION COMPLETE**  
**Date**: September 2026  
**Environment**: Windows 11, Intel Core i5-1135G7 @ 2.40GHz, Intel Iris Xe Graphics (iGPU), OpenVINO 2026.3.0  

---

## 1. Motivation from Phase G5

Phase G5 demonstrated workload-dependent CPU/GPU performance characteristics and identified limitations in the initial rule-based adaptive routing policy. Specifically, on the evaluated Intel Iris Xe architecture:
- On small, low-complexity pages (`CLASS_A`), GPU outperformed CPU despite light compute volume.
- On standard clinical documents (`CLASS_B`), execution was mixed: CPU was faster for small 5-page batches (due to dispatch/memory staging overhead), while GPU surpassed CPU at 10 and 20 pages.
- On dense biomedical layouts (`CLASS_C`), CPU slightly outperformed GPU across all tested batch sizes because sequential token recognition on complex textual tables was bottlenecked by integrated EU execution.
- G4's static rule-based heuristic ($C \ge 0.35 \rightarrow \text{GPU}$) relied solely on structural layout complexity, ignoring total pixel volume, cold-start compilation penalties ($2.16\text{s}$), and dynamic queue congestion.

---

## 2. Research Question

> *Can a runtime-only, data-driven routing policy make better CPU/GPU decisions than the initial rule-based policy while generalizing to previously unseen workloads, without accessing ground-truth labels or future execution timings?*

---

## 3. G6 System Architecture

The G6 architecture decouples workload characterization, cost estimation, suitability evaluation, and execution fallback into distinct modular components:

```text
Incoming Page / Batch
        |
        v
[WorkloadCharacterizer]
        |
        +---> WorkloadProfile (dimensions, text chars, image density, complexity)
        |
        v
[WorkloadCostEstimator] (workload_cost_model.py)
        |
        +---> WorkloadCostProfile (normalized compute intensity, total pixels, work units)
        |
        v
[BackendSuitabilityEstimator] (backend_suitability.py)
        |  <--- ResourceSnapshot (CPU %, GPU %, RAM, GPU Mem, Queue Depth)
        |  <--- Compilation State (is_gpu_warm: bool)
        |
        +---> BackendSuitabilityResult (CPU Score, GPU Score, Confidence, Rationale)
        |
        v
[AdaptiveWorkRouterG6] (adaptive_router_g6.py)
        |  <--- Anti-Oscillation Hysteresis (2.0s cooldown + critical overrides)
        |
        +---> RoutingDecisionG6 (selected device, calibrated confidence, audit record)
        |
        v
[AdaptiveRoutingStrategyProxy]
        |
        +---> Route to CPU Execution Strategy
        |
        +---> Route to GPU Execution Strategy
        |        |
        |        X (Runtime Driver Failure)
        |        v
        +---> Seamless CPU Fallback (0.34 ms, 100% data preservation)
```

---

## 4. Workload Features and Cost Model

The `WorkloadCostEstimator` maps physical document properties to an explainable compute intensity score ($I \in [0.0, 1.0]$) using documented, pre-calibrated weights without inspecting clinical text:

$$I = 0.35 \cdot P_{\text{norm}} + 0.25 \cdot D_{\text{img}} + 0.25 \cdot C + 0.15 \cdot \text{Ch}_{\text{norm}}$$

Where:
- $P_{\text{norm}} = \min(1.0, \text{TotalPixels} / (2550 \times 3300))$: Pixel volume scaling.
- $D_{\text{img}} \in [0.0, 1.0]$: Fraction of page occupied by raster graphics.
- $C \in [0.0, 1.0]$: G4 structural layout complexity score.
- $\text{Ch}_{\text{norm}} = \min(1.0, \text{CharCount} / 3000.0)$: Native text character density before OCR.

---

## 5. Backend Suitability Scoring

The `BackendSuitabilityEstimator` evaluates calibrated suitability scores ($S_{\text{CPU}}, S_{\text{GPU}} \in [0.0, 1.0]$):

1. **Base Suitability**:
   - $S_{\text{GPU\_base}} = 0.20 + 0.75 \cdot I$
   - $S_{\text{CPU\_base}} = 0.85 - 0.65 \cdot I$
2. **Dense Text Adjustment**: If $\text{Ch}_{\text{norm}} > 0.40$ and $I > 0.60$, $S_{\text{CPU}}$ is increased by $+0.12 \cdot \text{Ch}_{\text{norm}}$ and $S_{\text{GPU}}$ discounted by $-0.08 \cdot \text{Ch}_{\text{norm}}$.
3. **Cold-Start Amortization**: If `is_gpu_warm == False`, $S_{\text{GPU}}$ is discounted:
   $$\text{Discount}_{\text{cold}} = 0.35 + 0.65 \cdot \min\left(1.0, \frac{\text{PageCount}}{15}\right)$$
4. **Hardware Saturation**: If GPU utilization $> 80\%$, $S_{\text{GPU}}$ is penalized up to $80\%$; if GPU free memory $< 256\text{ MB}$, penalized by $70\%$. If CPU utilization $> 85\%$, $S_{\text{CPU}}$ is penalized.
5. **Confidence**: Derived from the suitability margin:
   $$\text{Confidence} = \min(0.99, \max(0.15, |S_{\text{GPU}} - S_{\text{CPU}}| \cdot 1.6 + 0.10))$$

---

## 6. Strict Leakage Prevention & Dataset Partitioning

To avoid benchmark contamination, the evaluation corpus was partitioned into three disjoint sets:

1. **Set A (Development)**: Synthetic simple and medium layouts used during cost model unit testing.
2. **Set B (Validation)**: Diverse layouts ($800\times 600$ to $1400\times 1000$) used to verify hysteresis stability and freeze the G6 policy.
3. **Set C (Held-Out Test)**: 13 completely unseen pages (6 medium clinical from `dataset/PDF_Original/Medium`, 6 dense hard clinical from `dataset/PDF_Original/Hard`, and 1 novel high-resolution heterogeneous page). **Evaluated exactly once with the frozen policy.**

**Zero-Leakage Invariants**:
- The router API (`AdaptiveWorkRouterG6.route`) has no parameter or access path for ground-truth labels, oracle execution times, future measurements, or G5 outcome values.

---

## 7. Empirical Results on Held-Out Test Set (Set C)

### Table 1: Execution Time and Latency Metrics on Held-Out Set (13 Pages, 5 Repetitions)
| Policy | Mean Time (s) | Std Dev (s) | Median Time (s) | Throughput (p/s) | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU_ONLY** | 5.2217 s | 0.2131 s | 5.1430 s | 2.49 p/s | 348.69 ms | 352.46 ms | 536.51 ms |
| **GPU_ONLY** | 5.1706 s | 0.1077 s | 5.1460 s | 2.51 p/s | 344.83 ms | 361.69 ms | 419.83 ms |
| **ADAPTIVE_G4** | 5.1179 s | 0.0749 s | 5.1301 s | 2.54 p/s | 340.66 ms | 361.14 ms | 415.12 ms |
| **ADAPTIVE_G6** | 5.2565 s | 0.0901 s | 5.1971 s | 2.47 p/s | 351.47 ms | 375.81 ms | 427.08 ms |
| **OFFLINE ORACLE** | 5.1706 s | - | 5.1460 s | 2.51 p/s | 344.83 ms | 361.69 ms | 419.83 ms |

### Table 2: Speedup and Oracle Regret Analysis
| Metric | G6 vs CPU | G6 vs GPU | G6 vs G4 | GPU vs CPU |
| :--- | :---: | :---: | :---: | :---: |
| **Speedup Factor** | 0.993x | 0.984x | 0.974x | 1.010x |
| **Time Difference (%)** | +0.67% | +1.66% | +2.71% | -0.98% |
| **Oracle Policy** | **GPU_ONLY** | - | - | - |
| **G4 Oracle Regret (%)** | **-1.02%** | (Outperformed oracle mean due to run-to-run sampling variance) |
| **G6 Oracle Regret (%)** | **+1.66%** | (Minimal regret within sampling variance) |
| **Oracle Agreement Rate** | **100.0%** (G4) | **100.0%** (G6) | Both policies correctly routed 100% to GPU |

### Table 3: Resource Utilization and Overhead
| Policy | Mean Host CPU (%) | Mean Iris Xe GPU (%) | Mean Routing Latency (ms) | Overhead Classification |
| :--- | :---: | :---: | :---: | :---: |
| **CPU_ONLY** | 7.3% | 0.6% | 0.000 ms | - |
| **GPU_ONLY** | 9.9% | 45.8% | 0.000 ms | - |
| **ADAPTIVE_G4** | 7.2% | 46.3% | 0.075 ms | Negligible (< 0.1 ms) |
| **ADAPTIVE_G6** | 8.9% | 45.7% | 0.141 ms | Negligible (< 0.15 ms) |

---

## 8. Cold-Start Profiling and Fault Tolerance

- **Cold-Start Model Compilation**:
  - Host CPU: **678.32 ms**
  - Iris Xe GPU: **1,273.45 ms**
- **First vs. Warm Inference**:
  - CPU: First = 552.32 ms, Warm = 558.69 ms
  - GPU: First = 536.93 ms, Warm = 360.17 ms (1.55x faster once warmed)
- **Fault-Tolerance Fallback**:
  - Injected GPU driver failure was caught by `AdaptiveRoutingStrategyProxy`.
  - Automatically redirected to CPU OCR in **0.34 ms** with **100% data preservation** (`ocr_device = CPU_FALLBACK`).

---

## 9. Research Interpretation and Limitations

### Objective Evaluation
1. **Decision Quality**: The G6 data-driven router demonstrated **100% oracle agreement** on the held-out clinical batch, correctly identifying that the aggregate workload intensity warranted GPU execution.
2. **End-to-End Latency**: The data-driven G6 router **did not produce a statistically significant runtime speedup** over fixed GPU execution ($5.2565\text{s}$ vs $5.1706\text{s}$, $+1.66\%$ delta).
3. **Overhead vs. Benefit**: In a single-node setup with an integrated GPU sharing system RAM, the additional cost modeling ($0.008\text{ms}$) and suitability scoring ($0.141\text{ms}$), while strictly negligible, cannot overcome the physical memory bandwidth ceiling of integrated graphics when the entire batch is homogeneous in hardware preference.
4. **Valid Scientific Finding**: The data-driven router successfully formulated explainable, multi-factor scheduling decisions and achieved perfect oracle agreement, but under warm execution on homogeneous clinical batches, adaptive scheduling yields performance equivalent to the best fixed backend rather than a super-linear speedup.

---

## 10. Regression and Verification Status

- **G6 Adaptive Routing Unit Tests**: **14 passed**
- **G1–G6 GPU Track Regression**: **133 passed**
- **RAG Regression Suite**: **336 passed, 1 skipped**
- **Full Framework Unit Suite**: **1,084 passed, 0 failed**

**Phase G6 is COMPLETE. Framework stops here.** (Do NOT proceed to G7).
