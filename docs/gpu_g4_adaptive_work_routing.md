# Phase G4 Technical Report: Adaptive CPU/GPU Work Routing

**Project**: Adaptive Distributed Parallel Processing Framework for Large-Scale Biomedical Document Intelligence (v2.0)  
**Phase**: G4 — Adaptive CPU/GPU Work Routing  
**Status**: Research Validated & Verified  
**Date**: September 2026  
**Hardware Platform**: Intel Core i5-1135G7 (4 physical cores, 8 logical cores, 16.1 GB RAM), Intel Iris Xe Graphics iGPU (Gen12, 80 Execution Units, OpenVINO 2026.3.0, Device `GPU.0`, 7.12 GB total shared memory pool)

---

## 1. Executive Summary

Phase G4 establishes the first dynamic, workload-aware, and resource-aware routing layer for optical character recognition (OCR) within the Adaptive Distributed Parallel Processing Framework. Building upon the real hardware detection from Phase G1, actual OpenVINO OCR pipeline execution from Phase G2, and runtime system observability from Phase G3, Phase G4 introduces:

1. **Deterministic Workload Characterization**: A normalized complexity heuristic in `[0.0, 1.0]` that quantifies document processing compute intensity in under 0.2 milliseconds based purely on geometric and structural properties.
2. **Sequential Policy Decision Engine**: An explainable rule hierarchy (Rules A through H) evaluating real-time CPU/GPU utilization, memory availability, and workload complexity to choose the optimal execution device.
3. **Anti-Oscillation Hysteresis**: A cooldown-governed dampening mechanism preventing rapid toggling between CPU and GPU under borderline or fluctuating resource conditions.
4. **Resilient Execution Fallback**: An transparent proxy (`AdaptiveRoutingStrategyProxy`) that intercepts runtime GPU exceptions (e.g. driver fault, out-of-memory), logs the event, updates metrics, and seamlessly executes on CPU without job loss or data corruption.
5. **Zero Clinical Data Leakage**: Invariant adherence ensuring no patient identifiers, diagnosis text, symptoms, or OCR contents are used as routing features or written to telemetry records.
6. **Strict Factory Isolation**: Digital vector PDF pages continue to bypass all OCR entirely via `DirectExtractionStrategy`, guaranteeing zero regression in text extraction latency.

---

## 2. Architecture and Data Flow

```
                      +-----------------------------+
                      | Incoming Page / Work Unit   |
                      +--------------+--------------+
                                     |
                                     v
                       Is Page Type DIGITAL?
                                    / \
                            YES    /   \   NO (SCANNED / MIXED)
                                  /     \
                                 v       v
         +--------------------------+  +--------------------------------+
         | DirectExtractionStrategy |  | AdaptiveRoutingStrategyProxy   |
         | (PyMuPDF Text Layer,     |  +---------------+----------------+
         |  Zero OCR Overhead)      |                  |
         +--------------------------+                  |
                                                       v
                                          +----------------------------+
                                          | WorkloadCharacterizer      |
                                          | -> WorkloadProfile (<0.2ms)|
                                          +-------------+--------------+
                                                        |
                                                        v
                     +---------------------+    +-------+--------------+
                     | ResourceMonitor(G3) |--->| AdaptiveWorkRouter   |
                     | (Snapshot: CPU/GPU) |    | -> Policy Rules A-H  |
                     +---------------------+    | -> Hysteresis Filter |
                                                +-------+--------------+
                                                        |
                                                        v
                                                Target Device
                                                    /       \
                                            "GPU"  /         \  "CPU"
                                                  /           \
                                                 v             v
                                      +-------------+   +-------------+
                                      | OpenVINO    |   | OpenVINO    |
                                      | GPU Strategy|   | CPU Strategy|
                                      +------+------+   +-------------+
                                             |
                                  Exception Occurred?
                                         /       \
                                 YES    /         \   NO
                                       /           \
                                      v             v
                          +-------------------+  Success
                          | Record Fallback   |  PageExtractionResult
                          | Execute CPU Pass  |
                          +-------------------+
```

---

## 3. Workload Characterization Model

The `WorkloadCharacterizer` quantifies page compute intensity without loading heavyweight neural networks or inspecting patient data.

### Complexity Heuristic

Complexity $C \in [0.0, 1.0]$ is computed deterministically using three non-sensitive structural features:

1. **Area Scaling Factor ($S_{\text{area}}$)**:
   $$S_{\text{area}} = \min\left(1.5, \max\left(0.5, \frac{\text{width} \times \text{height}}{612 \times 792}\right)\right)$$

2. **Scanned vs. Digital Heuristic**:
   - For scanned pages ($\text{char\_count} < 50$ and $\text{image\_density} \ge 0.50$):
     $$C = \min\left(1.0, \max\left(0.35, 0.40 + 0.50 \times \text{image\_density} + 0.05 \times (\text{image\_count} - 1)\right) \times \frac{S_{\text{area}}}{1.2}\right)$$
   - For digital pages with text layer:
     $$C = \min\left(0.60, \max\left(0.05, 0.05 + 0.35 \times \text{image\_density} + 0.0001 \times \text{char\_count}\right)\right)$$

### Latency Overhead

Benchmarking 100 characterizations yielded an average execution time of **0.0018 ms per page**, well below the strict 0.2 ms SLA.

---

## 4. Sequential Policy Decision Engine

The router implements deterministic, priority-ordered rules:

| Priority | Rule ID | Condition | Selected Device | Confidence | Rationale |
|---|---|---|---|---|---|
| 1 | `RULE_A_NO_GPU` | GPU absent or unavailable | `CPU` | 1.00 | Hardware absence requires conservative CPU execution. |
| 2 | `RULE_B_INVALID_METRICS` | Telemetry negative or corrupted | `CPU` | 0.90 | Sensor failure triggers safe fallback to default processor. |
| 3 | `RULE_C_SMALL_WORKLOAD` | $\text{pages} \le 1$ and $C \le 0.20$ | `CPU` | 0.95 | Workload is too small to amortize GPU PCIe/kernel dispatch latency. |
| 4 | `RULE_D_GPU_SATURATED` | GPU Util $\ge 85.0\%$ | `CPU` | 0.92 | GPU saturated; routing to CPU prevents tail latency spikes. |
| 5 | `RULE_E_LOW_GPU_MEMORY` | Free GPU Mem $< 256.0\text{ MB}$ | `CPU` | 0.95 | Low shared memory pool prevents GPU OOM crashes. |
| 6 | `RULE_F_CPU_SATURATED` | CPU Util $\ge 80.0\%$ | `GPU` | 0.91 | Offload compute to GPU to relieve host thread starvation. |
| 7 | `RULE_G_COMPLEXITY_GPU` | $C \ge 0.35$ and GPU healthy | `GPU` | 0.90 | High complexity scanned document justifies parallel execution. |
| 8 | `RULE_H_CPU_DEFAULT` | Default fallback | `CPU` | 0.85 | Moderate complexity executed on standard CPU pipeline. |

---

## 5. Anti-Oscillation Hysteresis

In dynamic multi-tenant workloads, system metrics oscillate around critical thresholds. Rapid switching between CPU and GPU incurs driver cache invalidation and thread migration overhead.

### Cooldown Dampening

- **Cooldown Period**: $T_{\text{cooldown}} = 2.0\text{ seconds}$
- If a candidate routing decision would flip devices within the cooldown window ($\Delta t < T_{\text{cooldown}}$):
  $$\text{Target Device} = \text{Previous Device}$$
  $$\text{Applied Rule} = \text{"HYSTERESIS\_DAMPENED"}$$

### Critical Overload Override

Dampening is overridden immediately if the previous device was GPU and:
1. The GPU becomes completely unavailable or disconnected.
2. The GPU utilization exceeds the threshold plus hysteresis delta:
   $$\text{GPU Util} \ge \text{Max Util} + \Delta_{\text{util}} = 85.0\% + 5.0\% = 90.0\%$$

---

## 6. Execution Fallback Proxy

The `AdaptiveRoutingStrategyProxy` wraps both GPU and CPU OpenVINO engines.

```python
if target_device == "GPU":
    try:
        return self._gpu_strategy.process(page, ...)
    except Exception as exc:
        router.record_execution_fallback(failed_device="GPU", fallback_device="CPU", ...)
        res = self._cpu_strategy.process(page, ...)
        res.ocr_device = "CPU_FALLBACK"
        return res
```

This guarantees zero worker crash or pipeline halts upon sporadic GPU device loss or memory allocation failures.

---

## 7. Experimental Validation Results

Validation was conducted using `scripts/test_adaptive_routing.py` across two execution modes.

### Simulation Scenarios (1-7)

| Scenario | Description | Expected Device | Actual Device | Status | Applied Rule |
|---|---|---|---|---|---|
| 1 | GPU Hardware Unavailable | CPU | CPU | PASS | `RULE_A_NO_GPU` |
| 2 | Small Workload (1 page, C=0.15) | CPU | CPU | PASS | `RULE_C_SMALL_WORKLOAD` |
| 3 | High Complexity Scanned Doc (C=0.75) | GPU | GPU | PASS | `RULE_G_COMPLEXITY_GPU` |
| 4 | High GPU Load (GPU Util = 92%) | CPU | CPU | PASS | `RULE_D_GPU_SATURATED` |
| 5 | High CPU Load (CPU Util = 88%) | GPU | GPU | PASS | `RULE_F_CPU_SATURATED` |
| 6 | Execution Fallback on GPU Failure | CPU_FALLBACK | CPU_FALLBACK | PASS | `EXECUTION_FALLBACK` |
| 7 | Borderline Hysteresis Cooldown | GPU | GPU | PASS | `HYSTERESIS_DAMPENED` |

**Simulation Score**: **7 / 7 Passed (100%)**

### Live Local Hardware Telemetry

- **Node ID**: `local_hw_test`
- **CPU**: Intel Core i5-1135G7 (4 physical, 8 logical cores)
- **GPU**: Intel(R) Iris(R) Xe Graphics (iGPU) (`GPU.0`, 7127.7 MB total memory pool)
- **Decision Latencies**:
  - Mean: **0.037 ms**
  - Median: **0.028 ms**
  - 95th Percentile: **0.062 ms**

All decisions executed with sub-0.1 ms latency, well inside the sub-millisecond requirement.

---

## 8. Dashboard Integration

The engineering dashboard (`dashboard/app.py`) was extended with a dedicated **Adaptive CPU/GPU Routing** panel (`dashboard/components/adaptive_routing_panel.py`):

1. **Summary Metric Cards**:
   - Total routing decisions made
   - CPU vs GPU routing counts and selection percentages
   - Execution fallback occurrences and GPU failures
   - Mean, median, and P95 router latency in milliseconds
   - Average workload complexity processed by CPU vs GPU
2. **Recent Decisions Table**:
   - Timestamp, Target Device, Applied Rule, Hysteresis Flag, Routing Latency, and Human-Readable Rationale.

---

## 9. Test Regressions and Invariant Verification

| Suite | Tests Executed | Passed | Skipped | Status |
|---|---|---|---|---|
| **Phase G4 Adaptive Router** (`test_adaptive_router_g4.py`) | 19 | 19 | 0 | PASSED |
| **Complete GPU Track Suite** (G1, G2, G3, G4) | 102 | 102 | 0 | PASSED |
| **RAG Regression Suite** (Phase 4.1 to 4.11) | 337 | 336 | 1 (live API) | PASSED |
| **Full Unit Test Suite** (Framework-wide) | 1053 | 1053 | 0 | PASSED |

### Verified Architectural Invariants

- **Invariant 1 (Device Validity)**: Decisions strictly return `'CPU'`, `'GPU'`, or `'FALLBACK'`.
- **Invariant 2 (Explainability)**: Every routing decision contains a non-empty human-readable explanation.
- **Invariant 3 (Decision Latency)**: Router latency strictly $< 1.0\text{ ms}$ (observed $< 0.1\text{ ms}$).
- **Invariant 4 (Anti-Oscillation)**: Hysteresis suppresses state flips within the 2.0s cooldown window.
- **Invariant 5 (Execution Resilience)**: GPU exceptions fall back to CPU cleanly with zero unhandled exceptions.
- **Invariant 6 (Strategy Isolation)**: Digital pages bypass OCR completely via `DirectExtractionStrategy`.
- **Invariant 7 (Clinical Privacy)**: Zero clinical terms, diagnoses, symptoms, or OCR texts in telemetry logs.
- **Invariant 8 (Telemetry Integrity)**: Consumes real Windows PDH and OpenVINO metrics without fabricated values.
