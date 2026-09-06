# Phase G3: GPU Utilization and Resource Monitoring

## Executive Summary

Phase G3 establishes a robust, read-only runtime resource-monitoring and observability layer for the Adaptive Distributed Parallel Processing Framework (v2.0). Grounded in the real hardware environment of **Intel(R) Iris(R) Xe Graphics (iGPU)** with OpenVINO Runtime (version 2026.3.0) on Windows 11, G3 observes host CPU, system RAM, Intel GPU engine utilization, GPU memory allocation, task queue length, worker states, and real OCR inference latency and throughput.

In accordance with strict project guidelines, Phase G3 is **purely observational**:
- **No adaptive routing** is implemented.
- **No dynamic worker allocation** is implemented.
- **No CPU/GPU scheduling decisions** are made.
- **No work-stealing alterations** are made.
- **No CPU vs GPU performance comparisons** are performed.
- All routing and scheduling adaptations remain strictly deferred to Phases G4, G5, and G6.

---

## 1. Target Hardware & Telemetry Grounding

All monitoring implementations were verified against the real local machine hardware:

| Component | Detected Specification | Telemetry Mechanism | Status |
|---|---|---|---|
| **CPU Brand** | Intel(R) Core(TM) i5-1135G7 @ 2.40GHz | `platform.processor()` / `psutil` | Real Telemetry |
| **CPU Architecture** | AMD64 (x86_64, 4 physical cores, 8 logical threads) | `psutil.cpu_count()` | Real Telemetry |
| **CPU Utilization** | Real-time percent across all cores | `psutil.cpu_percent()` | Real Telemetry |
| **System RAM** | 16,122.8 MB (~15.75 GB total) | `psutil.virtual_memory()` | Real Telemetry |
| **GPU Device** | Intel(R) Iris(R) Xe Graphics (iGPU) | OpenVINO Core `FULL_DEVICE_NAME` | Real Telemetry |
| **GPU Architecture** | `GPU: vendor=0x8086 arch=v12.0.0` (Gen12) | OpenVINO Core `DEVICE_ARCHITECTURE` | Real Telemetry |
| **OpenVINO Device ID** | `GPU` / `GPU.0` (80 Execution Units) | OpenVINO Core `EXECUTION_DEVICES` | Real Telemetry |
| **GPU Total Memory Pool** | 7,473,942,528 bytes (~7,127.71 MB) | OpenVINO `GPU_DEVICE_TOTAL_MEM_SIZE` | Real Telemetry |
| **GPU Allocated Memory** | OpenVINO memory buffer tracking | OpenVINO `GPU_MEMORY_STATISTICS` | Real Telemetry |
| **GPU Utilization (%)** | Windows GPU 3D/Compute/Copy Engine percent | Windows Performance Data Helper (PDH) | Real Telemetry |
| **GPU Temperature** | Discrete GPU temperature sensor | Not present on Intel iGPU (CPU die) | **UNAVAILABLE** (`None`) |
| **GPU Power (W)** | Discrete GPU power rail sensor | Not present on Intel iGPU (SoC rail) | **UNAVAILABLE** (`None`) |

### Strict Adherence to the No-Fake-Values Rule
As required by the specification:
- Sensors that do not physically exist on integrated graphics (e.g. discrete GPU temperature probes or discrete board power sensors) report **`None`** internally and format as **`UNAVAILABLE`**.
- Invented values such as 0% or 50% are strictly forbidden and never emitted.

---

## 2. Architecture & Components

```
                      +-----------------------------+
                      |       ResourceMonitor       |
                      +-----------------------------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
     +-----------------+    +-----------------+    +-----------------+
     |   SystemUtils   |    |   GPUMonitor    |    |  State Providers|
     |   (psutil)      |    | (OpenVINO + PDH)|    | (Queue / Worker)|
     +-----------------+    +-----------------+    +-----------------+
              |                      |                      |
              +----------------------+----------------------+
                                     |
                                     v
                        +--------------------------+
                        |     ResourceSnapshot     |
                        | (Immutable Structured)   |
                        +--------------------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
        +-----------------------+         +-----------------------+
        |  DashboardStateStore  |         | scripts/monitor_...   |
        |  (Time-Series History)|         | (Diagnostic CLI Tool) |
        +-----------------------+         +-----------------------+
```

### Component Details

1. **`GPUMonitor` (`src/adaptive_framework/acceleration/gpu_monitor.py`)**:
   - Manages OpenVINO GPU device handles and queries hardware capabilities.
   - Manages Windows Performance Data Helper (`pdh.dll`) query handles to sample GPU Engine utilization in-process with sub-millisecond query latency.
   - Implements context manager and resource cleanup to prevent handle leaks.
   - Gracefully reports `None` for unsupported metrics (`temperature_c`, `power_w`).

2. **`ResourceSnapshot` (`src/adaptive_framework/models/runtime.py`)**:
   - Structured, immutable point-in-time telemetry snapshot model.
   - Extended with logical/physical CPU core counts, RAM used/available MB, GPU availability, GPU device info, GPU utilization %, GPU memory total/used/free MB, queue depth, active worker count, and OCR inference latency and throughput.
   - 100% backward-compatible with legacy coordinator and heartbeat instantiation signatures.

3. **`ResourceMonitor` (`src/adaptive_framework/acceleration/resource_monitor.py`)**:
   - Sampling engine combining host CPU, RAM, GPU, queue, worker state, and OCR measurements.
   - Supports both on-demand point-in-time sampling (`sample()`) and daemon background sampling (`start_background_sampling()`).
   - Thread-safe circular buffer (`deque`) storing time-series history for sparklines and analysis.

4. **Dashboard Integration (`dashboard/state/dashboard_state.py` & `dashboard/components/cluster_overview.py`)**:
   - Extended `DashboardStateStore` to track `_gpu_util_history` and `_gpu_mem_history`.
   - Updated `ClusterOverview` panel to render live GPU telemetry cards, memory usage, and sensor availability status.

5. **Diagnostic CLI Tool (`scripts/monitor_resources.py`)**:
   - Allows command-line inspection of runtime telemetry with configurable sample counts and intervals.
   - Supports `--ocr-bench` to trigger real OpenVINO OCR inference on the GPU and record warm latency and throughput.

---

## 3. Verification & Diagnostic Output

### CLI Execution Sample

```
==============================================================================
ADF RUNTIME RESOURCE MONITORING -- PHASE G3
==============================================================================
Node ID:          local_node
CPU Cores:        8 logical, 4 physical
System RAM:       16122.8 MB total
GPU Device:       Intel(R) Iris(R) Xe Graphics (iGPU) (GPU)
Architecture:     GPU: vendor=0x8086 arch=v12.0.0
Total GPU Pool:   7127.71 MB
------------------------------------------------------------------------------
Running OpenVINO OCR pass to measure inference latency and throughput...
Observed OCR Latency: 50.99 ms | Throughput: 19.61 p/s
------------------------------------------------------------------------------
#   | Timestamp           | CPU%  | RAM% (Used/Tot)   | GPU%    | GPU Mem (MB)   | Temp   | Power 
--------------------------------------------------------------------------------------------------
1   | 15:24:17            | 13.0  | 86.4% (13938M)    | 3.0%    | 0/7128         | UNAVAIL | UNAVAIL
2   | 15:24:17            | 11.1  | 86.2% (13904M)    | 0.9%    | 0/7128         | UNAVAIL | UNAVAIL
------------------------------------------------------------------------------
TELEMETRY CAPABILITY AUDIT SUMMARY:
  1. CPU Utilization:       SUPPORTED (via psutil)
  2. CPU Core Count:        SUPPORTED (logical & physical)
  3. RAM Utilization:       SUPPORTED (percent, used MB, avail MB)
  4. GPU Availability:      SUPPORTED (OpenVINO Core)
  5. GPU Device Info:       SUPPORTED (Iris Xe, Gen12, vendor 0x8086)
  6. GPU Utilization:       SUPPORTED (Windows PDH GPU Engine counter)
  7. GPU Memory:            SUPPORTED (OpenVINO total pool & buffer stats)
  8. GPU Temperature:       UNAVAILABLE (iGPU unified on CPU die)
  9. GPU Power:             UNAVAILABLE (iGPU unified on SoC package rail)
 10. OCR Latency:           SUPPORTED (OpenVINO OCR inference observation)
 11. OCR Throughput:        SUPPORTED (pages/sec calculation)
 12. Queue Length:          SUPPORTED (ResourceSnapshot queue_length)
 13. Worker State:          SUPPORTED (ResourceSnapshot active_workers)
 14. Resource Snapshots:    SUPPORTED (immutable timestamped dataclass)
==============================================================================
```

---

## 4. Test Suite Results

- **GPU Track Suite**: 83 passed in ~22s
  - `tests/unit/test_hardware_probe.py`: 33 passed
  - `tests/unit/test_openvino_ocr_strategy.py`: 23 passed
  - `tests/unit/test_openvino_ocr_g2.py`: 12 passed
  - `tests/unit/test_gpu_monitor_g3.py`: 15 passed
- **Backward Compatibility**: Fully preserved for all existing coordinator, scheduler, and worker models.
