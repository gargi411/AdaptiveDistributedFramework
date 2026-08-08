# Evaluation Engine Package

## Architecture (v2.0 §4)

Measures and reports all 6 primary metrics.

## Status: Phase 6 Placeholder

## Metrics

| # | Metric | Unit | Target |
|---|--------|------|--------|
| 1 | Speedup | ratio | > 1.0 |
| 2 | Throughput | pages/sec | maximize |
| 3 | CPU Utilization | % | monitor |
| 4 | GPU Utilization | % | monitor |
| 5 | Energy | Joules | minimize |
| 6 | **Scheduler Overhead** | **%** | **< 1%** |

## Configuration

`configs/evaluation.yaml`

## Output Formats

JSON, CSV, Markdown (configurable via `report_formats`).