"""AdaptiveWorkRouter -- Dynamic, workload-aware, and resource-aware CPU/GPU work routing.

Determines whether an OCR workload executes on CPU or GPU based on:
  - Workload complexity (from WorkloadProfile)
  - Runtime resource measurements (from ResourceSnapshot)
  - Configurable policy thresholds (from AdaptiveRoutingConfig)
  - Anti-oscillation hysteresis and cooldown dampening

Strict Design Principles:
  - Systems-only optimization: Zero clinical data is evaluated or logged.
  - Transparent & explainable: Every decision includes an explicit rationale.
  - Sub-millisecond execution: Consumes pre-sampled telemetry without blocking calls.
  - Fault-tolerant: Records execution failures and supports seamless CPU fallback.
"""

from __future__ import annotations

import collections
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.acceleration.workload_characterizer import WorkloadProfile
from adaptive_framework.config.models import AdaptiveRoutingConfig
from adaptive_framework.models.runtime import ResourceSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing Decision Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingDecision:
    """Structured, immutable outcome of an adaptive work-routing evaluation.

    Attributes:
        decision_id: Unique identifier for this routing decision.
        timestamp: ISO 8601 UTC timestamp when the decision was made.
        target_device: Selected device ('CPU', 'GPU', or 'FALLBACK').
        reason: Human-readable factor-specific explanation.
        confidence: Decision confidence score [0.0, 1.0].
        workload_profile: Input workload profile.
        resource_snapshot: Input runtime resource snapshot (may be None).
        policy_version: Version identifier of the routing policy.
        routing_latency_ms: Time spent inside the router in milliseconds.
        previous_device: Device selected in the previous decision (if any).
        was_switched: True if this decision changed the active device.
    """

    decision_id: str
    timestamp: str
    target_device: str
    reason: str
    confidence: float
    workload_profile: WorkloadProfile
    resource_snapshot: ResourceSnapshot | None
    policy_version: str = "v1.0"
    routing_latency_ms: float = 0.0
    previous_device: str | None = None
    was_switched: bool = False
    applied_rule: str = ""
    hysteresis_applied: bool = False

    @property
    def explanation(self) -> str:
        """Alias for reason for backward/test compatibility."""
        return self.reason

    @property
    def fallback_occurred(self) -> bool:
        """True if this decision represents an execution fallback."""
        return self.target_device == "FALLBACK" or "fallback" in self.reason.lower()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for logging and JSON persistence."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "target_device": self.target_device,
            "reason": self.reason,
            "applied_rule": self.applied_rule,
            "hysteresis_applied": self.hysteresis_applied,
            "confidence": round(self.confidence, 3),
            "policy_version": self.policy_version,
            "routing_latency_ms": round(self.routing_latency_ms, 3),
            "previous_device": self.previous_device,
            "was_switched": self.was_switched,
            "workload": {
                "document_id": self.workload_profile.document_id,
                "page_number": self.workload_profile.page_number,
                "page_count": self.workload_profile.page_count,
                "complexity": round(self.workload_profile.estimated_complexity, 3),
                "is_scanned": self.workload_profile.is_scanned,
                "image_density": round(self.workload_profile.image_density, 3),
            },
            "resources": (
                {
                    "cpu_percent": self.resource_snapshot.cpu_percent,
                    "ram_percent": self.resource_snapshot.memory_percent,
                    "gpu_available": self.resource_snapshot.gpu_available,
                    "gpu_utilization": self.resource_snapshot.gpu_utilization_percent,
                    "gpu_memory_used_mb": self.resource_snapshot.gpu_memory_used_mb,
                    "gpu_memory_total_mb": self.resource_snapshot.gpu_memory_total_mb,
                    "queue_length": self.resource_snapshot.queue_length,
                }
                if self.resource_snapshot is not None
                else None
            ),
        }

    def format_summary(self) -> str:
        """Format as a clean ASCII report line."""
        return (
            f"RoutingDecision [{self.target_device}] (conf={self.confidence:.2f}, "
            f"latency={self.routing_latency_ms:.3f}ms) -> {self.reason}"
        )


# ---------------------------------------------------------------------------
# Metrics Collector
# ---------------------------------------------------------------------------

class RoutingMetricsCollector:
    """Tracks and aggregates adaptive routing decisions and runtime metrics."""

    def __init__(self, history_size: int = 100) -> None:
        self._history_size = max(10, history_size)
        self._decisions: collections.deque[RoutingDecision] = collections.deque(
            maxlen=self._history_size
        )
        self._total_decisions: int = 0
        self._cpu_decisions: int = 0
        self._gpu_decisions: int = 0
        self._fallback_decisions: int = 0
        self._gpu_failure_count: int = 0
        self._latencies_ms: list[float] = []
        self._cpu_complexities: list[float] = []
        self._gpu_complexities: list[float] = []

    def record_decision(self, decision: RoutingDecision) -> None:
        """Record a completed routing decision."""
        self._decisions.append(decision)
        self._total_decisions += 1
        if decision.target_device == "GPU":
            self._gpu_decisions += 1
            self._gpu_complexities.append(decision.workload_profile.estimated_complexity)
        elif decision.target_device == "FALLBACK":
            self._fallback_decisions += 1
        else:
            self._cpu_decisions += 1
            self._cpu_complexities.append(decision.workload_profile.estimated_complexity)

        self._latencies_ms.append(decision.routing_latency_ms)
        if len(self._latencies_ms) > 1000:
            self._latencies_ms.pop(0)

    def record_gpu_failure(self) -> None:
        """Increment observed GPU execution failure counter."""
        self._gpu_failure_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Return aggregated summary metrics."""
        total = max(1, self._total_decisions)
        cpu_ratio = self._cpu_decisions / total
        gpu_ratio = self._gpu_decisions / total

        mean_lat = sum(self._latencies_ms) / len(self._latencies_ms) if self._latencies_ms else 0.0
        sorted_lat = sorted(self._latencies_ms)
        median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0

        avg_cpu_c = (
            sum(self._cpu_complexities) / len(self._cpu_complexities)
            if self._cpu_complexities
            else 0.0
        )
        avg_gpu_c = (
            sum(self._gpu_complexities) / len(self._gpu_complexities)
            if self._gpu_complexities
            else 0.0
        )

        return {
            "total_decisions": self._total_decisions,
            "cpu_decisions": self._cpu_decisions,
            "gpu_decisions": self._gpu_decisions,
            "fallback_decisions": self._fallback_decisions,
            "gpu_failure_count": self._gpu_failure_count,
            "device_selection_ratio": {
                "cpu": round(cpu_ratio, 4),
                "gpu": round(gpu_ratio, 4),
            },
            "latency_ms": {
                "mean": round(mean_lat, 3),
                "median": round(median_lat, 3),
                "p95": round(p95_lat, 3),
            },
            "average_complexity": {
                "cpu": round(avg_cpu_c, 3),
                "gpu": round(avg_gpu_c, 3),
            },
        }

    def get_recent_decisions(self, n: int = 50) -> list[RoutingDecision]:
        """Return up to n recent routing decisions (oldest first)."""
        count = min(n, len(self._decisions))
        if count == 0:
            return []
        return list(self._decisions)[-count:]

    def get_metrics(self) -> dict[str, Any]:
        """Alias for get_summary for dashboard/script compatibility."""
        return self.get_summary()

    def get_history(self) -> list[RoutingDecision]:
        """Return full list of recorded routing decisions in memory."""
        return list(self._decisions)


# ---------------------------------------------------------------------------
# AdaptiveWorkRouter Class
# ---------------------------------------------------------------------------

class AdaptiveWorkRouter:
    """Dynamic, workload-aware, and resource-aware CPU/GPU work router.

    Usage:
        router = AdaptiveWorkRouter()
        decision = router.route(workload_profile, resource_snapshot)
        print(decision.target_device, decision.reason)
    """

    def __init__(
        self,
        config: AdaptiveRoutingConfig | None = None,
        metrics_collector: RoutingMetricsCollector | None = None,
    ) -> None:
        """Initialize the adaptive work router.

        Args:
            config: Configured thresholds. Uses defaults if None.
            metrics_collector: Optional custom metrics aggregator.
        """
        self._config = config or AdaptiveRoutingConfig()
        self._metrics = metrics_collector or RoutingMetricsCollector()

        # State tracking for anti-oscillation hysteresis
        self._previous_device: str | None = None
        self._last_switch_timestamp: float = 0.0

    def reset_state(self) -> None:
        """Reset internal device history and hysteresis cooldown timers."""
        self._previous_device = None
        self._last_switch_timestamp = 0.0

    @property
    def config(self) -> AdaptiveRoutingConfig:
        """Return the active configuration."""
        return self._config

    @property
    def metrics(self) -> RoutingMetricsCollector:
        """Return the metrics collector."""
        return self._metrics

    @property
    def collector(self) -> RoutingMetricsCollector:
        """Alias for metrics collector."""
        return self._metrics

    def route(
        self,
        workload: WorkloadProfile,
        resource_snapshot: ResourceSnapshot | None = None,
    ) -> RoutingDecision:
        """Evaluate workload and system resources to select target execution device.

        Args:
            workload: Quantitative profile of the incoming page/document.
            resource_snapshot: Real-time system telemetry snapshot from G3.

        Returns:
            RoutingDecision populated with target device and explanation.
        """
        t0 = time.perf_counter()
        now_mono = time.monotonic()
        decision_id = str(uuid.uuid4())
        ts_utc = datetime.now(timezone.utc).isoformat()

        # ── 1. Policy Disablement Check ───────────────────────────────────
        if not self._config.enabled:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            dec = RoutingDecision(
                decision_id=decision_id,
                timestamp=ts_utc,
                target_device="CPU",
                reason="Adaptive routing disabled by configuration; routing to CPU default.",
                confidence=1.0,
                workload_profile=workload,
                resource_snapshot=resource_snapshot,
                policy_version=self._config.policy_version,
                routing_latency_ms=latency_ms,
                previous_device=self._previous_device,
                was_switched=self._previous_device != "CPU",
                applied_rule="POLICY_DISABLED",
                hysteresis_applied=False,
            )
            self._metrics.record_decision(dec)
            return dec

        # ── 2. Primary Deterministic Evaluation ────────────────────────────
        candidate_device, base_reason, confidence, rule_name = self._evaluate_rules(
            workload=workload,
            resource=resource_snapshot,
        )

        # ── 3. Hysteresis & Anti-Oscillation Filter ────────────────────────
        final_device = candidate_device
        final_reason = base_reason
        final_rule = rule_name
        was_switched = False
        hysteresis_applied = False

        if self._config.hysteresis.enabled and self._previous_device is not None:
            if candidate_device != self._previous_device:
                elapsed = now_mono - self._last_switch_timestamp
                cooldown = self._config.hysteresis.cooldown_seconds

                if elapsed < cooldown:
                    # In cooldown period: damp rapid switching unless a critical condition is met
                    can_switch = False

                    # Critical condition: If switching away from GPU because GPU is absent or hard failure
                    if self._previous_device == "GPU" and (
                        resource_snapshot is None
                        or not resource_snapshot.gpu_available
                        or (
                            resource_snapshot.gpu_utilization_percent is not None
                            and resource_snapshot.gpu_utilization_percent
                            >= (
                                self._config.gpu.max_utilization_percent
                                + self._config.hysteresis.utilization_delta_threshold
                            )
                        )
                    ):
                        can_switch = True  # Override cooldown due to strong failure or heavy overload

                    if not can_switch:
                        # Suppress the flip
                        final_device = self._previous_device
                        final_reason = (
                            f"{base_reason} [Anti-oscillation hysteresis retained "
                            f"{self._previous_device}; cooldown {cooldown - elapsed:.1f}s remaining]"
                        )
                        final_rule = "HYSTERESIS_DAMPENED"
                        hysteresis_applied = True
                    else:
                        was_switched = True
                        self._last_switch_timestamp = now_mono
                else:
                    was_switched = True
                    self._last_switch_timestamp = now_mono
        else:
            if self._previous_device != candidate_device:
                was_switched = True
                self._last_switch_timestamp = now_mono

        self._previous_device = final_device
        latency_ms = (time.perf_counter() - t0) * 1000.0

        decision = RoutingDecision(
            decision_id=decision_id,
            timestamp=ts_utc,
            target_device=final_device,
            reason=final_reason,
            confidence=confidence,
            workload_profile=workload,
            resource_snapshot=resource_snapshot,
            policy_version=self._config.policy_version,
            routing_latency_ms=latency_ms,
            previous_device=self._previous_device,
            was_switched=was_switched,
            applied_rule=final_rule,
            hysteresis_applied=hysteresis_applied,
        )

        self._metrics.record_decision(decision)
        logger.debug("AdaptiveWorkRouter: %s", decision.format_summary())
        return decision

    # ------------------------------------------------------------------ #
    # Rule Evaluation Logic
    # ------------------------------------------------------------------ #

    def _evaluate_rules(
        self,
        workload: WorkloadProfile,
        resource: ResourceSnapshot | None,
    ) -> tuple[str, str, float, str]:
        """Apply sequential deterministic routing rules."""
        # Rule A: GPU Hardware Availability
        if resource is None or not resource.gpu_available:
            return (
                "CPU",
                "CPU selected because GPU acceleration is unavailable on this node.",
                1.0,
                "RULE_A_NO_GPU",
            )

        # Rule B: Missing / corrupted telemetry
        # If GPU is detected but resource snapshot lacks core metrics, fallback safely
        if resource.cpu_percent < 0.0 or resource.memory_percent < 0.0:
            return (
                "CPU",
                "CPU selected due to invalid resource telemetry (conservative fallback).",
                0.90,
                "RULE_B_INVALID_METRICS",
            )

        # Rule C: Small Workload Threshold
        # Small simple pages are faster on CPU due to GPU kernel dispatch & transfer overhead
        small_cfg = self._config.small_workload
        if workload.page_count <= small_cfg.max_pages and workload.estimated_complexity <= small_cfg.max_complexity:
            return (
                "CPU",
                (
                    f"CPU selected because workload is small (pages={workload.page_count}, "
                    f"complexity={workload.estimated_complexity:.2f} <= {small_cfg.max_complexity:.2f}); "
                    "GPU transfer overhead is not justified."
                ),
                0.95,
                "RULE_C_SMALL_WORKLOAD",
            )

        # Rule D: High GPU Utilization (Saturation)
        gpu_util = resource.gpu_utilization_percent
        if gpu_util is not None and gpu_util >= self._config.gpu.max_utilization_percent:
            return (
                "CPU",
                (
                    f"CPU selected because GPU utilization ({gpu_util:.1f}%) "
                    f"exceeds capacity threshold ({self._config.gpu.max_utilization_percent:.1f}%)."
                ),
                0.92,
                "RULE_D_GPU_SATURATED",
            )

        # Rule E: Insufficient GPU Memory
        if resource.gpu_memory_total_mb is not None and resource.gpu_memory_used_mb is not None:
            free_mem = max(0.0, resource.gpu_memory_total_mb - resource.gpu_memory_used_mb)
            if free_mem < self._config.gpu.min_memory_available_mb:
                return (
                    "CPU",
                    (
                        f"CPU selected because available GPU memory ({free_mem:.1f} MB) "
                        f"is below required minimum ({self._config.gpu.min_memory_available_mb:.1f} MB)."
                    ),
                    0.95,
                    "RULE_E_LOW_GPU_MEMORY",
                )

        # Rule F: High CPU Saturation with Available GPU
        if resource.cpu_percent >= self._config.cpu.max_utilization_percent:
            return (
                "GPU",
                (
                    f"GPU selected because CPU is heavily saturated ({resource.cpu_percent:.1f}% >= "
                    f"{self._config.cpu.max_utilization_percent:.1f}%) while GPU has available capacity."
                ),
                0.91,
                "RULE_F_CPU_SATURATED",
            )

        # Rule G: Workload Complexity Justifies Acceleration
        if workload.estimated_complexity >= self._config.gpu.min_workload_complexity:
            gpu_u_str = f"{gpu_util:.1f}%" if gpu_util is not None else "UNAVAILABLE"
            return (
                "GPU",
                (
                    f"GPU selected because workload complexity ({workload.estimated_complexity:.2f} >= "
                    f"{self._config.gpu.min_workload_complexity:.2f}) justifies acceleration and "
                    f"GPU utilization ({gpu_u_str}) is within operating limits."
                ),
                0.90,
                "RULE_G_COMPLEXITY_GPU",
            )

        # Rule H: Default Fallback
        return (
            "CPU",
            (
                f"CPU selected as default device; workload complexity ({workload.estimated_complexity:.2f}) "
                f"is below GPU threshold ({self._config.gpu.min_workload_complexity:.2f})."
            ),
            0.85,
            "RULE_H_CPU_DEFAULT",
        )

    def record_execution_fallback(
        self,
        failed_device: str,
        fallback_device: str,
        error_message: str,
        workload: WorkloadProfile,
    ) -> RoutingDecision:
        """Explicitly record a runtime execution fallback event (e.g. GPU crash/OOM)."""
        self._metrics.record_gpu_failure()
        decision_id = str(uuid.uuid4())
        ts_utc = datetime.now(timezone.utc).isoformat()

        reason = (
            f"Execution fallback to {fallback_device} triggered after {failed_device} "
            f"runtime failure: {error_message}"
        )
        dec = RoutingDecision(
            decision_id=decision_id,
            timestamp=ts_utc,
            target_device="FALLBACK",
            reason=reason,
            confidence=1.0,
            workload_profile=workload,
            resource_snapshot=None,
            policy_version=self._config.policy_version,
            routing_latency_ms=0.0,
            previous_device=failed_device,
            was_switched=True,
            applied_rule="EXECUTION_FALLBACK",
            hysteresis_applied=False,
        )
        self._metrics.record_decision(dec)
        logger.warning("AdaptiveWorkRouter: %s", reason)
        return dec
