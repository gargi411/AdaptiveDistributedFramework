"""Adaptive Work Router G6: Data-Driven CPU/GPU Work Scheduling.

Provides:
  - RoutingDecisionG6: Detailed routing audit record with suitability scores.
  - AdaptiveWorkRouterG6: Data-driven, cold-start aware, queue-aware routing engine.

Integrates:
  - WorkloadCostEstimator (G6 computational cost profiling)
  - BackendSuitabilityEstimator (Calibrated suitability scoring)
  - Anti-oscillation hysteresis (Stable boundary behavior)
  - Seamless execution fallback recording
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from adaptive_framework.acceleration.adaptive_router import (
    AdaptiveRoutingConfig,
    RoutingMetricsCollector,
)
from adaptive_framework.acceleration.backend_suitability import (
    BackendSuitabilityEstimator,
    BackendSuitabilityResult,
)
from adaptive_framework.acceleration.workload_characterizer import WorkloadProfile
from adaptive_framework.acceleration.workload_cost_model import (
    WorkloadCostEstimator,
    WorkloadCostProfile,
)
from adaptive_framework.models.runtime import ResourceSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecisionG6:
    """Audit record of a G6 data-driven routing decision.

    Attributes:
        decision_id: Unique UUID string for this decision event.
        timestamp: ISO 8601 UTC timestamp string.
        target_device: Selected device for processing ("CPU" or "GPU").
        confidence: Normalized confidence margin [0.0, 1.0].
        policy_name: Applied policy identifier ("G6_DATA_DRIVEN" or "G4_RULE_BASED").
        primary_reason: Human-readable machine explanation string.
        cpu_score: Computed CPU suitability score [0.0, 1.0].
        gpu_score: Computed GPU suitability score [0.0, 1.0].
        is_gpu_warm: Cold/warm flag used during evaluation.
        workload_cost: WorkloadCostProfile descriptor.
        resource_snapshot: Point-in-time hardware telemetry snapshot.
        routing_latency_ms: Time taken to evaluate the routing decision in ms.
        previous_device: Device assigned in the previous decision.
        was_switched: Whether this decision changed target devices.
        hysteresis_applied: Whether anti-oscillation dampened a proposed switch.
        fallback_occurred: True if this decision represents an execution fallback.
    """

    decision_id: str
    timestamp: str
    target_device: str
    confidence: float
    policy_name: str = "G6_DATA_DRIVEN"
    primary_reason: str = ""
    cpu_score: float = 0.5
    gpu_score: float = 0.5
    is_gpu_warm: bool = True
    workload_cost: WorkloadCostProfile | None = None
    resource_snapshot: ResourceSnapshot | None = None
    routing_latency_ms: float = 0.0
    previous_device: str | None = None
    was_switched: bool = False
    hysteresis_applied: bool = False
    fallback_occurred: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with clean formatting."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "target_device": self.target_device,
            "confidence": round(self.confidence, 4),
            "policy_name": self.policy_name,
            "primary_reason": self.primary_reason,
            "cpu_score": round(self.cpu_score, 4),
            "gpu_score": round(self.gpu_score, 4),
            "is_gpu_warm": self.is_gpu_warm,
            "workload_cost": self.workload_cost.to_dict() if self.workload_cost else None,
            "routing_latency_ms": round(self.routing_latency_ms, 4),
            "previous_device": self.previous_device,
            "was_switched": self.was_switched,
            "hysteresis_applied": self.hysteresis_applied,
            "fallback_occurred": self.fallback_occurred,
        }

    def format_summary(self) -> str:
        """Format as a concise summary string."""
        return (
            f"RoutingDecisionG6(id={self.decision_id[:8]}, target={self.target_device}, "
            f"conf={self.confidence:.2f}, cpu={self.cpu_score:.3f}, gpu={self.gpu_score:.3f}, "
            f"warm={self.is_gpu_warm}, reason='{self.primary_reason}', "
            f"lat={self.routing_latency_ms:.3f}ms)"
        )


class AdaptiveWorkRouterG6:
    """Data-driven adaptive work router for CPU/GPU acceleration.

    Integrates:
      - WorkloadCostEstimator for pre-OCR computational cost modeling
      - BackendSuitabilityEstimator for calibrated scoring & cold-start handling
      - Anti-oscillation hysteresis with critical override triggers
      - High-fidelity decision auditing with sub-0.1ms execution latency
    """

    def __init__(
        self,
        config: AdaptiveRoutingConfig | None = None,
        cost_estimator: WorkloadCostEstimator | None = None,
        suitability_estimator: BackendSuitabilityEstimator | None = None,
        metrics_collector: RoutingMetricsCollector | None = None,
    ) -> None:
        """Initialize G6 adaptive work router with components."""
        self._config = config or AdaptiveRoutingConfig()
        self._cost_estimator = cost_estimator or WorkloadCostEstimator()
        self._suitability_estimator = suitability_estimator or BackendSuitabilityEstimator()
        self._metrics = metrics_collector or RoutingMetricsCollector()

        # State tracking for hysteresis
        self._previous_device: str | None = None
        self._last_switch_timestamp: float = 0.0

    @property
    def metrics(self) -> RoutingMetricsCollector:
        """Return decision metrics collector."""
        return self._metrics

    @property
    def config(self) -> AdaptiveRoutingConfig:
        """Return active routing configuration."""
        return self._config

    def route(
        self,
        workload: WorkloadProfile,
        resource: ResourceSnapshot | None = None,
        is_gpu_warm: bool = True,
        queue_length: int = 0,
        active_workers: int = 1,
        pixel_dimensions: tuple[int, int] | None = None,
    ) -> RoutingDecisionG6:
        """Compute the optimal execution backend for the given workload.

        Args:
            workload: WorkloadProfile descriptor.
            resource: Optional ResourceSnapshot from G3 monitor.
            is_gpu_warm: Whether the GPU model is already compiled and warm.
            queue_length: In-flight task queue depth.
            active_workers: Active worker thread count.
            pixel_dimensions: Optional pre-rendered pixel dimensions (width, height).

        Returns:
            RoutingDecisionG6 with chosen backend, scores, confidence, and audit trail.
        """
        t0 = time.perf_counter()
        now_mono = time.monotonic()
        decision_id = str(uuid.uuid4())
        ts_utc = datetime.now(timezone.utc).isoformat()

        # 1. Estimate Workload Cost Profile
        cost_profile = self._cost_estimator.estimate_cost(
            workload=workload,
            pixel_dimensions=pixel_dimensions,
        )

        # 2. Check if Routing is Globally Disabled by Config
        if not self._config.enabled:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return RoutingDecisionG6(
                decision_id=decision_id,
                timestamp=ts_utc,
                target_device="CPU",
                confidence=1.0,
                policy_name="POLICY_DISABLED",
                primary_reason="Adaptive routing disabled by configuration; defaulting to CPU.",
                cpu_score=1.0,
                gpu_score=0.0,
                is_gpu_warm=is_gpu_warm,
                workload_cost=cost_profile,
                resource_snapshot=resource,
                routing_latency_ms=latency_ms,
                previous_device=self._previous_device,
                was_switched=False,
                hysteresis_applied=False,
            )

        # 3. Calculate Backend Suitability Scores
        suitability = self._suitability_estimator.estimate_suitability(
            cost=cost_profile,
            resource=resource,
            is_gpu_warm=is_gpu_warm,
            queue_length=queue_length,
            active_workers=active_workers,
        )

        candidate_device = suitability.recommended_backend
        final_device = candidate_device
        final_reason = suitability.primary_reason
        was_switched = False
        hysteresis_applied = False

        # 4. Anti-Oscillation Hysteresis Filter
        if self._previous_device is None:
            # Initial decision: establish baseline device and switch timestamp
            self._last_switch_timestamp = now_mono
        elif self._config.hysteresis.enabled and candidate_device != self._previous_device:
            elapsed = now_mono - self._last_switch_timestamp
            cooldown = self._config.hysteresis.cooldown_seconds

            if elapsed < cooldown:
                # Cooldown active: verify if critical override applies
                can_switch = False

                # Critical condition 1: GPU hardware became unavailable
                if candidate_device == "CPU" and (
                    resource is not None and not resource.gpu_available
                ):
                    can_switch = True

                # Critical condition 2: Severe GPU overload
                if candidate_device == "CPU" and resource is not None:
                    if (
                        resource.gpu_utilization_percent is not None
                        and resource.gpu_utilization_percent >= 90.0
                    ):
                        can_switch = True

                if not can_switch:
                    # Suppress device flip
                    final_device = self._previous_device
                    final_reason = (
                        f"{suitability.primary_reason} [Hysteresis retained {self._previous_device}; "
                        f"{cooldown - elapsed:.1f}s cooldown remaining]"
                    )
                    hysteresis_applied = True
                else:
                    was_switched = True
                    self._last_switch_timestamp = now_mono
            else:
                was_switched = True
                self._last_switch_timestamp = now_mono
        elif candidate_device != self._previous_device:
            was_switched = True
            self._last_switch_timestamp = now_mono

        self._previous_device = final_device
        latency_ms = (time.perf_counter() - t0) * 1000.0

        decision = RoutingDecisionG6(
            decision_id=decision_id,
            timestamp=ts_utc,
            target_device=final_device,
            confidence=suitability.confidence,
            policy_name="G6_DATA_DRIVEN",
            primary_reason=final_reason,
            cpu_score=suitability.cpu_score,
            gpu_score=suitability.gpu_score,
            is_gpu_warm=is_gpu_warm,
            workload_cost=cost_profile,
            resource_snapshot=resource,
            routing_latency_ms=latency_ms,
            previous_device=self._previous_device,
            was_switched=was_switched,
            hysteresis_applied=hysteresis_applied,
            fallback_occurred=False,
        )

        # Record metrics using adapter
        self._record_metrics_adapter(decision, cost_profile, workload, resource)
        logger.debug("AdaptiveWorkRouterG6: %s", decision.format_summary())
        return decision

    def record_execution_fallback(
        self,
        failed_device: str,
        fallback_device: str,
        error_message: str,
        workload: WorkloadProfile,
    ) -> RoutingDecisionG6:
        """Record an execution failure and seamless fallback event."""
        logger.warning(
            "AdaptiveWorkRouterG6: Fallback to %s after %s failure: %s",
            fallback_device, failed_device, error_message,
        )
        self._metrics.record_gpu_failure()
        decision_id = str(uuid.uuid4())
        ts_utc = datetime.now(timezone.utc).isoformat()
        reason = (
            f"Execution fallback to {fallback_device} triggered after {failed_device} "
            f"runtime failure: {error_message}"
        )

        cost = self._cost_estimator.estimate_cost(workload)
        fb_decision = RoutingDecisionG6(
            decision_id=decision_id,
            timestamp=ts_utc,
            target_device="FALLBACK",
            confidence=1.0,
            policy_name="G6_DATA_DRIVEN",
            primary_reason=reason,
            cpu_score=1.0,
            gpu_score=0.0,
            is_gpu_warm=True,
            workload_cost=cost,
            resource_snapshot=None,
            routing_latency_ms=0.0,
            previous_device=failed_device,
            was_switched=True,
            hysteresis_applied=False,
            fallback_occurred=True,
        )
        self._record_metrics_adapter(fb_decision, cost, workload, None)
        return fb_decision

    def _record_metrics_adapter(
        self,
        decision: RoutingDecisionG6,
        cost: WorkloadCostProfile,
        workload: WorkloadProfile,
        resource: ResourceSnapshot | None,
    ) -> None:
        """Adapt G6 decision to internal metrics collector."""
        from adaptive_framework.acceleration.adaptive_router import RoutingDecision

        g4_dec = RoutingDecision(
            decision_id=decision.decision_id,
            timestamp=decision.timestamp,
            target_device=decision.target_device,
            reason=decision.primary_reason,
            confidence=decision.confidence,
            workload_profile=workload,
            resource_snapshot=resource,
            policy_version="2.0.0-g6",
            routing_latency_ms=decision.routing_latency_ms,
            previous_device=decision.previous_device,
            was_switched=decision.was_switched,
            applied_rule=decision.primary_reason,
            hysteresis_applied=decision.hysteresis_applied,
        )
        self._metrics.record_decision(g4_dec)
