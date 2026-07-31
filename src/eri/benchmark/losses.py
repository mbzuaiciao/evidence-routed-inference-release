"""Explicit policy-weighted Phase 3A loss calculations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from eri.benchmark.registry import eligible_method_ids, label_for
from eri.benchmark.schemas import (
    EligibilityLabel,
    EligibilityStatus,
    LossProfile,
    MethodExecution,
    SimulationResult,
)


@dataclass(frozen=True, slots=True)
class ScoredExecution:
    covered: bool | None
    interval_width: float | None
    selected_label: EligibilityStatus | None
    valid_selection: bool
    invalid_selection: bool
    harmful_nonabstention: bool
    unnecessary_abstention: bool
    coverage_applicable: bool
    components: dict[str, float]
    total: float


def score_execution(
    execution: MethodExecution,
    simulation: SimulationResult,
    labels: tuple[EligibilityLabel, ...],
    profile: LossProfile,
    *,
    confidence_level: float,
) -> ScoredExecution:
    """Score one observed action; weights remain explicit policy choices."""
    selected_label = (
        label_for(labels, execution.selected_method).status
        if execution.selected_method is not None
        else None
    )
    valid_selection = (
        selected_label is EligibilityStatus.ELIGIBLE
        and execution.execution_failure is None
    )
    invalid_selection = (
        execution.selected_method is not None
        and selected_label
        in (
            EligibilityStatus.INELIGIBLE,
            EligibilityStatus.TARGET_NOT_WELL_DEFINED,
        )
    ) or execution.execution_failure is not None
    has_eligible_method = bool(eligible_method_ids(labels))
    harmful_nonabstention = not execution.abstained and (
        not simulation.target_defined or not has_eligible_method
    )
    unnecessary_abstention = (
        execution.abstained and simulation.target_defined and has_eligible_method
    )
    interval_width = None
    covered = None
    coverage_applicable = simulation.target_defined and not execution.abstained
    if execution.confidence_interval is not None:
        lower, upper = execution.confidence_interval
        interval_width = upper - lower
        if simulation.target_defined and simulation.true_sharpe is not None:
            covered = lower <= simulation.true_sharpe <= upper
    elif coverage_applicable:
        covered = False
    coverage_error = (
        abs(float(covered) - confidence_level) if covered is not None else 0.0
    )
    normalized_width = (
        interval_width
        / max(profile.interval_width_scale, abs(simulation.true_sharpe or 0.0))
        if interval_width is not None and simulation.target_defined
        else 0.0
    )
    runtime = execution.routing_runtime_seconds + execution.inference_runtime_seconds
    components = {
        "coverage_error": coverage_error,
        "normalized_interval_width": normalized_width,
        "invalid_method": float(invalid_selection),
        "harmful_nonabstention": float(harmful_nonabstention),
        "unnecessary_abstention": float(unnecessary_abstention),
        "execution_failure": float(execution.execution_failure is not None),
        "abstention_cost": float(execution.abstained),
        "normalized_runtime": runtime / profile.runtime_scale_seconds,
    }
    total = _weighted_total(components, profile)
    return ScoredExecution(
        covered=covered,
        interval_width=interval_width,
        selected_label=selected_label,
        valid_selection=valid_selection,
        invalid_selection=invalid_selection,
        harmful_nonabstention=harmful_nonabstention,
        unnecessary_abstention=unnecessary_abstention,
        coverage_applicable=coverage_applicable,
        components=components,
        total=total,
    )


def aggregate_expected_loss(
    scored: tuple[ScoredExecution, ...],
    profile: LossProfile,
    *,
    confidence_level: float,
) -> float:
    """Estimate regime-level expected loss with aggregate coverage error."""
    if not scored:
        raise ValueError("expected loss requires at least one scored execution")
    covered = [float(item.covered) for item in scored if item.covered is not None]
    widths = [
        item.components["normalized_interval_width"]
        for item in scored
        if item.interval_width is not None
    ]
    components = {
        "coverage_error": abs(mean(covered) - confidence_level) if covered else 0.0,
        "normalized_interval_width": mean(widths) if widths else 0.0,
        "invalid_method": mean(item.components["invalid_method"] for item in scored),
        "harmful_nonabstention": mean(
            item.components["harmful_nonabstention"] for item in scored
        ),
        "unnecessary_abstention": mean(
            item.components["unnecessary_abstention"] for item in scored
        ),
        "execution_failure": mean(
            item.components["execution_failure"] for item in scored
        ),
        "abstention_cost": mean(item.components["abstention_cost"] for item in scored),
        "normalized_runtime": mean(
            item.components["normalized_runtime"] for item in scored
        ),
    }
    return _weighted_total(components, profile)


def _weighted_total(components: dict[str, float], profile: LossProfile) -> float:
    return (
        profile.coverage_error * components["coverage_error"]
        + profile.interval_width * components["normalized_interval_width"]
        + profile.invalid_method * components["invalid_method"]
        + profile.harmful_nonabstention * components["harmful_nonabstention"]
        + profile.unnecessary_abstention * components["unnecessary_abstention"]
        + profile.execution_failure * components["execution_failure"]
        + profile.abstention * components["abstention_cost"]
        + profile.runtime * components["normalized_runtime"]
    )
