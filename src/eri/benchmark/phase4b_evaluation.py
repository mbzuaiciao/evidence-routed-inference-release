"""Paired metrics and claim gates for Phase 4B."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean

from eri.benchmark.losses import score_execution
from eri.benchmark.monte_carlo import normal_mean_interval, wilson_interval
from eri.benchmark.phase4b_schemas import (
    BenchmarkTier,
    ClaimLevel,
    Phase4BConfig,
    Phase4BSystem,
    SharedCell,
    SystemOutput,
    ToolCallRecord,
)
from eri.benchmark.registry import eligible_method_ids, label_for
from eri.benchmark.schemas import (
    EligibilityStatus,
    LossProfile,
    MethodExecution,
    SimulationResult,
)

PRIMARY_OUTCOMES = (
    "invalid_primary_method_selection",
    "harmful_nonabstention",
    "routing_regret",
)


def permitted_claim_level(
    config: Phase4BConfig,
    *,
    live_outputs_complete: bool = True,
    provider_metadata_complete: bool = True,
    unreviewed_scientific_override: bool = False,
) -> ClaimLevel:
    """Mock and pilots cannot support agent-performance claims."""
    if config.tier is BenchmarkTier.SMOKE:
        return ClaimLevel.IMPLEMENTATION_ONLY
    if not live_outputs_complete:
        return ClaimLevel.IMPLEMENTATION_ONLY
    if unreviewed_scientific_override:
        return ClaimLevel.IMPLEMENTATION_ONLY
    if config.tier is BenchmarkTier.LIVE_PILOT:
        return ClaimLevel.PILOT_DESCRIPTIVE
    if not provider_metadata_complete:
        return ClaimLevel.IMPLEMENTATION_ONLY
    if config.confirmation_run and config.independent_review_complete:
        return ClaimLevel.CONFIRMATION_SUPPORTED
    if config.tier is BenchmarkTier.DEADLINE and config.independent_review_complete:
        return ClaimLevel.BENCHMARK_SUPPORTED
    return ClaimLevel.IMPLEMENTATION_ONLY


def score_output(
    cell: SharedCell,
    output: SystemOutput,
    calls: Sequence[ToolCallRecord],
    simulation: SimulationResult,
    profile: LossProfile,
    *,
    confidence_level: float,
) -> dict[str, float | int | str]:
    selected_status = None
    if output.selected_method is not None:
        selected_status = label_for(cell.truth_labels, output.selected_method).status
    eligible = eligible_method_ids(cell.truth_labels)
    target_defined = bool(cell.truth.get("target_defined"))
    harmful = not output.abstained and (not target_defined or not eligible)
    unnecessary = output.abstained and target_defined and bool(eligible)
    covered = None
    true_sharpe = cell.truth.get("true_sharpe")
    if output.interval is not None and isinstance(true_sharpe, int | float):
        covered = output.interval[0] <= float(true_sharpe) <= output.interval[1]
    scored = score_execution(
        MethodExecution(
            selected_method=output.selected_method,
            abstained=output.abstained,
            estimate=output.estimate,
            confidence_interval=output.interval,
            execution_failure=output.execution_failure,
            routing_runtime_seconds=0.0,
            inference_runtime_seconds=0.0,
        ),
        simulation,
        cell.truth_labels,
        profile,
        confidence_level=confidence_level,
    )
    relevant_calls = sum(call.relevant for call in calls)
    evidence_gained = sum(call.evidence_items_gained for call in calls)
    critical_discovered = sum(call.critical_properties_discovered for call in calls)
    critical_total = len(cell.critical_groups)
    diagnostic_cost = sum(float(call.cost) for call in calls)
    return {
        "cell_id": cell.cell_id,
        "scenario_id": cell.scenario.scenario_id,
        "regime": cell.regime.value,
        "condition_id": cell.condition_id,
        "system": output.system.value,
        "selected_action": output.selected_method or "abstention",
        "valid_method_selection": int(selected_status is EligibilityStatus.ELIGIBLE),
        "invalid_primary_method_selection": int(scored.invalid_selection),
        "selective_coverage": "" if covered is None else int(covered),
        "harmful_nonabstention": int(harmful),
        "unnecessary_abstention": int(unnecessary),
        "execution_failure": int(output.execution_failure is not None),
        "oracle_agreement": 0,
        "routing_loss": scored.total,
        "routing_regret": 0.0,
        "diagnostic_calls": len(calls),
        "relevant_diagnostic_requests": relevant_calls,
        "irrelevant_diagnostic_requests": len(calls) - relevant_calls,
        "duplicate_requests": sum(call.duplicate for call in calls),
        "relevant_diagnostic_request_rate": (
            relevant_calls / len(calls) if calls else 0.0
        ),
        "irrelevant_diagnostic_request_rate": (
            (len(calls) - relevant_calls) / len(calls) if calls else 0.0
        ),
        "duplicate_request_rate": (
            sum(call.duplicate for call in calls) / len(calls) if calls else 0.0
        ),
        "diagnostic_cost": diagnostic_cost,
        "budget_efficiency": (
            critical_discovered / diagnostic_cost
            if diagnostic_cost
            else float(critical_total == 0)
        ),
        "evidence_gained": evidence_gained,
        "evidence_gained_per_call": evidence_gained / len(calls) if calls else 0.0,
        "critical_discovery_fraction": (
            critical_discovered / critical_total if critical_total else 1.0
        ),
        "missed_critical_diagnostic": int(
            critical_total > 0 and critical_discovered < critical_total
        ),
        "eligibility_override_attempts": int(
            output.recommended_method_eligible is False
        ),
        "unsupported_claims": int(
            bool(output.final_claims) and not output.claim_evidence_ids
        ),
        "unknown_tool_requests": len(output.unknown_tool_requests),
        "untraceable_report_claims": int(
            bool(output.final_claims) and not output.claim_evidence_ids
        ),
        "guardrail_rejections": len(output.guardrail_rejections),
        "audit_complete": int(bool(output.audit_events)),
        "final_claim_valid": int(
            not output.execution_failure and not scored.invalid_selection
        ),
    }


def aggregate_metric_rows(
    rows: Sequence[Mapping[str, float | int | str]], *, group_keys: tuple[str, ...]
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, float | int | str]]] = defaultdict(
        list
    )
    for source_row in rows:
        grouped[tuple(str(source_row[key]) for key in group_keys)].append(source_row)
    metrics = (
        "valid_method_selection",
        "invalid_primary_method_selection",
        "selective_coverage",
        "harmful_nonabstention",
        "unnecessary_abstention",
        "execution_failure",
        "oracle_agreement",
        "routing_regret",
        "diagnostic_calls",
        "relevant_diagnostic_requests",
        "irrelevant_diagnostic_requests",
        "duplicate_requests",
        "relevant_diagnostic_request_rate",
        "irrelevant_diagnostic_request_rate",
        "duplicate_request_rate",
        "diagnostic_cost",
        "budget_efficiency",
        "evidence_gained",
        "evidence_gained_per_call",
        "critical_discovery_fraction",
        "missed_critical_diagnostic",
        "eligibility_override_attempts",
        "unsupported_claims",
        "unknown_tool_requests",
        "untraceable_report_claims",
        "guardrail_rejections",
        "audit_complete",
        "final_claim_valid",
    )
    rate_metrics = {
        "valid_method_selection",
        "invalid_primary_method_selection",
        "selective_coverage",
        "harmful_nonabstention",
        "unnecessary_abstention",
        "execution_failure",
        "oracle_agreement",
        "missed_critical_diagnostic",
        "eligibility_override_attempts",
        "unsupported_claims",
        "untraceable_report_claims",
        "audit_complete",
        "final_claim_valid",
    }
    output: list[dict[str, float | int | str]] = []
    for group, items in sorted(grouped.items()):
        row: dict[str, float | int | str] = {
            **dict(zip(group_keys, group, strict=True)),
            "n": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items if item[metric] != ""]
            if not values:
                continue
            row[metric] = mean(values)
            if metric in rate_metrics:
                numerator = sum(int(value) for value in values)
                interval = wilson_interval(
                    numerator, len(values), confidence_level=0.95
                )
                row[f"{metric}_numerator"] = numerator
                row[f"{metric}_denominator"] = len(values)
                row[f"{metric}_wilson_lower"] = interval.lower
                row[f"{metric}_wilson_upper"] = interval.upper
        output.append(row)
    return output


def paired_comparisons(
    rows: Sequence[Mapping[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    by_cell: dict[str, dict[str, Mapping[str, float | int | str]]] = defaultdict(dict)
    for row in rows:
        by_cell[str(row["cell_id"])][str(row["system"])] = row
    reference = Phase4BSystem.STATIC_DETERMINISTIC.value
    output: list[dict[str, float | int | str]] = []
    for system in Phase4BSystem:
        if system.value == reference:
            continue
        matched = [
            (cell[system.value], cell[reference])
            for cell in by_cell.values()
            if system.value in cell and reference in cell
        ]
        for metric in PRIMARY_OUTCOMES:
            differences = [
                float(left[metric]) - float(right[metric]) for left, right in matched
            ]
            by_scenario: dict[str, list[float]] = defaultdict(list)
            for left, right in matched:
                by_scenario[str(left["scenario_id"])].append(
                    float(left[metric]) - float(right[metric])
                )
            cluster_means = [mean(values) for values in by_scenario.values()]
            interval = normal_mean_interval(
                cluster_means or [0.0],
                confidence_level=0.95,
                method="normal-scenario-clustered-descriptive",
                clusters=len(cluster_means),
            )
            output.append(
                {
                    "system": system.value,
                    "reference": reference,
                    "metric": metric,
                    "paired_n": len(differences),
                    "available_cells": len(by_cell),
                    "excluded_unmatched_cells": len(by_cell) - len(differences),
                    "paired_mean_difference": mean(differences) if differences else 0.0,
                    "scenario_clusters": len(cluster_means),
                    "clustered_lower": interval.lower,
                    "clustered_upper": interval.upper,
                    "clustered_standard_error": interval.standard_error or 0.0,
                    "uncertainty_unit": (
                        "matched cell; scenario-clustered descriptive"
                    ),
                }
            )
    return output


def rate_interval(numerator: int, denominator: int) -> dict[str, float | int | str]:
    return wilson_interval(numerator, denominator, confidence_level=0.95).model_dump()
