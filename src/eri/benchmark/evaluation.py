"""Per-run evaluation and benchmark metric calculations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from statistics import mean

import numpy as np

from eri.benchmark.losses import score_execution
from eri.benchmark.schemas import (
    BaselineName,
    BenchmarkConfig,
    EligibilityLabel,
    LedgerRecord,
    LossProfile,
    MethodExecution,
    SimulationResult,
)

_REGRET_TOLERANCE = 1e-12


def draw_fingerprint(simulation: SimulationResult) -> str:
    """Hash the exact return sequence in a platform-independent float64 form."""
    values = np.asarray(simulation.returns, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def build_record(
    *,
    run_id: str,
    scenario_id: str,
    baseline: BaselineName,
    simulation: SimulationResult,
    labels: tuple[EligibilityLabel, ...],
    execution: MethodExecution,
    oracle_method: str,
    oracle_execution: MethodExecution,
    config: BenchmarkConfig,
    profile: LossProfile,
) -> LedgerRecord:
    """Combine independent truth, observed action, and oracle counterfactual."""
    selected_score = score_execution(
        execution,
        simulation,
        labels,
        profile,
        confidence_level=config.confidence_level,
    )
    oracle_score = score_execution(
        oracle_execution,
        simulation,
        labels,
        profile,
        confidence_level=config.confidence_level,
    )
    raw_regret = selected_score.total - oracle_score.total
    regret = 0.0 if baseline is BaselineName.ORACLE else raw_regret
    if abs(regret) <= _REGRET_TOLERANCE:
        regret = 0.0
    return LedgerRecord(
        run_id=run_id,
        benchmark_version=config.benchmark_version,
        scenario_id=scenario_id,
        process_family=simulation.process_family,
        process_version=simulation.process_version,
        process_parameters=simulation.parameters,
        seed=simulation.seed,
        sample_size=simulation.sample_size,
        draw_fingerprint=draw_fingerprint(simulation),
        inference_truth=simulation.inference_truth,
        target_defined=simulation.target_defined,
        true_mean=simulation.true_mean,
        true_standard_deviation=simulation.true_standard_deviation,
        true_sharpe=simulation.true_sharpe,
        descriptive_full_sample_sharpe=simulation.descriptive_full_sample_sharpe,
        eligibility_labels=labels,
        baseline=baseline,
        selected_method=execution.selected_method,
        sensitivity_methods=execution.sensitivity_methods,
        abstained=execution.abstained,
        evidence_ids=execution.evidence_ids,
        unresolved_conflicts=execution.unresolved_conflicts,
        estimate=execution.estimate,
        confidence_interval=execution.confidence_interval,
        coverage_applicable=selected_score.coverage_applicable,
        covered=selected_score.covered,
        interval_width=selected_score.interval_width,
        execution_failure=execution.execution_failure,
        routing_runtime_seconds=execution.routing_runtime_seconds,
        inference_runtime_seconds=execution.inference_runtime_seconds,
        oracle_method=oracle_method,
        oracle_total_loss=oracle_score.total,
        loss_profile=profile.name,
        selected_label=selected_score.selected_label,
        valid_selection=selected_score.valid_selection,
        invalid_selection=selected_score.invalid_selection,
        harmful_nonabstention=selected_score.harmful_nonabstention,
        unnecessary_abstention=selected_score.unnecessary_abstention,
        loss_components=selected_score.components,
        total_loss=selected_score.total,
        regret=regret,
        warnings=execution.warnings,
    )


def metric_row(records: Sequence[LedgerRecord]) -> dict[str, int | float | str]:
    """Compute documented selection and operating metrics for one group."""
    if not records:
        raise ValueError("metric group must not be empty")
    count = len(records)
    nonabstaining = [record for record in records if not record.abstained]
    coverage_records = [
        record
        for record in records
        if record.target_defined is True
        and not record.abstained
        or record.target_defined is None
        and record.covered is not None
    ]
    coverage = [record.covered is True for record in coverage_records]
    widths = [
        float(record.interval_width)
        for record in records
        if record.interval_width is not None and record in coverage_records
    ]
    invalid_count = sum(record.invalid_selection for record in nonabstaining)
    covered_count = sum(value is True for value in coverage)
    abstention_count = sum(record.abstained for record in records)
    harmful_count = sum(record.harmful_nonabstention for record in records)
    unnecessary_count = sum(record.unnecessary_abstention for record in records)
    agreement_count = sum(
        (record.selected_method or "abstention") == record.oracle_method
        for record in records
    )
    failure_count = sum(record.execution_failure is not None for record in records)
    return {
        "runs": count,
        "valid_selection_numerator": sum(record.valid_selection for record in records),
        "valid_selection_denominator": count,
        "valid_method_selection_rate": mean(
            float(record.valid_selection) for record in records
        ),
        "invalid_method_selection_rate": (
            invalid_count / len(nonabstaining) if nonabstaining else 0.0
        ),
        "invalid_selection_numerator": invalid_count,
        "invalid_selection_denominator": len(nonabstaining),
        "confidence_interval_coverage": (
            mean(float(value) for value in coverage) if coverage else "not-applicable"
        ),
        "coverage_numerator": covered_count,
        "coverage_denominator": len(coverage),
        "selective_coverage": (
            mean(float(value) for value in coverage) if coverage else "not-applicable"
        ),
        "mean_interval_width": mean(widths) if widths else "not-applicable",
        "interval_width_denominator": len(widths),
        "abstention_rate": abstention_count / count,
        "abstention_numerator": abstention_count,
        "abstention_denominator": count,
        "harmful_nonabstention_rate": mean(
            float(record.harmful_nonabstention) for record in records
        ),
        "harmful_nonabstention_numerator": harmful_count,
        "harmful_nonabstention_denominator": count,
        "unnecessary_abstention_rate": mean(
            float(record.unnecessary_abstention) for record in records
        ),
        "unnecessary_abstention_numerator": unnecessary_count,
        "unnecessary_abstention_denominator": count,
        "oracle_agreement_rate": mean(
            float((record.selected_method or "abstention") == record.oracle_method)
            for record in records
        ),
        "oracle_agreement_numerator": agreement_count,
        "oracle_agreement_denominator": count,
        "mean_routing_regret": mean(record.regret for record in records),
        "mean_runtime_seconds": mean(
            record.routing_runtime_seconds + record.inference_runtime_seconds
            for record in records
        ),
        "execution_failure_rate": mean(
            float(record.execution_failure is not None) for record in records
        ),
        "execution_failure_numerator": failure_count,
        "execution_failure_denominator": count,
    }


def validate_shared_draws(records: Sequence[LedgerRecord]) -> None:
    """Require one immutable draw and exactly one row per baseline in each cell."""
    groups: dict[tuple[str, int, int], list[LedgerRecord]] = {}
    for record in records:
        key = (record.scenario_id, record.seed, record.sample_size)
        groups.setdefault(key, []).append(record)
    for key, group in groups.items():
        if len(group) != len(BaselineName):
            raise ValueError(f"shared-draw cell {key} does not contain every baseline")
        if {record.baseline for record in group} != set(BaselineName):
            raise ValueError(f"shared-draw cell {key} has duplicate/missing baselines")
        if len({record.draw_fingerprint for record in group}) != 1:
            raise ValueError(f"shared-draw cell {key} contains different return draws")
