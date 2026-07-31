"""Regime-level empirical oracle with disjoint training draws."""

from __future__ import annotations

from eri.benchmark.baselines import abstention_execution, execute_method
from eri.benchmark.losses import (
    ScoredExecution,
    aggregate_expected_loss,
    score_execution,
)
from eri.benchmark.registry import (
    METHOD_IDS,
    eligibility_for,
    eligible_method_ids,
    generate_scenario,
    label_for,
)
from eri.benchmark.schemas import (
    BenchmarkConfig,
    EligibilityStatus,
    LossProfile,
    OracleChoice,
    ScenarioConfig,
)


def estimate_oracle(
    scenario: ScenarioConfig,
    config: BenchmarkConfig,
    profile: LossProfile,
) -> OracleChoice:
    """Select the lowest expected-loss eligible action using training seeds."""
    training_draws = tuple(
        generate_scenario(scenario, seed=seed) for seed in config.oracle_training_seeds
    )
    reference = training_draws[0]
    labels = eligibility_for(reference)
    candidates = eligible_method_ids(labels)
    if not candidates:
        if label_for(labels, "abstention").status is not EligibilityStatus.ELIGIBLE:
            raise ValueError("no eligible inference method or abstention action")
        candidates = ("abstention",)
    expected_losses: dict[str, float] = {}
    for method_id in candidates:
        scored: list[ScoredExecution] = []
        for simulation in training_draws:
            execution = (
                abstention_execution()
                if method_id == "abstention"
                else execute_method(method_id, simulation, config)
            )
            scored.append(
                score_execution(
                    execution,
                    simulation,
                    eligibility_for(simulation),
                    profile,
                    confidence_level=config.confidence_level,
                )
            )
        expected_losses[method_id] = aggregate_expected_loss(
            tuple(scored),
            profile,
            confidence_level=config.confidence_level,
        )
    tie_order = {method_id: index for index, method_id in enumerate(METHOD_IDS)}
    selected = min(
        expected_losses,
        key=lambda method_id: (expected_losses[method_id], tie_order[method_id]),
    )
    return OracleChoice(
        scenario_id=scenario.scenario_id,
        method_id=selected,
        expected_losses=expected_losses,
        training_seeds=config.oracle_training_seeds,
        tie_break_rule="minimum expected loss, then registered METHOD_IDS order",
    )
