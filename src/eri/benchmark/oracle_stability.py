"""Deterministic oracle-action stability analysis."""

from __future__ import annotations

import json
from collections import Counter
from statistics import mean

from eri.benchmark.oracle import estimate_oracle
from eri.benchmark.phase3b_schemas import Phase3BConfig, SeedManifest
from eri.benchmark.schemas import (
    BaselineName,
    BenchmarkConfig,
    LedgerRecord,
    LossProfile,
    ScenarioConfig,
)


def oracle_stability_rows(
    scenarios: tuple[ScenarioConfig, ...],
    benchmark_config: BenchmarkConfig,
    phase3b_config: Phase3BConfig,
    seed_manifest: SeedManifest,
    profile: LossProfile,
) -> list[dict[str, int | float | str | bool]]:
    """Classify oracle action stability across deterministic seed blocks."""
    rows: list[dict[str, int | float | str | bool]] = []
    for scenario in scenarios:
        for training_size in phase3b_config.oracle_stability_sizes:
            actions: list[str] = []
            losses: list[dict[str, float]] = []
            margins: list[float] = []
            unique_action = True
            for block in seed_manifest.oracle_stability_blocks:
                seeds = block[:training_size]
                configured = benchmark_config.model_copy(
                    update={"oracle_training_seeds": seeds}
                )
                choice = estimate_oracle(scenario, configured, profile)
                actions.append(choice.method_id)
                block_losses = dict(choice.expected_losses)
                losses.append(block_losses)
                ordered = sorted(block_losses.values())
                if len(ordered) > 1:
                    unique_action = False
                    margins.append(ordered[1] - ordered[0])
            counts = Counter(actions)
            change_frequency = 1.0 - max(counts.values()) / len(actions)
            minimum_margin = min(margins) if margins else 0.0
            average_margin = mean(margins) if margins else 0.0
            practically_tied = bool(
                margins and minimum_margin <= phase3b_config.oracle_weak_margin
            )
            status = classify_oracle_stability(
                actions,
                margins,
                unique_action=unique_action,
                weak_margin=float(phase3b_config.oracle_weak_margin),
            )
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "loss_profile": profile.name,
                    "training_size": training_size,
                    "blocks": len(actions),
                    "selected_action": counts.most_common(1)[0][0],
                    "block_actions": json.dumps(actions),
                    "block_expected_losses": json.dumps(losses, sort_keys=True),
                    "minimum_best_second_margin": minimum_margin,
                    "mean_best_second_margin": average_margin,
                    "action_change_frequency": change_frequency,
                    "practically_tied": practically_tied,
                    "stability_status": status,
                }
            )
    return rows


def classify_oracle_stability(
    actions: list[str],
    margins: list[float],
    *,
    unique_action: bool,
    weak_margin: float,
) -> str:
    """Classify block agreement and the best-versus-second-best loss margin."""
    if not actions:
        raise ValueError("oracle stability requires at least one block action")
    if unique_action:
        return "unique eligible action"
    if len(set(actions)) > 1:
        return "unstable"
    if margins and min(margins) <= weak_margin:
        return "weak-margin"
    return "stable"


def final_stability_summary(
    rows: list[dict[str, int | float | str | bool]], *, training_size: int
) -> dict[str, str]:
    """Return scenario-to-status labels at the final training size."""
    return {
        str(row["scenario_id"]): str(row["stability_status"])
        for row in rows
        if int(row["training_size"]) == training_size
    }


def oracle_agreement_by_stability(
    records: list[LedgerRecord],
    rows: list[dict[str, int | float | str | bool]],
    *,
    training_size: int,
) -> dict[str, dict[str, int | float]]:
    """Stratify router-oracle agreement by audited final stability status."""
    statuses = final_stability_summary(rows, training_size=training_size)
    router = [
        record
        for record in records
        if record.baseline is BaselineName.DETERMINISTIC_ROUTER
    ]
    groups = {
        "stable": {"stable"},
        "unique-eligible-action": {"unique eligible action"},
        "stable-or-unique": {"stable", "unique eligible action"},
        "weak-margin": {"weak-margin"},
        "unstable": {"unstable"},
    }
    output: dict[str, dict[str, int | float]] = {}
    for name, accepted in groups.items():
        selected = [
            record for record in router if statuses.get(record.scenario_id) in accepted
        ]
        numerator = sum(
            (record.selected_method or "abstention") == record.oracle_method
            for record in selected
        )
        denominator = len(selected)
        output[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator if denominator else 0.0,
        }
    return output
