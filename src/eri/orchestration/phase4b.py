"""Construction and execution of matched Phase 4B benchmark cells."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from eri.agents.provider import TypedProvider
from eri.benchmark.agent_baselines import SystemRun, run_system
from eri.benchmark.information_regimes import mask_evidence, shared_fingerprints
from eri.benchmark.phase4b_schemas import (
    EvidenceConditionConfig,
    Phase4BConfig,
    SharedCell,
)
from eri.benchmark.provenance_benchmark import (
    generate_provenance_case,
    structured_provenance,
)
from eri.benchmark.registry import eligibility_for, generate_scenario
from eri.benchmark.schemas import (
    BenchmarkConfig,
    LossProfile,
    ScenarioConfig,
    SimulationResult,
)
from eri.config import Phase2Config, load_yaml_model
from eri.diagnostics.registry import run_diagnostics
from eri.routing.method_catalog import METHOD_CATALOG


def load_scenarios(source: Path) -> tuple[ScenarioConfig, ...]:
    """Reuse the exact Phase 3 scenario source."""
    return load_yaml_model(source, BenchmarkConfig).scenarios


def build_shared_cell(
    *,
    scenario: ScenarioConfig,
    seed: int,
    condition: EvidenceConditionConfig,
    phase2: Phase2Config,
    config: Phase4BConfig,
) -> tuple[SharedCell, SimulationResult]:
    allowed_tools = config.allowed_tools
    simulation = generate_scenario(scenario, seed=seed)
    provenance = generate_provenance_case(
        condition.provenance_template, order_variant=seed
    )
    structured = (
        structured_provenance(provenance)
        if "structured-provenance" in condition.initial_groups
        else None
    )
    full_evidence = run_diagnostics(
        simulation.returns, phase2.diagnostics, provenance=structured
    )
    initial = mask_evidence(full_evidence, condition.initial_groups)
    method_metadata: dict[str, JsonValue] = {
        item.method_id: cast(JsonValue, json.loads(json.dumps(asdict(item))))
        for item in METHOD_CATALOG
        if item.method_id in allowed_tools or item.method_id == "abstention"
    }
    budget: dict[str, object] = {
        "diagnostic_budget": condition.diagnostic_budget,
        "diagnostic_costs": dict(condition.diagnostic_costs),
        "sensitivity_budget": condition.sensitivity_budget,
    }
    conventions: dict[str, object] = {
        "input_frequency": config.input_frequency,
        "periods_per_year": config.periods_per_year,
        "benchmark_sharpe": config.benchmark_sharpe,
    }
    fingerprints = shared_fingerprints(
        returns=simulation.returns,
        evidence=initial,
        prose=provenance.text,
        allowed_tools=allowed_tools,
        budget=budget,
        method_metadata=method_metadata,
        inference_conventions=conventions,
    )
    identity = (
        f"{scenario.scenario_id}:{seed}:{condition.regime.value}:"
        f"{condition.condition_id}"
    )
    cell = SharedCell(
        cell_id=identity,
        scenario=scenario,
        return_seed=seed,
        regime=condition.regime,
        condition_id=condition.condition_id,
        returns=simulation.returns,
        full_evidence=full_evidence,
        initial_evidence=initial,
        provenance_text=provenance.text,
        provenance_case_id=provenance.case_id,
        allowed_tools=allowed_tools,
        diagnostic_budget=condition.diagnostic_budget,
        sensitivity_budget=condition.sensitivity_budget,
        diagnostic_costs=condition.diagnostic_costs,
        critical_groups=condition.critical_groups,
        input_frequency=config.input_frequency,
        periods_per_year=config.periods_per_year,
        benchmark_sharpe=config.benchmark_sharpe,
        method_metadata=method_metadata,
        fingerprints=fingerprints,
        truth_labels=eligibility_for(simulation),
        truth={
            "target_defined": simulation.target_defined,
            "true_sharpe": simulation.true_sharpe,
            "process_family": simulation.process_family.value,
        },
    )
    return cell, simulation


def run_matched_cell(
    cell: SharedCell,
    simulation: SimulationResult,
    *,
    phase2: Phase2Config,
    benchmark: BenchmarkConfig,
    config: Phase4BConfig,
    profile: LossProfile,
    provider: TypedProvider | None = None,
) -> tuple[SystemRun, ...]:
    """Pair all six systems on the same immutable cell."""
    return tuple(
        run_system(
            system,
            cell,
            simulation,
            phase2,
            benchmark,
            config,
            provider,
            profile,
        )
        for system in config.systems
    )
