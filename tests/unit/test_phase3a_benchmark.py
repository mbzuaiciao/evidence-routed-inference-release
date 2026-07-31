"""Unit tests for Phase 3A contracts, registry, loss, oracle, and ledger."""

from math import isclose
from pathlib import Path

import pytest
from pydantic import ValidationError

import eri.benchmark.oracle as oracle_module
from eri.benchmark.baselines import abstention_execution, execute_method
from eri.benchmark.evaluation import build_record, draw_fingerprint, metric_row
from eri.benchmark.ledger import read_ledger, write_ledger
from eri.benchmark.losses import score_execution
from eri.benchmark.oracle import estimate_oracle
from eri.benchmark.registry import eligibility_for
from eri.benchmark.runner import load_benchmark_config, load_loss_profile
from eri.benchmark.schemas import (
    BaselineName,
    BenchmarkConfig,
    EligibilityStatus,
    LossProfile,
    MethodExecution,
    ScenarioConfig,
    SimulationResult,
)
from eri.simulation.ar1 import simulate_ar1
from eri.simulation.garch import simulate_garch
from eri.simulation.iid_gaussian import simulate_iid_gaussian
from eri.simulation.structural_break import simulate_structural_break

ROOT = Path(__file__).resolve().parents[2]


def _config() -> BenchmarkConfig:
    return load_benchmark_config(ROOT / "configs/benchmark/phase3a-smoke.yaml")


def _profile() -> LossProfile:
    return load_loss_profile(
        ROOT / "configs/benchmark/loss-profiles.yaml", "validity-first"
    )


def test_generators_are_reproducible() -> None:
    simulations = (
        simulate_iid_gaussian(
            sample_size=60,
            seed=1,
            parameters={"mean": 0.01, "volatility": 0.1},
        ),
        simulate_ar1(
            sample_size=60,
            seed=2,
            parameters={
                "mean": 0.01,
                "phi": 0.3,
                "innovation_std": 0.1,
                "burn_in": 20,
            },
        ),
        simulate_garch(
            sample_size=60,
            seed=3,
            parameters={
                "mean": 0.01,
                "omega": 0.0002,
                "alpha": 0.1,
                "beta": 0.8,
                "burn_in": 20,
            },
        ),
        simulate_structural_break(
            sample_size=60,
            seed=4,
            parameters={
                "variant": "mean",
                "mean_1": -0.02,
                "mean_2": 0.03,
                "volatility_1": 0.1,
                "volatility_2": 0.1,
                "break_fraction": 0.5,
            },
        ),
    )
    repeated = (
        simulate_iid_gaussian(
            sample_size=60,
            seed=1,
            parameters={"mean": 0.01, "volatility": 0.1},
        ),
        simulate_ar1(
            sample_size=60,
            seed=2,
            parameters={
                "mean": 0.01,
                "phi": 0.3,
                "innovation_std": 0.1,
                "burn_in": 20,
            },
        ),
        simulate_garch(
            sample_size=60,
            seed=3,
            parameters={
                "mean": 0.01,
                "omega": 0.0002,
                "alpha": 0.1,
                "beta": 0.8,
                "burn_in": 20,
            },
        ),
        simulate_structural_break(
            sample_size=60,
            seed=4,
            parameters={
                "variant": "mean",
                "mean_1": -0.02,
                "mean_2": 0.03,
                "volatility_1": 0.1,
                "volatility_2": 0.1,
                "break_fraction": 0.5,
            },
        ),
    )
    assert simulations == repeated


def test_generator_parameter_validation() -> None:
    with pytest.raises(ValueError, match=r"abs\(phi\)"):
        simulate_ar1(sample_size=60, seed=1, parameters={"phi": 1.0})
    with pytest.raises(ValueError, match=r"alpha \+ beta"):
        simulate_garch(
            sample_size=60,
            seed=1,
            parameters={"omega": 0.1, "alpha": 0.6, "beta": 0.4},
        )
    with pytest.raises(ValueError, match="distinct means"):
        simulate_structural_break(
            sample_size=60,
            seed=1,
            parameters={
                "variant": "mean",
                "mean_1": 0.0,
                "mean_2": 0.0,
                "volatility_1": 0.1,
                "volatility_2": 0.1,
            },
        )
    with pytest.raises(ValueError, match="equal volatility"):
        simulate_structural_break(
            sample_size=60,
            seed=1,
            parameters={
                "variant": "mean",
                "mean_1": 0.0,
                "mean_2": 0.1,
                "volatility_1": 0.1,
                "volatility_2": 0.2,
            },
        )
    with pytest.raises(ValueError, match="equal means"):
        simulate_structural_break(
            sample_size=60,
            seed=1,
            parameters={
                "variant": "variance",
                "mean_1": 0.0,
                "mean_2": 0.1,
                "volatility_1": 0.1,
                "volatility_2": 0.2,
            },
        )


def test_benchmark_configuration_rejects_overlapping_seeds() -> None:
    raw = _config().model_dump()
    with pytest.raises(ValidationError, match="must be disjoint"):
        BenchmarkConfig.model_validate(
            {**raw, "evaluation_seeds": raw["oracle_training_seeds"]}
        )


def test_eligibility_registry_is_independent_and_versioned() -> None:
    ar = simulate_ar1(
        sample_size=100,
        seed=5,
        parameters={"phi": 0.4, "innovation_std": 0.1, "burn_in": 20},
    )
    labels = {label.method_id: label for label in eligibility_for(ar)}
    assert labels["iid-gaussian-sharpe"].status is EligibilityStatus.INELIGIBLE
    assert labels["hac-sharpe"].status is EligibilityStatus.ELIGIBLE
    assert all(label.version == "phase3a-eligibility-1.0" for label in labels.values())


def test_registry_handles_zero_ar_dependence_and_garch_moment_boundary() -> None:
    zero_ar = simulate_ar1(
        sample_size=100,
        seed=6,
        parameters={"phi": 0.0, "innovation_std": 0.1, "burn_in": 20},
    )
    zero_ar_labels = {label.method_id: label for label in eligibility_for(zero_ar)}
    assert zero_ar_labels["iid-gaussian-sharpe"].status is EligibilityStatus.ELIGIBLE

    heavy_garch = simulate_garch(
        sample_size=100,
        seed=6,
        parameters={
            "omega": 0.0002,
            "alpha": 0.30,
            "beta": 0.65,
            "burn_in": 20,
        },
    )
    assert heavy_garch.inference_truth["finite_fourth_moment"] is False
    labels = {label.method_id: label for label in eligibility_for(heavy_garch)}
    assert labels["hac-sharpe"].status is EligibilityStatus.INELIGIBLE
    assert labels["circular-block-bootstrap"].status is EligibilityStatus.INELIGIBLE
    assert labels["abstention"].status is EligibilityStatus.ELIGIBLE


def test_loss_penalizes_harmful_and_unnecessary_actions() -> None:
    structural = simulate_structural_break(
        sample_size=80,
        seed=7,
        parameters={
            "variant": "variance",
            "mean_1": 0.01,
            "mean_2": 0.01,
            "volatility_1": 0.05,
            "volatility_2": 0.2,
        },
    )
    forced = execute_method("iid-gaussian-sharpe", structural, _config())
    harmful = score_execution(
        forced,
        structural,
        eligibility_for(structural),
        _profile(),
        confidence_level=0.95,
    )
    assert harmful.harmful_nonabstention and harmful.invalid_selection

    iid = simulate_iid_gaussian(
        sample_size=80,
        seed=8,
        parameters={"mean": 0.01, "volatility": 0.1},
    )
    unnecessary = score_execution(
        abstention_execution(),
        iid,
        eligibility_for(iid),
        _profile(),
        confidence_level=0.95,
    )
    assert unnecessary.unnecessary_abstention


def test_failed_execution_is_explicit_and_not_counted_as_valid() -> None:
    simulation = simulate_iid_gaussian(
        sample_size=80,
        seed=8,
        parameters={"mean": 0.01, "volatility": 0.1},
    )
    failed = MethodExecution(
        selected_method="iid-gaussian-sharpe",
        abstained=False,
        execution_failure="ArithmeticError: fixture",
        routing_runtime_seconds=0.0,
        inference_runtime_seconds=0.0,
    )
    scored = score_execution(
        failed,
        simulation,
        eligibility_for(simulation),
        _profile(),
        confidence_level=0.95,
    )
    assert not scored.valid_selection
    assert scored.invalid_selection
    assert scored.coverage_applicable
    assert scored.covered is False
    assert scored.components["execution_failure"] == 1.0
    assert scored.components["normalized_interval_width"] == 0.0


def test_loss_total_has_only_explicit_configured_components() -> None:
    simulation = simulate_iid_gaussian(
        sample_size=80,
        seed=18,
        parameters={"mean": 0.01, "volatility": 0.1},
    )
    execution = execute_method("iid-gaussian-sharpe", simulation, _config())
    profile = _profile()
    scored = score_execution(
        execution,
        simulation,
        eligibility_for(simulation),
        profile,
        confidence_level=0.95,
    )
    expected = (
        profile.coverage_error * scored.components["coverage_error"]
        + profile.interval_width * scored.components["normalized_interval_width"]
        + profile.invalid_method * scored.components["invalid_method"]
        + profile.harmful_nonabstention
        * scored.components["harmful_nonabstention"]
        + profile.unnecessary_abstention
        * scored.components["unnecessary_abstention"]
        + profile.execution_failure * scored.components["execution_failure"]
        + profile.abstention * scored.components["abstention_cost"]
        + profile.runtime * scored.components["normalized_runtime"]
    )
    assert isclose(scored.total, expected, rel_tol=0.0, abs_tol=1e-12)


def test_oracle_tie_breaking_is_deterministic() -> None:
    zero_profile = LossProfile(
        name="zero",
        coverage_error=0.0,
        interval_width=0.0,
        invalid_method=0.0,
        harmful_nonabstention=0.0,
        unnecessary_abstention=0.0,
        execution_failure=0.0,
        abstention=0.0,
        runtime=0.0,
        interval_width_scale=1.0,
        runtime_scale_seconds=1.0,
        notes="Tie-breaking test only.",
    )
    scenario = _config().scenarios[0]
    first = estimate_oracle(scenario, _config(), zero_profile)
    second = estimate_oracle(scenario, _config(), zero_profile)
    assert first == second
    assert first.method_id == "iid-gaussian-sharpe"


def test_oracle_reuses_one_training_draw_per_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = oracle_module.generate_scenario

    def counted_generate(
        scenario: ScenarioConfig, *, seed: int
    ) -> SimulationResult:
        nonlocal calls
        calls += 1
        return original(scenario, seed=seed)

    monkeypatch.setattr(oracle_module, "generate_scenario", counted_generate)
    estimate_oracle(_config().scenarios[0], _config(), _profile())
    assert calls == len(_config().oracle_training_seeds)


def test_ledger_serialization_and_metrics(tmp_path: Path) -> None:
    simulation = simulate_iid_gaussian(
        sample_size=80,
        seed=9,
        parameters={"mean": 0.01, "volatility": 0.1},
    )
    labels = eligibility_for(simulation)
    execution = execute_method("iid-gaussian-sharpe", simulation, _config())
    record = build_record(
        run_id="unit",
        scenario_id="iid-unit",
        baseline=BaselineName.ALWAYS_GAUSSIAN,
        simulation=simulation,
        labels=labels,
        execution=execution,
        oracle_method="iid-gaussian-sharpe",
        oracle_execution=execution,
        config=_config(),
        profile=_profile(),
    )
    path = tmp_path / "ledger.jsonl"
    write_ledger(path, (record,))
    assert read_ledger(path) == (record,)
    metrics = metric_row((record,))
    assert metrics["valid_method_selection_rate"] == 1.0
    assert metrics["oracle_agreement_rate"] == 1.0
    assert metrics["coverage_numerator"] in {0, 1}
    assert metrics["coverage_denominator"] == 1
    assert record.draw_fingerprint == draw_fingerprint(simulation)
    assert record.oracle_total_loss is not None
    assert isclose(
        record.regret,
        record.total_loss - record.oracle_total_loss,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
