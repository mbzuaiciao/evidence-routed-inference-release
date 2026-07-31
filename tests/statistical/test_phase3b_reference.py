"""Statistical reference properties for Phase 3B uncertainty and stability."""

from __future__ import annotations

from math import comb, isclose, sqrt
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import cast

import pytest

import eri.benchmark.oracle_stability as stability_module
from eri.benchmark.monte_carlo import (
    normal_mean_interval,
    paired_comparison_rows,
    wilson_interval,
)
from eri.benchmark.oracle_stability import oracle_stability_rows
from eri.benchmark.phase3b_runner import build_seed_manifest, load_phase3b_config
from eri.benchmark.runner import load_benchmark_config, load_loss_profile
from eri.benchmark.schemas import (
    BaselineName,
    BenchmarkConfig,
    EligibilityStatus,
    LedgerRecord,
    LossProfile,
    OracleChoice,
    ScenarioConfig,
)

ROOT = Path(__file__).resolve().parents[2]


def test_wilson_nominal_behavior_on_controlled_binomial_fixture() -> None:
    probability = 0.3
    total = 50
    exact_coverage = 0.0
    for successes in range(total + 1):
        interval = wilson_interval(successes, total, confidence_level=0.95)
        if interval.lower <= probability <= interval.upper:
            exact_coverage += (
                comb(total, successes)
                * probability**successes
                * (1.0 - probability) ** (total - successes)
            )
    assert 0.94 <= exact_coverage <= 0.98


@pytest.mark.parametrize(
    ("successes", "total"),
    [(0, 1459), (1363, 1459), (541, 2000), (1710, 2000), (25, 25)],
)
def test_wilson_matches_independent_standard_library_implementation(
    successes: int, total: int
) -> None:
    z_value = NormalDist().inv_cdf(0.975)
    estimate = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (estimate + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * sqrt(estimate * (1.0 - estimate) / total + z_value**2 / (4.0 * total**2))
        / denominator
    )
    expected_lower = 0.0 if successes == 0 else center - half_width
    expected_upper = 1.0 if successes == total else center + half_width
    observed = wilson_interval(successes, total, confidence_level=0.95)
    assert isclose(observed.lower, expected_lower, abs_tol=1e-15)
    assert isclose(observed.upper, expected_upper, abs_tol=1e-15)


def test_mean_interval_tightens_with_more_deterministic_draws() -> None:
    small = normal_mean_interval([0.0, 1.0] * 10, confidence_level=0.95)
    large = normal_mean_interval([0.0, 1.0] * 100, confidence_level=0.95)
    assert small.estimate == large.estimate == 0.5
    assert large.standard_error is not None and small.standard_error is not None
    assert large.standard_error < small.standard_error
    assert large.upper - large.lower < small.upper - small.lower


def _paired_record(
    scenario: str, seed: int, baseline: BaselineName, invalid: bool
) -> LedgerRecord:
    return cast(
        LedgerRecord,
        SimpleNamespace(
            scenario_id=scenario,
            seed=seed,
            baseline=baseline,
            invalid_selection=invalid,
            target_defined=True,
            selected_label=EligibilityStatus.ELIGIBLE,
            covered=True,
            abstained=False,
            regret=0.0,
        ),
    )


def test_paired_difference_uses_scenario_cluster_means() -> None:
    records: list[LedgerRecord] = []
    for scenario, router_invalid, gaussian_invalid in (
        ("a", False, True),
        ("b", True, False),
    ):
        for seed in (1, 2):
            for baseline in BaselineName:
                invalid = (
                    router_invalid
                    if baseline is BaselineName.DETERMINISTIC_ROUTER
                    else gaussian_invalid
                    if baseline is BaselineName.ALWAYS_GAUSSIAN
                    else False
                )
                records.append(_paired_record(scenario, seed, baseline, invalid))
    row = next(
        item
        for item in paired_comparison_rows(records, confidence_level=0.95)
        if item["system_b"] == BaselineName.ALWAYS_GAUSSIAN.value
        and item["metric"] == "invalid_selection"
        and item["scenario_id"] == "all"
    )
    assert row["mean_paired_difference"] == 0.0
    assert isclose(float(row["standard_error"]), 1.0, abs_tol=1e-12)
    assert row["paired_cells"] == 4
    assert row["scenario_clusters"] == 2


def test_oracle_converges_with_wider_training_and_has_no_evaluation_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase3b = load_phase3b_config(
        ROOT / "configs/benchmark/phase3b-pilot.yaml"
    ).model_copy(
        update={
            "oracle_training_draws": 2,
            "oracle_stability_sizes": (1, 2),
            "oracle_stability_blocks": 2,
        }
    )
    manifest = build_seed_manifest(phase3b)
    benchmark = load_benchmark_config(
        ROOT / "configs/benchmark/phase3a-mini.yaml"
    ).model_copy(
        update={
            "oracle_training_seeds": manifest.oracle_training,
            "evaluation_seeds": (999_001,),
        }
    )
    scenario = benchmark.scenarios[0]
    profile = load_loss_profile(
        ROOT / "configs/benchmark/loss-profiles.yaml", "validity-first"
    )
    seen_training: list[tuple[int, ...]] = []

    def deterministic_loss_matrix(
        scenario: ScenarioConfig,
        config: BenchmarkConfig,
        profile: LossProfile,
    ) -> OracleChoice:
        del profile
        training = config.oracle_training_seeds
        seen_training.append(training)
        if len(training) == 1:
            action = "circular-block-bootstrap"
            losses = {"hac-sharpe": 0.3, "circular-block-bootstrap": 0.2}
        else:
            action = "hac-sharpe"
            losses = {"hac-sharpe": 0.1, "circular-block-bootstrap": 0.2}
        return OracleChoice(
            scenario_id=scenario.scenario_id,
            method_id=action,
            expected_losses=losses,
            training_seeds=training,
            tie_break_rule="fixture",
        )

    monkeypatch.setattr(stability_module, "estimate_oracle", deterministic_loss_matrix)
    first = oracle_stability_rows((scenario,), benchmark, phase3b, manifest, profile)
    changed_evaluation = benchmark.model_copy(update={"evaluation_seeds": (123_456,)})
    second = oracle_stability_rows(
        (scenario,), changed_evaluation, phase3b, manifest, profile
    )
    assert first == second
    assert first[0]["selected_action"] == "circular-block-bootstrap"
    assert first[-1]["selected_action"] == "hac-sharpe"
    assert all(999_001 not in seeds and 123_456 not in seeds for seeds in seen_training)
