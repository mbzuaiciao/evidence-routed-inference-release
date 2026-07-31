"""Unit checks for Phase 3B seeds, intervals, pairing, and confirmation."""

from __future__ import annotations

import shutil
from math import isclose
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from eri.benchmark.confirmation import (
    create_freeze_manifest,
    critical_paths,
    validate_freeze_manifest,
)
from eri.benchmark.monte_carlo import (
    normal_mean_interval,
    paired_comparison_rows,
    scenario_clustered_values,
    wilson_interval,
)
from eri.benchmark.oracle_stability import (
    classify_oracle_stability,
    oracle_agreement_by_stability,
)
from eri.benchmark.phase3b_runner import (
    build_seed_manifest,
    determine_claim_level,
    load_phase3b_config,
)
from eri.benchmark.phase3b_schemas import (
    Phase3BConfig,
    Phase3BRunResult,
)
from eri.benchmark.schemas import BaselineName, EligibilityStatus, LedgerRecord

ROOT = Path(__file__).resolve().parents[2]


def _config() -> Phase3BConfig:
    return load_phase3b_config(ROOT / "configs/benchmark/phase3b-pilot.yaml")


def _record(
    scenario: str,
    seed: int,
    baseline: BaselineName,
    *,
    invalid: bool,
    covered: bool | None,
    abstained: bool,
    regret: float,
) -> LedgerRecord:
    return cast(
        LedgerRecord,
        SimpleNamespace(
            scenario_id=scenario,
            seed=seed,
            baseline=baseline,
            invalid_selection=invalid,
            target_defined=covered is not None,
            selected_label=(None if abstained else EligibilityStatus.ELIGIBLE),
            covered=covered,
            abstained=abstained,
            regret=regret,
        ),
    )


def _paired_fixture() -> list[LedgerRecord]:
    records: list[LedgerRecord] = []
    for index, seed in enumerate((1, 2)):
        for baseline in BaselineName:
            records.append(
                _record(
                    "scenario-a",
                    seed,
                    baseline,
                    invalid=(
                        baseline is BaselineName.ALWAYS_GAUSSIAN
                        or baseline is BaselineName.DETERMINISTIC_ROUTER
                        and index == 1
                    ),
                    covered=(
                        index == 0
                        if baseline is BaselineName.DETERMINISTIC_ROUTER
                        else False
                    ),
                    abstained=False,
                    regret=(
                        float(index)
                        if baseline is BaselineName.DETERMINISTIC_ROUTER
                        else 0.0
                    ),
                )
            )
    return records


def test_seed_namespaces_are_deterministic_and_disjoint() -> None:
    first = build_seed_manifest(_config())
    second = build_seed_manifest(_config())
    assert first == second
    oracle: set[int] = set()
    for block in first.oracle_stability_blocks:
        oracle.update(block)
    namespaces: tuple[set[int], ...] = (
        oracle,
        set(first.development_evaluation),
        set(first.hackathon_evaluation),
        set(first.aaai_evaluation),
        set(first.frozen_confirmation),
    )
    for index, left in enumerate(namespaces):
        assert all(not left & right for right in namespaces[index + 1 :])


def test_seed_configuration_rejects_overlapping_namespaces() -> None:
    raw = _config().model_dump()
    seed_raw = dict(raw["seed_namespaces"])
    seed_raw["hackathon_evaluation_start"] = seed_raw["development_evaluation_start"]
    with pytest.raises(ValidationError, match="must not overlap"):
        Phase3BConfig.model_validate({**raw, "seed_namespaces": seed_raw})


def test_wilson_interval_matches_reference_and_serializes() -> None:
    interval = wilson_interval(5, 10, confidence_level=0.95)
    assert interval.estimate == 0.5
    assert isclose(interval.lower, 0.236593, abs_tol=1e-6)
    assert isclose(interval.upper, 0.763407, abs_tol=1e-6)
    assert interval.model_validate_json(interval.model_dump_json()) == interval


def test_wilson_boundary_interval_contains_observed_rate_exactly() -> None:
    zero = wilson_interval(0, 25, confidence_level=0.95)
    one = wilson_interval(25, 25, confidence_level=0.95)
    assert zero.estimate == zero.lower == 0.0
    assert one.estimate == one.upper == 1.0


def test_wilson_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="successes <= total"):
        wilson_interval(0, 0, confidence_level=0.95)


def test_regret_interval_reports_standard_error() -> None:
    interval = normal_mean_interval([1.0, 2.0, 3.0, 4.0], confidence_level=0.95)
    assert interval.estimate == 2.5
    assert interval.standard_error is not None
    assert isclose(interval.standard_error, 0.6454972244, abs_tol=1e-10)
    assert interval.lower < interval.estimate < interval.upper


def test_paired_differences_and_scenario_clustered_values() -> None:
    records = _paired_fixture()
    rows = paired_comparison_rows(records, confidence_level=0.95)
    comparison = next(
        row
        for row in rows
        if row["system_b"] == BaselineName.ALWAYS_GAUSSIAN.value
        and row["metric"] == "invalid_selection"
        and row["scenario_id"] == "all"
    )
    assert comparison["mean_paired_difference"] == -0.5
    assert comparison["paired_cells"] == 2
    assert comparison["scenario_clusters"] == 1
    assert scenario_clustered_values(records, "regret") == [0.125]


def test_paired_coverage_retains_ineligible_method_with_valid_outcome() -> None:
    records = _paired_fixture()
    gaussian = next(
        record for record in records if record.baseline is BaselineName.ALWAYS_GAUSSIAN
    )
    gaussian.selected_label = EligibilityStatus.INELIGIBLE
    rows = paired_comparison_rows(records, confidence_level=0.95)
    coverage = next(
        row
        for row in rows
        if row["system_b"] == BaselineName.ALWAYS_GAUSSIAN.value
        and row["metric"] == "coverage"
        and row["scenario_id"] == "all"
    )
    assert coverage["paired_cells"] == 2
    assert coverage["excluded_cells"] == 0
    assert coverage["denominator_definition"] == (
        "cells with defined targets and valid coverage outcomes for both systems"
    )


def test_paired_comparison_rejects_duplicate_system_rows() -> None:
    records = _paired_fixture()
    with pytest.raises(ValueError, match="duplicate system row"):
        paired_comparison_rows([*records, records[0]], confidence_level=0.95)


@pytest.mark.parametrize(
    ("actions", "margins", "unique", "expected"),
    [
        (["abstention", "abstention"], [], True, "unique eligible action"),
        (["hac", "bootstrap"], [0.1, 0.2], False, "unstable"),
        (["hac", "hac"], [0.001, 0.02], False, "weak-margin"),
        (["hac", "hac"], [0.2, 0.3], False, "stable"),
    ],
)
def test_oracle_stability_classification(
    actions: list[str], margins: list[float], unique: bool, expected: str
) -> None:
    assert (
        classify_oracle_stability(
            actions, margins, unique_action=unique, weak_margin=0.01
        )
        == expected
    )


def test_oracle_agreement_is_stratified_by_stability() -> None:
    records = [
        cast(
            LedgerRecord,
            SimpleNamespace(
                baseline=BaselineName.DETERMINISTIC_ROUTER,
                scenario_id=scenario,
                selected_method=selected,
                oracle_method=oracle,
            ),
        )
        for scenario, selected, oracle in (
            ("stable", "hac", "hac"),
            ("unstable", "hac", "bootstrap"),
            ("unique", None, "abstention"),
        )
    ]
    stability: list[dict[str, int | float | str | bool]] = [
        {
            "scenario_id": "stable",
            "training_size": 25,
            "stability_status": "stable",
        },
        {
            "scenario_id": "unstable",
            "training_size": 25,
            "stability_status": "unstable",
        },
        {
            "scenario_id": "unique",
            "training_size": 25,
            "stability_status": "unique eligible action",
        },
    ]
    result = oracle_agreement_by_stability(records, stability, training_size=25)
    assert result["stable"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert result["unstable"] == {
        "numerator": 0,
        "denominator": 1,
        "rate": 0.0,
    }
    assert result["stable-or-unique"] == {
        "numerator": 2,
        "denominator": 2,
        "rate": 1.0,
    }


def _copy_critical_state(tmp_path: Path) -> Path:
    source_config = ROOT / "configs/benchmark/phase3b-aaai.yaml"
    for source in critical_paths(ROOT, source_config):
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return tmp_path / source_config.relative_to(ROOT)


def test_freeze_manifest_hash_and_policy_mutation_invalidation(
    tmp_path: Path,
) -> None:
    config_path = _copy_critical_state(tmp_path)
    manifest_path = tmp_path / "freeze.json"
    manifest = create_freeze_manifest(tmp_path, config_path, manifest_path)
    assert isinstance(manifest["manifest_hash"], str)
    assert validate_freeze_manifest(
        tmp_path, manifest_path, expected_config_path=config_path
    ).valid

    wrong_config = tmp_path / "configs/benchmark/phase3b-pilot.yaml"
    wrong_config.write_text("fixture\n", encoding="utf-8")
    mismatch = validate_freeze_manifest(
        tmp_path, manifest_path, expected_config_path=wrong_config
    )
    assert not mismatch.valid
    assert "different benchmark configuration" in mismatch.reason

    policy = tmp_path / "src/eri/routing/router.py"
    policy.write_text(policy.read_text("utf-8") + "\n# mutated\n", encoding="utf-8")
    invalid = validate_freeze_manifest(tmp_path, manifest_path)
    assert not invalid.valid
    assert "src/eri/routing/router.py" in invalid.changed_paths


def test_claim_level_is_conservative() -> None:
    pilot = _config()
    assert determine_claim_level(pilot, confirmation_valid=False) == "smoke-only"
    assert determine_claim_level(pilot, confirmation_valid=True) == "smoke-only"
    unreviewed = pilot.model_copy(
        update={"run_label": "phase3b-hackathon", "evaluation_draws": 250}
    )
    assert (
        determine_claim_level(unreviewed, confirmation_valid=False)
        == "preliminary-descriptive"
    )
    reviewed = unreviewed.model_copy(update={"independent_review_complete": True})
    assert (
        determine_claim_level(reviewed, confirmation_valid=False)
        == "benchmark-supported"
    )
    assert (
        determine_claim_level(reviewed, confirmation_valid=True)
        == "benchmark-supported"
    )
    aaai = load_phase3b_config(ROOT / "configs/benchmark/phase3b-aaai.yaml").model_copy(
        update={"independent_review_complete": True}
    )
    assert determine_claim_level(aaai, confirmation_valid=True) == (
        "confirmation-supported"
    )


def test_result_contract_serialization() -> None:
    result = Phase3BRunResult(
        profile="validity-first",
        run_id="fixture",
        run_directory=Path("runs/phase3b/fixture"),
        record_count=64,
        permitted_claim_level="smoke-only",
        summary={"confirmation_status": "not-run", "estimate": 0.25},
    )
    assert Phase3BRunResult.model_validate_json(result.model_dump_json()) == result
