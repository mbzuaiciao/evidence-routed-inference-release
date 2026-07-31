"""Lightweight end-to-end checks for the two-profile Phase 3B workflow."""

from __future__ import annotations

import csv
import json
import subprocess
from math import isclose
from pathlib import Path

import pytest
import yaml

from eri.benchmark.confirmation import (
    create_freeze_manifest,
    validate_freeze_manifest,
)
from eri.benchmark.ledger import read_ledger
from eri.benchmark.phase3b_runner import load_phase3b_config, run_phase3b
from eri.benchmark.phase3b_schemas import Phase3BRunResult
from eri.benchmark.runner import load_loss_profile
from eri.benchmark.schemas import BaselineName, LedgerRecord

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def phase3b_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[tuple[Phase3BRunResult, ...], dict[str, tuple[LedgerRecord, ...]]]:
    output_root = tmp_path_factory.mktemp("phase3b-output")
    config_directory = tmp_path_factory.mktemp("phase3b-config")
    source = load_phase3b_config(ROOT / "configs/benchmark/phase3b-pilot.yaml")
    raw = source.model_dump(mode="json")
    seed_namespaces = dict(raw["seed_namespaces"])
    seed_namespaces["development_evaluation_count"] = 2
    raw.update(
        {
            "run_label": "phase3b-fixture",
            "output_root": output_root.as_posix(),
            "bootstrap_replications": 2,
            "oracle_training_draws": 2,
            "evaluation_draws": 2,
            "oracle_stability_sizes": [1, 2],
            "oracle_stability_blocks": 2,
            "seed_namespaces": seed_namespaces,
            "generate_figures": False,
        }
    )
    config_path = config_directory / "fixture.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    results = run_phase3b(repo_root=ROOT, config_path=config_path)
    ledgers = {
        result.profile: read_ledger(result.run_directory / "ledger.jsonl")
        for result in results
    }
    return results, ledgers


def test_pilot_fixture_runs_both_profiles_and_writes_required_outputs(
    phase3b_run: tuple[
        tuple[Phase3BRunResult, ...], dict[str, tuple[LedgerRecord, ...]]
    ],
) -> None:
    results, ledgers = phase3b_run
    assert {result.profile for result in results} == {
        "validity-first",
        "efficiency-balanced",
    }
    required = {
        "resolved_config.yaml",
        "metadata.json",
        "seed_manifest.json",
        "scenario_manifest.json",
        "oracle_training_summary.csv",
        "oracle_stability.csv",
        "ledger.jsonl",
        "metrics_by_scenario.csv",
        "metrics_by_system.csv",
        "paired_comparisons.csv",
        "uncertainty_intervals.csv",
        "loss_profile_summary.csv",
        "routing_confusion.csv",
        "summary.json",
        "summary.md",
        "result_claims.json",
    }
    for result in results:
        assert result.record_count == 8 * 2 * len(BaselineName)
        assert result.run_id == result.run_directory.name
        assert required <= {path.name for path in result.run_directory.iterdir()}
        assert len(ledgers[result.profile]) == result.record_count


def test_shared_fingerprints_and_structural_coverage_are_preserved(
    phase3b_run: tuple[
        tuple[Phase3BRunResult, ...], dict[str, tuple[LedgerRecord, ...]]
    ],
) -> None:
    _, ledgers = phase3b_run
    for records in ledgers.values():
        cells: dict[tuple[str, int], set[str]] = {}
        for record in records:
            cells.setdefault((record.scenario_id, record.seed), set()).add(
                record.draw_fingerprint
            )
        assert all(len(fingerprints) == 1 for fingerprints in cells.values())
        structural = [
            record for record in records if record.process_family == "structural-break"
        ]
        assert structural
        assert all(record.coverage_applicable is False for record in structural)
        assert all(record.covered is None for record in structural)


def test_intervals_pairing_stability_and_result_claims_are_serialized(
    phase3b_run: tuple[
        tuple[Phase3BRunResult, ...], dict[str, tuple[LedgerRecord, ...]]
    ],
) -> None:
    results, _ = phase3b_run
    for result in results:
        with (result.run_directory / "uncertainty_intervals.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            intervals = list(csv.DictReader(handle))
        with (result.run_directory / "paired_comparisons.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            paired = list(csv.DictReader(handle))
        with (result.run_directory / "oracle_stability.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            stability = list(csv.DictReader(handle))
        claims = json.loads(
            (result.run_directory / "result_claims.json").read_text("utf-8")
        )
        assert any(row["interval_method"] == "wilson-score" for row in intervals)
        assert any(row["aggregation"] == "scenario-clustered" for row in paired)
        assert {int(row["training_size"]) for row in stability} == {1, 2}
        assert claims["permitted_claim_level"] == "preliminary-descriptive"
        assert len(claims["router_invalid_selection_ci"]) == 2
        assert claims["confirmation_status"] == "not-run"
        assert "oracle_agreement_by_stability" in claims


def test_summary_denominators_abstention_and_regret_reconstruct_from_ledger(
    phase3b_run: tuple[
        tuple[Phase3BRunResult, ...], dict[str, tuple[LedgerRecord, ...]]
    ],
) -> None:
    results, ledgers = phase3b_run
    for result in results:
        records = ledgers[result.profile]
        router = [
            record
            for record in records
            if record.baseline is BaselineName.DETERMINISTIC_ROUTER
        ]
        nonabstaining = [record for record in router if not record.abstained]
        coverage = [
            record
            for record in router
            if record.target_defined is True and not record.abstained
        ]
        summary = json.loads(
            (result.run_directory / "summary.json").read_text("utf-8")
        )["metrics_by_baseline"][BaselineName.DETERMINISTIC_ROUTER.value]
        assert summary["invalid_selection_numerator"] == sum(
            record.invalid_selection for record in nonabstaining
        )
        assert summary["invalid_selection_denominator"] == len(nonabstaining)
        assert summary["coverage_numerator"] == sum(
            record.covered is True for record in coverage
        )
        assert summary["coverage_denominator"] == len(coverage)
        assert summary["abstention_numerator"] == sum(
            record.abstained for record in router
        )
        assert summary["unnecessary_abstention_numerator"] == sum(
            record.unnecessary_abstention for record in router
        )
        assert summary["harmful_nonabstention_numerator"] == sum(
            record.harmful_nonabstention for record in router
        )

        profile = load_loss_profile(
            ROOT / "configs/benchmark/loss-profiles.yaml", result.profile
        )
        for record in records:
            components = record.loss_components
            reconstructed = (
                profile.coverage_error * components["coverage_error"]
                + profile.interval_width * components["normalized_interval_width"]
                + profile.invalid_method * components["invalid_method"]
                + profile.harmful_nonabstention * components["harmful_nonabstention"]
                + profile.unnecessary_abstention * components["unnecessary_abstention"]
                + profile.execution_failure * components["execution_failure"]
                + profile.abstention * components["abstention_cost"]
                + profile.runtime * components["normalized_runtime"]
            )
            assert isclose(reconstructed, record.total_loss, abs_tol=1e-12)
            assert record.oracle_total_loss is not None
            expected_regret = (
                0.0
                if record.baseline is BaselineName.ORACLE
                else record.total_loss - record.oracle_total_loss
            )
            assert isclose(expected_regret, record.regret, abs_tol=1e-12)


def test_confirmation_manifest_can_be_created_and_validated(tmp_path: Path) -> None:
    manifest_path = tmp_path / "freeze.json"
    create_freeze_manifest(
        ROOT,
        ROOT / "configs/benchmark/phase3b-aaai.yaml",
        manifest_path,
    )
    validation = validate_freeze_manifest(ROOT, manifest_path)
    assert validation.valid
    assert validation.manifest_hash


def test_runs_are_ignored_and_personal_obsidian_is_outside_benchmark_logic() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "runs/phase3b/fixture"],
        cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 0
    benchmark_sources = tuple((ROOT / "src/eri/benchmark").glob("*.py"))
    assert all(
        "inference-obsidian" not in path.read_text("utf-8")
        for path in benchmark_sources
    )
