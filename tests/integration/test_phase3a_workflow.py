"""End-to-end Phase 3A smoke benchmark tests."""

import json
from math import isclose
from pathlib import Path

import pytest

from eri.benchmark.evaluation import draw_fingerprint, validate_shared_draws
from eri.benchmark.ledger import read_ledger
from eri.benchmark.registry import generate_scenario
from eri.benchmark.runner import load_benchmark_config, run_benchmark
from eri.benchmark.schemas import BaselineName, BenchmarkRunResult, LedgerRecord

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def smoke_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[BenchmarkRunResult, tuple[LedgerRecord, ...]]:
    run_directory = tmp_path_factory.mktemp("phase3a") / "run"
    result = run_benchmark(
        repo_root=ROOT,
        config_path=ROOT / "configs/benchmark/phase3a-smoke.yaml",
        loss_profiles_path=ROOT / "configs/benchmark/loss-profiles.yaml",
        output_directory=run_directory,
    )
    return result, read_ledger(run_directory / "ledger.jsonl")


def test_end_to_end_smoke_outputs_and_scope(
    smoke_run: tuple[BenchmarkRunResult, tuple[LedgerRecord, ...]],
) -> None:
    result, records = smoke_run
    assert result.record_count == 16
    assert {record.process_family.value for record in records} == {
        "iid-gaussian",
        "ar1",
        "garch",
        "structural-break",
    }
    assert {record.baseline for record in records} == set(BaselineName)
    required = {
        "summary.json",
        "summary.md",
        "ledger.jsonl",
        "resolved_config.yaml",
        "metadata.json",
        "metrics_by_scenario.csv",
        "metrics_by_baseline.csv",
        "oracle_choices.csv",
        "routing_confusion.csv",
        "invalid_selection_rate.png",
        "coverage.png",
        "abstention_rate.png",
        "mean_routing_regret.png",
    }
    assert required <= {path.name for path in result.run_directory.iterdir()}
    assert "inference-obsidian" not in result.run_directory.as_posix()


def test_shared_draws_and_oracle_split_are_recorded(
    smoke_run: tuple[BenchmarkRunResult, tuple[LedgerRecord, ...]],
) -> None:
    result, records = smoke_run
    groups: dict[tuple[str, int], set[str]] = {}
    for record in records:
        groups.setdefault((record.scenario_id, record.seed), set()).add(
            record.draw_fingerprint
        )
    assert all(len(fingerprints) == 1 for fingerprints in groups.values())
    evaluation_seeds = {record.seed for record in records}
    assert all(
        not (set(choice.training_seeds) & evaluation_seeds)
        for choice in result.oracle_choices
    )
    validate_shared_draws(records)
    config = load_benchmark_config(ROOT / "configs/benchmark/phase3a-smoke.yaml")
    scenarios = {scenario.scenario_id: scenario for scenario in config.scenarios}
    for (scenario_id, seed), fingerprints in groups.items():
        regenerated = generate_scenario(scenarios[scenario_id], seed=seed)
        assert fingerprints == {draw_fingerprint(regenerated)}

    first_cell = [
        record
        for record in records
        if (record.scenario_id, record.seed)
        == (records[0].scenario_id, records[0].seed)
    ]
    mismatched = (
        first_cell[0].model_copy(update={"draw_fingerprint": "different"}),
        *first_cell[1:],
    )
    with pytest.raises(ValueError, match="different return draws"):
        validate_shared_draws(mismatched)


def test_structural_coverage_invalid_selection_and_router_abstention(
    smoke_run: tuple[BenchmarkRunResult, tuple[LedgerRecord, ...]],
) -> None:
    _, records = smoke_run
    structural = [
        record for record in records if record.process_family == "structural-break"
    ]
    assert all(record.covered is None for record in structural)
    assert any(
        record.baseline is BaselineName.ALWAYS_GAUSSIAN
        and record.invalid_selection
        for record in records
    )
    router = next(
        record
        for record in structural
        if record.baseline is BaselineName.DETERMINISTIC_ROUTER
    )
    assert router.abstained and not router.harmful_nonabstention
    oracle = next(
        record for record in structural if record.baseline is BaselineName.ORACLE
    )
    assert oracle.oracle_method == "abstention" and oracle.regret == 0.0


def test_metric_denominators_and_summary_are_reconstructable(
    smoke_run: tuple[BenchmarkRunResult, tuple[LedgerRecord, ...]],
) -> None:
    result, records = smoke_run
    raw = json.loads((result.run_directory / "summary.json").read_text("utf-8"))
    for baseline in BaselineName:
        group = [record for record in records if record.baseline is baseline]
        metrics = raw["metrics_by_baseline"][baseline.value]
        nonabstaining = [record for record in group if not record.abstained]
        covered = [record for record in group if record.covered is not None]
        assert metrics["runs"] == len(group)
        assert metrics["invalid_selection_numerator"] == sum(
            record.invalid_selection for record in nonabstaining
        )
        assert metrics["invalid_selection_denominator"] == len(nonabstaining)
        assert metrics["coverage_numerator"] == sum(
            record.covered is True for record in covered
        )
        assert metrics["coverage_denominator"] == len(covered)
        assert metrics["oracle_agreement_numerator"] == sum(
            (record.selected_method or "abstention") == record.oracle_method
            for record in group
        )
        assert metrics["oracle_agreement_denominator"] == len(group)
        for record in group:
            assert record.oracle_total_loss is not None
            expected_regret = (
                0.0
                if baseline is BaselineName.ORACLE
                else record.total_loss - record.oracle_total_loss
            )
            assert isclose(
                record.regret, expected_regret, rel_tol=0.0, abs_tol=1e-12
            )
        mean_regret = sum(record.regret for record in group) / len(group)
        assert mean_regret >= -1e-12
    structural = [
        record for record in records if record.process_family == "structural-break"
    ]
    assert all(record.coverage_applicable is False for record in structural)
    assert all(record.covered is None for record in structural)
