"""End-to-end tests for the deterministic Phase 2 workflow."""

from pathlib import Path

import numpy as np

from eri.config import load_phase2_config
from eri.diagnostics.provenance import (
    ProvenanceCompleteness,
    ResearchProvenance,
)
from eri.workflows.deterministic import run_deterministic_workflow

ROOT = Path(__file__).resolve().parents[2]


def test_clean_iid_like_data_runs_primary_and_sensitivity_methods() -> None:
    config = load_phase2_config(ROOT / "configs")
    values = np.random.default_rng(31).normal(0.01, 0.1, 500)
    result = run_deterministic_workflow(values, config)
    eligibility = {item.method_id: item for item in result.decision.eligibility}
    assert eligibility["iid-gaussian-sharpe"].eligible
    assert eligibility["iid-gaussian-sharpe"].sensitivity_only
    assert eligibility["mertens-psr"].eligible
    assert result.decision.selected_method == "hac-sharpe"
    assert len(result.inference_results) >= 2
    fingerprints = {item.data_fingerprint for item in result.inference_results}
    assert len(fingerprints) == 1 and None not in fingerprints
    assert all(
        item.sample_frequency == "per-period"
        and item.annualization_convention == "not-annualized"
        and item.metadata["benchmark_scale"] == "per-period"
        for item in result.inference_results
    )


def test_workflow_is_reproducible_for_identical_inputs() -> None:
    config = load_phase2_config(ROOT / "configs")
    values = np.random.default_rng(310).normal(0.01, 0.1, 300)
    first = run_deterministic_workflow(values, config)
    second = run_deterministic_workflow(values, config)
    assert first.model_dump() == second.model_dump()


def test_missing_provenance_preserves_ordinary_inference() -> None:
    config = load_phase2_config(ROOT / "configs")
    experiment = config.experiment.model_copy(
        update={"claim_requires_selection_provenance": True}
    )
    configured = config.model_copy(update={"experiment": experiment})
    values = np.random.default_rng(311).normal(0.01, 0.1, 300)
    result = run_deterministic_workflow(values, configured)
    assert result.decision.selected_method == "hac-sharpe"
    assert "multiplicity_adjusted_sharpe" in result.decision.claim_abstentions
    assert result.inference_results


def test_provenance_reference_flows_to_inference_results() -> None:
    config = load_phase2_config(ROOT / "configs")
    provenance = ResearchProvenance(
        trials=5,
        completeness=ProvenanceCompleteness.PARTIAL,
        reference="research-log:v1",
    )
    values = np.random.default_rng(312).normal(0.01, 0.1, 300)
    result = run_deterministic_workflow(values, config, provenance=provenance)
    assert all(
        item.data_provenance == "research-log:v1"
        for item in result.inference_results
    )


def test_linear_dependence_never_selects_iid_primary() -> None:
    config = load_phase2_config(ROOT / "configs")
    shocks = np.random.default_rng(32).normal(size=500)
    values = np.zeros(500)
    for index in range(1, values.size):
        values[index] = 0.8 * values[index - 1] + shocks[index]
    result = run_deterministic_workflow(values, config)
    assert result.decision.selected_method not in (
        "iid-gaussian-sharpe",
        "mertens-psr",
    )


def test_structural_instability_abstains() -> None:
    config = load_phase2_config(ROOT / "configs")
    generator = np.random.default_rng(33)
    values = np.concatenate(
        [generator.normal(0.0, 1.0, 200), generator.normal(2.0, 1.0, 200)]
    )
    result = run_deterministic_workflow(values, config)
    assert result.decision.abstention is not None
    assert not result.inference_results


def test_invalid_and_degenerate_data_abstain_cleanly() -> None:
    config = load_phase2_config(ROOT / "configs")
    for values in ([0.1, float("nan")], [0.0] * 100, []):
        result = run_deterministic_workflow(values, config)
        assert result.decision.abstention is not None
        assert not result.inference_results
