"""Unit tests for deterministic Phase 2 diagnostics."""

from datetime import UTC, datetime, timedelta
from math import isclose
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from eri.config import (
    DependenceConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    StabilityConfig,
    load_yaml_model,
)
from eri.diagnostics._math import chi_square_survival
from eri.diagnostics.data_quality import diagnose_data_quality
from eri.diagnostics.dependence import (
    diagnose_linear_dependence,
    diagnose_squared_dependence,
)
from eri.diagnostics.distribution import diagnose_distribution
from eri.diagnostics.evidence_factory import make_evidence
from eri.diagnostics.provenance import (
    ProvenanceCompleteness,
    ResearchProvenance,
    diagnose_provenance,
)
from eri.diagnostics.stability import diagnose_stability
from eri.schemas.evidence import EvidenceFinding

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config() -> DiagnosticsConfig:
    return load_yaml_model(ROOT / "configs/diagnostics/default.yaml", DiagnosticsConfig)


def test_evidence_identifier_is_stable_and_serializable() -> None:
    first = make_evidence(
        claim="linear_independence",
        finding=EvidenceFinding.INCONCLUSIVE,
        diagnostic_name="fixture",
        sample_size=100,
        structured_output={"lag": 1},
    )
    second = make_evidence(
        claim="linear_independence",
        finding=EvidenceFinding.INCONCLUSIVE,
        diagnostic_name="fixture",
        sample_size=100,
        structured_output={"lag": 1},
    )
    assert first.evidence_id == second.evidence_id
    assert first.model_validate_json(first.model_dump_json()) == first


def test_evidence_identifier_covers_semantic_metadata() -> None:
    baseline = make_evidence(
        claim="claim",
        finding=EvidenceFinding.INCONCLUSIVE,
        diagnostic_name="fixture",
        sample_size=10,
    )
    warned = make_evidence(
        claim="claim",
        finding=EvidenceFinding.INCONCLUSIVE,
        diagnostic_name="fixture",
        sample_size=10,
        warnings=("Material caveat.",),
    )
    assert baseline.evidence_id != warned.evidence_id


def test_chi_square_survival_matches_closed_form_references() -> None:
    assert isclose(chi_square_survival(20.0, 2), float(np.exp(-10.0)))
    assert isclose(
        chi_square_survival(20.0, 4), float(np.exp(-10.0) * 11.0)
    )


def test_data_quality_reports_nonfinite_without_imputation(
    config: DiagnosticsConfig,
) -> None:
    evidence = diagnose_data_quality([0.1, float("nan"), 0.2], config.data_quality)
    assert len(evidence) == 1
    assert evidence[0].finding is EvidenceFinding.CONTRADICTS
    assert evidence[0].warnings == ("No values were imputed or removed.",)
    assert evidence[0].data_fingerprint is not None
    assert evidence[0].structured_output["all_finite"] is False


def test_timestamp_and_repetition_warnings(config: DiagnosticsConfig) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [start, start, start + timedelta(days=3), None]
    items = diagnose_data_quality(
        [0.0, 0.0, 0.0, 0.1], config.data_quality, timestamps=timestamps
    )
    by_claim = {item.claim: item for item in items}
    concentration = by_claim["return_concentration_risk"]
    assert concentration.finding is EvidenceFinding.SUPPORTS
    assert "heuristic warning" in concentration.does_not_establish[0]
    assert by_claim["timestamp_quality"].finding is EvidenceFinding.CONTRADICTS
    assert by_claim["timestamp_quality"].structured_output["duplicate_count"] == 1


def test_linear_and_squared_dependence_are_distinct(config: DiagnosticsConfig) -> None:
    generator = np.random.default_rng(11)
    innovations = generator.normal(size=400)
    ar = np.empty(400)
    ar[0] = innovations[0]
    for index in range(1, ar.size):
        ar[index] = 0.75 * ar[index - 1] + innovations[index]
    linear = diagnose_linear_dependence(ar, config.linear_dependence)
    squared = diagnose_squared_dependence(ar, config.squared_dependence)
    assert linear.claim == "linear_independence"
    assert squared.claim == "absence_of_squared_return_dependence"
    assert linear.finding is EvidenceFinding.CONTRADICTS
    assert linear.p_value is not None and 0.0 <= linear.p_value <= 1.0


def test_no_rejection_remains_inconclusive(config: DiagnosticsConfig) -> None:
    values = np.random.default_rng(123).normal(size=500)
    evidence = diagnose_linear_dependence(values, config.linear_dependence)
    assert evidence.finding is EvidenceFinding.INCONCLUSIVE
    assert "does not prove independence" in evidence.does_not_establish[0]


def test_excessive_lag_and_constant_transform_are_inconclusive() -> None:
    excessive = DependenceConfig(
        lags=(1, 5),
        significance_level=0.05,
        practical_autocorrelation=0.15,
        minimum_sample_size=10,
    )
    constant_transform = excessive.model_copy(update={"lags": (1, 2)})
    too_large = diagnose_linear_dependence(np.arange(4.0), excessive)
    constant = diagnose_squared_dependence(
        np.asarray([-1.0, 1.0] * 10), constant_transform
    )
    assert too_large.finding is EvidenceFinding.INCONCLUSIVE
    assert any("below sample size" in warning for warning in too_large.warnings)
    assert constant.finding is EvidenceFinding.INCONCLUSIVE
    assert "degenerate variance" in constant.warnings[0]


def test_distribution_uses_raw_kurtosis_and_heuristic_label(
    config: DiagnosticsConfig,
) -> None:
    values = np.random.default_rng(9).standard_t(df=3, size=600)
    shape, fourth = diagnose_distribution(values, config.distribution)
    raw_kurtosis = shape.structured_output["raw_kurtosis"]
    assert isinstance(raw_kurtosis, (int, float)) and raw_kurtosis > 3.0
    assert fourth.finding is EvidenceFinding.INCONCLUSIVE
    assert "heuristic" in fourth.warnings[0]


def test_split_sample_break_is_detected(config: DiagnosticsConfig) -> None:
    generator = np.random.default_rng(42)
    values = np.concatenate(
        [generator.normal(0.0, 1.0, 200), generator.normal(2.0, 1.0, 200)]
    )
    evidence = diagnose_stability(values, config.stability)
    assert evidence.finding is EvidenceFinding.CONTRADICTS
    assert evidence.structured_output["mean_instability_detected"] is True
    assert "exact change point" in evidence.does_not_establish[1]


def test_deterministic_piecewise_constant_break_is_serializable(
    config: DiagnosticsConfig,
) -> None:
    evidence = diagnose_stability(
        np.asarray([0.0] * 100 + [1.0] * 100), config.stability
    )
    assert evidence.finding is EvidenceFinding.CONTRADICTS
    assert evidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_provenance_completeness_is_explicit() -> None:
    missing = diagnose_provenance(None)
    complete = diagnose_provenance(
        ResearchProvenance(
            trials=10,
            parameter_searches=2,
            markets_or_universes=("US equities",),
            start_dates_or_windows=("2000-2025",),
            failed_trials_retained=True,
            trial_dependence_known=True,
            completeness=ProvenanceCompleteness.COMPLETE,
        )
    )
    assert missing.finding is EvidenceFinding.INCONCLUSIVE
    assert complete.finding is EvidenceFinding.SUPPORTS
    assert complete.structured_output["markets_or_universes"] == ["US equities"]


def test_complete_provenance_rejects_missing_required_details() -> None:
    with pytest.raises(ValidationError):
        ResearchProvenance(completeness=ProvenanceCompleteness.COMPLETE)


def test_configuration_rejects_unknown_keys(config: DiagnosticsConfig) -> None:
    with pytest.raises(ValidationError):
        DiagnosticsConfig.model_validate(
            {**config.model_dump(), "hidden_threshold": 0.5}
        )


def test_configuration_rejects_unsorted_lags_and_invalid_segments() -> None:
    with pytest.raises(ValidationError):
        DependenceConfig(
            lags=(2, 1),
            significance_level=0.05,
            practical_autocorrelation=0.15,
            minimum_sample_size=30,
        )
    with pytest.raises(ValidationError):
        StabilityConfig(
            split_fraction=0.5,
            material_mean_z=3.0,
            material_variance_ratio=3.0,
            minimum_sample_size=30,
            minimum_segment_size=20,
        )


def test_configuration_rejects_nonfinite_benchmark() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            benchmark_sharpe=float("nan"),
            confidence_level=0.95,
            hac_bandwidth=5,
            bootstrap_block_length=10,
            bootstrap_replications=100,
            bootstrap_seed=1,
            claim_requires_selection_provenance=False,
        )
