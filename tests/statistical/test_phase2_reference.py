"""Fixed-seed reference properties for inference-relevant diagnostics."""

from math import isclose
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from eri.config import DependenceConfig, DiagnosticsConfig, load_yaml_model
from eri.diagnostics.dependence import (
    diagnose_linear_dependence,
    diagnose_squared_dependence,
)
from eri.diagnostics.registry import run_diagnostics
from eri.schemas.evidence import EvidenceFinding, EvidenceItem

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_yaml_model(ROOT / "configs/diagnostics/default.yaml", DiagnosticsConfig)


def _claims(values: NDArray[np.float64]) -> dict[str, EvidenceItem]:
    return {item.claim: item for item in run_diagnostics(values, CONFIG)}


def test_iid_gaussian_does_not_require_exact_dgp_label() -> None:
    claims = _claims(np.random.default_rng(20).normal(size=800))
    assert claims["linear_independence"].finding is EvidenceFinding.INCONCLUSIVE
    assert claims["gaussian_shape"].finding is EvidenceFinding.INCONCLUSIVE


def test_skewed_iid_is_non_gaussian() -> None:
    values = np.random.default_rng(21).exponential(size=800) - 1.0
    claims = _claims(values)
    assert claims["gaussian_shape"].finding is EvidenceFinding.CONTRADICTS


def test_ar_returns_trigger_linear_dependence() -> None:
    generator = np.random.default_rng(22)
    values = np.zeros(800)
    shocks = generator.normal(size=800)
    for index in range(1, values.size):
        values[index] = 0.7 * values[index - 1] + shocks[index]
    claims = _claims(values)
    assert claims["linear_independence"].finding is EvidenceFinding.CONTRADICTS


def test_volatility_clustering_triggers_squared_dependence() -> None:
    generator = np.random.default_rng(23)
    values = np.zeros(1000)
    variance = 1.0
    for index in range(values.size):
        variance = (
            0.05 + 0.12 * (values[index - 1] ** 2 if index else 0.0) + 0.83 * variance
        )
        values[index] = np.sqrt(variance) * generator.normal()
    claims = _claims(values)
    linear_correlation = float(np.corrcoef(values[1:], values[:-1])[0, 1])
    assert abs(linear_correlation) < 0.1
    assert (
        claims["absence_of_squared_return_dependence"].finding
        is EvidenceFinding.CONTRADICTS
    )


def test_repeated_returns_and_insufficient_samples_are_visible() -> None:
    repeated = _claims(np.repeat([0.0, 0.01], 100))
    short = _claims(np.asarray([0.01, -0.01, 0.02, -0.02]))
    assert repeated["return_concentration_risk"].finding is EvidenceFinding.SUPPORTS
    assert short["linear_independence"].finding is EvidenceFinding.INCONCLUSIVE


def test_ljung_box_matches_statsmodels_reference_fixture() -> None:
    """Reference values are from statsmodels 0.14 acorr_ljungbox."""
    config = DependenceConfig(
        lags=(1, 2, 5, 10),
        significance_level=0.05,
        practical_autocorrelation=0.99,
        minimum_sample_size=11,
    )
    values = np.asarray(
        [
            0.4,
            -0.2,
            0.1,
            0.7,
            -0.5,
            0.3,
            0.2,
            -0.1,
            0.6,
            -0.4,
            0.5,
            -0.3,
            0.8,
            -0.6,
            0.0,
            0.9,
            -0.7,
            0.2,
            -0.2,
            0.4,
        ]
    )
    linear = diagnose_linear_dependence(values, config)
    squared = diagnose_squared_dependence(values, config)
    assert linear.statistic is not None and isclose(
        linear.statistic, 29.48193116987596, rel_tol=1e-12
    )
    assert linear.p_value is not None and isclose(
        linear.p_value, 0.0010406779490299896, rel_tol=1e-12
    )
    assert squared.statistic is not None and isclose(
        squared.statistic, 5.812812840853946, rel_tol=1e-12
    )
    assert squared.p_value is not None and isclose(
        squared.p_value, 0.8307369919838092, rel_tol=1e-12
    )
    assert linear.structured_output["degrees_of_freedom"] == 10


def test_distribution_matches_scipy_jarque_bera_reference() -> None:
    """Reference values use SciPy's bias=True moment conventions."""
    values = np.asarray(
        [
            0.4, -0.2, 0.1, 0.7, -0.5, 0.3, 0.2, -0.1, 0.6, -0.4,
            0.5, -0.3, 0.8, -0.6, 0.0, 0.9, -0.7, 0.2, -0.2, 0.4,
        ]
        * 2
    )
    shape = _claims(values)["gaussian_shape"]
    assert shape.statistic is not None and isclose(
        shape.statistic, 1.867310197204791, rel_tol=1e-12
    )
    assert shape.p_value is not None and isclose(
        shape.p_value, 0.39311421001673247, rel_tol=1e-12
    )
    skewness = shape.structured_output["skewness"]
    raw_kurtosis = shape.structured_output["raw_kurtosis"]
    assert isinstance(skewness, (int, float)) and isclose(
        skewness, -0.031959510582875446
    )
    assert isinstance(raw_kurtosis, (int, float)) and isclose(
        raw_kurtosis, 1.943448781621976
    )


def test_structural_variance_break_and_gradual_drift_raise_instability_risk() -> None:
    generator = np.random.default_rng(24)
    variance_break = np.concatenate(
        [generator.normal(0.0, 0.5, 250), generator.normal(0.0, 2.0, 250)]
    )
    gradual_drift = generator.normal(0.0, 0.5, 500) + np.linspace(0.0, 2.0, 500)
    variance_evidence = _claims(variance_break)["structural_stability"]
    drift_evidence = _claims(gradual_drift)["structural_stability"]
    assert variance_evidence.finding is EvidenceFinding.CONTRADICTS
    assert variance_evidence.structured_output["variance_instability_detected"] is True
    assert drift_evidence.finding is EvidenceFinding.CONTRADICTS
