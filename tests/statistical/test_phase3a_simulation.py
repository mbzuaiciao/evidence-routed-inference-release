"""Fixed-seed statistical reference checks for Phase 3A generators."""

from math import isclose

import numpy as np

from eri.simulation.ar1 import simulate_ar1
from eri.simulation.garch import simulate_garch
from eri.simulation.iid_gaussian import simulate_iid_gaussian
from eri.simulation.structural_break import simulate_structural_break


def test_iid_gaussian_empirical_moments() -> None:
    result = simulate_iid_gaussian(
        sample_size=30_000,
        seed=101,
        parameters={"mean": 0.02, "volatility": 0.15},
    )
    values = np.asarray(result.returns)
    assert abs(float(np.mean(values)) - 0.02) < 0.003
    assert abs(float(np.std(values)) - 0.15) < 0.003
    assert result.true_sharpe is not None
    assert isclose(float(result.true_sharpe), 0.02 / 0.15)
    assert result.inference_truth["target_scale"] == "unannualized-per-period"
    assert result.parameters == {"mean": 0.02, "volatility": 0.15}


def test_ar1_autocorrelation_and_unconditional_variance() -> None:
    phi = 0.45
    innovation_std = 0.08
    result = simulate_ar1(
        sample_size=30_000,
        seed=102,
        parameters={
            "mean": 0.01,
            "phi": phi,
            "innovation_std": innovation_std,
            "burn_in": 500,
        },
    )
    values = np.asarray(result.returns)
    autocorrelation = float(np.corrcoef(values[1:], values[:-1])[0, 1])
    expected_variance = innovation_std**2 / (1.0 - phi**2)
    assert abs(autocorrelation - phi) < 0.03
    assert abs(float(np.var(values)) - expected_variance) < 0.0003
    assert result.true_mean == 0.01
    assert result.true_standard_deviation == expected_variance**0.5
    assert result.true_sharpe == 0.01 / expected_variance**0.5
    assert result.parameters["innovation_variance"] == innovation_std**2


def test_garch_unconditional_variance_and_squared_dependence() -> None:
    omega, alpha, beta = 0.0002, 0.12, 0.82
    result = simulate_garch(
        sample_size=40_000,
        seed=103,
        parameters={
            "mean": 0.0,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "burn_in": 1000,
        },
    )
    values = np.asarray(result.returns)
    expected_variance = omega / (1.0 - alpha - beta)
    squared_autocorrelation = float(
        np.corrcoef(values[1:] ** 2, values[:-1] ** 2)[0, 1]
    )
    linear_autocorrelation = float(np.corrcoef(values[1:], values[:-1])[0, 1])
    assert abs(float(np.var(values)) - expected_variance) < 0.0007
    assert squared_autocorrelation > 0.1
    assert abs(linear_autocorrelation) < 0.05
    assert result.inference_truth["finite_fourth_moment"] is True
    assert result.parameters["initial_conditional_variance"] == expected_variance


def test_structural_mean_and_variance_breaks() -> None:
    mean_break = simulate_structural_break(
        sample_size=10_000,
        seed=104,
        parameters={
            "variant": "mean",
            "mean_1": -0.04,
            "mean_2": 0.05,
            "volatility_1": 0.1,
            "volatility_2": 0.1,
            "break_fraction": 0.5,
        },
    )
    mean_values = np.asarray(mean_break.returns)
    assert float(np.mean(mean_values[:5000])) < -0.035
    assert float(np.mean(mean_values[5000:])) > 0.045
    assert mean_break.true_sharpe is None and not mean_break.target_defined

    variance_break = simulate_structural_break(
        sample_size=10_000,
        seed=105,
        parameters={
            "variant": "variance",
            "mean_1": 0.01,
            "mean_2": 0.01,
            "volatility_1": 0.04,
            "volatility_2": 0.16,
            "break_fraction": 0.5,
        },
    )
    variance_values = np.asarray(variance_break.returns)
    ratio = float(np.var(variance_values[5000:]) / np.var(variance_values[:5000]))
    assert ratio > 10.0
    assert variance_break.true_sharpe is None and not variance_break.target_defined
    assert mean_break.descriptive_full_sample_sharpe is not None
    assert variance_break.descriptive_full_sample_sharpe is not None


def test_generators_follow_exact_fixed_seed_recursions() -> None:
    iid = simulate_iid_gaussian(
        sample_size=5,
        seed=201,
        parameters={"mean": 0.02, "volatility": 0.15},
    )
    iid_rng = np.random.default_rng(201)
    assert np.array_equal(np.asarray(iid.returns), iid_rng.normal(0.02, 0.15, 5))

    phi, innovation_std, burn_in = 0.4, 0.08, 3
    ar = simulate_ar1(
        sample_size=5,
        seed=202,
        parameters={
            "mean": 0.01,
            "phi": phi,
            "innovation_std": innovation_std,
            "burn_in": burn_in,
        },
    )
    ar_rng = np.random.default_rng(202)
    centered = ar_rng.normal(0.0, innovation_std / (1.0 - phi**2) ** 0.5)
    expected_ar: list[float] = []
    for index in range(burn_in + 5):
        centered = phi * centered + ar_rng.normal(0.0, innovation_std)
        if index >= burn_in:
            expected_ar.append(0.01 + centered)
    assert np.array_equal(np.asarray(ar.returns), np.asarray(expected_ar))

    omega, alpha, beta, burn_in = 0.0002, 0.1, 0.8, 3
    garch = simulate_garch(
        sample_size=5,
        seed=203,
        parameters={
            "mean": 0.01,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "burn_in": burn_in,
        },
    )
    garch_rng = np.random.default_rng(203)
    variance = omega / (1.0 - alpha - beta)
    residual = 0.0
    expected_garch: list[float] = []
    for index in range(burn_in + 5):
        variance = omega + alpha * residual**2 + beta * variance
        residual = variance**0.5 * garch_rng.normal()
        if index >= burn_in:
            expected_garch.append(0.01 + residual)
    assert np.array_equal(np.asarray(garch.returns), np.asarray(expected_garch))


def test_structural_break_exact_index_and_mixture_functional() -> None:
    result = simulate_structural_break(
        sample_size=11,
        seed=204,
        parameters={
            "variant": "mean",
            "break_fraction": 0.4,
            "mean_1": -0.02,
            "mean_2": 0.03,
            "volatility_1": 0.1,
            "volatility_2": 0.1,
        },
    )
    assert result.parameters["break_index"] == 4
    rng = np.random.default_rng(204)
    expected = np.concatenate((rng.normal(-0.02, 0.1, 4), rng.normal(0.03, 0.1, 7)))
    assert np.array_equal(np.asarray(result.returns), expected)
    weight = 4 / 11
    mixture_mean = weight * -0.02 + (1.0 - weight) * 0.03
    mixture_variance = weight * (0.1**2 + (-0.02 - mixture_mean) ** 2) + (
        1.0 - weight
    ) * (0.1**2 + (0.03 - mixture_mean) ** 2)
    assert result.descriptive_full_sample_sharpe is not None
    assert isclose(
        result.descriptive_full_sample_sharpe,
        mixture_mean / mixture_variance**0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result.true_sharpe is None
