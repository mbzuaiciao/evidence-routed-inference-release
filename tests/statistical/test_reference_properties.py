"""Deterministic analytical and simulation-based statistical references."""

from math import isclose, sqrt

import numpy as np

from eri.inference.gaussian import (
    iid_gaussian_sharpe_standard_error,
    iid_gaussian_sharpe_variance,
)
from eri.inference.hac import (
    bartlett_long_run_variance,
    hac_sharpe_standard_error,
    influence_function_series,
)
from eri.inference.mertens_psr import mertens_sharpe_variance
from eri.inference.power import gaussian_sharpe_power, minimum_sample_length
from eri.inference.sharpe import sample_sharpe


def test_constant_returns_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        sample_sharpe([0.01, 0.01, 0.01])


def test_iid_gaussian_variance_matches_documented_formula() -> None:
    sharpe = 0.75
    sample_size = 120

    actual = iid_gaussian_sharpe_variance(sharpe, sample_size)

    assert isclose(actual, (1.0 + sharpe**2 / 2.0) / sample_size)


def test_mertens_variance_reduces_to_gaussian_formula() -> None:
    sharpe = 0.75
    sample_size = 120

    gaussian = iid_gaussian_sharpe_variance(sharpe, sample_size)
    mertens = mertens_sharpe_variance(
        sharpe,
        sample_size,
        skewness=0.0,
        raw_kurtosis=3.0,
    )

    assert isclose(mertens, gaussian)


def test_hac_bandwidth_zero_matches_lag_zero_influence_variance() -> None:
    returns = np.array([-0.04, 0.01, 0.03, 0.02, -0.01, 0.05, 0.015])
    influence = influence_function_series(returns)
    expected_long_run_variance = float(np.dot(influence, influence) / len(influence))

    long_run_variance = bartlett_long_run_variance(influence, bandwidth=0)
    result = hac_sharpe_standard_error(returns, bandwidth=0)

    assert isclose(long_run_variance, expected_long_run_variance)
    assert isclose(result.long_run_variance, expected_long_run_variance)
    assert isclose(
        result.standard_error,
        sqrt(expected_long_run_variance / len(returns)),
    )


def test_hac_and_iid_standard_errors_are_close_for_large_iid_gaussian_sample() -> None:
    generator = np.random.default_rng(20260717)
    returns = generator.normal(loc=0.08, scale=1.0, size=5_000)

    iid = iid_gaussian_sharpe_standard_error(returns)
    hac = hac_sharpe_standard_error(returns, bandwidth=5)

    assert isclose(
        hac.standard_error / iid.standard_error,
        1.0,
        rel_tol=0.12,
    )


def test_hac_detects_mean_and_squared_return_dependence() -> None:
    generator = np.random.default_rng(991)
    innovations = generator.normal(size=3_000)
    dependent = np.empty_like(innovations)
    dependent[0] = innovations[0]
    for index in range(1, len(dependent)):
        dependent[index] = 0.75 * dependent[index - 1] + innovations[index]
    returns = 0.08 + dependent

    iid = iid_gaussian_sharpe_standard_error(returns)
    hac = hac_sharpe_standard_error(returns, bandwidth=20)

    assert hac.standard_error > 1.3 * iid.standard_error


def test_power_increases_with_sample_size_and_effect_size() -> None:
    base = gaussian_sharpe_power(
        sample_size=80, true_sharpe=0.2, benchmark=0.0
    )
    larger_sample = gaussian_sharpe_power(
        sample_size=160, true_sharpe=0.2, benchmark=0.0
    )
    larger_effect = gaussian_sharpe_power(
        sample_size=80, true_sharpe=0.4, benchmark=0.0
    )

    assert larger_sample.power > base.power
    assert larger_effect.power > base.power


def test_required_sample_increases_for_harder_benchmark_and_higher_power() -> None:
    baseline = minimum_sample_length(
        true_sharpe=0.4, benchmark=0.0, target_power=0.8
    )
    closer_benchmark = minimum_sample_length(
        true_sharpe=0.4, benchmark=0.2, target_power=0.8
    )
    higher_target = minimum_sample_length(
        true_sharpe=0.4, benchmark=0.0, target_power=0.9
    )

    assert closer_benchmark.sample_size > baseline.sample_size
    assert higher_target.sample_size > baseline.sample_size
