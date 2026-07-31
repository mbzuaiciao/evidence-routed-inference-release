"""Independent reference calculations for the Phase 1 mathematical audit."""

from math import isclose, sqrt

import numpy as np
import pytest

from eri.inference.block_bootstrap import (
    circular_block_bootstrap,
    circular_block_indices,
)
from eri.inference.common import normal_cdf, normal_quantile
from eri.inference.gaussian import (
    gaussian_sharpe_confidence_interval,
    gaussian_sharpe_test,
    iid_gaussian_sharpe_variance,
)
from eri.inference.hac import (
    bartlett_long_run_variance,
    hac_sharpe_standard_error,
    influence_function_series,
)
from eri.inference.mertens_psr import probabilistic_sharpe_ratio
from eri.inference.power import gaussian_sharpe_power, minimum_sample_length
from eri.inference.sharpe import sample_sharpe


def _manual_influence(returns: np.ndarray) -> np.ndarray:
    """Separately calculate the empirical-distribution Sharpe influence."""
    mean = float(sum(float(value) for value in returns) / len(returns))
    residuals = np.array([float(value) - mean for value in returns])
    variance = float(sum(float(value**2) for value in residuals) / len(returns))
    sigma = sqrt(variance)
    sharpe = mean / sigma
    influence = np.array(
        [
            residual / sigma
            - sharpe * (residual**2 - variance) / (2.0 * variance)
            for residual in residuals
        ]
    )
    return influence - float(sum(influence) / len(influence))


def _manual_bartlett_lrv(series: np.ndarray, bandwidth: int) -> float:
    """Separately calculate Newey-West LRV with denominator n at every lag."""
    sample_size = len(series)
    mean = float(sum(float(value) for value in series) / sample_size)
    centered = [float(value) - mean for value in series]
    result = sum(value * value for value in centered) / sample_size
    for lag in range(1, bandwidth + 1):
        covariance = sum(
            centered[index] * centered[index - lag]
            for index in range(lag, sample_size)
        ) / sample_size
        result += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * covariance
    return result


def test_hac_matches_independent_influence_and_newey_west_calculation() -> None:
    returns = np.array([0.02, -0.01, 0.04, 0.03, -0.02, 0.01, 0.05, -0.015])
    bandwidth = 3
    manual_influence = _manual_influence(returns)
    manual_lrv = _manual_bartlett_lrv(manual_influence, bandwidth)

    implementation_influence = influence_function_series(returns)
    implementation_lrv = bartlett_long_run_variance(
        implementation_influence,
        bandwidth=bandwidth,
    )
    result = hac_sharpe_standard_error(returns, bandwidth=bandwidth)

    assert np.allclose(implementation_influence, manual_influence)
    assert isclose(implementation_lrv, manual_lrv)
    assert isclose(result.long_run_variance, manual_lrv)
    assert isclose(result.standard_error, sqrt(manual_lrv / len(returns)))


def test_iid_gaussian_interval_has_reasonable_fixed_seed_coverage() -> None:
    generator = np.random.default_rng(8031)
    true_sharpe = 0.2
    replications = 600
    covered = 0
    for _ in range(replications):
        returns = generator.normal(loc=true_sharpe, scale=1.0, size=180)
        interval = gaussian_sharpe_confidence_interval(returns)
        covered += interval.lower <= true_sharpe <= interval.upper

    coverage = covered / replications

    assert 0.92 <= coverage <= 0.98


def test_hac_is_better_calibrated_than_iid_for_autocorrelated_series() -> None:
    generator = np.random.default_rng(4407)
    replications = 120
    sample_size = 300
    estimates = np.empty(replications)
    iid_errors = np.empty(replications)
    hac_errors = np.empty(replications)
    for replication in range(replications):
        innovations = generator.normal(size=sample_size)
        dependent = np.empty(sample_size)
        dependent[0] = innovations[0]
        for index in range(1, sample_size):
            dependent[index] = 0.6 * dependent[index - 1] + innovations[index]
        returns = 0.15 + dependent
        estimates[replication] = sample_sharpe(returns)
        iid_errors[replication] = sqrt(
            iid_gaussian_sharpe_variance(
                estimates[replication],
                sample_size,
            )
        )
        hac_errors[replication] = hac_sharpe_standard_error(
            returns,
            bandwidth=8,
        ).standard_error

    empirical_error = float(np.std(estimates, ddof=1))
    mean_iid_error = float(np.mean(iid_errors))
    mean_hac_error = float(np.mean(hac_errors))

    assert mean_hac_error > 1.2 * mean_iid_error
    assert abs(mean_hac_error - empirical_error) < abs(
        mean_iid_error - empirical_error
    )


def test_squared_return_dependence_changes_hac_with_weak_linear_correlation() -> None:
    generator = np.random.default_rng(2026)
    sample_size = 4_000
    innovations = generator.normal(size=sample_size)
    conditional_variance = np.empty(sample_size)
    shocks = np.empty(sample_size)
    conditional_variance[0] = 1.0
    shocks[0] = innovations[0]
    for index in range(1, sample_size):
        conditional_variance[index] = (
            0.05
            + 0.15 * shocks[index - 1] ** 2
            + 0.8 * conditional_variance[index - 1]
        )
        shocks[index] = sqrt(conditional_variance[index]) * innovations[index]
    returns = 0.5 + shocks
    centered = returns - np.mean(returns)
    linear_correlation = float(np.corrcoef(centered[1:], centered[:-1])[0, 1])
    squared_correlation = float(
        np.corrcoef(centered[1:] ** 2, centered[:-1] ** 2)[0, 1]
    )

    full_hac = hac_sharpe_standard_error(returns, bandwidth=20)
    mean_only_influence = centered / sqrt(float(np.mean(centered**2)))
    mean_only_error = sqrt(
        _manual_bartlett_lrv(mean_only_influence, 20) / sample_size
    )

    assert abs(linear_correlation) < 0.06
    assert squared_correlation > 0.1
    assert abs(full_hac.standard_error / mean_only_error - 1.0) > 0.05


def test_psr_benchmark_tail_direction_and_midpoint() -> None:
    returns = np.array([-0.03, 0.01, 0.04, 0.02, -0.01, 0.05, 0.015, 0.025])
    observed = sample_sharpe(returns)
    midpoint = probabilistic_sharpe_ratio(returns, benchmark=observed)
    lower = probabilistic_sharpe_ratio(returns, benchmark=observed - 0.2)
    upper = probabilistic_sharpe_ratio(returns, benchmark=observed + 0.2)

    assert isclose(midpoint.probabilistic_sharpe_ratio, 0.5)
    assert lower.probabilistic_sharpe_ratio > 0.5
    assert upper.probabilistic_sharpe_ratio < 0.5


def test_circular_block_indices_preserve_continuity_and_exact_length() -> None:
    indices = circular_block_indices(
        sample_size=5,
        block_length=3,
        starts=[4, 1],
    )

    assert indices.tolist() == [4, 0, 1, 1, 2]
    assert len(indices) == 5


def test_bootstrap_reports_degenerate_resample_instead_of_nonfinite_draw() -> None:
    with pytest.raises(ValueError, match="replication 0 has degenerate variance"):
        circular_block_bootstrap(
            [-1.0, 1.0],
            block_length=1,
            replications=2,
            seed=0,
        )


def test_gaussian_test_uses_null_scale_and_correct_tail_directions() -> None:
    returns = np.array([-0.1, 0.2, 0.3, 0.4, -0.05, 0.25, 0.15])
    benchmark = 0.4
    result = gaussian_sharpe_test(
        returns,
        benchmark=benchmark,
        alternative="greater",
    )
    less = gaussian_sharpe_test(
        returns,
        benchmark=benchmark,
        alternative="less",
    )
    two_sided = gaussian_sharpe_test(
        returns,
        benchmark=benchmark,
        alternative="two-sided",
    )
    expected_error = sqrt(
        iid_gaussian_sharpe_variance(benchmark, len(returns))
    )

    assert isclose(result.standard_error, expected_error)
    assert isclose(result.p_value + less.p_value, 1.0)
    assert isclose(two_sided.p_value, 2.0 * min(result.p_value, less.p_value))


def test_power_matches_independent_null_threshold_calculation() -> None:
    sample_size = 150
    true_sharpe = 0.7
    benchmark = 0.2
    alpha = 0.05
    null_error = sqrt((1.0 + benchmark**2 / 2.0) / sample_size)
    alternative_error = sqrt((1.0 + true_sharpe**2 / 2.0) / sample_size)
    threshold = benchmark + normal_quantile(1.0 - alpha) * null_error
    expected = normal_cdf((true_sharpe - threshold) / alternative_error)

    result = gaussian_sharpe_power(
        sample_size=sample_size,
        true_sharpe=true_sharpe,
        benchmark=benchmark,
        significance_level=alpha,
    )
    more_permissive = gaussian_sharpe_power(
        sample_size=sample_size,
        true_sharpe=true_sharpe,
        benchmark=benchmark,
        significance_level=0.1,
    )

    assert isclose(result.power, expected)
    assert isclose(result.null_standard_error, null_error)
    assert isclose(result.alternative_standard_error, alternative_error)
    assert isclose(result.rejection_threshold, threshold)
    assert more_permissive.power > result.power


def test_minimum_sample_search_reports_unreachable_target() -> None:
    with pytest.raises(ValueError, match="not reached"):
        minimum_sample_length(
            true_sharpe=0.01,
            benchmark=0.0,
            target_power=0.99,
            maximum_sample_size=10,
        )
