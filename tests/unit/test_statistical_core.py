"""Unit tests for validation, conventions, and deterministic result objects."""

from dataclasses import asdict
from math import isclose, sqrt

import numpy as np
import pytest

from eri.data.frequency import validate_periods_per_year
from eri.data.returns import validate_returns
from eri.diagnostics.summary import summarize_returns
from eri.inference.block_bootstrap import circular_block_bootstrap
from eri.inference.gaussian import (
    gaussian_sharpe_confidence_interval,
    gaussian_sharpe_test,
)
from eri.inference.power import gaussian_sharpe_power, minimum_sample_length
from eri.inference.sharpe import (
    annualized_sample_sharpe,
    estimate_sharpe,
    sample_mean,
    sample_sharpe,
    sample_standard_deviation,
)


def test_sample_statistics_use_ddof_one_and_explicit_annualization() -> None:
    returns = np.array([0.01, 0.03, -0.01, 0.02], dtype=np.float64)
    expected_mean = float(np.mean(returns))
    expected_standard_deviation = float(np.std(returns, ddof=1))
    expected_sharpe = expected_mean / expected_standard_deviation

    assert isclose(sample_mean(returns), expected_mean)
    assert isclose(sample_standard_deviation(returns), expected_standard_deviation)
    assert isclose(sample_sharpe(returns), expected_sharpe)
    assert isclose(
        annualized_sample_sharpe(returns, periods_per_year=252),
        expected_sharpe * sqrt(252),
    )

    estimate = estimate_sharpe(returns, annualize=True, periods_per_year=252)
    assert estimate.annualized
    assert estimate.annualization_convention == "square-root-of-time"
    assert estimate.periods_per_year == 252
    assert estimate.warnings


@pytest.mark.parametrize(
    "returns",
    [
        [],
        [0.1],
        [0.1, float("nan")],
        [0.1, float("inf")],
        [0.1, 0.1, 0.1],
    ],
)
def test_sharpe_rejects_invalid_or_degenerate_inputs(returns: list[float]) -> None:
    with pytest.raises(ValueError):
        sample_sharpe(returns)


def test_return_validation_rejects_non_vector_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_returns(np.ones((2, 2), dtype=np.float64))


@pytest.mark.parametrize("periods", [0.0, -1.0, float("inf"), float("nan")])
def test_periods_per_year_must_be_positive_and_finite(periods: float) -> None:
    with pytest.raises(ValueError):
        validate_periods_per_year(periods)


def test_summary_reports_raw_kurtosis() -> None:
    summary = summarize_returns([-1.0, 0.0, 1.0])

    assert summary.standard_deviation_ddof == 1
    assert isclose(summary.skewness, 0.0, abs_tol=1e-15)
    assert isclose(summary.raw_kurtosis, 1.5)


def test_gaussian_interval_and_p_values_are_valid() -> None:
    returns = np.array([-0.02, 0.01, 0.03, 0.015, -0.005, 0.025])
    interval = gaussian_sharpe_confidence_interval(returns, confidence_level=0.9)
    greater = gaussian_sharpe_test(
        returns, benchmark=0.0, alternative="greater"
    )
    two_sided = gaussian_sharpe_test(
        returns, benchmark=0.0, alternative="two-sided"
    )

    assert interval.lower <= interval.estimate <= interval.upper
    assert 0.0 <= greater.p_value <= 1.0
    assert 0.0 <= two_sided.p_value <= 1.0

    with pytest.raises(ValueError):
        gaussian_sharpe_confidence_interval(returns, confidence_level=1.0)
    with pytest.raises(ValueError):
        gaussian_sharpe_test(returns, benchmark=float("nan"))


def test_seeded_circular_bootstrap_is_exactly_reproducible_and_serializable() -> None:
    returns = np.array(
        [-0.03, 0.02, 0.01, 0.04, -0.01, 0.03, 0.005, -0.015],
        dtype=np.float64,
    )
    first = circular_block_bootstrap(
        returns,
        block_length=3,
        replications=40,
        seed=1729,
        retain_draws=True,
    )
    second = circular_block_bootstrap(
        returns,
        block_length=3,
        replications=40,
        seed=1729,
        retain_draws=True,
    )

    assert first == second
    assert first.draws is not None
    assert first.lower <= first.upper
    assert asdict(first)["seed"] == 1729


@pytest.mark.parametrize("block_length", [0, 9])
def test_bootstrap_rejects_invalid_block_length(block_length: int) -> None:
    with pytest.raises(ValueError):
        circular_block_bootstrap(
            [-0.03, 0.02, 0.01, 0.04, -0.01, 0.03, 0.005, -0.015],
            block_length=block_length,
            replications=10,
            seed=7,
        )


def test_power_and_minimum_sample_length_conventions() -> None:
    low_n = gaussian_sharpe_power(
        sample_size=50,
        true_sharpe=0.3,
        benchmark=0.0,
    )
    high_n = gaussian_sharpe_power(
        sample_size=200,
        true_sharpe=0.3,
        benchmark=0.0,
    )
    minimum = minimum_sample_length(
        true_sharpe=0.3,
        benchmark=0.0,
        target_power=0.8,
        periods_per_year=252,
    )

    assert high_n.power > low_n.power
    assert minimum.power >= 0.8
    assert minimum.sample_length_years is not None
    assert isclose(minimum.sample_length_years, minimum.sample_size / 252)
    if minimum.sample_size > 2:
        previous = gaussian_sharpe_power(
            sample_size=minimum.sample_size - 1,
            true_sharpe=0.3,
            benchmark=0.0,
        )
        assert previous.power < 0.8
