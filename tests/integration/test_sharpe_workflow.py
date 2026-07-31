"""Lightweight deterministic end-to-end statistical workflow."""

import json
from dataclasses import asdict

import numpy as np

from eri.data.returns import validate_returns
from eri.diagnostics.summary import summarize_returns
from eri.inference.block_bootstrap import circular_block_bootstrap
from eri.inference.gaussian import (
    gaussian_sharpe_confidence_interval,
    gaussian_sharpe_test,
)
from eri.inference.hac import hac_sharpe_standard_error
from eri.inference.mertens_psr import probabilistic_sharpe_ratio
from eri.inference.sharpe import estimate_sharpe


def test_numeric_returns_flow_through_all_phase1_methods() -> None:
    generator = np.random.default_rng(314159)
    returns = validate_returns(generator.normal(0.015, 0.08, size=240))

    summary = summarize_returns(returns)
    estimate = estimate_sharpe(returns)
    gaussian_interval = gaussian_sharpe_confidence_interval(returns)
    gaussian_test = gaussian_sharpe_test(
        returns, benchmark=0.0, alternative="greater"
    )
    psr = probabilistic_sharpe_ratio(returns, benchmark=0.0)
    hac = hac_sharpe_standard_error(returns, bandwidth=4)
    bootstrap = circular_block_bootstrap(
        returns,
        block_length=5,
        replications=80,
        seed=2718,
    )

    structured = {
        "summary": asdict(summary),
        "estimate": asdict(estimate),
        "gaussian_interval": asdict(gaussian_interval),
        "gaussian_test": asdict(gaussian_test),
        "psr": asdict(psr),
        "hac": asdict(hac),
        "bootstrap": asdict(bootstrap),
    }
    serialized = json.dumps(structured, sort_keys=True)

    assert summary.sample_size == 240
    assert gaussian_interval.lower <= gaussian_interval.upper
    assert 0.0 <= gaussian_test.p_value <= 1.0
    assert 0.0 <= psr.probabilistic_sharpe_ratio <= 1.0
    assert hac.bandwidth == 4
    assert bootstrap.block_length == 5
    assert "mertens-psr" in serialized
