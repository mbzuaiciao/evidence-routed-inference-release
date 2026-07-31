"""Framework-independent statistical inference functions."""

from eri.inference.block_bootstrap import (
    circular_block_bootstrap,
    circular_block_indices,
)
from eri.inference.gaussian import (
    gaussian_sharpe_confidence_interval,
    gaussian_sharpe_test,
    iid_gaussian_sharpe_standard_error,
)
from eri.inference.hac import hac_sharpe_standard_error
from eri.inference.mertens_psr import probabilistic_sharpe_ratio
from eri.inference.power import gaussian_sharpe_power, minimum_sample_length
from eri.inference.sharpe import (
    annualized_sample_sharpe,
    sample_mean,
    sample_sharpe,
    sample_standard_deviation,
)

__all__ = [
    "annualized_sample_sharpe",
    "circular_block_bootstrap",
    "circular_block_indices",
    "gaussian_sharpe_confidence_interval",
    "gaussian_sharpe_power",
    "gaussian_sharpe_test",
    "hac_sharpe_standard_error",
    "iid_gaussian_sharpe_standard_error",
    "minimum_sample_length",
    "probabilistic_sharpe_ratio",
    "sample_mean",
    "sample_sharpe",
    "sample_standard_deviation",
]
