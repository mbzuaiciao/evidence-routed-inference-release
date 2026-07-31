"""Reproducible Phase 3A return-generating processes."""

from eri.simulation.ar1 import simulate_ar1
from eri.simulation.garch import simulate_garch
from eri.simulation.iid_gaussian import simulate_iid_gaussian
from eri.simulation.structural_break import simulate_structural_break

__all__ = [
    "simulate_ar1",
    "simulate_garch",
    "simulate_iid_gaussian",
    "simulate_structural_break",
]
