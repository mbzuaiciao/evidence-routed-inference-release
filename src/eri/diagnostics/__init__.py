"""Typed deterministic diagnostics for statistical inference."""

from eri.diagnostics.provenance import ProvenanceCompleteness, ResearchProvenance
from eri.diagnostics.registry import run_diagnostics
from eri.diagnostics.summary import ReturnSummary, summarize_returns

__all__ = [
    "ProvenanceCompleteness",
    "ResearchProvenance",
    "ReturnSummary",
    "run_diagnostics",
    "summarize_returns",
]
