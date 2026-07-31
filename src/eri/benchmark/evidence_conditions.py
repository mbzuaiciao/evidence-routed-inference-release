"""Registered Phase 4B information conditions and adaptive baseline policy."""

from __future__ import annotations

from eri.benchmark.information_regimes import ACQUIRABLE_GROUPS
from eri.schemas.evidence import EvidenceFinding, EvidenceItem

REGISTERED_CONDITIONS = (
    "full-structured",
    "returns-only",
    "linear-only",
    "distribution-only",
    "conflicting-weak",
    "squared-hidden",
    "stability-hidden",
    "provenance-explicit",
    "provenance-incomplete",
    "provenance-hedged",
    "provenance-conflicting",
    "provenance-missing-failures",
    "provenance-multiple-markets",
    "provenance-unknown-dependence",
    "provenance-irrelevant",
    "provenance-injection-data",
)

ADAPTIVE_POLICY_VERSION = "phase4b-adaptive-v1.0"


def adaptive_diagnostic_requests(
    visible: tuple[EvidenceItem, ...], *, budget: int
) -> tuple[str, ...]:
    """Fixed non-LLM priority policy based only on currently visible evidence."""
    visible_groups = {
        {
            "linear-dependence": "linear-dependence",
            "squared-return-dependence": "squared-dependence",
            "distribution-shape": "distribution",
            "split-sample-stability": "stability",
        }.get(item.diagnostic_name or "", "returns-summary")
        for item in visible
    }
    findings = {
        {
            "linear-dependence": "linear-dependence",
            "squared-return-dependence": "squared-dependence",
            "distribution-shape": "distribution",
            "split-sample-stability": "stability",
        }.get(item.diagnostic_name or "", "returns-summary"): item.finding
        for item in visible
    }
    # Structural instability already forces the conservative router to abstain;
    # additional diagnostics cannot change that action.
    if findings.get("stability") is EvidenceFinding.CONTRADICTS:
        return ()
    requests: list[str] = []
    for group in (
        "squared-dependence",
        "stability",
        "linear-dependence",
        "distribution",
    ):
        if group not in visible_groups:
            requests.append(group)
        if len(requests) >= budget:
            break
        # Squared-return dependence plus stability are the minimum evidence
        # needed to distinguish a robust route from structural abstention.
        combined = visible_groups | set(requests)
        if {"squared-dependence", "stability"}.issubset(combined):
            squared = findings.get("squared-dependence")
            if squared is EvidenceFinding.CONTRADICTS:
                break
    return tuple(item for item in requests if item in ACQUIRABLE_GROUPS)
