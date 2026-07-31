"""Evidence masking and immutable shared-cell fingerprints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256

from eri.schemas.evidence import EvidenceItem

GROUP_BY_DIAGNOSTIC = {
    "data-quality": "returns-summary",
    "linear-dependence": "linear-dependence",
    "squared-return-dependence": "squared-dependence",
    "distribution-shape": "distribution",
    "split-sample-stability": "stability",
    "research-provenance": "structured-provenance",
}

ACQUIRABLE_GROUPS = (
    "linear-dependence",
    "squared-dependence",
    "distribution",
    "stability",
)


def evidence_group(item: EvidenceItem) -> str:
    """Map audited diagnostic output to one benchmark-visible group."""
    return GROUP_BY_DIAGNOSTIC.get(item.diagnostic_name or "", "returns-summary")


def mask_evidence(
    evidence: tuple[EvidenceItem, ...], groups: tuple[str, ...]
) -> tuple[EvidenceItem, ...]:
    """Expose only explicitly named evidence groups."""
    allowed = set(groups)
    return tuple(item for item in evidence if evidence_group(item) in allowed)


def reveal_group(
    full_evidence: tuple[EvidenceItem, ...],
    visible_evidence: tuple[EvidenceItem, ...],
    group: str,
) -> tuple[EvidenceItem, ...]:
    """Reveal a diagnostic group without duplicating existing evidence."""
    existing = {item.evidence_id for item in visible_evidence}
    additions = tuple(
        item
        for item in full_evidence
        if evidence_group(item) == group and item.evidence_id not in existing
    )
    return (*visible_evidence, *additions)


def stable_fingerprint(value: object) -> str:
    """Hash a canonical JSON representation used by every paired system."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode()).hexdigest()


def shared_fingerprints(
    *,
    returns: tuple[float, ...],
    evidence: tuple[EvidenceItem, ...],
    prose: str,
    allowed_tools: tuple[str, ...],
    budget: Mapping[str, object],
    method_metadata: Mapping[str, object] | None = None,
    inference_conventions: Mapping[str, object] | None = None,
) -> dict[str, str]:
    return {
        "returns": stable_fingerprint(returns),
        "initial_evidence": stable_fingerprint(
            [item.model_dump(mode="json") for item in evidence]
        ),
        "prose_metadata": stable_fingerprint(prose),
        "allowed_tools": stable_fingerprint(allowed_tools),
        "budget": stable_fingerprint(budget),
        "method_metadata": stable_fingerprint(method_metadata or {}),
        "inference_conventions": stable_fingerprint(inference_conventions or {}),
    }
