"""Unit tests for eligibility, abstention, and deterministic tie-breaking."""

from math import isclose
from pathlib import Path

import pytest
from pydantic import ValidationError

from eri.config import Phase2Config, RoutingConfig, load_phase2_config
from eri.diagnostics.evidence_factory import make_evidence
from eri.routing.router import route_methods
from eri.routing.sensitivity import compare_conclusions
from eri.schemas.evidence import EvidenceFinding, EvidenceItem
from eri.schemas.inference import InferenceResult

ROOT = Path(__file__).resolve().parents[2]


def _config() -> Phase2Config:
    return load_phase2_config(ROOT / "configs")


def _item(claim: str, finding: EvidenceFinding, sample_size: int = 200) -> EvidenceItem:
    return make_evidence(
        claim=claim,
        finding=finding,
        diagnostic_name="routing-fixture",
        sample_size=sample_size,
    )


def _base_evidence() -> tuple[EvidenceItem, ...]:
    return (
        _item("data_validity", EvidenceFinding.SUPPORTS),
        _item("variance_adequate", EvidenceFinding.SUPPORTS),
        _item("linear_independence", EvidenceFinding.INCONCLUSIVE),
        _item("absence_of_squared_return_dependence", EvidenceFinding.INCONCLUSIVE),
        _item("gaussian_shape", EvidenceFinding.INCONCLUSIVE),
        _item("structural_stability", EvidenceFinding.INCONCLUSIVE),
        _item("selection_provenance_complete", EvidenceFinding.INCONCLUSIVE),
    )


def test_configured_priority_is_deterministic() -> None:
    decision = route_methods(_base_evidence(), _config().routing, _config().experiment)
    assert decision.selected_method == "hac-sharpe"
    assert decision.tie_break_rule == "first eligible method in routing.method_priority"
    assert "circular-block-bootstrap" in decision.sensitivity_methods
    eligibility = {item.method_id: item for item in decision.eligibility}
    assert eligibility["iid-gaussian-sharpe"].sensitivity_only
    assert eligibility["mertens-psr"].sensitivity_only


def test_detected_dependence_rules_out_iid_methods() -> None:
    evidence = tuple(
        _item(item.claim, EvidenceFinding.CONTRADICTS)
        if item.claim == "linear_independence"
        else item
        for item in _base_evidence()
    )
    decision = route_methods(evidence, _config().routing, _config().experiment)
    eligibility = {item.method_id: item for item in decision.eligibility}
    assert not eligibility["iid-gaussian-sharpe"].eligible
    assert not eligibility["mertens-psr"].eligible
    assert decision.selected_method == "hac-sharpe"


def test_non_gaussianity_makes_gaussian_sensitivity_only() -> None:
    evidence = tuple(
        _item(item.claim, EvidenceFinding.CONTRADICTS)
        if item.claim == "gaussian_shape"
        else item
        for item in _base_evidence()
    )
    decision = route_methods(evidence, _config().routing, _config().experiment)
    gaussian = next(
        item for item in decision.eligibility if item.method_id == "iid-gaussian-sharpe"
    )
    assert gaussian.eligible and gaussian.weakened and gaussian.sensitivity_only


def test_invalid_data_requires_abstention() -> None:
    evidence = (_item("data_validity", EvidenceFinding.CONTRADICTS),)
    decision = route_methods(evidence, _config().routing, _config().experiment)
    assert decision.selected_method is None
    assert decision.abstention is not None and decision.abstention.abstain


def test_missing_provenance_blocks_multiplicity_sensitive_claim() -> None:
    config = _config()
    experiment = config.experiment.model_copy(
        update={"claim_requires_selection_provenance": True}
    )
    decision = route_methods(_base_evidence(), config.routing, experiment)
    assert decision.selected_method == "hac-sharpe"
    assert decision.abstention is None
    claim_abstention = decision.claim_abstentions["multiplicity_adjusted_sharpe"]
    assert "complete trial/search provenance" in claim_abstention.missing_evidence


def test_contradictory_claims_are_exposed() -> None:
    evidence = (
        *_base_evidence(),
        _item("gaussian_shape", EvidenceFinding.SUPPORTS),
        _item("gaussian_shape", EvidenceFinding.CONTRADICTS),
    )
    decision = route_methods(evidence, _config().routing, _config().experiment)
    assert decision.unresolved_conflicts
    gaussian = next(
        item for item in decision.eligibility if item.method_id == "iid-gaussian-sharpe"
    )
    assert gaussian.weakened
    assert set(gaussian.evidence_references) <= set(decision.evidence_references)


def test_contradictory_dependence_is_conservative_under_reordering() -> None:
    contradiction = _item("linear_independence", EvidenceFinding.CONTRADICTS)
    support = _item("linear_independence", EvidenceFinding.SUPPORTS)
    base = tuple(
        item for item in _base_evidence() if item.claim != "linear_independence"
    )
    decisions = (
        route_methods(
            (*base, support, contradiction), _config().routing, _config().experiment
        ),
        route_methods(
            (*base, contradiction, support), _config().routing, _config().experiment
        ),
    )
    for decision in decisions:
        gaussian = next(
            item
            for item in decision.eligibility
            if item.method_id == "iid-gaussian-sharpe"
        )
        assert not gaussian.eligible


def test_sensitivity_threshold_is_fully_configured() -> None:
    psr = InferenceResult(
        inference_id="psr",
        method_id="mertens-psr",
        estimate=0.2,
        metadata={"frequentist_probability": 0.92},
    )
    strict = compare_conclusions((psr,), benchmark=0.0, significance_level=0.05)
    relaxed = compare_conclusions((psr,), benchmark=0.0, significance_level=0.10)
    assert isclose(strict.probability_threshold, 0.95)
    assert strict.conclusions["mertens-psr"] is False
    assert relaxed.conclusions["mertens-psr"] is True


def test_conflicting_valid_method_conclusions_are_exposed() -> None:
    positive = InferenceResult(
        inference_id="positive",
        method_id="method-a",
        estimate=0.2,
        confidence_interval=(0.1, 0.3),
    )
    negative = InferenceResult(
        inference_id="negative",
        method_id="method-b",
        estimate=-0.1,
        confidence_interval=(-0.2, 0.0),
    )
    comparison = compare_conclusions(
        (positive, negative), benchmark=0.0, significance_level=0.05
    )
    assert comparison.disagreement


def test_routing_configuration_rejects_unavailable_or_missing_methods() -> None:
    raw = _config().routing.model_dump()
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(
            {**raw, "method_priority": ["iid-gaussian-sharpe", "unknown"]}
        )
