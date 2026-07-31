"""Unit tests for individual ERI schema invariants."""

from math import inf

import pytest
from pydantic import ValidationError

from eri.schemas import (
    AbstentionDecision,
    DiagnosticResult,
    EvidenceFinding,
    EvidenceItem,
    InferenceResult,
    MethodDecision,
    MethodEligibility,
)


def test_evidence_item_rejects_blank_claim() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="e-1",
            source="study-a",
            claim="   ",
            finding=EvidenceFinding.INCONCLUSIVE,
        )


def test_evidence_item_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="e-1",
            source="study-a",
            claim="A measured claim.",
            finding=EvidenceFinding.SUPPORT,
            unsupported=True,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("p_value", [-0.01, 1.01])
def test_diagnostic_result_rejects_invalid_probability(p_value: float) -> None:
    with pytest.raises(ValidationError):
        DiagnosticResult(
            diagnostic_id="normality",
            passed=False,
            p_value=p_value,
        )


def test_ineligible_method_requires_a_reason() -> None:
    with pytest.raises(ValidationError, match="at least one reason"):
        MethodEligibility(method_id="ols", eligible=False)


def test_method_decision_requires_selected_method_to_be_eligible() -> None:
    rejected = MethodEligibility(
        method_id="ols",
        eligible=False,
        reasons=("heteroskedastic residuals",),
    )

    with pytest.raises(ValidationError, match="selected method must be eligible"):
        MethodDecision(
            selected_method="ols",
            rationale="candidate considered first",
            eligibility=(rejected,),
        )


def test_method_decision_supports_abstention_and_sensitivity_methods() -> None:
    primary = MethodEligibility(method_id="hac", eligible=True)
    sensitivity = MethodEligibility(method_id="bootstrap", eligible=True)
    selected = MethodDecision(
        selected_method="hac",
        rationale="Dependence-robust primary estimate.",
        eligibility=(primary, sensitivity),
        sensitivity_methods=("bootstrap",),
    )
    abstained = MethodDecision(
        rationale="No method met minimum data requirements.",
        eligibility=(
            MethodEligibility(
                method_id="hac",
                eligible=False,
                reasons=("insufficient observations",),
            ),
        ),
        abstention=AbstentionDecision(
            abstain=True,
            reasons=("No eligible methods.",),
        ),
    )

    assert selected.sensitivity_methods == ("bootstrap",)
    assert abstained.selected_method is None
    assert abstained.abstention is not None


def test_inference_result_rejects_invalid_numeric_outputs() -> None:
    with pytest.raises(ValidationError):
        InferenceResult(
            inference_id="i-1",
            method_id="welch-t",
            estimate=inf,
        )

    with pytest.raises(ValidationError, match="lower bound exceeds"):
        InferenceResult(
            inference_id="i-2",
            method_id="welch-t",
            estimate=1.5,
            confidence_interval=(2.0, 1.0),
        )


def test_inference_result_records_audit_metadata() -> None:
    result = InferenceResult(
        inference_id="i-audit",
        method_id="gaussian-sharpe",
        method_version="1.0",
        method_configuration={"ddof": 1},
        estimate=0.5,
        sample_frequency="daily",
        periods_per_year=252,
        annualization_convention="square-root-of-time",
        benchmark_sharpe=0.0,
        data_provenance="returns.csv",
        data_fingerprint="sha256:abc",
        warnings=("Serial dependence not assessed.",),
        assumptions=("Returns are excess returns.",),
    )

    assert result.method_version == "1.0"
    assert result.periods_per_year == 252


def test_abstention_requires_a_reason() -> None:
    with pytest.raises(ValidationError, match="at least one reason"):
        AbstentionDecision(abstain=True)


def test_valid_method_decision_is_frozen() -> None:
    eligible = MethodEligibility(method_id="welch-t", eligible=True)
    decision = MethodDecision(
        selected_method="welch-t",
        rationale="unequal variances are permitted",
        eligibility=(eligible,),
    )

    with pytest.raises(ValidationError):
        decision.rationale = "changed"  # type: ignore[misc]
