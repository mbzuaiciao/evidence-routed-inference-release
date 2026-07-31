"""Round-trip tests across related ERI schemas."""

from datetime import UTC, datetime

from eri.schemas import (
    AbstentionDecision,
    DiagnosticResult,
    EvidenceFinding,
    EvidenceItem,
    InferenceResult,
    MethodDecision,
    MethodEligibility,
)


def test_evidence_item_json_round_trip() -> None:
    evidence = EvidenceItem(
        evidence_id="e-001",
        source="trial-registry",
        claim="Treatment A reduced the measured outcome.",
        finding=EvidenceFinding.SUPPORT,
        observed_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
        attributes={"arm_count": 2, "preregistered": True},
    )

    restored = EvidenceItem.model_validate_json(evidence.model_dump_json())

    assert restored == evidence
    assert restored.model_dump(mode="json")["observed_at"] == "2026-01-02T03:04:00Z"


def test_decision_and_inference_results_serialize_with_nested_diagnostics() -> None:
    diagnostic = DiagnosticResult(
        diagnostic_id="variance-check",
        passed=True,
        statistic=1.2,
        p_value=0.31,
        message="No blocking variance issue detected.",
    )
    eligibility = MethodEligibility(
        method_id="welch-t",
        eligible=True,
        diagnostics=(diagnostic,),
    )
    decision = MethodDecision(
        selected_method="welch-t",
        rationale="The candidate passed the required diagnostics.",
        eligibility=(eligibility,),
    )
    assert decision.selected_method is not None
    result = InferenceResult(
        inference_id="i-001",
        method_id=decision.selected_method,
        method_version="1.0",
        method_configuration={"ddof": 1},
        estimate=1.4,
        standard_error=0.2,
        confidence_interval=(1.0, 1.8),
        p_value=0.02,
        diagnostics=(diagnostic,),
        metadata={"evidence_ids": ["e-001"]},
        sample_frequency="daily",
        periods_per_year=252,
        annualization_convention="square-root-of-time",
        benchmark_sharpe=0.0,
        data_provenance="trial-registry",
        data_fingerprint="sha256:abc",
        warnings=("Approximate finite-sample inference.",),
        assumptions=("IID Gaussian returns.",),
    )

    restored_decision = MethodDecision.model_validate_json(decision.model_dump_json())
    restored_result = InferenceResult.model_validate_json(result.model_dump_json())

    assert restored_decision == decision
    assert restored_result == result
    assert restored_result.diagnostics[0].diagnostic_id == "variance-check"


def test_abstention_decision_json_round_trip() -> None:
    decision = AbstentionDecision(
        abstain=True,
        reasons=("No candidate method passed its required diagnostics.",),
        missing_evidence=("A larger independent sample is required.",),
    )

    restored = AbstentionDecision.model_validate_json(decision.model_dump_json())

    assert restored == decision
