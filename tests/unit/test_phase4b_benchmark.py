"""Unit contracts for the Phase 4B information benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.estimate_phase4b_cost import estimate
from scripts.run_phase4b_benchmark import load_provider_factory

from eri.agents.monolithic import validate_monolithic_output
from eri.agents.unconstrained import validate_unconstrained_output
from eri.benchmark.agent_baselines import apply_execution_guardrail
from eri.benchmark.evidence_conditions import (
    ADAPTIVE_POLICY_VERSION,
    adaptive_diagnostic_requests,
)
from eri.benchmark.information_regimes import (
    mask_evidence,
    shared_fingerprints,
    stable_fingerprint,
)
from eri.benchmark.phase4b_evaluation import (
    paired_comparisons,
    permitted_claim_level,
)
from eri.benchmark.phase4b_runner import load_phase4b_config
from eri.benchmark.phase4b_schemas import (
    BenchmarkTier,
    ClaimLevel,
    InformationRegime,
    Phase4BSystem,
    ProviderMode,
)
from eri.benchmark.provenance_benchmark import (
    generate_provenance_case,
    public_provenance,
    score_provenance,
)
from eri.config import load_phase2_config
from eri.diagnostics.registry import run_diagnostics
from eri.routing.router import route_methods

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def evidence() -> tuple[object, ...]:
    config = load_phase2_config(ROOT / "configs")
    return run_diagnostics(
        [0.01, -0.005, 0.006, -0.002, 0.008, 0.001] * 40,
        config.diagnostics,
    )


def test_smoke_config_registers_exact_systems_and_regimes() -> None:
    config = load_phase4b_config(ROOT / "configs/benchmark/phase4b-smoke.yaml")
    assert set(config.systems) == set(Phase4BSystem)
    assert set(config.regimes) == set(InformationRegime)


def test_deadline_reuses_eight_scenarios_and_all_evidence_conditions() -> None:
    config = load_phase4b_config(ROOT / "configs/benchmark/phase4b-deadline.yaml")
    assert len(config.scenario_ids) == 8
    ids = {item.condition_id for item in config.conditions}
    assert {"returns-only", "linear-only", "distribution-only"} <= ids
    assert {"conflicting-weak", "squared-hidden", "stability-hidden"} <= ids
    assert "provenance-injection-data" in ids


def test_information_mask_exposes_only_named_groups(
    evidence: tuple[object, ...],
) -> None:
    from eri.schemas.evidence import EvidenceItem

    typed = tuple(item for item in evidence if isinstance(item, EvidenceItem))
    masked = mask_evidence(typed, ("returns-summary", "linear-dependence"))
    assert {item.diagnostic_name for item in masked} <= {
        "data-quality",
        "linear-dependence",
    }
    assert all(item.diagnostic_name != "squared-return-dependence" for item in masked)


def test_fingerprints_are_shared_and_content_sensitive(
    evidence: tuple[object, ...],
) -> None:
    from eri.schemas.evidence import EvidenceItem

    typed = tuple(item for item in evidence if isinstance(item, EvidenceItem))
    first = shared_fingerprints(
        returns=(0.1, 0.2),
        evidence=typed,
        prose="same",
        allowed_tools=("stability",),
        budget={"diagnostic_budget": 1},
    )
    assert first == shared_fingerprints(
        returns=(0.1, 0.2),
        evidence=typed,
        prose="same",
        allowed_tools=("stability",),
        budget={"diagnostic_budget": 1},
    )
    assert first["returns"] != stable_fingerprint((0.1, 0.3))


def test_adaptive_policy_is_deterministic_and_respects_budget(
    evidence: tuple[object, ...],
) -> None:
    from eri.schemas.evidence import EvidenceItem

    typed = tuple(item for item in evidence if isinstance(item, EvidenceItem))
    initial = mask_evidence(typed, ("returns-summary",))
    requests = adaptive_diagnostic_requests(initial, budget=2)
    assert requests == adaptive_diagnostic_requests(initial, budget=2)
    assert len(requests) == 2
    assert requests == ("squared-dependence", "stability")
    assert ADAPTIVE_POLICY_VERSION == "phase4b-adaptive-v1.0"


def test_adaptive_policy_requests_hidden_stability_and_stops_when_sufficient(
    evidence: tuple[object, ...],
) -> None:
    from eri.schemas.evidence import EvidenceFinding, EvidenceItem

    typed = tuple(item for item in evidence if isinstance(item, EvidenceItem))
    stability_hidden = mask_evidence(
        typed,
        ("returns-summary", "linear-dependence", "squared-dependence", "distribution"),
    )
    assert adaptive_diagnostic_requests(stability_hidden, budget=1) == ("stability",)
    assert adaptive_diagnostic_requests(typed, budget=4) == ()
    assert adaptive_diagnostic_requests(
        mask_evidence(typed, ("returns-summary",)), budget=1
    ) == ("squared-dependence",)
    stability = next(
        item for item in typed if item.diagnostic_name == "split-sample-stability"
    )
    unstable = tuple(
        item.model_copy(update={"finding": EvidenceFinding.CONTRADICTS})
        if item.evidence_id == stability.evidence_id
        else item
        for item in mask_evidence(typed, ("returns-summary", "stability"))
    )
    assert adaptive_diagnostic_requests(unstable, budget=4) == ()


def test_provenance_truth_is_hidden_from_public_payload() -> None:
    case = generate_provenance_case("explicit-complete")
    assert "truth" not in public_provenance(case)
    assert "24 trials" in case.text


def test_prompt_injection_is_benchmark_data_not_an_instruction() -> None:
    case = generate_provenance_case("prompt-injection-data")
    assert case.truth.trials is None
    assert case.truth.completeness == "missing"


def test_provenance_scoring_rewards_exact_supported_fields() -> None:
    case = generate_provenance_case("explicit-complete")
    scores = score_provenance(
        {"trials": 24, "parameter_searches": 6, "completeness": "complete"},
        {"trials": "24 trials", "parameter_searches": "6 parameter searches"},
        case,
    )
    assert scores["exact_field_precision"] == 1.0
    assert scores["source_span_validity"] == 1.0


def test_provenance_scoring_rejects_fabricated_number() -> None:
    case = generate_provenance_case("incomplete")
    scores = score_provenance({"trials": 999}, {}, case)
    assert scores["fabricated_number_rate"] == 1.0


def test_hedged_and_conflicting_counts_are_not_exact_truth() -> None:
    hedged = generate_provenance_case("hedged-counts")
    conflict = generate_provenance_case("conflicting-counts")
    assert hedged.truth.trials is None
    assert hedged.truth.parameter_searches is None
    assert "trials" in hedged.truth.unresolved_fields
    assert conflict.truth.trials is None
    assert "trials" in conflict.truth.unresolved_fields


def test_provenance_sequence_normalization_and_unresolved_scoring() -> None:
    case = generate_provenance_case("multiple-markets-windows")
    scores = score_provenance(
        {
            "markets_or_universes": ["US", "EU", "Japan"],
            "start_dates_or_windows": ["2008-2015", "2016-2024"],
            "completeness": "partial",
        },
        {
            "markets_or_universes": "US, EU, and Japan markets",
            "start_dates_or_windows": "2008-2015 and 2016-2024",
        },
        case,
        case.truth.unresolved_fields,
    )
    assert scores["exact_field_precision"] == 1.0
    assert scores["source_span_validity"] == 1.0
    assert scores["unresolved_field_accuracy"] == 1.0


def test_monolithic_output_validation() -> None:
    parsed = validate_monolithic_output(
        {
            "diagnostic_requests": ["stability"],
            "provenance_interpretation": {},
            "method_recommendation": "hac-sharpe",
            "abstain": False,
            "report": "Supported recommendation.",
            "evidence_references": ["e1"],
        }
    )
    assert parsed.method_recommendation == "hac-sharpe"
    with pytest.raises(ValidationError):
        validate_monolithic_output(
            {"method_recommendation": None, "abstain": False, "report": "bad"}
        )


def test_unconstrained_output_is_recordable() -> None:
    parsed = validate_unconstrained_output(
        {
            "role_messages": [
                {"role": "planner", "recommendation": "Request stability."}
            ],
            "method_recommendation": None,
            "abstain": True,
            "report": "Insufficient evidence.",
        }
    )
    assert parsed.role_messages[0].role == "planner"


def test_guardrail_vetoes_ineligible_recommendation(
    evidence: tuple[object, ...],
) -> None:
    from eri.schemas.evidence import EvidenceItem

    config = load_phase2_config(ROOT / "configs")
    typed = tuple(item for item in evidence if isinstance(item, EvidenceItem))
    decision = route_methods(typed, config.routing, config.experiment)
    ineligible = decision.eligibility[0].model_copy(update={"eligible": False})
    recommendation, rejected = apply_execution_guardrail(
        ineligible.method_id,
        (ineligible, *decision.eligibility[1:]),
    )
    assert recommendation is None
    assert rejected == (f"eligibility-veto:{ineligible.method_id}",)


def test_mock_claim_level_is_implementation_only() -> None:
    config = load_phase4b_config(ROOT / "configs/benchmark/phase4b-smoke.yaml")
    assert permitted_claim_level(config) is ClaimLevel.IMPLEMENTATION_ONLY


def test_live_pilot_claim_level_is_descriptive_only() -> None:
    config = load_phase4b_config(ROOT / "configs/benchmark/phase4b-live-pilot.yaml")
    assert permitted_claim_level(config) is ClaimLevel.PILOT_DESCRIPTIVE
    assert (
        permitted_claim_level(config, live_outputs_complete=False)
        is ClaimLevel.IMPLEMENTATION_ONLY
    )
    assert (
        permitted_claim_level(config, unreviewed_scientific_override=True)
        is ClaimLevel.IMPLEMENTATION_ONLY
    )


def test_missing_provider_metadata_blocks_benchmark_claim() -> None:
    config = load_phase4b_config(ROOT / "configs/benchmark/phase4b-deadline.yaml")
    reviewed = config.model_copy(update={"independent_review_complete": True})
    assert (
        permitted_claim_level(reviewed, provider_metadata_complete=False)
        is ClaimLevel.IMPLEMENTATION_ONLY
    )


def test_mock_cannot_be_configured_as_scientific_tier() -> None:
    config = load_phase4b_config(ROOT / "configs/benchmark/phase4b-smoke.yaml")
    with pytest.raises(ValidationError):
        config.model_copy(
            update={"tier": BenchmarkTier.DEADLINE, "provider_mode": ProviderMode.MOCK}
        ).__class__.model_validate(
            {
                **config.model_dump(mode="python"),
                "tier": BenchmarkTier.DEADLINE,
            }
        )


def test_cost_accounting_matches_cell_and_call_budgets() -> None:
    result = estimate(ROOT / "configs/benchmark/phase4b-live-pilot.yaml")
    assert result["matched_cells"] == 18
    assert result["maximum_model_calls"] == 216
    assert result["logical_model_calls"] == 180
    assert result["worst_case_model_attempts_with_retries"] == 432
    assert (
        float(result["estimated_cost_expected_usd"])
        < float(result["estimated_cost_upper_bound_usd"])
        < float(result["estimated_cost_worst_case_usd"])
    )
    assert float(result["estimated_cost_upper_bound_usd"]) > 0


def test_deadline_cost_accounting_includes_all_model_systems_and_retries() -> None:
    result = estimate(ROOT / "configs/benchmark/phase4b-deadline.yaml")
    assert result["matched_cells"] == 240
    assert result["logical_model_calls"] == 2400
    assert result["maximum_model_calls"] == 2880
    assert result["worst_case_model_attempts_with_retries"] == 5760


def test_live_provider_factory_must_be_explicit_module_function() -> None:
    with pytest.raises(ValueError, match="module:function"):
        load_provider_factory("implicit-provider")


def test_paired_metric_uses_matched_cells() -> None:
    rows = [
        {
            "cell_id": "c1",
            "scenario_id": "s1",
            "system": Phase4BSystem.STATIC_DETERMINISTIC.value,
            "invalid_primary_method_selection": 1,
            "harmful_nonabstention": 0,
            "routing_regret": 1.0,
        },
        {
            "cell_id": "c1",
            "scenario_id": "s1",
            "system": Phase4BSystem.ADAPTIVE_DETERMINISTIC.value,
            "invalid_primary_method_selection": 0,
            "harmful_nonabstention": 0,
            "routing_regret": 0.0,
        },
    ]
    comparisons = paired_comparisons(rows)
    target = next(
        row
        for row in comparisons
        if row["system"] == Phase4BSystem.ADAPTIVE_DETERMINISTIC.value
        and row["metric"] == "invalid_primary_method_selection"
    )
    assert target["paired_mean_difference"] == -1.0
