"""Audit regression fixtures for Phase 4B fairness and provider boundaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from eri.agents.monolithic import MonolithicOutput
from eri.agents.provider import ProviderError
from eri.agents.schemas import (
    AgentRole,
    DiagnosticRequest,
    InformationGain,
    PlannerOutput,
    ProviderMessage,
)
from eri.agents.semantic_validation import validate_planner_output
from eri.agents.unconstrained import UnconstrainedOutput
from eri.benchmark.agent_baselines import run_system
from eri.benchmark.information_regimes import mask_evidence
from eri.benchmark.phase4b_runner import load_phase4b_config
from eri.benchmark.phase4b_schemas import Phase4BConfig, Phase4BSystem, SharedCell
from eri.benchmark.runner import load_benchmark_config, load_loss_profile
from eri.benchmark.schemas import BenchmarkConfig, LossProfile, SimulationResult
from eri.config import Phase2Config, load_phase2_config
from eri.diagnostics.registry import run_diagnostics
from eri.orchestration.adjudicator import (
    DeterministicAdjudicator,
    build_planner_semantic_context,
)
from eri.orchestration.audit import AuditTrail
from eri.orchestration.budgets import AgentBudgets, BudgetLedger
from eri.orchestration.phase4b import build_shared_cell

ROOT = Path(__file__).resolve().parents[2]
ModelT = TypeVar("ModelT", bound=BaseModel)


class FakeLiveProvider:
    """Deterministic typed provider fixture with usage metadata and no network."""

    def __init__(self, recommendation: str = "unknown-unregistered-method") -> None:
        self.recommendation = recommendation
        self.audit_records: list[dict[str, object]] = []

    def generate_typed(
        self,
        *,
        role: AgentRole,
        messages: Sequence[ProviderMessage],
        output_schema: type[ModelT],
    ) -> ModelT:
        del role
        payload = json.loads(messages[-1].content)
        assert "truth" not in payload
        serialized = json.dumps(payload, sort_keys=True)
        assert '"sampling_seed"' not in serialized
        assert '"return_seed"' in serialized
        self.audit_records.append(
            {
                "attempt": 1,
                "input_tokens": 40,
                "output_tokens": 20,
                "latency_seconds": 0.01,
                "configuration_resolution": {
                    "model": {
                        "canonical_value": "gpt-5.6-sol",
                        "override_supplied": True,
                        "override_source": "environment",
                        "final_value": "reviewed-override",
                        "secret": False,
                        "changes_scientific_comparability": True,
                    }
                },
            }
        )
        if output_schema is MonolithicOutput:
            value: object = {
                "diagnostic_requests": [],
                "provenance_interpretation": {},
                "method_recommendation": self.recommendation,
                "abstain": False,
                "report": "Fixture recommendation recorded for safe execution.",
                "evidence_references": [],
            }
        elif output_schema is UnconstrainedOutput:
            value = {
                "role_messages": [
                    {"role": "reviewer", "recommendation": "Record the raw action."}
                ],
                "diagnostic_requests": [],
                "provenance_interpretation": {},
                "method_recommendation": self.recommendation,
                "abstain": False,
                "report": "Fixture recommendation recorded for safe execution.",
                "evidence_references": [],
            }
        else:  # pragma: no cover - the fixture intentionally tests two-call systems
            raise AssertionError(output_schema)
        return output_schema.model_validate(value)


class FailingLiveProvider:
    audit_records: list[dict[str, object]] = []

    def generate_typed(
        self,
        *,
        role: AgentRole,
        messages: Sequence[ProviderMessage],
        output_schema: type[ModelT],
    ) -> ModelT:
        del role, messages, output_schema
        raise ProviderError("deterministic provider failure")


class ZeroRetryTimeoutProvider:
    max_retries = 0

    def __init__(self) -> None:
        self.audit_records: list[dict[str, object]] = []

    def generate_typed(
        self,
        *,
        role: AgentRole,
        messages: Sequence[ProviderMessage],
        output_schema: type[ModelT],
    ) -> ModelT:
        del messages, output_schema
        self.audit_records.append(
            {
                "provider": "lmstudio",
                "requested_model": "openai/gpt-oss-20b",
                "role": role.value,
                "attempt": 1,
                "retry_count": 0,
                "repair_attempt": False,
                "failure_kind": "timeout",
                "timeout_seconds": 120.0,
                "elapsed_seconds": 120.01,
                "provider_sdk_automatic_retries": 0,
                "repository_max_retries": 0,
                "maximum_physical_attempts_per_logical_call": 1,
                "repository_retry_delay_seconds": 0.0,
                "configuration_resolution": {},
            }
        )
        raise ProviderError("provider attempts exhausted after 1 attempt")


class SemanticallyInvalidPlannerProvider:
    max_retries = 0

    def __init__(self, *, malformed_text: bool = False) -> None:
        self.malformed_text = malformed_text
        self.audit_records: list[dict[str, object]] = []
        self.calls = 0

    def generate_typed(
        self,
        *,
        role: AgentRole,
        messages: Sequence[ProviderMessage],
        output_schema: type[ModelT],
    ) -> ModelT:
        del messages
        self.calls += 1
        self.audit_records.append(
            {
                "provider": "lmstudio",
                "requested_model": "openai/gpt-oss-20b",
                "role": role.value,
                "attempt": 1,
                "retry_count": 0,
                "repair_attempt": False,
                "input_tokens": 10,
                "output_tokens": 10,
                "total_tokens": 20,
                "latency_seconds": 0.01,
                "provider_sdk_automatic_retries": 0,
                "repository_max_retries": 0,
                "maximum_physical_attempts_per_logical_call": 1,
                "repository_retry_delay_seconds": 0.0,
                "configuration_resolution": {},
            }
        )
        assert output_schema is PlannerOutput
        value = PlannerOutput(
            requested_diagnostics=(
                DiagnosticRequest(
                    diagnostic_id="unknown-diagnostic",
                    rationale=(
                        "bad <|reserved_200016|>"
                        if self.malformed_text
                        else "Unknown request fixture."
                    ),
                    unresolved_claims=("absence_of_squared_return_dependence",),
                    priority=1,
                    expected_information_gain=InformationGain.HIGH,
                ),
            ),
            rationale="Schema-valid semantic failure fixture.",
            stop_recommendation=False,
        )
        return output_schema.model_validate(value.model_dump(mode="json"))


def _fixture_cell() -> tuple[
    SharedCell,
    SimulationResult,
    Phase2Config,
    BenchmarkConfig,
    Phase4BConfig,
    LossProfile,
]:
    config = load_phase4b_config(ROOT / "configs/benchmark/phase4b-live-pilot.yaml")
    phase2 = load_phase2_config(ROOT / "configs")
    source = load_benchmark_config(ROOT / config.scenario_source)
    scenario = source.scenarios[0]
    condition = next(
        item for item in config.conditions if item.condition_id == "squared-hidden"
    )
    cell, simulation = build_shared_cell(
        scenario=scenario,
        seed=config.return_seeds[0],
        condition=condition,
        phase2=phase2,
        config=config,
    )
    benchmark = source.model_copy(update={"scenarios": (scenario,)})
    profile = load_loss_profile(
        ROOT / "configs/benchmark/loss-profiles.yaml", config.loss_profile
    )
    return cell, simulation, phase2, benchmark, config, profile


def test_fake_live_provider_preserves_raw_recommendation_and_usage() -> None:
    cell, simulation, phase2, benchmark, config, profile = _fixture_cell()
    provider = FakeLiveProvider()
    result = run_system(
        Phase4BSystem.UNCONSTRAINED_AGENTS,
        cell,
        simulation,
        phase2,
        benchmark,
        config,
        provider,
        profile,
    )
    assert result.output.recommended_method == "unknown-unregistered-method"
    assert result.output.selected_method is None
    assert result.output.unknown_tool_requests == ("unknown-unregistered-method",)
    assert result.output.input_fingerprints == cell.fingerprints
    assert result.provider.input_tokens == 160
    assert result.provider.output_tokens == 80
    assert result.provider.usage_reported
    assert result.provider.requested_sampling_seed is None
    assert result.provider.sampling_seed is None
    assert result.provider.configuration_resolution["model"] == {
        "canonical_value": "gpt-5.6-sol",
        "override_supplied": True,
        "override_source": "environment",
        "final_value": "reviewed-override",
        "secret": False,
        "changes_scientific_comparability": True,
    }


def test_live_provider_failure_is_retained_without_mock_fallback() -> None:
    cell, simulation, phase2, benchmark, config, profile = _fixture_cell()
    result = run_system(
        Phase4BSystem.MONOLITHIC_AGENT,
        cell,
        simulation,
        phase2,
        benchmark,
        config,
        FailingLiveProvider(),
        profile,
    )
    assert (
        result.output.execution_failure
        == "ProviderError: deterministic provider failure"
    )
    assert result.output.abstained
    assert result.provider.failure == result.output.execution_failure
    assert result.provider.mode.value == "live"


def test_safe_failure_uses_concrete_zero_retry_policy() -> None:
    cell, simulation, phase2, benchmark, config, profile = _fixture_cell()
    result = run_system(
        Phase4BSystem.MONOLITHIC_AGENT,
        cell,
        simulation,
        phase2,
        benchmark,
        config,
        ZeroRetryTimeoutProvider(),
        profile,
    )
    assert config.model.retries == 1
    assert result.provider.retry_policy == "at-most-0-retries"
    assert result.provider.repository_max_retries == 0
    assert result.provider.maximum_physical_attempts_per_logical_call == 1


@pytest.mark.parametrize("malformed_text", (False, True))
def test_semantic_planner_failure_stops_after_one_successful_provider_call(
    malformed_text: bool,
) -> None:
    cell, simulation, phase2, benchmark, config, profile = _fixture_cell()
    provider = SemanticallyInvalidPlannerProvider(malformed_text=malformed_text)
    result = run_system(
        Phase4BSystem.CONSTRAINED_AGENTS,
        cell,
        simulation,
        phase2,
        benchmark,
        config,
        provider,
        profile,
    )
    assert provider.calls == 1
    assert result.output.abstained
    assert result.output.execution_failure is not None
    assert "semantic-validation-failed" in result.output.execution_failure
    assert result.output.planner_semantic_validation is not None
    assert result.provider.physical_attempts == 1
    assert result.provider.retry_count == 0
    assert result.provider.repair_attempts == 0
    assert all(call.status.startswith("rejected:") for call in result.tool_calls)
    assert all(call.evidence_items_gained == 0 for call in result.tool_calls)
    audit = provider.audit_records[0]
    assert audit["transport_status"] == "success"
    assert audit["schema_validation_status"] == "passed"
    assert audit["semantic_validation_status"] == ("schema-valid-semantically-invalid")
    assert "failure_kind" not in audit
    assert result.provider.provider_sdk_automatic_retries == 0
    assert result.provider.repository_retry_delay_seconds == 0.0
    assert result.provider.logical_model_calls == 1
    assert result.provider.physical_attempts == 1
    assert result.provider.retry_count == 0


def test_full_structured_provenance_and_hidden_diagnostic_masks() -> None:
    config = load_phase4b_config(ROOT / "configs/benchmark/phase4b-smoke.yaml")
    phase2 = load_phase2_config(ROOT / "configs")
    source = load_benchmark_config(ROOT / config.scenario_source)
    scenario = source.scenarios[0]
    full_condition = next(
        item for item in config.conditions if item.condition_id == "full-structured"
    )
    hidden_condition = next(
        item for item in config.conditions if item.condition_id == "squared-hidden"
    )
    full, _ = build_shared_cell(
        scenario=scenario,
        seed=config.return_seeds[0],
        condition=full_condition,
        phase2=phase2,
        config=config,
    )
    hidden, _ = build_shared_cell(
        scenario=scenario,
        seed=config.return_seeds[0],
        condition=hidden_condition,
        phase2=phase2,
        config=config,
    )
    provenance = next(
        item
        for item in full.initial_evidence
        if item.diagnostic_name == "research-provenance"
    )
    assert provenance.finding.value == "supports"
    assert all(
        item.diagnostic_name != "squared-return-dependence"
        for item in hidden.initial_evidence
    )
    assert any(
        item.diagnostic_name == "squared-return-dependence"
        for item in hidden.full_evidence
    )


def test_phase4a_adjudicator_controls_hidden_diagnostic_acquisition() -> None:
    phase2 = load_phase2_config(ROOT / "configs")
    returns = tuple([0.01, -0.005, 0.006, -0.002, 0.008, 0.001] * 40)
    all_evidence = run_diagnostics(returns, phase2.diagnostics)
    initial = mask_evidence(all_evidence, ("returns-summary",))
    adjudicator = DeterministicAdjudicator(
        BudgetLedger(
            AgentBudgets(
                maximum_diagnostic_requests=1,
                maximum_sensitivity_requests=0,
                maximum_review_iterations=1,
                maximum_model_calls=4,
                maximum_repeated_identical_request_count=1,
            )
        ),
        AuditTrail(),
        allow_unobserved_diagnostic_claims=True,
    )
    planner = PlannerOutput(
        requested_diagnostics=(
            DiagnosticRequest(
                diagnostic_id="squared-return-dependence",
                rationale="The hidden squared-return property is decision-relevant.",
                unresolved_claims=("absence_of_squared_return_dependence",),
                priority=1,
                expected_information_gain=InformationGain.HIGH,
            ),
        ),
        rationale="Acquire one registered property.",
        stop_recommendation=False,
    )
    visible, accepted, rejected = adjudicator.execute_planner_requests(
        planner,
        semantic_validation=validate_planner_output(
            planner,
            build_planner_semantic_context(
                ("squared-return-dependence",),
                allowed_evidence_references=("absence_of_squared_return_dependence",),
            ),
        ),
        returns=returns,
        config=phase2,
        provenance=None,
        evidence=initial,
    )
    assert accepted == ("squared-return-dependence:default",)
    assert not rejected
    assert any(item.diagnostic_name == "squared-return-dependence" for item in visible)
