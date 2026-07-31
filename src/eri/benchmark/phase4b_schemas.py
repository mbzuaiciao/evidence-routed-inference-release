"""Strict contracts for the Phase 4B information-availability benchmark."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import Field, FiniteFloat, JsonValue, model_validator

from eri.benchmark.schemas import BenchmarkModel, EligibilityLabel, ScenarioConfig
from eri.schemas.evidence import EvidenceItem


class Phase4BSystem(StrEnum):
    STATIC_DETERMINISTIC = "static-deterministic-router"
    ADAPTIVE_DETERMINISTIC = "adaptive-deterministic-policy"
    MONOLITHIC_AGENT = "monolithic-agent"
    UNCONSTRAINED_AGENTS = "unconstrained-specialized-agents"
    CONSTRAINED_AGENTS = "evidence-constrained-phase4a-agents"
    ORACLE = "oracle"


class InformationRegime(StrEnum):
    FULL = "full-information"
    BUDGETED = "budgeted-observation"
    PROVENANCE = "open-world-provenance"


class BenchmarkTier(StrEnum):
    SMOKE = "smoke"
    LIVE_PILOT = "live-pilot"
    DEADLINE = "deadline"
    PAPER = "paper"


class ClaimLevel(StrEnum):
    IMPLEMENTATION_ONLY = "implementation-only"
    PILOT_DESCRIPTIVE = "pilot-descriptive"
    BENCHMARK_SUPPORTED = "benchmark-supported"
    CONFIRMATION_SUPPORTED = "confirmation-supported"


class ProviderMode(StrEnum):
    MOCK = "mock"
    LIVE = "live"


class EvidenceConditionConfig(BenchmarkModel):
    condition_id: str = Field(min_length=1)
    regime: InformationRegime
    initial_groups: tuple[str, ...]
    diagnostic_budget: int = Field(ge=0)
    sensitivity_budget: int = Field(ge=0)
    diagnostic_costs: dict[str, FiniteFloat]
    critical_groups: tuple[str, ...] = ()
    provenance_template: str = Field(min_length=1)
    ambiguity: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_groups(self) -> Self:
        if len(self.initial_groups) != len(set(self.initial_groups)):
            raise ValueError("initial evidence groups must be unique")
        if any(value < 0 for value in self.diagnostic_costs.values()):
            raise ValueError("diagnostic costs must be nonnegative")
        if len(self.critical_groups) != len(set(self.critical_groups)):
            raise ValueError("critical evidence groups must be unique")
        if set(self.critical_groups) & set(self.initial_groups):
            raise ValueError("critical evidence groups must initially be hidden")
        return self


class ModelFairnessConfig(BenchmarkModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_version: str | None = Field(default=None, min_length=1)
    temperature: FiniteFloat | None = Field(default=None, ge=0.0, le=2.0)
    maximum_response_tokens: int = Field(ge=1)
    maximum_input_tokens: int = Field(ge=1)
    maximum_model_calls: int = Field(ge=1)
    retries: int = Field(ge=0, le=2)
    timeout_seconds: FiniteFloat = Field(gt=0.0)
    input_cost_per_million_tokens: FiniteFloat = Field(ge=0.0)
    output_cost_per_million_tokens: FiniteFloat = Field(ge=0.0)
    expected_input_token_fraction: FiniteFloat = Field(gt=0.0, le=1.0)
    expected_output_token_fraction: FiniteFloat = Field(gt=0.0, le=1.0)
    expected_retry_rate: FiniteFloat = Field(ge=0.0, le=1.0)


class Phase4BConfig(BenchmarkModel):
    benchmark_version: str = Field(min_length=1)
    tier: BenchmarkTier
    run_label: str = Field(min_length=1)
    output_root: Path = Path("runs/phase4b")
    scenario_source: Path
    scenario_ids: tuple[str, ...] = Field(min_length=1)
    return_seeds: tuple[int, ...] = Field(min_length=1)
    seed_namespace: str = Field(min_length=1)
    systems: tuple[Phase4BSystem, ...]
    regimes: tuple[InformationRegime, ...]
    conditions: tuple[EvidenceConditionConfig, ...] = Field(min_length=1)
    provider_mode: ProviderMode
    model: ModelFairnessConfig
    allowed_tools: tuple[str, ...] = Field(min_length=1)
    input_frequency: str = Field(min_length=1)
    periods_per_year: FiniteFloat | None = Field(default=None, gt=0.0)
    benchmark_sharpe: FiniteFloat
    method_metadata_version: str = Field(min_length=1)
    confidence_level: FiniteFloat = Field(gt=0.0, lt=1.0)
    hac_bandwidth: int = Field(ge=0)
    bootstrap_block_length: int = Field(ge=1)
    bootstrap_replications: int = Field(ge=2)
    bootstrap_seed: int
    loss_profile: str = Field(min_length=1)
    independent_review_complete: bool = False
    confirmation_run: bool = False
    confirmation_freeze_hash: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_preregistered_design(self) -> Self:
        if set(self.systems) != set(Phase4BSystem) or len(self.systems) != 6:
            raise ValueError("Phase 4B requires exactly the six registered systems")
        if set(self.regimes) != set(InformationRegime) or len(self.regimes) != 3:
            raise ValueError("Phase 4B requires exactly regimes A, B, and C")
        if len(self.return_seeds) != len(set(self.return_seeds)):
            raise ValueError("return seeds must be unique")
        condition_ids = [item.condition_id for item in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition IDs must be unique")
        if {item.regime for item in self.conditions} != set(self.regimes):
            raise ValueError("every configured regime needs an evidence condition")
        if (
            self.provider_mode is ProviderMode.MOCK
            and self.tier is not BenchmarkTier.SMOKE
        ):
            raise ValueError("mock providers are restricted to the smoke tier")
        if self.provider_mode is ProviderMode.LIVE and self.model.provider == "mock":
            raise ValueError("live tiers require an explicit non-mock provider")
        if self.model.maximum_model_calls < 4:
            raise ValueError("matched agent systems require a four-call budget")
        if self.confirmation_run and (
            self.tier is not BenchmarkTier.PAPER
            or not self.independent_review_complete
            or self.model.model_version is None
            or self.confirmation_freeze_hash is None
        ):
            raise ValueError(
                "confirmation runs require paper tier, review, pinned model version, "
                "and a preregistered freeze hash"
            )
        return self


class ProvenanceTruth(BenchmarkModel):
    trials: int | None = Field(default=None, ge=1)
    parameter_searches: int | None = Field(default=None, ge=0)
    markets_or_universes: tuple[str, ...] = ()
    start_dates_or_windows: tuple[str, ...] = ()
    failed_trials_retained: bool | None = None
    trial_dependence_known: bool | None = None
    completeness: str = Field(min_length=1)
    contradictions: tuple[str, ...] = ()
    unresolved_fields: tuple[str, ...] = ()


class ProvenanceCase(BenchmarkModel):
    case_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    truth: ProvenanceTruth
    source_spans: dict[str, str]
    template_version: str = Field(min_length=1)


class SharedCell(BenchmarkModel):
    cell_id: str = Field(min_length=1)
    scenario: ScenarioConfig
    return_seed: int
    regime: InformationRegime
    condition_id: str = Field(min_length=1)
    returns: tuple[FiniteFloat, ...] = Field(min_length=2)
    full_evidence: tuple[EvidenceItem, ...]
    initial_evidence: tuple[EvidenceItem, ...]
    provenance_text: str
    provenance_case_id: str
    allowed_tools: tuple[str, ...]
    diagnostic_budget: int = Field(ge=0)
    sensitivity_budget: int = Field(ge=0)
    diagnostic_costs: dict[str, FiniteFloat]
    critical_groups: tuple[str, ...]
    input_frequency: str
    periods_per_year: FiniteFloat | None = None
    benchmark_sharpe: FiniteFloat
    method_metadata: dict[str, JsonValue]
    fingerprints: dict[str, str]
    truth_labels: tuple[EligibilityLabel, ...]
    truth: dict[str, JsonValue]


class ToolCallRecord(BenchmarkModel):
    cell_id: str
    system: Phase4BSystem
    tool_id: str
    status: str
    cost: FiniteFloat = Field(ge=0.0)
    relevant: bool
    duplicate: bool
    evidence_items_gained: int = Field(ge=0)
    critical_properties_discovered: int = Field(ge=0)


class ProviderRecord(BenchmarkModel):
    cell_id: str
    system: Phase4BSystem
    provider: str
    model: str
    model_version: str | None = None
    requested_sampling_seed: int | None = None
    sampling_seed: int | None = None
    temperature: FiniteFloat | None = None
    maximum_response_tokens: int
    maximum_input_tokens: int
    maximum_model_calls: int = Field(ge=1)
    retry_policy: str = Field(min_length=1)
    repository_max_retries: int = Field(default=0, ge=0)
    maximum_physical_attempts_per_logical_call: int = Field(default=1, ge=1)
    provider_sdk_automatic_retries: int = Field(default=0, ge=0)
    repository_retry_delay_seconds: FiniteFloat = Field(default=0.0, ge=0.0)
    diagnostic_budget: int = Field(ge=0)
    sensitivity_budget: int = Field(ge=0)
    allowed_tools_fingerprint: str = Field(min_length=1)
    model_calls: int = Field(ge=0)
    logical_model_calls: int = Field(default=0, ge=0)
    physical_attempts: int = Field(default=0, ge=0)
    http_requests_dispatched: int = Field(default=0, ge=0)
    provider_responses_received: int = Field(default=0, ge=0)
    pre_dispatch_failures: int = Field(default=0, ge=0)
    transport_failures_after_dispatch: int = Field(default=0, ge=0)
    structured_output_compatibility_instructions: tuple[str, ...] = ()
    structured_output_compatibility_instruction_count: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_seconds: FiniteFloat = Field(ge=0.0)
    retry_count: int = Field(default=0, ge=0)
    repair_attempts: int = Field(ge=0)
    provider_request_ids: tuple[str, ...] = ()
    provider_failure_kinds: tuple[str, ...] = ()
    request_fingerprints: tuple[str, ...] = ()
    response_schema_fingerprints: tuple[str, ...] = ()
    provider_reported_models: tuple[str, ...] = ()
    finish_reasons: tuple[str, ...] = ()
    safety_statuses: tuple[str, ...] = ()
    reproducibility_status: str | None = None
    provider_usage_extra: tuple[dict[str, JsonValue], ...] = ()
    configuration_resolution: dict[str, JsonValue] = Field(default_factory=dict)
    failure: str | None = None
    usage_reported: bool = False
    mode: ProviderMode


class SystemOutput(BenchmarkModel):
    cell_id: str
    system: Phase4BSystem
    scientific_agent_reasoning: bool = False
    selected_method: str | None = None
    abstained: bool
    recommended_method: str | None = None
    recommended_method_eligible: bool | None = None
    diagnostic_requests: tuple[str, ...] = ()
    acquired_evidence_ids: tuple[str, ...] = ()
    visible_evidence_ids: tuple[str, ...] = ()
    provenance_fields: dict[str, JsonValue] = Field(default_factory=dict)
    provenance_source_spans: dict[str, str] = Field(default_factory=dict)
    unresolved_provenance_fields: tuple[str, ...] = ()
    final_claims: tuple[str, ...] = ()
    claim_evidence_ids: tuple[str, ...] = ()
    execution_failure: str | None = None
    planner_semantic_validation: dict[str, JsonValue] | None = None
    guardrail_rejections: tuple[str, ...] = ()
    unknown_tool_requests: tuple[str, ...] = ()
    audit_events: tuple[str, ...] = ()
    interval: tuple[FiniteFloat, FiniteFloat] | None = None
    estimate: FiniteFloat | None = None
    input_fingerprints: dict[str, str]


class Phase4BRunResult(BenchmarkModel):
    run_id: str
    run_directory: Path
    cell_count: int = Field(ge=1)
    output_count: int = Field(ge=1)
    claim_level: ClaimLevel
    output_files: tuple[str, ...]
