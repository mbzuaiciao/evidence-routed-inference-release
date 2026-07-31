"""End-to-end mock tests for the deadline-aware agentic layer."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from eri.agents.mock import MockRawProvider, SequenceRawProvider
from eri.agents.provider import ValidatingProvider
from eri.agents.schemas import AgentRole, ProviderMessage
from eri.config import load_phase2_config
from eri.demo.examples import get_demo_example
from eri.demo.render import write_demo_artifacts
from eri.orchestration.budgets import load_agent_config
from eri.orchestration.state import AgenticWorkflowState
from eri.orchestration.workflow import run_agentic_workflow

ROOT = Path(__file__).resolve().parents[2]


class ReviewerOverrideRawProvider:
    """Use normal mock behavior except for one adversarial review response."""

    def __init__(self, review: object) -> None:
        self._review = review
        self._mock = MockRawProvider()

    def generate_raw(
        self,
        *,
        role: AgentRole,
        messages: Sequence[ProviderMessage],
        schema_name: str,
        repair_error: str | None,
    ) -> object:
        if role is AgentRole.SKEPTICAL_REVIEWER:
            return self._review
        return self._mock.generate_raw(
            role=role,
            messages=messages,
            schema_name=schema_name,
            repair_error=repair_error,
        )


class ReportOverrideRawProvider:
    """Inject an untraceable report while preserving all earlier mock roles."""

    def __init__(self) -> None:
        self._mock = MockRawProvider()

    def generate_raw(
        self,
        *,
        role: AgentRole,
        messages: Sequence[ProviderMessage],
        schema_name: str,
        repair_error: str | None,
    ) -> object:
        raw = self._mock.generate_raw(
            role=role,
            messages=messages,
            schema_name=schema_name,
            repair_error=repair_error,
        )
        if role is not AgentRole.REPORT:
            return raw
        report = cast(dict[str, object], raw)
        return {**report, "evidence_references": [], "estimate": "Invented 999."}


def _run(example_id: str) -> AgenticWorkflowState:
    example = get_demo_example(example_id)
    return run_agentic_workflow(
        example.returns,
        load_phase2_config(ROOT / "configs"),
        load_agent_config(ROOT / "configs/agents/phase4a-mock.yaml"),
        provenance_text=example.provenance_text,
        example_id=example_id,
    )


def test_mock_iid_demo_uses_deterministic_primary_and_sensitivities() -> None:
    state = _run("iid")
    assert state.final_decision.selected_method == "hac-sharpe"
    assert state.inference_results
    assert state.planner_output.stop_recommendation


def test_mock_garch_demo_surfaces_squared_dependence() -> None:
    state = _run("garch")
    squared = next(
        item
        for item in state.evidence
        if item.claim == "absence_of_squared_return_dependence"
    )
    eligibility = {item.method_id: item for item in state.final_decision.eligibility}
    assert squared.finding.value in {"contradicts", "contradiction"}
    assert not eligibility["iid-gaussian-sharpe"].eligible
    assert "circular-block-bootstrap" in state.accepted_agent_requests


def test_mock_structural_break_demo_abstains() -> None:
    state = _run("break")
    assert state.final_decision.selected_method is None
    assert state.final_decision.abstention is not None
    assert not state.inference_results
    assert "Deterministic abstention" in state.report.abstention_or_qualifications


def test_incomplete_provenance_demo_does_not_invent_counts() -> None:
    state = _run("provenance")
    assert state.provenance_output is not None
    assert state.provenance_output.proposed_provenance.trials is None
    assert "trials" in state.provenance_output.unresolved_fields


def test_agent_proposal_cannot_override_eligibility() -> None:
    example = get_demo_example("break")
    provider = ValidatingProvider(
        ReviewerOverrideRawProvider(
            {
                "requested_sensitivity_methods": ["iid-gaussian-sharpe"],
                "recommendation": "run-sensitivity-methods",
                "rationale": "Adversarial request for an ineligible method.",
            }
        )
    )
    state = run_agentic_workflow(
        example.returns,
        load_phase2_config(ROOT / "configs"),
        load_agent_config(ROOT / "configs/agents/phase4a-mock.yaml"),
        example_id="override-attempt",
        provider=provider,
    )
    assert state.final_decision.selected_method is None
    assert state.final_decision.abstention is not None
    assert any(
        item.reason == "agent cannot override deterministic ineligibility"
        for item in state.rejected_proposals
    )


def test_rejected_proposals_appear_in_audit_log() -> None:
    example = get_demo_example("iid")
    provider = ValidatingProvider(
        ReviewerOverrideRawProvider(
            {
                "supported_challenges": [
                    {
                        "challenge_id": "invented-evidence",
                        "statement": "An unsupported challenge.",
                        "evidence_references": ["evidence:not-real"],
                    }
                ],
                "recommendation": "accept",
                "rationale": "Attempt to cite evidence outside typed state.",
            }
        )
    )
    state = run_agentic_workflow(
        example.returns,
        load_phase2_config(ROOT / "configs"),
        load_agent_config(ROOT / "configs/agents/phase4a-mock.yaml"),
        example_id="rejected-proposal",
        provider=provider,
    )
    assert any(
        item.proposal_id == "invented-evidence" for item in state.rejected_proposals
    )
    assert any(item.event_type == "proposal-rejected" for item in state.audit_events)
    assert any(
        item.event_type == "final-deterministic-decision" for item in state.audit_events
    )


def test_provider_failure_falls_back_safely() -> None:
    example = get_demo_example("iid")
    state = run_agentic_workflow(
        example.returns,
        load_phase2_config(ROOT / "configs"),
        load_agent_config(ROOT / "configs/agents/phase4a-live.yaml"),
        example_id="provider-failure",
        provider=ValidatingProvider(SequenceRawProvider([{}, {}])),
    )
    assert state.final_decision.selected_method is not None
    assert state.fallback_used
    assert any(
        item.event_type == "agent-output-rejected" for item in state.audit_events
    )
    assert any(item.event_type == "fallback-accepted" for item in state.audit_events)


def test_report_cannot_suppress_evidence_or_invent_a_statistic() -> None:
    example = get_demo_example("iid")
    state = run_agentic_workflow(
        example.returns,
        load_phase2_config(ROOT / "configs"),
        load_agent_config(ROOT / "configs/agents/phase4a-mock.yaml"),
        provenance_text=example.provenance_text,
        example_id="report-override",
        provider=ValidatingProvider(ReportOverrideRawProvider()),
    )
    assert state.fallback_used
    assert set(state.report.evidence_references) == {
        item.evidence_id for item in state.evidence
    }
    assert "999" not in state.report.estimate
    assert any(
        item.event_type == "agent-output-rejected" and item.actor == "report"
        for item in state.audit_events
    )


def test_identical_mock_input_produces_identical_state() -> None:
    assert _run("iid").model_dump() == _run("iid").model_dump()


def test_demo_artifacts_are_complete_and_outside_obsidian(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    written = write_demo_artifacts(_run("iid"), directory)
    assert {path.name for path in written} == {
        "final_report.md",
        "final_state.json",
        "evidence.json",
        "audit_log.jsonl",
        "agent_requests.json",
        "deterministic_decision.json",
        "method_results.json",
    }
    assert all("inference-obsidian" not in path.as_posix() for path in written)
    final_state = json.loads((directory / "final_state.json").read_text("utf-8"))
    evidence = json.loads((directory / "evidence.json").read_text("utf-8"))
    decision = json.loads(
        (directory / "deterministic_decision.json").read_text("utf-8")
    )
    results = json.loads((directory / "method_results.json").read_text("utf-8"))
    audit = [
        json.loads(line)
        for line in (directory / "audit_log.jsonl").read_text("utf-8").splitlines()
    ]
    assert evidence == final_state["evidence"]
    assert decision == final_state["final_decision"]
    assert results == final_state["inference_results"]
    assert audit == final_state["audit_events"]
    report = (directory / "final_report.md").read_text("utf-8")
    assert "Agent mode: **mock**" in report
    assert "## Agent requests and adjudication" in report
    assert "## Deterministic route" in report


def test_audit_log_is_ordered_budgeted_and_replayable() -> None:
    state = _run("garch")
    events = state.audit_events
    assert [item.sequence for item in events] == list(range(1, len(events) + 1))
    assert [item.event_id for item in events] == [
        f"audit:{index:04d}" for index in range(1, len(events) + 1)
    ]
    assert events[0].event_type == "workflow-started"
    assert events[-1].event_type == "workflow-completed"
    assert events[-1].payload["budget_usage"] == state.budget_usage
    assert events[-1].payload["fallback_used"] is state.fallback_used
    assert any(item.event_type == "diagnostic-executed" for item in events)
    assert any(item.event_type == "sensitivity-accepted" for item in events)
    for event in events:
        if event.event_type not in {"budget-consumed", "budget-rejected"}:
            continue
        before = cast(dict[str, object], event.payload["budget_before"])
        after = cast(dict[str, object], event.payload["budget_after"])
        for key in (
            "diagnostic_requests",
            "sensitivity_requests",
            "review_iterations",
            "model_calls",
        ):
            before_value = before[key]
            after_value = after[key]
            assert isinstance(before_value, int)
            assert isinstance(after_value, int)
            assert after_value >= before_value


def test_mock_execution_terminates_within_all_budgets() -> None:
    config = load_agent_config(ROOT / "configs/agents/phase4a-mock.yaml")
    for example_id in ("iid", "garch", "break", "provenance"):
        state = _run(example_id)
        model_calls = state.budget_usage["model_calls"]
        review_iterations = state.budget_usage["review_iterations"]
        assert isinstance(model_calls, int)
        assert isinstance(review_iterations, int)
        assert model_calls <= config.budgets.maximum_model_calls
        assert review_iterations <= config.budgets.maximum_review_iterations
        assert state.audit_events[-1].event_type == "workflow-completed"


def test_generated_phase4a_runs_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text("utf-8").splitlines()
    assert "runs/" in ignore


def test_personal_obsidian_paths_are_not_workflow_inputs() -> None:
    fields = set(AgenticWorkflowState.model_fields)
    assert "personal_notes" not in fields
    assert "obsidian_workspace" not in fields
