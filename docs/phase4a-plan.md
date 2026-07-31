# Phase 4A plan: deadline-aware agentic hackathon layer

## Objective

Phase 4A adds a demonstrable agent layer without changing the audited
statistical core, eligibility rules, deterministic router, Phase 3B results, or
claims policy. It is a systems implementation, not a comparative agent study.

## Workflow

Returns and optional prose provenance flow through deterministic diagnostics
and typed evidence. A Diagnostic Planner Agent may request allowlisted
diagnostics. A Provenance Agent proposes only explicitly supported structured
fields. The deterministic router then establishes eligibility and an initial
route. A Skeptical Reviewer Agent may cite evidence and request eligible
sensitivity methods. Deterministic re-adjudication remains final. A Report
Agent renders only traceable typed state.

## Modes

- Mock mode is deterministic, offline, keyless, and required for tests.
- Live mode is optional and configured only through environment variables and
  an injected provider transport. Failure falls back safely to mock mode or
  stops explicitly before an unsafe agent output can be accepted.

## Scope boundaries

Agents do not calculate statistics, invent evidence, alter eligibility,
select methods by favorable outcomes, or suppress contradictions. LangGraph,
UI work, live-provider dependencies, and comparative agent evaluation are
deferred.

## Deliverables

The phase supplies four typed roles, explicit budgets, an allowlisted tool
registry, an audit log, three deterministic demos, CLI entry points, and seven
artifacts per demo run under ignored `runs/phase4a/` directories.

The audit log records ordered event IDs, typed input references, provider mode,
redacted provider responses, budget transitions, tool configurations,
acceptance or rejection, both routing stages, fallback status, and workflow
completion. It does not store hidden reasoning.
