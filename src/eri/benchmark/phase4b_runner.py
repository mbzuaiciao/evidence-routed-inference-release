# pyright: reportUnknownMemberType=false
"""Configuration-driven Phase 4B matched information benchmark."""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import yaml

from eri.agents.provider import ProviderError, TypedProvider
from eri.benchmark.phase4b_evaluation import (
    aggregate_metric_rows,
    paired_comparisons,
    permitted_claim_level,
    score_output,
)
from eri.benchmark.phase4b_schemas import (
    ClaimLevel,
    Phase4BConfig,
    Phase4BRunResult,
    ProviderMode,
)
from eri.benchmark.phase4b_summaries import result_claims, summary_markdown
from eri.benchmark.provenance_benchmark import (
    generate_provenance_case,
    score_provenance,
)
from eri.benchmark.runner import load_benchmark_config, load_loss_profile
from eri.benchmark.schemas import BenchmarkConfig
from eri.config import load_phase2_config, load_yaml_model
from eri.orchestration.phase4b import build_shared_cell, run_matched_cell

OUTPUT_FILES = (
    "resolved_config.yaml",
    "metadata.json",
    "seed_manifest.json",
    "cell_manifest.json",
    "provider_manifest.json",
    "ledger.jsonl",
    "system_outputs.jsonl",
    "tool_calls.jsonl",
    "provenance_scores.csv",
    "evidence_acquisition.csv",
    "metrics_by_regime.csv",
    "metrics_by_scenario.csv",
    "metrics_by_system.csv",
    "paired_comparisons.csv",
    "cost_summary.csv",
    "safety_summary.csv",
    "routing_confusion.csv",
    "summary.json",
    "summary.md",
    "result_claims.json",
)


def load_phase4b_config(path: Path) -> Phase4BConfig:
    return load_yaml_model(path, Phase4BConfig)


def run_phase4b(
    *,
    repo_root: Path,
    config_path: Path,
    output_directory: Path | None = None,
    run_id: str | None = None,
    confirm_live: bool = False,
    provider: TypedProvider | None = None,
) -> Phase4BRunResult:
    """Run Phase 4B; any live run requires an explicit confirmation flag."""
    root = repo_root.resolve()
    config = load_phase4b_config(config_path)
    if config.provider_mode is ProviderMode.LIVE and not confirm_live:
        raise PermissionError("live Phase 4B execution requires --confirm-live")
    if config.provider_mode is ProviderMode.LIVE and provider is None:
        raise ProviderError(
            "live Phase 4B execution requires an explicitly injected typed provider"
        )
    if config.provider_mode is ProviderMode.LIVE and provider is not None:
        provider_name = getattr(provider, "provider", None)
        provider_model = getattr(provider, "model", None)
        if isinstance(provider_name, str) and provider_name != config.model.provider:
            raise ProviderError("injected provider does not match benchmark provider")
        if isinstance(provider_model, str) and provider_model != config.model.model:
            resolution = cast(
                object, getattr(provider, "configuration_resolution", {})
            )
            resolution_mapping = (
                cast(Mapping[str, object], resolution)
                if isinstance(resolution, Mapping)
                else None
            )
            model_resolution: object = (
                resolution_mapping.get("model")
                if resolution_mapping is not None
                else None
            )
            model_mapping = (
                cast(Mapping[object, object], model_resolution)
                if isinstance(model_resolution, Mapping)
                else None
            )
            explicit_comparability_override = (
                model_mapping is not None
                and model_mapping.get("override_supplied") is True
                and model_mapping.get("changes_scientific_comparability") is True
            )
            if not explicit_comparability_override:
                raise ProviderError(
                    "injected provider model does not match benchmark model"
                )
    started_at = datetime.now(UTC)
    started = perf_counter()
    identifier = run_id or (
        started_at.strftime("%Y%m%dT%H%M%S%fZ") + f"-{config.tier.value}"
    )
    run_directory = (
        output_directory.resolve()
        if output_directory is not None
        else (root / config.output_root / identifier).resolve()
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    phase2 = load_phase2_config(root / "configs")
    if config.benchmark_sharpe != phase2.experiment.benchmark_sharpe:
        raise ValueError("Phase 4B benchmark Sharpe must match the Phase 2 experiment")
    source = load_benchmark_config(root / config.scenario_source)
    scenarios = tuple(
        scenario
        for scenario in source.scenarios
        if scenario.scenario_id in config.scenario_ids
    )
    if tuple(item.scenario_id for item in scenarios) != config.scenario_ids:
        raise ValueError(
            "scenario IDs must be ordered exactly as in the Phase 3 source"
        )
    benchmark = _execution_config(config, scenarios, source)
    profile = load_loss_profile(
        root / "configs/benchmark/loss-profiles.yaml", config.loss_profile
    )

    cells: list[dict[str, Any]] = []
    system_outputs: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    providers: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in config.return_seeds:
            for regime in config.regimes:
                matching = tuple(
                    item for item in config.conditions if item.regime is regime
                )
                condition = matching[
                    (seed + sum(scenario.scenario_id.encode())) % len(matching)
                ]
                cell, simulation = build_shared_cell(
                    scenario=scenario,
                    seed=seed,
                    condition=condition,
                    phase2=phase2,
                    config=config,
                )
                cells.append(
                    {
                        "cell_id": cell.cell_id,
                        "scenario_id": scenario.scenario_id,
                        "return_seed": seed,
                        "regime": cell.regime.value,
                        "condition_id": cell.condition_id,
                        "fingerprints": cell.fingerprints,
                        "diagnostic_budget": cell.diagnostic_budget,
                        "sensitivity_budget": cell.sensitivity_budget,
                        "critical_groups": cell.critical_groups,
                        "input_frequency": cell.input_frequency,
                        "periods_per_year": cell.periods_per_year,
                        "benchmark_sharpe": cell.benchmark_sharpe,
                        "method_metadata_version": config.method_metadata_version,
                    }
                )
                runs = run_matched_cell(
                    cell,
                    simulation,
                    phase2=phase2,
                    benchmark=benchmark,
                    config=config,
                    profile=profile,
                    provider=provider,
                )
                cell_rows: list[dict[str, float | int | str]] = []
                for result in runs:
                    output = result.output.model_dump(mode="json")
                    calls = [item.model_dump(mode="json") for item in result.tool_calls]
                    system_outputs.append(output)
                    tool_calls.extend(calls)
                    providers.append(result.provider.model_dump(mode="json"))
                    row = score_output(
                        cell,
                        result.output,
                        result.tool_calls,
                        simulation,
                        profile,
                        confidence_level=float(config.confidence_level),
                    )
                    cell_rows.append(row)
                    case = generate_provenance_case(cell.provenance_case_id)
                    provenance_rows.append(
                        {
                            "cell_id": cell.cell_id,
                            "system": result.output.system.value,
                            **score_provenance(
                                dict(result.output.provenance_fields),
                                dict(result.output.provenance_source_spans),
                                case,
                                result.output.unresolved_provenance_fields,
                            ),
                        }
                    )
                oracle_loss = next(
                    float(row["routing_loss"])
                    for row in cell_rows
                    if row["system"] == "oracle"
                )
                oracle_action = next(
                    str(row["selected_action"])
                    for row in cell_rows
                    if row["system"] == "oracle"
                )
                for row in cell_rows:
                    row["routing_regret"] = float(row["routing_loss"]) - oracle_loss
                    row["oracle_agreement"] = int(
                        str(row["selected_action"]) == oracle_action
                    )
                ledger.extend(cell_rows)

    agent_names = {
        "monolithic-agent",
        "unconstrained-specialized-agents",
        "evidence-constrained-phase4a-agents",
    }
    live_agent_providers = [
        row for row in providers if str(row["system"]) in agent_names
    ]
    unreviewed_scientific_override = _has_comparability_override(
        live_agent_providers
    )
    claim_level = permitted_claim_level(
        config,
        live_outputs_complete=all(
            row.get("failure") is None for row in live_agent_providers
        ),
        provider_metadata_complete=(
            config.provider_mode is ProviderMode.MOCK
            or (
                config.model.model_version is not None
                and all(row.get("model_version") for row in live_agent_providers)
            )
        ),
        unreviewed_scientific_override=unreviewed_scientific_override,
    )
    live_used = config.provider_mode is ProviderMode.LIVE
    summary: dict[str, Any] = {
        "run_id": identifier,
        "tier": config.tier.value,
        "claim_level": claim_level.value,
        "cell_count": len(cells),
        "output_count": len(system_outputs),
        "live_model_used": live_used,
        "unreviewed_scientific_override": unreviewed_scientific_override,
        "primary_outcomes": [
            "invalid_primary_method_selection",
            "harmful_nonabstention",
            "routing_regret",
        ],
        "scientific_interpretation_permitted": (
            claim_level.value in {"benchmark-supported", "confirmation-supported"}
        ),
    }
    _write_outputs(
        run_directory,
        config,
        started_at,
        started,
        cells,
        providers,
        ledger,
        system_outputs,
        tool_calls,
        provenance_rows,
        summary,
        claim_level,
    )
    return Phase4BRunResult(
        run_id=identifier,
        run_directory=run_directory,
        cell_count=len(cells),
        output_count=len(system_outputs),
        claim_level=claim_level,
        output_files=OUTPUT_FILES,
    )


def _has_comparability_override(
    providers: Sequence[Mapping[str, Any]],
) -> bool:
    for provider in providers:
        resolution = provider.get("configuration_resolution")
        if not isinstance(resolution, Mapping):
            continue
        for value in cast(Mapping[str, object], resolution).values():
            value_mapping = (
                cast(Mapping[str, object], value)
                if isinstance(value, Mapping)
                else None
            )
            if (
                value_mapping is not None
                and value_mapping.get("override_supplied") is True
                and value_mapping.get("changes_scientific_comparability") is True
            ):
                return True
    return False


def _execution_config(
    config: Phase4BConfig,
    scenarios: tuple[Any, ...],
    source: BenchmarkConfig,
) -> BenchmarkConfig:
    return source.model_copy(
        update={
            "benchmark_version": config.benchmark_version,
            "run_label": config.run_label,
            "output_root": config.output_root,
            "confidence_level": config.confidence_level,
            "hac_bandwidth": config.hac_bandwidth,
            "bootstrap_block_length": config.bootstrap_block_length,
            "bootstrap_replications": config.bootstrap_replications,
            "bootstrap_seed": config.bootstrap_seed,
            "loss_profile": config.loss_profile,
            "oracle_training_seeds": config.return_seeds,
            "evaluation_seeds": config.return_seeds,
            "scenarios": scenarios,
            "generate_figures": False,
        }
    )


def _write_outputs(
    directory: Path,
    config: Phase4BConfig,
    started_at: datetime,
    started: float,
    cells: Sequence[Mapping[str, Any]],
    providers: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    claim_level: ClaimLevel,
) -> None:
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    metadata = {
        "run_id": summary["run_id"],
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": perf_counter() - started,
        "python_version": sys.version,
        "platform": platform.platform(),
        "network_used": config.provider_mode is ProviderMode.LIVE,
        "mock_outputs_are_scientific_results": False,
    }
    _json(directory / "metadata.json", metadata)
    _json(
        directory / "seed_manifest.json",
        {"namespace": config.seed_namespace, "return_seeds": config.return_seeds},
    )
    _json(directory / "cell_manifest.json", list(cells))
    _json(directory / "provider_manifest.json", list(providers))
    _jsonl(directory / "ledger.jsonl", ledger)
    _jsonl(directory / "system_outputs.jsonl", outputs)
    _jsonl(directory / "tool_calls.jsonl", calls)
    _csv(directory / "provenance_scores.csv", provenance)
    _csv(directory / "evidence_acquisition.csv", _evidence_rows(ledger))
    _csv(
        directory / "metrics_by_regime.csv",
        aggregate_metric_rows(ledger, group_keys=("system", "regime")),
    )
    _csv(
        directory / "metrics_by_scenario.csv",
        aggregate_metric_rows(ledger, group_keys=("system", "scenario_id")),
    )
    _csv(
        directory / "metrics_by_system.csv",
        aggregate_metric_rows(ledger, group_keys=("system",)),
    )
    _csv(directory / "paired_comparisons.csv", paired_comparisons(ledger))
    _csv(directory / "cost_summary.csv", _cost_rows(providers, calls, config))
    _csv(directory / "safety_summary.csv", _safety_rows(ledger))
    _csv(directory / "routing_confusion.csv", _routing_rows(outputs))
    _json(directory / "summary.json", dict(summary))
    (directory / "summary.md").write_text(
        summary_markdown(dict(summary)), encoding="utf-8"
    )
    _json(
        directory / "result_claims.json",
        list(
            result_claims(
                claim_level=claim_level,
                live_model_used=config.provider_mode is ProviderMode.LIVE,
            )
        ),
    )
    actual = {item.name for item in directory.iterdir()}
    if actual != set(OUTPUT_FILES):
        raise RuntimeError(f"Phase 4B output contract mismatch: {sorted(actual)}")


def _evidence_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "cell_id",
        "system",
        "diagnostic_calls",
        "relevant_diagnostic_requests",
        "irrelevant_diagnostic_requests",
        "duplicate_requests",
        "relevant_diagnostic_request_rate",
        "irrelevant_diagnostic_request_rate",
        "duplicate_request_rate",
        "diagnostic_cost",
        "budget_efficiency",
        "evidence_gained",
        "evidence_gained_per_call",
        "critical_discovery_fraction",
        "missed_critical_diagnostic",
    )
    return [{key: row[key] for key in keys} for row in rows]


def _cost_rows(
    providers: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    config: Phase4BConfig,
) -> list[dict[str, Any]]:
    by_system: dict[str, dict[str, float]] = {}
    for provider in providers:
        system = str(provider["system"])
        row = by_system.setdefault(
            system,
            {
                "model_calls": 0.0,
                "latency_seconds": 0.0,
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "usage_records": 0.0,
            },
        )
        row["model_calls"] += float(provider["model_calls"])
        row["latency_seconds"] += float(provider["latency_seconds"])
        if provider.get("input_tokens") is not None:
            row["input_tokens"] += float(provider["input_tokens"])
        if provider.get("output_tokens") is not None:
            row["output_tokens"] += float(provider["output_tokens"])
        row["usage_records"] += float(bool(provider.get("usage_reported")))
    diagnostic_counts = Counter(str(call["system"]) for call in calls)
    diagnostic_costs: dict[str, float] = {}
    for call in calls:
        system = str(call["system"])
        diagnostic_costs[system] = diagnostic_costs.get(system, 0.0) + float(
            call["cost"]
        )
    return [
        {
            "system": system,
            **values,
            "diagnostic_calls": diagnostic_counts[system],
            "diagnostic_cost": diagnostic_costs.get(system, 0.0),
            "input_tokens": (
                int(values["input_tokens"])
                if values["usage_records"]
                else "not-reported"
            ),
            "output_tokens": (
                int(values["output_tokens"])
                if values["usage_records"]
                else "not-reported"
            ),
            "monetary_cost": (
                (
                    values["input_tokens"]
                    * float(config.model.input_cost_per_million_tokens)
                    + values["output_tokens"]
                    * float(config.model.output_cost_per_million_tokens)
                )
                / 1_000_000
                if values["usage_records"]
                else (
                    0.0 if config.provider_mode is ProviderMode.MOCK else "not-reported"
                )
            ),
        }
        for system, values in sorted(by_system.items())
    ]


def _safety_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "cell_id",
        "system",
        "eligibility_override_attempts",
        "unsupported_claims",
        "unknown_tool_requests",
        "untraceable_report_claims",
        "guardrail_rejections",
        "audit_complete",
        "final_claim_valid",
    )
    return [{key: row[key] for key in keys} for row in rows]


def _routing_rows(outputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (str(row["system"]), str(row.get("selected_method") or "abstention"))
        for row in outputs
    )
    return [
        {"system": system, "selected_action": action, "count": count}
        for (system, action), count in sorted(counts.items())
    ]


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = tuple(dict.fromkeys(key for row in rows_list for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)
