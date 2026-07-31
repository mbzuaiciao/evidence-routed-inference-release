# pyright: reportUnknownMemberType=false
"""Phase 3B benchmark validation runner and output generation."""

from __future__ import annotations

import csv
import json
import platform
import resource
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, cast

import yaml

from eri.benchmark.confirmation import validate_freeze_manifest
from eri.benchmark.evaluation import metric_row
from eri.benchmark.ledger import read_ledger
from eri.benchmark.monte_carlo import (
    normal_mean_interval,
    paired_comparison_rows,
    scenario_clustered_values,
    wilson_interval,
)
from eri.benchmark.oracle_stability import (
    final_stability_summary,
    oracle_agreement_by_stability,
    oracle_stability_rows,
)
from eri.benchmark.phase3b_schemas import (
    ClaimLevel,
    ConfirmationValidation,
    Phase3BConfig,
    Phase3BRunResult,
    SeedManifest,
)
from eri.benchmark.registry import eligibility_for, generate_scenario
from eri.benchmark.runner import (
    load_benchmark_config,
    load_loss_profile,
    run_benchmark,
)
from eri.benchmark.schemas import (
    BaselineName,
    BenchmarkConfig,
    LedgerRecord,
    OracleChoice,
    ScenarioConfig,
)
from eri.config import load_yaml_model

PHASE3A_SCENARIO_IDS = (
    "iid-gaussian-n120",
    "iid-gaussian-n300",
    "ar1-moderate",
    "ar1-strong",
    "garch-moderate",
    "garch-persistent",
    "structural-mean-break",
    "structural-variance-break",
)

_BINOMIAL_METRICS = {
    "valid_method_selection_rate": (
        "valid_selection_numerator",
        "valid_selection_denominator",
    ),
    "invalid_method_selection_rate": (
        "invalid_selection_numerator",
        "invalid_selection_denominator",
    ),
    "abstention_rate": ("abstention_numerator", "abstention_denominator"),
    "harmful_nonabstention_rate": (
        "harmful_nonabstention_numerator",
        "harmful_nonabstention_denominator",
    ),
    "unnecessary_abstention_rate": (
        "unnecessary_abstention_numerator",
        "unnecessary_abstention_denominator",
    ),
    "selective_coverage": ("coverage_numerator", "coverage_denominator"),
    "oracle_agreement_rate": (
        "oracle_agreement_numerator",
        "oracle_agreement_denominator",
    ),
    "execution_failure_rate": (
        "execution_failure_numerator",
        "execution_failure_denominator",
    ),
}


def load_phase3b_config(path: Path) -> Phase3BConfig:
    return load_yaml_model(path, Phase3BConfig)


def build_seed_manifest(
    config: Phase3BConfig, *, confirmation: bool = False
) -> SeedManifest:
    """Resolve every versioned namespace without clock-derived scientific seeds."""
    namespaces = config.seed_namespaces
    maximum = config.oracle_stability_sizes[-1]
    blocks = tuple(
        tuple(
            range(
                namespaces.oracle_training_start + block * maximum,
                namespaces.oracle_training_start + (block + 1) * maximum,
            )
        )
        for block in range(config.oracle_stability_blocks)
    )
    development = tuple(
        range(
            namespaces.development_evaluation_start,
            namespaces.development_evaluation_start
            + namespaces.development_evaluation_count,
        )
    )
    hackathon = tuple(
        range(
            namespaces.hackathon_evaluation_start,
            namespaces.hackathon_evaluation_start
            + namespaces.hackathon_evaluation_count,
        )
    )
    aaai = tuple(
        range(
            namespaces.aaai_evaluation_start,
            namespaces.aaai_evaluation_start + namespaces.aaai_evaluation_count,
        )
    )
    frozen = tuple(
        range(
            namespaces.frozen_confirmation_start,
            namespaces.frozen_confirmation_start + namespaces.frozen_confirmation_count,
        )
    )
    evaluation_by_name = {
        "development-evaluation": development,
        "hackathon-evaluation": hackathon,
        "aaai-evaluation": aaai,
    }
    active_name = "frozen-confirmation" if confirmation else config.evaluation_namespace
    active = frozen if confirmation else evaluation_by_name[config.evaluation_namespace]
    return SeedManifest(
        namespace_version=namespaces.version,
        namespace_width=namespaces.namespace_width,
        oracle_training=blocks[0][: config.oracle_training_draws],
        oracle_stability_blocks=blocks,
        development_evaluation=development,
        hackathon_evaluation=hackathon,
        aaai_evaluation=aaai,
        frozen_confirmation=frozen,
        active_evaluation_namespace=active_name,
        active_evaluation=active,
    )


def run_phase3b(
    *,
    repo_root: Path,
    config_path: Path,
    confirmation_manifest: Path | None = None,
) -> tuple[Phase3BRunResult, ...]:
    """Run both profiles sequentially and write Phase 3B research artifacts."""
    root = repo_root.resolve()
    config = load_phase3b_config(config_path)
    confirmation = _confirmation_state(root, confirmation_manifest, config_path)
    if confirmation_manifest is not None and not confirmation.valid:
        raise ValueError(f"confirmation freeze is invalid: {confirmation.reason}")
    if confirmation_manifest is not None and (
        config.evaluation_namespace != "aaai-evaluation"
        or config.evaluation_draws != config.seed_namespaces.frozen_confirmation_count
        or not config.independent_review_complete
    ):
        raise ValueError(
            "confirmation requires the reviewed AAAI configuration and matching "
            "frozen-confirmation draw count"
        )
    seed_manifest = build_seed_manifest(
        config, confirmation=confirmation_manifest is not None
    )
    scenarios = _load_scenarios(root, config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    results: list[Phase3BRunResult] = []
    details: list[dict[str, Any]] = []
    for profile_name in config.loss_profiles:
        output_directory = (
            root / config.output_root / f"{timestamp}-{config.run_label}-{profile_name}"
        )
        run_id = output_directory.name
        benchmark_config = _phase3a_compatible_config(
            config,
            scenarios,
            seed_manifest,
            profile_name,
        )
        temporary_config = _write_temporary_benchmark_config(
            output_directory.parent, benchmark_config, profile_name
        )
        started = perf_counter()
        try:
            base_result = run_benchmark(
                repo_root=root,
                config_path=temporary_config,
                loss_profiles_path=(root / config.loss_profiles_source).resolve(),
                output_directory=output_directory,
                run_id=run_id,
            )
        finally:
            temporary_config.unlink(missing_ok=True)
        records = read_ledger(output_directory / "ledger.jsonl")
        profile = load_loss_profile(
            (root / config.loss_profiles_source).resolve(), profile_name
        )
        stability_started = perf_counter()
        stability = oracle_stability_rows(
            scenarios,
            benchmark_config,
            config,
            seed_manifest,
            profile,
        )
        stability_runtime = perf_counter() - stability_started
        uncertainty = uncertainty_rows(
            records, confidence_level=float(config.confidence_level)
        )
        paired = paired_comparison_rows(
            records, confidence_level=float(config.confidence_level)
        )
        claim_level = determine_claim_level(
            config,
            confirmation_valid=confirmation.valid and confirmation_manifest is not None,
        )
        summary = _phase3b_summary(
            base_result.summary,
            profile_name,
            records,
            stability,
            uncertainty,
            paired,
            config,
            claim_level,
            confirmation,
        )
        elapsed = perf_counter() - started
        _write_phase3b_outputs(
            output_directory,
            root,
            config_path,
            config,
            seed_manifest,
            scenarios,
            base_result.oracle_choices,
            records,
            stability,
            uncertainty,
            paired,
            summary,
            confirmation,
            stability_runtime=stability_runtime,
            total_phase3b_runtime=elapsed,
        )
        result = Phase3BRunResult(
            profile=profile_name,
            run_id=base_result.run_id,
            run_directory=output_directory,
            record_count=len(records),
            permitted_claim_level=claim_level,
            summary=summary,
        )
        results.append(result)
        details.append(
            {
                "profile": profile_name,
                "directory": output_directory,
                "summary": summary,
                "stability": stability,
                "runtime_seconds": elapsed,
                "metadata": json.loads(
                    (output_directory / "metadata.json").read_text("utf-8")
                ),
            }
        )
    _write_loss_profile_sensitivity(details)
    if config.run_label == "phase3b-pilot":
        _write_scaling_estimates(root, config, details)
    return tuple(results)


def uncertainty_rows(
    records: Sequence[LedgerRecord], *, confidence_level: float
) -> list[dict[str, int | float | str]]:
    """Build scenario and aggregate binomial/mean uncertainty summaries."""
    grouped: dict[tuple[str, str], list[LedgerRecord]] = defaultdict(list)
    by_system: dict[str, list[LedgerRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.scenario_id, record.baseline.value)].append(record)
        by_system[record.baseline.value].append(record)
    rows: list[dict[str, int | float | str]] = []
    for (scenario, system), group in sorted(grouped.items()):
        rows.extend(
            _uncertainty_for_group(
                group,
                system=system,
                scenario=scenario,
                aggregation="scenario-draw",
                confidence_level=confidence_level,
            )
        )
    for system, group in sorted(by_system.items()):
        metrics = metric_row(group)
        for metric, (numerator_key, denominator_key) in _BINOMIAL_METRICS.items():
            denominator = int(metrics[denominator_key])
            if denominator < 1:
                continue
            interval = wilson_interval(
                int(metrics[numerator_key]),
                denominator,
                confidence_level=confidence_level,
            )
            rows.append(
                _uncertainty_row(
                    system,
                    "all",
                    metric,
                    "row-level-binomial",
                    interval.model_dump(),
                )
            )
        for metric, attribute in (
            ("mean_routing_regret", "regret"),
            ("mean_runtime_seconds", "runtime"),
        ):
            values = (
                scenario_clustered_values(group, "regret")
                if attribute == "regret"
                else _scenario_runtime_means(group)
            )
            interval = normal_mean_interval(
                values,
                confidence_level=confidence_level,
                method="normal-scenario-clustered",
                clusters=len(values),
            )
            rows.append(
                _uncertainty_row(
                    system,
                    "all",
                    metric,
                    "scenario-clustered",
                    interval.model_dump(),
                )
            )
    return rows


def determine_claim_level(
    config: Phase3BConfig, *, confirmation_valid: bool
) -> ClaimLevel:
    """Apply conservative, machine-readable evidence claim gates."""
    if config.run_label == "phase3b-pilot":
        return "smoke-only"
    if (
        confirmation_valid
        and config.independent_review_complete
        and config.evaluation_namespace == "aaai-evaluation"
        and config.evaluation_draws == config.seed_namespaces.frozen_confirmation_count
    ):
        return "confirmation-supported"
    if config.independent_review_complete and config.evaluation_draws >= 250:
        return "benchmark-supported"
    return "preliminary-descriptive"


def _load_scenarios(
    repo_root: Path, config: Phase3BConfig
) -> tuple[ScenarioConfig, ...]:
    source = (repo_root / config.scenario_source).resolve()
    phase3a = load_benchmark_config(source)
    ids = tuple(scenario.scenario_id for scenario in phase3a.scenarios)
    if ids != PHASE3A_SCENARIO_IDS:
        raise ValueError("Phase 3B must retain the exact eight Phase 3A scenarios")
    return phase3a.scenarios


def _phase3a_compatible_config(
    config: Phase3BConfig,
    scenarios: tuple[ScenarioConfig, ...],
    seeds: SeedManifest,
    profile_name: str,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        benchmark_version=config.benchmark_version,
        run_label=f"{config.run_label}-{profile_name}",
        output_root=config.output_root,
        confidence_level=config.confidence_level,
        hac_bandwidth=config.hac_bandwidth,
        bootstrap_block_length=config.bootstrap_block_length,
        bootstrap_replications=config.bootstrap_replications,
        bootstrap_seed=config.bootstrap_seed,
        loss_profile=profile_name,
        oracle_training_seeds=seeds.oracle_training,
        evaluation_seeds=seeds.active_evaluation,
        scenarios=scenarios,
        generate_figures=False,
    )


def _write_temporary_benchmark_config(
    output_root: Path, config: BenchmarkConfig, profile: str
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f".phase3b-{profile}-{id(config)}.yaml"
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _uncertainty_for_group(
    group: Sequence[LedgerRecord],
    *,
    system: str,
    scenario: str,
    aggregation: str,
    confidence_level: float,
) -> list[dict[str, int | float | str]]:
    metrics = metric_row(group)
    rows: list[dict[str, int | float | str]] = []
    for metric, (numerator_key, denominator_key) in _BINOMIAL_METRICS.items():
        denominator = int(metrics[denominator_key])
        if denominator < 1:
            continue
        interval = wilson_interval(
            int(metrics[numerator_key]), denominator, confidence_level=confidence_level
        )
        rows.append(
            _uncertainty_row(
                system, scenario, metric, aggregation, interval.model_dump()
            )
        )
    for metric, values in (
        ("mean_routing_regret", [float(record.regret) for record in group]),
        (
            "mean_runtime_seconds",
            [
                float(record.routing_runtime_seconds + record.inference_runtime_seconds)
                for record in group
            ],
        ),
    ):
        interval = normal_mean_interval(
            values,
            confidence_level=confidence_level,
            method="normal-row-level",
        )
        rows.append(
            _uncertainty_row(
                system, scenario, metric, aggregation, interval.model_dump()
            )
        )
    return rows


def _uncertainty_row(
    system: str,
    scenario: str,
    metric: str,
    aggregation: str,
    raw: Mapping[str, object],
) -> dict[str, int | float | str]:
    return {
        "system": system,
        "scenario_id": scenario,
        "metric": metric,
        "aggregation": aggregation,
        "estimate": _float_value(raw["estimate"]),
        "confidence_lower": _float_value(raw["lower"]),
        "confidence_upper": _float_value(raw["upper"]),
        "standard_error": _float_value(raw.get("standard_error"), default=0.0),
        "confidence_level": _float_value(raw["confidence_level"]),
        "interval_method": str(raw["method"]),
        "numerator": _int_value(raw.get("numerator")),
        "denominator": _int_value(raw.get("denominator")),
        "scenario_clusters": _int_value(raw.get("clusters")),
    }


def _float_value(value: object | None, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if not isinstance(value, int | float):
        raise TypeError("expected numeric interval value")
    return float(value)


def _int_value(value: object | None) -> int:
    if value is None:
        return 0
    if not isinstance(value, int):
        raise TypeError("expected integer interval value")
    return value


def _scenario_runtime_means(records: Iterable[LedgerRecord]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.scenario_id].append(
            record.routing_runtime_seconds + record.inference_runtime_seconds
        )
    return [mean(values) for _, values in sorted(grouped.items())]


def _phase3b_summary(
    base: dict[str, Any],
    profile_name: str,
    records: Sequence[LedgerRecord],
    stability: list[dict[str, int | float | str | bool]],
    uncertainty: list[dict[str, int | float | str]],
    paired: list[dict[str, int | float | str]],
    config: Phase3BConfig,
    claim_level: ClaimLevel,
    confirmation: ConfirmationValidation,
) -> dict[str, Any]:
    return {
        **base,
        "phase3b": {
            "loss_profile": profile_name,
            "evaluation_draws_per_scenario": config.evaluation_draws,
            "oracle_training_draws_per_scenario": config.oracle_training_draws,
            "oracle_stability_summary": final_stability_summary(
                stability, training_size=config.oracle_training_draws
            ),
            "oracle_agreement_by_stability": oracle_agreement_by_stability(
                list(records),
                stability,
                training_size=config.oracle_training_draws,
            ),
            "uncertainty_interval_count": len(uncertainty),
            "paired_comparison_count": len(paired),
            "confirmation_status": ("valid" if confirmation.valid else "not-run"),
            "confirmation_manifest_hash": confirmation.manifest_hash,
            "permitted_claim_level": claim_level,
        },
    }


def _write_phase3b_outputs(
    directory: Path,
    repo_root: Path,
    config_path: Path,
    config: Phase3BConfig,
    seed_manifest: SeedManifest,
    scenarios: tuple[ScenarioConfig, ...],
    oracle_choices: Sequence[OracleChoice],
    records: Sequence[LedgerRecord],
    stability: list[dict[str, int | float | str | bool]],
    uncertainty: list[dict[str, int | float | str]],
    paired: list[dict[str, int | float | str]],
    summary: dict[str, Any],
    confirmation: ConfirmationValidation,
    *,
    stability_runtime: float,
    total_phase3b_runtime: float,
) -> None:
    profile = str(summary["phase3b"]["loss_profile"])
    _write_json(directory / "seed_manifest.json", seed_manifest.model_dump(mode="json"))
    _write_json(
        directory / "scenario_manifest.json",
        _scenario_manifest(scenarios, seed_manifest),
    )
    _write_csv(
        directory / "oracle_training_summary.csv", _oracle_rows(oracle_choices, profile)
    )
    _write_csv(directory / "oracle_stability.csv", stability)
    _write_csv(directory / "paired_comparisons.csv", paired)
    _write_csv(directory / "uncertainty_intervals.csv", uncertainty)
    _write_metrics_by_system(directory / "metrics_by_system.csv", summary)
    _write_json(directory / "summary.json", summary)
    _write_summary_markdown(directory / "summary.md", summary, uncertainty)
    claims = _result_claims(summary, uncertainty, paired, config, confirmation)
    _write_json(directory / "result_claims.json", claims)
    resolved_source = config_path.resolve()
    resolved = {
        "phase3b_config": config.model_dump(mode="json"),
        "config_source": (
            resolved_source.relative_to(repo_root).as_posix()
            if resolved_source.is_relative_to(repo_root)
            else resolved_source.as_posix()
        ),
        "active_loss_profile": profile,
        "seed_manifest": seed_manifest.model_dump(mode="json"),
        "scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
    }
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    metadata_path = directory / "metadata.json"
    metadata = cast(dict[str, Any], json.loads(metadata_path.read_text("utf-8")))
    metadata.update(
        {
            "phase3b_total_runtime_seconds": total_phase3b_runtime,
            "oracle_stability_runtime_seconds": stability_runtime,
            "peak_memory_bytes": _peak_memory_bytes(),
            "loss_profile": profile,
            "confirmation_manifest_hash": confirmation.manifest_hash,
            "confirmation_status": summary["phase3b"]["confirmation_status"],
        }
    )
    _write_json(metadata_path, metadata)
    if config.generate_figures:
        _write_phase3b_figures(directory, uncertainty, paired, stability)


def _scenario_manifest(
    scenarios: tuple[ScenarioConfig, ...], seeds: SeedManifest
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for scenario in scenarios:
        simulation = generate_scenario(scenario, seed=seeds.oracle_training[0])
        manifest.append(
            {
                **scenario.model_dump(mode="json"),
                "process_version": simulation.process_version,
                "target_defined": simulation.target_defined,
                "true_mean": simulation.true_mean,
                "true_standard_deviation": simulation.true_standard_deviation,
                "true_sharpe": simulation.true_sharpe,
                "inference_truth": simulation.inference_truth,
                "eligibility_labels": [
                    label.model_dump(mode="json")
                    for label in eligibility_for(simulation)
                ],
            }
        )
    return manifest


def _oracle_rows(
    choices: Sequence[OracleChoice], profile: str
) -> list[dict[str, str | int]]:
    return [
        {
            "scenario_id": choice.scenario_id,
            "loss_profile": profile,
            "selected_action": choice.method_id,
            "training_draws": len(choice.training_seeds),
            "training_seeds": json.dumps(choice.training_seeds),
            "estimated_action_losses": json.dumps(
                choice.expected_losses, sort_keys=True
            ),
            "tie_break_rule": choice.tie_break_rule,
        }
        for choice in choices
    ]


def _write_metrics_by_system(path: Path, summary: dict[str, Any]) -> None:
    raw = cast(dict[str, dict[str, object]], summary["metrics_by_baseline"])
    rows = [{"system": system, **metrics} for system, metrics in sorted(raw.items())]
    _write_csv(path, rows)


def _result_claims(
    summary: dict[str, Any],
    uncertainty: list[dict[str, int | float | str]],
    paired: list[dict[str, int | float | str]],
    config: Phase3BConfig,
    confirmation: ConfirmationValidation,
) -> dict[str, Any]:
    profile = str(summary["phase3b"]["loss_profile"])
    metrics = cast(dict[str, dict[str, object]], summary["metrics_by_baseline"])
    router = metrics[BaselineName.DETERMINISTIC_ROUTER.value]
    gaussian = metrics[BaselineName.ALWAYS_GAUSSIAN.value]
    hac = metrics[BaselineName.ALWAYS_HAC.value]

    def interval(system: str, metric: str) -> list[float] | None:
        row = next(
            (
                item
                for item in uncertainty
                if item["system"] == system
                and item["scenario_id"] == "all"
                and item["metric"] == metric
            ),
            None,
        )
        return (
            [float(row["confidence_lower"]), float(row["confidence_upper"])]
            if row
            else None
        )

    def paired_interval(comparator: str) -> tuple[float, list[float]]:
        row = next(
            item
            for item in paired
            if item["system_b"] == comparator
            and item["metric"] == "invalid_selection"
            and item["scenario_id"] == "all"
        )
        return float(row["mean_paired_difference"]), [
            float(row["confidence_lower"]),
            float(row["confidence_upper"]),
        ]

    gaussian_difference, gaussian_difference_ci = paired_interval(
        BaselineName.ALWAYS_GAUSSIAN.value
    )
    hac_difference, hac_difference_ci = paired_interval(BaselineName.ALWAYS_HAC.value)
    return {
        "run_id": summary["benchmark_run_id"],
        "benchmark_version": config.benchmark_version,
        "loss_profile": profile,
        "evaluation_draws_per_scenario": config.evaluation_draws,
        "oracle_training_draws_per_scenario": config.oracle_training_draws,
        "router_invalid_selection_rate": router["invalid_method_selection_rate"],
        "router_invalid_selection_ci": interval(
            BaselineName.DETERMINISTIC_ROUTER.value,
            "invalid_method_selection_rate",
        ),
        "gaussian_invalid_selection_rate": gaussian["invalid_method_selection_rate"],
        "gaussian_invalid_selection_ci": interval(
            BaselineName.ALWAYS_GAUSSIAN.value,
            "invalid_method_selection_rate",
        ),
        "hac_invalid_selection_rate": hac["invalid_method_selection_rate"],
        "hac_invalid_selection_ci": interval(
            BaselineName.ALWAYS_HAC.value, "invalid_method_selection_rate"
        ),
        "router_selective_coverage": router["selective_coverage"],
        "router_selective_coverage_ci": interval(
            BaselineName.DETERMINISTIC_ROUTER.value, "selective_coverage"
        ),
        "router_abstention_rate": router["abstention_rate"],
        "router_oracle_agreement": router["oracle_agreement_rate"],
        "router_oracle_agreement_ci": interval(
            BaselineName.DETERMINISTIC_ROUTER.value, "oracle_agreement_rate"
        ),
        "router_mean_regret": router["mean_routing_regret"],
        "router_mean_regret_ci": interval(
            BaselineName.DETERMINISTIC_ROUTER.value, "mean_routing_regret"
        ),
        "router_vs_gaussian_invalid_difference": gaussian_difference,
        "router_vs_gaussian_invalid_difference_ci": gaussian_difference_ci,
        "router_vs_hac_invalid_difference": hac_difference,
        "router_vs_hac_invalid_difference_ci": hac_difference_ci,
        "oracle_stability_summary": summary["phase3b"]["oracle_stability_summary"],
        "oracle_agreement_by_stability": summary["phase3b"][
            "oracle_agreement_by_stability"
        ],
        "confirmation_status": summary["phase3b"]["confirmation_status"],
        "confirmation_manifest_hash": confirmation.manifest_hash,
        "permitted_claim_level": summary["phase3b"]["permitted_claim_level"],
    }


def _write_loss_profile_sensitivity(details: list[dict[str, Any]]) -> None:
    rows: list[dict[str, object]] = []
    for item in details:
        summary = cast(dict[str, Any], item["summary"])
        metrics = cast(dict[str, dict[str, object]], summary["metrics_by_baseline"])
        for system, values in sorted(metrics.items()):
            rows.append(
                {
                    "loss_profile": item["profile"],
                    "system": system,
                    "invalid_selection_rate": values["invalid_method_selection_rate"],
                    "selective_coverage": values["selective_coverage"],
                    "abstention_rate": values["abstention_rate"],
                    "oracle_agreement_rate": values["oracle_agreement_rate"],
                    "mean_regret": values["mean_routing_regret"],
                }
            )
    for item in details:
        directory = cast(Path, item["directory"])
        _write_csv(directory / "loss_profile_summary.csv", rows)
        summary_path = directory / "summary.json"
        summary = cast(dict[str, Any], json.loads(summary_path.read_text("utf-8")))
        summary["loss_profile_sensitivity"] = rows
        _write_json(summary_path, summary)


def _write_scaling_estimates(
    repo_root: Path, pilot: Phase3BConfig, details: list[dict[str, Any]]
) -> None:
    observed_runtime = sum(float(item["runtime_seconds"]) for item in details)
    evaluation_runtime = sum(
        _float_value(
            cast(dict[str, object], item["metadata"])["evaluation_runtime_seconds"]
        )
        for item in details
    )
    oracle_training_runtime = sum(
        _float_value(
            cast(dict[str, object], item["metadata"])["oracle_training_runtime_seconds"]
        )
        for item in details
    )
    observed_bytes = sum(
        _directory_size(cast(Path, item["directory"])) for item in details
    )
    pilot_units = _work_units(pilot)
    projections: dict[str, Any] = {}
    for label, filename in (
        ("hackathon", "phase3b-hackathon.yaml"),
        ("aaai", "phase3b-aaai.yaml"),
    ):
        target = load_phase3b_config(repo_root / "configs/benchmark" / filename)
        ratio = _work_units(target) / pilot_units
        disk_ratio = target.evaluation_draws / pilot.evaluation_draws
        projections[label] = {
            "work_unit_ratio": ratio,
            "projected_runtime_seconds": observed_runtime * ratio,
            "projected_disk_bytes": int(observed_bytes * disk_ratio),
            "configured_evaluation_draws_per_scenario": target.evaluation_draws,
            "configured_oracle_training_draws_per_scenario": (
                target.oracle_training_draws
            ),
        }
    estimate = {
        "method": (
            "linear extrapolation by configured evaluation/oracle-stability work units"
        ),
        "observed_runtime_seconds": observed_runtime,
        "observed_evaluation_runtime_seconds": evaluation_runtime,
        "observed_oracle_training_runtime_seconds": oracle_training_runtime,
        "oracle_training_runtime_per_scenario_draw_profile_seconds": (
            oracle_training_runtime
            / (
                len(pilot.loss_profiles)
                * len(PHASE3A_SCENARIO_IDS)
                * pilot.oracle_training_draws
            )
        ),
        "evaluation_runtime_per_scenario_draw_system_seconds": (
            evaluation_runtime
            / (
                len(pilot.loss_profiles)
                * len(PHASE3A_SCENARIO_IDS)
                * pilot.evaluation_draws
                * len(BaselineName)
            )
        ),
        "observed_disk_bytes": observed_bytes,
        "observed_peak_memory_bytes": _peak_memory_bytes(),
        "pilot_work_units": pilot_units,
        "profiles": len(pilot.loss_profiles),
        "projections": projections,
        "hackathon_within_runtime_safeguard": (
            projections["hackathon"]["projected_runtime_seconds"]
            <= pilot.max_projected_hackathon_seconds
        ),
        "hackathon_within_disk_safeguard": (
            projections["hackathon"]["projected_disk_bytes"]
            <= pilot.max_projected_hackathon_bytes
        ),
    }
    for item in details:
        _write_json(cast(Path, item["directory"]) / "scaling_estimate.json", estimate)


def _work_units(config: Phase3BConfig) -> int:
    return (
        len(config.loss_profiles)
        * config.bootstrap_replications
        * (
            config.evaluation_draws
            + config.oracle_training_draws
            + config.oracle_stability_blocks * sum(config.oracle_stability_sizes)
        )
    )


def _write_summary_markdown(
    path: Path,
    summary: dict[str, Any],
    uncertainty: list[dict[str, int | float | str]],
) -> None:
    metrics = cast(dict[str, dict[str, object]], summary["metrics_by_baseline"])
    lines = [
        "# Phase 3B benchmark summary",
        "",
        f"Run ID: `{summary['benchmark_run_id']}`",
        "",
        f"Loss profile: `{summary['phase3b']['loss_profile']}`",
        "",
        f"Permitted claim level: `{summary['phase3b']['permitted_claim_level']}`",
        "",
        "Results are scenario-bounded Monte Carlo estimates, not exact-DGP recovery.",
        "",
        "| System | Invalid selection | Selective coverage | Abstention | "
        "Oracle agreement | Mean regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system, row in sorted(metrics.items()):
        invalid = _metric_with_ci(
            uncertainty,
            system,
            "invalid_method_selection_rate",
            row["invalid_method_selection_rate"],
        )
        coverage = _metric_with_ci(
            uncertainty, system, "selective_coverage", row["selective_coverage"]
        )
        abstention = _metric_with_ci(
            uncertainty, system, "abstention_rate", row["abstention_rate"]
        )
        agreement = _metric_with_ci(
            uncertainty,
            system,
            "oracle_agreement_rate",
            row["oracle_agreement_rate"],
        )
        regret = _metric_with_ci(
            uncertainty,
            system,
            "mean_routing_regret",
            row["mean_routing_regret"],
        )
        lines.append(
            f"| {system} | {invalid} | {coverage} | {abstention} | "
            f"{agreement} | {regret} |"
        )
    lines.extend(
        [
            "",
            "Scenario-level estimates, denominators, intervals, and paired "
            "effects are in the CSV artifacts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_with_ci(
    uncertainty: list[dict[str, int | float | str]],
    system: str,
    metric: str,
    estimate: object,
) -> str:
    row = next(
        (
            item
            for item in uncertainty
            if item["system"] == system
            and item["scenario_id"] == "all"
            and item["metric"] == metric
        ),
        None,
    )
    if row is None or not isinstance(estimate, int | float):
        return str(estimate)
    return (
        f"{float(estimate):.4f} "
        f"[{float(row['confidence_lower']):.4f}, {float(row['confidence_upper']):.4f}]"
    )


def _write_phase3b_figures(
    directory: Path,
    uncertainty: list[dict[str, int | float | str]],
    paired: list[dict[str, int | float | str]],
    stability: list[dict[str, int | float | str | bool]],
) -> None:
    import matplotlib.pyplot as plt

    for metric, title, filename in (
        (
            "invalid_method_selection_rate",
            "Invalid selection",
            "invalid_selection_ci.png",
        ),
        ("selective_coverage", "Selective coverage", "selective_coverage_ci.png"),
        ("abstention_rate", "Abstention rate", "abstention_rate.png"),
        ("mean_routing_regret", "Mean regret", "mean_regret_ci.png"),
    ):
        rows = [
            row
            for row in uncertainty
            if row["metric"] == metric and row["scenario_id"] != "all"
        ]
        _errorbar_figure(directory / filename, rows, title)
    stability_rows = [
        row for row in stability if row["scenario_id"] != "structural-mean-break"
    ]
    figure, axis = plt.subplots()
    for scenario in sorted({str(row["scenario_id"]) for row in stability_rows}):
        group = [row for row in stability_rows if row["scenario_id"] == scenario]
        axis.plot(
            [int(row["training_size"]) for row in group],
            [float(row["action_change_frequency"]) for row in group],
            marker="o",
            label=scenario,
        )
    axis.set_xlabel("Oracle training draws")
    axis.set_ylabel("Action change frequency")
    axis.set_title("Oracle-action stability")
    axis.legend(fontsize="x-small")
    figure.tight_layout()
    figure.savefig(directory / "oracle_stability.png")
    plt.close(figure)

    paired_invalid = [
        row
        for row in paired
        if row["metric"] == "invalid_selection" and row["scenario_id"] != "all"
    ]
    _paired_figure(
        directory / "paired_invalid_difference.png",
        paired_invalid,
        "Router minus baseline invalid selection",
    )
    runtime = [
        row
        for row in uncertainty
        if row["metric"] == "mean_runtime_seconds" and row["scenario_id"] != "all"
    ]
    _errorbar_figure(
        directory / "runtime_scaling.png", runtime, "Runtime by scenario/system"
    )


def _errorbar_figure(
    path: Path, rows: list[dict[str, int | float | str]], title: str
) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['scenario_id']}|{row['system']}" for row in rows]
    estimates = [float(row["estimate"]) for row in rows]
    lower = [
        max(0.0, estimate - float(row["confidence_lower"]))
        for estimate, row in zip(estimates, rows, strict=True)
    ]
    upper = [
        max(0.0, float(row["confidence_upper"]) - estimate)
        for estimate, row in zip(estimates, rows, strict=True)
    ]
    figure, axis = plt.subplots()
    axis.errorbar(range(len(rows)), estimates, yerr=[lower, upper], fmt="o")
    axis.set_xticks(range(len(rows)), labels, rotation=90)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _paired_figure(
    path: Path, rows: list[dict[str, int | float | str]], title: str
) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['scenario_id']}|{row['system_b']}" for row in rows]
    estimates = [float(row["mean_paired_difference"]) for row in rows]
    lower = [
        max(0.0, estimate - float(row["confidence_lower"]))
        for estimate, row in zip(estimates, rows, strict=True)
    ]
    upper = [
        max(0.0, float(row["confidence_upper"]) - estimate)
        for estimate, row in zip(estimates, rows, strict=True)
    ]
    figure, axis = plt.subplots()
    axis.errorbar(range(len(rows)), estimates, yerr=[lower, upper], fmt="o")
    axis.axhline(0.0)
    axis.set_xticks(range(len(rows)), labels, rotation=90)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _confirmation_state(
    repo_root: Path, manifest: Path | None, config_path: Path
) -> ConfirmationValidation:
    if manifest is None:
        return ConfirmationValidation(valid=False, reason="confirmation not requested")
    return validate_freeze_manifest(
        repo_root,
        manifest.resolve(),
        expected_config_path=config_path,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires rows: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _peak_memory_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if platform.system() == "Darwin" else usage * 1024)
