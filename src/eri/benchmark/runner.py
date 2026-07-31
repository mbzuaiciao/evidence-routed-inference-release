"""Configuration-driven Phase 3A benchmark runner."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml

from eri.benchmark.baselines import run_baseline
from eri.benchmark.evaluation import build_record, validate_shared_draws
from eri.benchmark.ledger import write_ledger
from eri.benchmark.oracle import estimate_oracle
from eri.benchmark.registry import eligibility_for, generate_scenario
from eri.benchmark.schemas import (
    BaselineName,
    BenchmarkConfig,
    BenchmarkRunResult,
    LedgerRecord,
    LossProfile,
    LossProfiles,
)
from eri.benchmark.summaries import build_summary, write_outputs
from eri.config import load_phase2_config, load_yaml_model


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    return load_yaml_model(path, BenchmarkConfig)


def load_loss_profile(path: Path, name: str) -> LossProfile:
    profiles = load_yaml_model(path, LossProfiles)
    for profile in profiles.profiles:
        if profile.name == name:
            return profile
    raise KeyError(f"unknown loss profile: {name}")


def run_benchmark(
    *,
    repo_root: Path,
    config_path: Path,
    loss_profiles_path: Path,
    output_directory: Path | None = None,
    run_id: str | None = None,
) -> BenchmarkRunResult:
    """Run shared-draw baselines and write one auditable run directory."""
    started = perf_counter()
    started_at = datetime.now(UTC)
    config = load_benchmark_config(config_path)
    profile = load_loss_profile(loss_profiles_path, config.loss_profile)
    phase2_config = load_phase2_config(repo_root / "configs")
    resolved_run_id = run_id or (
        started_at.strftime("%Y%m%dT%H%M%S%fZ") + f"-{config.run_label}"
    )
    run_directory = (
        output_directory.resolve()
        if output_directory is not None
        else (repo_root / config.output_root / resolved_run_id).resolve()
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    oracle_started = perf_counter()
    oracle_choices = tuple(
        estimate_oracle(scenario, config, profile) for scenario in config.scenarios
    )
    oracle_runtime_seconds = perf_counter() - oracle_started
    oracle_by_scenario = {choice.scenario_id: choice for choice in oracle_choices}
    records: list[LedgerRecord] = []
    evaluation_started = perf_counter()
    for scenario in config.scenarios:
        oracle_choice = oracle_by_scenario[scenario.scenario_id]
        for seed in config.evaluation_seeds:
            simulation = generate_scenario(scenario, seed=seed)
            labels = eligibility_for(simulation)
            oracle_execution = run_baseline(
                BaselineName.ORACLE,
                simulation,
                phase2_config,
                config,
                oracle_method=oracle_choice.method_id,
            )
            for baseline in BaselineName:
                execution = (
                    oracle_execution
                    if baseline is BaselineName.ORACLE
                    else run_baseline(
                        baseline,
                        simulation,
                        phase2_config,
                        config,
                        oracle_method=oracle_choice.method_id,
                    )
                )
                records.append(
                    build_record(
                        run_id=resolved_run_id,
                        scenario_id=scenario.scenario_id,
                        baseline=baseline,
                        simulation=simulation,
                        labels=labels,
                        execution=execution,
                        oracle_method=oracle_choice.method_id,
                        oracle_execution=oracle_execution,
                        config=config,
                        profile=profile,
                    )
                )
    evaluation_runtime_seconds = perf_counter() - evaluation_started
    ledger_records = tuple(records)
    validate_shared_draws(ledger_records)
    write_ledger(run_directory / "ledger.jsonl", ledger_records)
    summary = build_summary(resolved_run_id, ledger_records)
    write_outputs(
        run_directory,
        summary,
        ledger_records,
        oracle_choices,
        generate_figures=config.generate_figures,
    )
    _write_resolved_config(run_directory, config, profile)
    completed_at = datetime.now(UTC)
    metadata = {
        "run_id": resolved_run_id,
        "benchmark_version": config.benchmark_version,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "runtime_seconds": perf_counter() - started,
        "python_version": sys.version,
        "platform": platform.platform(),
        "network_used": False,
        "oracle_training_runtime_seconds": oracle_runtime_seconds,
        "evaluation_runtime_seconds": evaluation_runtime_seconds,
        "oracle_training_seeds": list(config.oracle_training_seeds),
        "evaluation_seeds": list(config.evaluation_seeds),
    }
    (run_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return BenchmarkRunResult(
        run_id=resolved_run_id,
        run_directory=run_directory,
        record_count=len(ledger_records),
        oracle_choices=oracle_choices,
        summary=summary,
    )


def _write_resolved_config(
    run_directory: Path, config: BenchmarkConfig, profile: LossProfile
) -> None:
    raw = {
        "benchmark": config.model_dump(mode="json"),
        "selected_loss_profile": profile.model_dump(mode="json"),
    }
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=True), encoding="utf-8"
    )
