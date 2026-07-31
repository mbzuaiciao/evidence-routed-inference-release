# pyright: reportUnknownMemberType=false
"""Machine-readable, Markdown, CSV, and lightweight figure summaries."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from eri.benchmark.evaluation import metric_row
from eri.benchmark.schemas import BaselineName, LedgerRecord, OracleChoice

MetricValue = int | float | str
MetricRows = dict[str, dict[str, MetricValue]]


def build_summary(run_id: str, records: Sequence[LedgerRecord]) -> dict[str, Any]:
    """Build aggregate metrics and AAAI-placeholder-ready fields."""
    by_baseline: dict[str, list[LedgerRecord]] = defaultdict(list)
    by_scenario_baseline: dict[tuple[str, str], list[LedgerRecord]] = defaultdict(list)
    for record in records:
        by_baseline[record.baseline.value].append(record)
        by_scenario_baseline[(record.scenario_id, record.baseline.value)].append(record)
    baseline_metrics = {
        baseline: metric_row(group) for baseline, group in sorted(by_baseline.items())
    }
    scenario_metrics = {
        f"{scenario}|{baseline}": metric_row(group)
        for (scenario, baseline), group in sorted(by_scenario_baseline.items())
    }
    router = baseline_metrics[BaselineName.DETERMINISTIC_ROUTER.value]
    gaussian = baseline_metrics[BaselineName.ALWAYS_GAUSSIAN.value]
    hac = baseline_metrics[BaselineName.ALWAYS_HAC.value]
    return {
        "benchmark_run_id": run_id,
        "record_count": len(records),
        "metrics_by_baseline": baseline_metrics,
        "metrics_by_scenario_and_baseline": scenario_metrics,
        "result_ready": {
            "router_invalid_selection_rate": router["invalid_method_selection_rate"],
            "gaussian_invalid_selection_rate": gaussian[
                "invalid_method_selection_rate"
            ],
            "hac_invalid_selection_rate": hac["invalid_method_selection_rate"],
            "router_selective_coverage": router["selective_coverage"],
            "router_abstention_rate": router["abstention_rate"],
            "router_oracle_agreement": router["oracle_agreement_rate"],
            "router_mean_regret": router["mean_routing_regret"],
            "benchmark_run_id": run_id,
        },
    }


def write_outputs(
    run_directory: Path,
    summary: dict[str, Any],
    records: Sequence[LedgerRecord],
    oracle_choices: Sequence[OracleChoice],
    *,
    generate_figures: bool,
) -> None:
    """Write compact deterministic result artifacts."""
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary_markdown(run_directory / "summary.md", summary)
    _write_metric_csvs(run_directory, summary)
    _write_oracle_csv(run_directory / "oracle_choices.csv", oracle_choices)
    _write_confusion_csv(run_directory / "routing_confusion.csv", records)
    if generate_figures:
        _write_figures(run_directory, summary)


def summarize_existing_run(
    run_directory: Path, *, generate_figures: bool = True
) -> dict[str, Any]:
    """Rebuild summary artifacts from an existing validated ledger."""
    from eri.benchmark.ledger import read_ledger

    records = read_ledger(run_directory / "ledger.jsonl")
    if not records:
        raise ValueError("ledger contains no records")
    summary = build_summary(records[0].run_id, records)
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary_markdown(run_directory / "summary.md", summary)
    _write_metric_csvs(run_directory, summary)
    _write_confusion_csv(run_directory / "routing_confusion.csv", records)
    if generate_figures:
        _write_figures(run_directory, summary)
    return summary


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 3A benchmark summary",
        "",
        f"Run ID: `{summary['benchmark_run_id']}`",
        "",
        "These are descriptive benchmark outputs, not universal superiority claims.",
        "",
        "## Metrics by baseline",
        "",
        "| Baseline | Invalid selection | Coverage | Abstention | "
        "Oracle agreement | Mean regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics = cast(MetricRows, summary["metrics_by_baseline"])
    for baseline, raw in sorted(metrics.items()):
        lines.append(
            f"| {baseline} | {_format(raw['invalid_method_selection_rate'])} | "
            f"{_format(raw['selective_coverage'])} | "
            f"{_format(raw['abstention_rate'])} | "
            f"{_format(raw['oracle_agreement_rate'])} | "
            f"{_format(raw['mean_routing_regret'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metric_csvs(run_directory: Path, summary: dict[str, Any]) -> None:
    baseline_rows = cast(MetricRows, summary["metrics_by_baseline"])
    scenario_rows = cast(MetricRows, summary["metrics_by_scenario_and_baseline"])
    _write_mapping_csv(
        run_directory / "metrics_by_baseline.csv", "baseline", baseline_rows
    )
    _write_mapping_csv(
        run_directory / "metrics_by_scenario.csv",
        "scenario_and_baseline",
        scenario_rows,
    )


def _write_mapping_csv(path: Path, key_name: str, rows: MetricRows) -> None:
    normalized: list[dict[str, object]] = []
    for key, raw in sorted(rows.items(), key=lambda item: str(item[0])):
        normalized.append({key_name: str(key), **raw})
    if not normalized:
        raise ValueError("CSV output requires rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def _write_oracle_csv(path: Path, choices: Sequence[OracleChoice]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "oracle_method", "expected_losses"],
        )
        writer.writeheader()
        for choice in choices:
            writer.writerow(
                {
                    "scenario_id": choice.scenario_id,
                    "oracle_method": choice.method_id,
                    "expected_losses": json.dumps(
                        choice.expected_losses, sort_keys=True
                    ),
                }
            )


def _write_confusion_csv(path: Path, records: Sequence[LedgerRecord]) -> None:
    counts = Counter(
        (
            record.baseline.value,
            record.oracle_method,
            record.selected_method or "abstention",
        )
        for record in records
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "oracle_method", "selected_method", "count"])
        for (baseline, oracle, selected), count in sorted(counts.items()):
            writer.writerow([baseline, oracle, selected, count])


def _write_figures(run_directory: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    metrics = cast(MetricRows, summary["metrics_by_scenario_and_baseline"])
    specifications = (
        (
            "invalid_method_selection_rate",
            "Invalid selection rate",
            "invalid_selection_rate.png",
        ),
        ("selective_coverage", "Selective coverage", "coverage.png"),
        ("abstention_rate", "Abstention rate", "abstention_rate.png"),
        ("mean_routing_regret", "Mean routing regret", "mean_routing_regret.png"),
    )
    for metric, ylabel, filename in specifications:
        labels: list[str] = []
        values: list[float] = []
        for key, raw in sorted(metrics.items()):
            value = raw.get(metric)
            if isinstance(value, int | float):
                labels.append(str(key))
                values.append(float(value))
        figure, axis = plt.subplots()
        axis.bar(range(len(values)), values)
        axis.set_xticks(range(len(labels)), labels, rotation=90)
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel + " by scenario and baseline")
        figure.tight_layout()
        figure.savefig(run_directory / filename)
        plt.close(figure)


def _format(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else str(value)
