"""Monte Carlo intervals and paired benchmark comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import sqrt
from statistics import mean, stdev

from eri.benchmark.phase3b_schemas import IntervalEstimate
from eri.benchmark.schemas import BaselineName, LedgerRecord
from eri.inference.common import normal_quantile


def wilson_interval(
    successes: int, total: int, *, confidence_level: float
) -> IntervalEstimate:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    z_value = normal_quantile((1.0 + level) / 2.0)
    estimate = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (estimate + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * sqrt(estimate * (1.0 - estimate) / total + z_value**2 / (4.0 * total**2))
        / denominator
    )
    return IntervalEstimate(
        estimate=estimate,
        lower=0.0 if successes == 0 else max(0.0, center - half_width),
        upper=1.0 if successes == total else min(1.0, center + half_width),
        confidence_level=level,
        method="wilson-score",
        numerator=successes,
        denominator=total,
    )


def normal_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float,
    method: str = "normal-monte-carlo",
    clusters: int | None = None,
) -> IntervalEstimate:
    """Return a normal interval for a Monte Carlo mean."""
    if not values:
        raise ValueError("mean interval requires at least one observation")
    estimate = mean(values)
    standard_error = stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0
    z_value = normal_quantile((1.0 + confidence_level) / 2.0)
    return IntervalEstimate(
        estimate=estimate,
        lower=estimate - z_value * standard_error,
        upper=estimate + z_value * standard_error,
        standard_error=standard_error,
        confidence_level=confidence_level,
        method=method,
        denominator=len(values),
        clusters=clusters,
    )


def paired_comparison_rows(
    records: Sequence[LedgerRecord], *, confidence_level: float
) -> list[dict[str, int | float | str]]:
    """Compute paired router-minus-comparator effects by scenario and clustered."""
    by_cell: dict[tuple[str, int], dict[BaselineName, LedgerRecord]] = defaultdict(dict)
    for record in records:
        key = (record.scenario_id, record.seed)
        if record.baseline in by_cell[key]:
            raise ValueError(
                "paired comparison contains a duplicate system row for "
                f"{record.scenario_id} seed {record.seed}"
            )
        by_cell[key][record.baseline] = record
    expected = set(BaselineName)
    if any(set(cell) != expected for cell in by_cell.values()):
        raise ValueError("paired comparison requires every baseline in every cell")

    output: list[dict[str, int | float | str]] = []
    comparators = (
        BaselineName.ALWAYS_GAUSSIAN,
        BaselineName.ALWAYS_HAC,
        BaselineName.ORACLE,
    )
    metrics = ("invalid_selection", "coverage", "abstention", "regret")
    scenarios = sorted({scenario for scenario, _ in by_cell})
    for comparator in comparators:
        for metric in metrics:
            scenario_values: dict[str, list[float]] = {}
            scenario_excluded: dict[str, int] = {}
            for scenario in scenarios:
                differences: list[float] = []
                excluded = 0
                for (cell_scenario, _), cell in by_cell.items():
                    if cell_scenario != scenario:
                        continue
                    router = cell[BaselineName.DETERMINISTIC_ROUTER]
                    other = cell[comparator]
                    difference = _paired_value(router, other, metric)
                    if difference is None:
                        excluded += 1
                    else:
                        differences.append(difference)
                scenario_values[scenario] = differences
                scenario_excluded[scenario] = excluded
                if differences:
                    interval = normal_mean_interval(
                        differences,
                        confidence_level=confidence_level,
                        method="paired-normal-row-level",
                    )
                    output.append(
                        _paired_row(
                            comparator,
                            metric,
                            scenario,
                            "scenario-draw",
                            interval,
                            excluded,
                        )
                    )
            nonempty = [values for values in scenario_values.values() if values]
            if nonempty:
                scenario_means = [mean(values) for values in nonempty]
                interval = normal_mean_interval(
                    scenario_means,
                    confidence_level=confidence_level,
                    method="paired-normal-scenario-clustered",
                    clusters=len(scenario_means),
                )
                output.append(
                    _paired_row(
                        comparator,
                        metric,
                        "all",
                        "scenario-clustered",
                        interval,
                        sum(scenario_excluded.values()),
                        paired_cells=sum(len(values) for values in nonempty),
                    )
                )
    return output


def _paired_value(
    router: LedgerRecord, comparator: LedgerRecord, metric: str
) -> float | None:
    if metric == "invalid_selection":
        return float(router.invalid_selection) - float(comparator.invalid_selection)
    if metric == "abstention":
        return float(router.abstained) - float(comparator.abstained)
    if metric == "regret":
        return float(router.regret) - float(comparator.regret)
    if metric == "coverage":
        if (
            router.target_defined is not True
            or comparator.target_defined is not True
            or router.covered is None
            or comparator.covered is None
        ):
            return None
        return float(router.covered) - float(comparator.covered)
    raise KeyError(metric)


def _paired_row(
    comparator: BaselineName,
    metric: str,
    scenario: str,
    aggregation: str,
    interval: IntervalEstimate,
    excluded: int,
    *,
    paired_cells: int | None = None,
) -> dict[str, int | float | str]:
    return {
        "system_a": BaselineName.DETERMINISTIC_ROUTER.value,
        "system_b": comparator.value,
        "metric": metric,
        "scenario_id": scenario,
        "aggregation": aggregation,
        "mean_paired_difference": interval.estimate,
        "standard_error": float(interval.standard_error or 0.0),
        "confidence_lower": interval.lower,
        "confidence_upper": interval.upper,
        "confidence_level": interval.confidence_level,
        "interval_method": interval.method,
        "paired_cells": paired_cells or int(interval.denominator or 0),
        "excluded_cells": excluded,
        "scenario_clusters": int(interval.clusters or 0),
        "denominator_definition": _paired_denominator(metric),
    }


def _paired_denominator(metric: str) -> str:
    if metric == "coverage":
        return "cells with defined targets and valid coverage outcomes for both systems"
    return "shared scenario/evaluation-seed cells"


def scenario_clustered_values(
    records: Iterable[LedgerRecord], attribute: str
) -> list[float]:
    """Return equal-weight scenario means for a numeric record attribute."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.scenario_id].append(float(getattr(record, attribute)))
    return [mean(values) for _, values in sorted(grouped.items())]
