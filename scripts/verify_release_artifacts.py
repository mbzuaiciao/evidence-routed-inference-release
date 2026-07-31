"""Verify the scientific aggregates included in the curated release."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
PHASE4J = ROOT / "artifacts/phase4j"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (PHASE4J / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{actual!r} != {expected!r}")


def verify_phase4j() -> None:
    outcomes = _read_csv("primary_outcomes.csv")
    aggregates = _read_csv("aggregate_metrics.csv")
    comparisons = _read_csv("paired_comparisons.csv")
    regrets = _read_csv("routing_regret.csv")

    if len(outcomes) != 108 or len(regrets) != 108:
        raise AssertionError("Phase 4J must retain exactly 108 outcomes")
    if len(aggregates) != 6 or len(comparisons) != 15:
        raise AssertionError("Phase 4J aggregate inventory is incomplete")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in outcomes:
        grouped[row["system"]].append(row)
    if set(grouped) != {row["system"] for row in aggregates}:
        raise AssertionError("Phase 4J system inventory mismatch")

    by_system = {row["system"]: row for row in aggregates}
    for system, rows in grouped.items():
        if len(rows) != 18:
            raise AssertionError(f"{system} does not contain 18 cells")
        for metric in (
            "routing_regret",
            "invalid_primary_method_selection",
            "harmful_nonabstention",
        ):
            recomputed = sum(float(row[metric]) for row in rows) / len(rows)
            _assert_close(recomputed, float(by_system[system][metric]))


def verify_phase5b_ledger() -> None:
    ledger_path = ROOT / "paper/evidence_ledger.yaml"
    ledger = cast(dict[str, Any], yaml.safe_load(ledger_path.read_text()))
    claims = {
        claim["claim_id"]: claim["value"] for claim in ledger["claims"]
    }

    expected_conditions = {
        "phase5b-condition-a": (4.723122, 0.3889, 0.1667, 0.6111, 1),
        "phase5b-condition-b": (1.389345, 0.1111, 0.1111, 0.8889, 1),
        "phase5b-condition-c": (0.250287, 0.0, 0.0, 1.0, 0),
        "phase5b-condition-d": (3.598322, 0.2222, 0.1111, 0.7778, 0),
    }
    for claim_id, expected in expected_conditions.items():
        value = claims[claim_id]
        observed = (
            round(float(value["mean_regret"]), 6),
            round(float(value["invalid_selection"]), 4),
            round(float(value["harmful_nonabstention"]), 4),
            round(float(value["claim_validity"]), 4),
            int(value["failed"]),
        )
        if observed != expected:
            raise AssertionError(f"{claim_id}: {observed!r} != {expected!r}")

    expected_contrasts = {
        "phase5b-secondary-b-minus-a": -3.333777,
        "phase5b-primary-c-minus-b": -1.139058,
        "phase5b-secondary-d-minus-c": 3.348035,
    }
    for claim_id, expected in expected_contrasts.items():
        value = claims[claim_id]
        observed = round(float(value["all_cell_mean"]), 6)
        _assert_close(observed, expected)


def main() -> int:
    verify_phase4j()
    verify_phase5b_ledger()
    print("Release aggregate verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
