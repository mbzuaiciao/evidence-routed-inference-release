"""Estimate Phase 4B live call, token, cost, and serial-runtime upper bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eri.benchmark.phase4b_runner import load_phase4b_config

AGENT_SYSTEMS = 3
LOGICAL_CALLS_PER_CELL = (2, 4, 4)


def estimate(config_path: Path) -> dict[str, float | int | str]:
    config = load_phase4b_config(config_path)
    cells = len(config.scenario_ids) * len(config.return_seeds) * len(config.regimes)
    logical_calls = cells * sum(LOGICAL_CALLS_PER_CELL)
    configured_upper_calls = cells * AGENT_SYSTEMS * config.model.maximum_model_calls
    worst_attempts = configured_upper_calls * (1 + config.model.retries)
    expected_attempts = logical_calls * (1.0 + float(config.model.expected_retry_rate))

    def tokens(
        attempts: float, input_fraction: float, output_fraction: float
    ) -> tuple[float, float]:
        return (
            attempts * config.model.maximum_input_tokens * input_fraction,
            attempts * config.model.maximum_response_tokens * output_fraction,
        )

    def cost(input_tokens: float, output_tokens: float) -> float:
        return float(
            (
                input_tokens * config.model.input_cost_per_million_tokens
                + output_tokens * config.model.output_cost_per_million_tokens
            )
            / 1_000_000
        )

    expected_tokens = tokens(
        expected_attempts,
        float(config.model.expected_input_token_fraction),
        float(config.model.expected_output_token_fraction),
    )
    upper_tokens = tokens(configured_upper_calls, 1.0, 1.0)
    worst_tokens = tokens(worst_attempts, 1.0, 1.0)
    return {
        "tier": config.tier.value,
        "matched_cells": cells,
        "agent_systems": AGENT_SYSTEMS,
        "logical_model_calls": logical_calls,
        "maximum_model_calls": configured_upper_calls,
        "worst_case_model_attempts_with_retries": worst_attempts,
        "expected_input_tokens": round(expected_tokens[0]),
        "expected_output_tokens": round(expected_tokens[1]),
        "estimated_cost_expected_usd": round(cost(*expected_tokens), 2),
        "estimated_input_tokens_upper_bound": round(upper_tokens[0]),
        "estimated_output_tokens_upper_bound": round(upper_tokens[1]),
        "estimated_cost_upper_bound_usd": round(cost(*upper_tokens), 2),
        "estimated_input_tokens_worst_case": round(worst_tokens[0]),
        "estimated_output_tokens_worst_case": round(worst_tokens[1]),
        "estimated_cost_worst_case_usd": round(cost(*worst_tokens), 2),
        "serial_runtime_expected_seconds": float(
            expected_attempts * config.model.timeout_seconds
        ),
        "serial_runtime_upper_bound_seconds": float(
            configured_upper_calls * config.model.timeout_seconds
        ),
        "serial_runtime_worst_case_seconds": float(
            worst_attempts * config.model.timeout_seconds
        ),
        "assumptions": (
            "expected uses registered 2/4/4 logical calls and configured token/retry "
            "fractions; upper uses every configured logical call without retry; worst "
            "uses maximum tokens, timeout, and every allowed retry"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/phase4b-live-pilot.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(estimate(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
