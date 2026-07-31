"""Run audited agentic analysis for a local one-column return CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

from eri.config import load_phase2_config
from eri.demo.render import write_demo_artifacts
from eri.orchestration.budgets import load_agent_config
from eri.orchestration.workflow import run_agentic_workflow


def _read_returns(path: Path) -> tuple[float, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("return CSV is empty")
    start = 1 if rows[0] and rows[0][0].strip().lower() in {"return", "returns"} else 0
    values = tuple(float(row[0]) for row in rows[start:] if row and row[0].strip())
    if not values:
        raise ValueError("return CSV has no numeric observations")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--frequency", required=True)
    parser.add_argument("--benchmark-sharpe", required=True, type=float)
    parser.add_argument("--provenance", default="")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    phase2 = load_phase2_config(root / "configs")
    phase2 = phase2.model_copy(
        update={
            "experiment": phase2.experiment.model_copy(
                update={"benchmark_sharpe": args.benchmark_sharpe}
            )
        }
    )
    agent_config = load_agent_config(root / f"configs/agents/phase4a-{args.mode}.yaml")
    state = run_agentic_workflow(
        _read_returns(args.csv),
        phase2,
        agent_config,
        provenance_text=args.provenance,
        example_id=f"uploaded-{args.frequency}",
        input_frequency=args.frequency,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "runs/phase4a" / f"{timestamp}-uploaded"
    write_demo_artifacts(state, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
