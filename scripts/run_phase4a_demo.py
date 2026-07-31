"""Run fixed-seed Phase 4A demonstrations and write ignored audit artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from eri.config import load_phase2_config
from eri.demo.examples import DEMO_BUILDERS, get_demo_example
from eri.demo.render import write_demo_artifacts
from eri.orchestration.budgets import load_agent_config
from eri.orchestration.workflow import run_agentic_workflow


def run_examples(
    example_ids: tuple[str, ...], *, config_path: Path
) -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    phase2 = load_phase2_config(root / "configs")
    agents = load_agent_config(config_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directories: list[Path] = []
    for example_id in example_ids:
        example = get_demo_example(example_id)
        state = run_agentic_workflow(
            example.returns,
            phase2,
            agents,
            provenance_text=example.provenance_text,
            example_id=example.example_id,
        )
        directory = root / "runs/phase4a" / f"{timestamp}-{example.example_id}"
        write_demo_artifacts(state, directory)
        directories.append(directory)
    return tuple(directories)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("all", *DEMO_BUILDERS), default="all")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/agents/phase4a-mock.yaml")
    )
    args = parser.parse_args()
    examples = ("iid", "garch", "break") if args.example == "all" else (args.example,)
    for path in run_examples(examples, config_path=args.config):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
