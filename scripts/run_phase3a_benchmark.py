"""Run one configuration-driven Phase 3A benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from eri.benchmark.runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/phase3a-smoke.yaml"),
    )
    parser.add_argument(
        "--loss-profiles",
        type=Path,
        default=Path("configs/benchmark/loss-profiles.yaml"),
    )
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = run_benchmark(
        repo_root=repo_root,
        config_path=(repo_root / args.config).resolve(),
        loss_profiles_path=(repo_root / args.loss_profiles).resolve(),
        output_directory=args.output_directory,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
