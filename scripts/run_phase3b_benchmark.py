"""Run one explicit two-profile Phase 3B benchmark tier."""

from __future__ import annotations

import argparse
from pathlib import Path

from eri.benchmark.phase3b_runner import run_phase3b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/phase3b-pilot.yaml"),
    )
    parser.add_argument("--confirmation-manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = run_phase3b(
        repo_root=root,
        config_path=(root / args.config).resolve(),
        confirmation_manifest=(
            (root / args.confirmation_manifest).resolve()
            if args.confirmation_manifest is not None
            else None
        ),
    )
    for result in results:
        print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
