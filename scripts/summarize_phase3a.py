"""Rebuild Phase 3A summary artifacts from a research ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

from eri.benchmark.summaries import summarize_existing_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    summary = summarize_existing_run(
        args.run_directory.resolve(), generate_figures=not args.no_figures
    )
    print(summary["benchmark_run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
