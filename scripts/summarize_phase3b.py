"""Validate and print the headline claim artifact for a Phase 3B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eri.benchmark.ledger import read_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    records = read_ledger(directory / "ledger.jsonl")
    claims: object = json.loads((directory / "result_claims.json").read_text("utf-8"))
    if not records or not isinstance(claims, dict):
        raise ValueError("Phase 3B run is incomplete")
    print(json.dumps(claims, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
