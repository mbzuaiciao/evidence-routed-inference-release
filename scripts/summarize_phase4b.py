"""Reprint the machine-readable and Markdown summary for a Phase 4B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from eri.benchmark.phase4b_summaries import summary_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    raw: object = json.loads((args.run_directory / "summary.json").read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("summary.json must contain an object")
    summary = cast(dict[str, object], raw)
    rendered = summary_markdown(summary)
    (args.run_directory / "summary.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
