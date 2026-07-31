"""JSONL research-ledger persistence."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from eri.benchmark.schemas import LedgerRecord


def write_ledger(path: Path, records: Iterable[LedgerRecord]) -> None:
    """Write one validated compact JSON object per benchmark action."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def read_ledger(path: Path) -> tuple[LedgerRecord, ...]:
    """Load and revalidate a JSONL ledger."""
    records: list[LedgerRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw: object = json.loads(line)
                records.append(LedgerRecord.model_validate(raw))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid ledger row {line_number}") from error
    return tuple(records)
