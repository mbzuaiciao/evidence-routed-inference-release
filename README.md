# Evidence-Routed Statistical Inference

Evidence-Routed Statistical Inference (ERI) separates diagnostic evidence,
method eligibility, deterministic routing, and statistical execution. This
curated snapshot contains the typed scientific core, fixed-seed simulation and
benchmark code, non-live tests, aggregate result artifacts, and the AAAI-27
manuscript sources and final PDFs.

## Repository map

- `src/eri/`: typed schemas, diagnostics, routing, inference, simulation, and
  the deterministic and mock-agent benchmark layers.
- `configs/`: selected non-secret scientific and benchmark configurations.
- `scripts/`: deterministic benchmark runners and aggregate summarizers.
- `tests/`: unit, integration, and statistical-reference tests that require no
  hosted provider.
- `artifacts/phase4j/`: sanitized, machine-readable aggregate and per-cell
  scientific outcomes from the retained Phase 4J pilot.
- `paper/evidence_ledger.yaml`: sanitized claim ledger. Phase 5B source paths
  use neutral `external-artifact://` identifiers because the recovered raw
  package is not part of this release.
- `paper/aaai_submission_short/`: official AAAI-27 source, final main paper,
  supplement, and reproducibility checklist.
- `SOURCE_ALLOWLIST.txt` and `RELEASE_FILE_MANIFEST.txt`: exact source-selection
  and final-snapshot inventories.

## Install and validate

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
uv run python scripts/verify_release_artifacts.py
```

`make check` runs the non-live Python checks and aggregate verification. No
release target calls a hosted model.

## Reproduce deterministic benchmark material

Generate the small fixed-seed Phase 3A benchmark in a chosen output directory:

```bash
uv run python scripts/run_phase3a_benchmark.py \
  --config configs/benchmark/phase3a-smoke.yaml \
  --output-directory /tmp/eri-phase3a-smoke
```

Run deterministic and mock-provider workflow checks with:

```bash
uv run pytest tests/integration/test_phase3a_workflow.py \
  tests/integration/test_phase3b_workflow.py \
  tests/integration/test_phase4a_workflow.py \
  tests/integration/test_phase4b_workflow.py
```

The adaptive and static deterministic baselines operate on generated returns
and typed diagnostic evidence. Hosted-model workflows instead produce typed
intermediate objects through an external provider before deterministic
validation and evaluation. Provider adapters, credentials, live-control code,
authorization records, and raw transcripts are intentionally absent here.

## Verify reported aggregates

```bash
uv run python scripts/verify_release_artifacts.py
```

The verifier recomputes Phase 4J per-system mean routing regret and rates from
the included 108 per-cell outcomes, then checks the published aggregate CSV.
It also checks the Phase 5B condition and paired-contrast values in the
sanitized evidence ledger against the manuscript table. Phase 5B unit-level
records were not included in the source snapshot, so those values are
ledger-backed canonical aggregates rather than independently recomputable raw
outputs.

## Build the manuscripts

From `paper/aaai_submission_short/`:

```bash
make main
make supplement
make checklist
```

The official `aaai2027.sty` and `aaai2027.bst` files required by the manuscript
are included. A TeX distribution with `latexmk`, `pdflatex`, and BibTeX is
required.

## Reproducibility boundary

Synthetic data generation, deterministic routing, statistical methods, tests,
and the included aggregate checks are reproducible from this repository.
Hosted-provider outputs are not bit-for-bit reproducible: model services can
change and the raw request/response transcripts are deliberately omitted.
Phase 4J scientific outcome tables are artifact-backed by the included CSVs.
Some Phase 5B operational call, token, and cost quantities are transcript-backed
only, as marked in the evidence ledger, and are not independently verifiable
from this snapshot.

The reported experiments are small, benchmark-specific, single-provider
pilots. They do not establish population-level superiority of a workflow,
general inferiority of multi-agent systems, or general superiority of
deterministic routing.
