# Phase 3B benchmark-validation plan

## Objective

Phase 3B strengthens Monte Carlo evidence for the existing Phase 3A benchmark.
It does not add process families, statistical procedures, agents, orchestration,
UI, or network access. The evaluated systems remain always-IID-Gaussian,
always-HAC, the deterministic router, and the frozen empirical oracle. Both
registered loss profiles are reported separately.

The scientific questions concern invalid procedure selection, selective
coverage and abstention, routing regret, loss-profile sensitivity, and the
stability of the empirical oracle. The benchmark is closed-world and does not
claim exact DGP recovery.

## Registered run tiers

| Tier | Oracle draws per scenario | Evaluation draws per scenario | Purpose |
| --- | ---: | ---: | --- |
| Scaling pilot | 25 | 25 | Correctness and resource measurement only. |
| Hackathon | 250 | 250 | Reproducible descriptive benchmark evidence. |
| AAAI | 1,000 | 2,000 | Larger preregistered evidence, subject to review and resource gates. |

All tiers reuse the exact eight scenarios in `phase3a-mini.yaml`. The runner
rejects a changed scenario identity or order. Each system receives the same
simulated return draw, and oracle training is separate from evaluation.

## Execution gates

The pilot is run once before a larger benchmark. Its runtime, disk use, and
peak-memory observation are recorded and extrapolated. The hackathon tier is
explicit and is not part of `make check`. The AAAI and confirmation targets are
also explicit and must not run automatically.

A confirmation run additionally requires a valid freeze manifest. The manifest
hashes the benchmark configuration, registry/router policy inputs, loss
profiles, diagnostics configuration, and simulation and benchmark sources.

## Outputs and review

Each loss profile receives a timestamped ignored directory under
`runs/phase3b/`, with a ledger, resolved configuration, seed and scenario
manifests, scenario and system metrics, intervals, paired effects, oracle
training and stability summaries, loss-profile sensitivity, figures, and
machine-readable result claims. Numerical submission language is always a
manual review step; the benchmark never edits an abstract or hackathon draft.

## Boundaries

- Pilot output is `smoke-only`.
- Unreviewed larger output is `preliminary-descriptive`.
- `benchmark-supported` requires the configured independent-review gate and at
  least the hackathon evaluation size.
- `confirmation-supported` additionally requires an unchanged frozen critical
  state and the unseen confirmation namespace.

These levels govern wording, not mathematical truth or external validity.

## Independent audit corrections

Version 0.3 preserves the version-0.2 pilot and hackathon directories. It
corrects paired coverage so that a fixed method with a valid coverage outcome
is retained even when the registry labels that method ineligible. Eligibility
is evaluated separately through invalid selection. It also binds confirmation
to the exact reviewed AAAI configuration, prevents a pilot from being promoted
by a confirmation flag, aligns future run IDs with their directory names, and
accounts for configured bootstrap replications in runtime work units.

The version-0.2 hackathon ledgers and headline counts remain reconstructable and
scientifically usable for explicitly preliminary statements. Their saved
router-versus-Gaussian paired coverage rows used the old over-restrictive
denominator and must not be cited. No existing run may be overwritten.
