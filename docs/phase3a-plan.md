# Phase 3A minimum benchmark plan

## Research question

Does the current deterministic evidence router select more defensible
Sharpe-inference procedures than simple fixed-method baselines across a small
set of controlled return-generating processes?

## Implemented scope

- Families: IID Gaussian, stationary AR(1), stationary Gaussian-innovation
  GARCH(1,1)-style volatility clustering with explicit moment metadata, and a
  two-segment mean or variance break.
- Systems: always IID Gaussian, always HAC, the unchanged Phase 2 deterministic
  workflow, and a regime-level oracle.
- Outputs: validated JSONL ledger, resolved configuration, metadata, JSON and
  Markdown summaries, four CSV tables, and four separate matplotlib figures.
- Configurations: a contract-level smoke run and an explicit, laptop-sized mini
  run. The mini run is never part of `make check`.

All systems receive the same simulated draw for a scenario and seed. A SHA-256
draw fingerprint makes this invariant testable. Generated returns and targets
remain per-period and unannualized.

The ledger records process version and parameters, target applicability and
population moments, selected and oracle losses, configured loss-profile name,
coverage applicability, and explicit metric-accounting inputs. Generated
summary rows include numerators and denominators rather than rates alone.

## Scientific boundary

Truth labels describe inference-relevant validity, not exact process
classification. The eligibility registry is versioned and independent of
router evidence. A structural-break case has segment Sharpes and a documented
descriptive mixture functional, but no stable full-sample inferential Sharpe;
ordinary coverage is therefore not applicable.

Oracle-training seeds are disjoint from evaluation seeds. The oracle first
filters analytically ineligible methods, estimates expected policy-weighted
loss on training draws, and applies a deterministic tie-break. Loss weights are
policy choices reported under validity-first and efficiency-balanced profiles.

## Non-goals

No agents, LLMs, LangGraph, UI, network execution, new inference method,
open-world task, exact-DGP recovery, automatic large run, or claim of benchmark
superiority is included.
