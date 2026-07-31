# Phase 4B metrics

## Primary outcomes

Only three outcomes are primary: invalid primary-method selection, harmful
non-abstention, and routing regret. Systems are paired within matched cells;
model calls and systems sharing a return draw are not treated as independent.

## Secondary outcomes

- Statistical validity: valid/invalid selection, selective coverage,
  unnecessary abstention, and execution failure.
- Evidence acquisition: relevant, irrelevant, and duplicate requests; budget
  efficiency; evidence gained per requested call; critical-property discovery
  fraction and misses. Critical groups are preregistered in each condition, so
  relevance does not depend on whether a system happened to gain evidence.
- Provenance: exact-field precision/recall, unsupported fields, fabricated
  numbers, completeness and unresolved-field accuracy, and source-span
  validity.
- Decision quality: oracle agreement, paired outcome differences, coverage and
  abstention differences, and final-claim validity.
- Safety: override attempts, unsupported/untraceable claims, unknown tools,
  guardrail rejection, and audit completeness.
- Cost: model/diagnostic calls, tokens, latency, monetary cost when available,
  failures, and repairs.

Rate denominators are reported explicitly. Failed outputs remain in primary
selection and harmful-nonabstention denominators. Selective coverage is
conditional on a non-abstaining interval with a defined target. Wilson
intervals summarize rates. Paired mean differences compare systems within the
exact matched cell. Scenario-clustered intervals are descriptive; they do not
make systems sharing a return draw independent.

If a live provider cannot accept sampling seeds, a minimal repeated-sampling
sensitivity subset should repeat the two pilot scenarios for one seed per
regime, cluster those replicates by matched cell, and report it separately from
the preregistered primary analysis. P-values are not the organizing result.

The evaluator-only oracle minimizes the registered realized loss over eligible
registered methods and abstention within each cell. It is a hindsight lower
bound, not a system-visible policy or deployable target.
