# Evidence schema

Phase 2 routes only typed, deterministic evidence. An `EvidenceItem` records a
stable content-derived identifier, claim, polarity, diagnostic and version,
finite statistic or structured output, optional p-value, uncertainty metadata,
sample size, explicit configuration, assumptions, warnings, method effects,
non-claims, and a data fingerprint or provenance reference.

Polarity is deliberately three-valued:

- `supports` means the configured diagnostic supplies affirmative evidence for
  the narrowly stated claim;
- `contradicts` means configured statistical or practical thresholds supply
  evidence against the claim;
- `inconclusive` means neither conclusion is justified.

P-values remain numeric evidence. They are not converted into truth labels.
For dependence and shape checks, failure to reject remains `inconclusive`, not
support for IID or Gaussianity. Practical thresholds and p-value thresholds are
both stored in each evidence item.

Evidence identifiers are SHA-256-derived from the complete normalized evidence
payload except the identifier and observation time. Warnings, assumptions,
method effects, uncertainty, configuration, structured output, and provenance
therefore affect identity. Return fingerprints hash the exact little-endian
float64 shape and bytes. This supports audit linkage; it is not authentication
or a privacy mechanism.

`ResearchProvenance` separately records trials, parameter searches, searched
markets or windows, retention of failed trials, knowledge of trial dependence,
and completeness. Returns cannot reconstruct missing selection history.
`MethodDecision` separates global workflow abstention from claim-specific
abstentions such as an unavailable multiplicity-adjusted claim.

Backward-compatible Phase 0 `support` and `contradiction` enum values remain
accepted for serialized records. New diagnostics emit `supports`,
`contradicts`, or `inconclusive`.
