# Deterministic routing

The router maps typed evidence to method eligibility, then chooses the first
eligible non-sensitivity-only method in the configured priority. It never
optimizes for the narrowest interval or most favorable conclusion. The output
records every eligible and ineligible method, reasons, evidence references,
the primary method, sensitivity methods, unresolved conflicts, information
requests, warnings, abstention, and the tie-break rule.

## Rule table

| Method | Required evidence/configuration | Contraindications | Weakening or sensitivity-only conditions | Abstention relationship |
| --- | --- | --- | --- | --- |
| IID Gaussian | valid nondegenerate data, configured minimum sample | detected linear or squared-return dependence | material non-Gaussianity; sensitivity-only unless independence is affirmatively supported | never promoted to primary merely after non-rejection |
| Mertens/PSR | valid nondegenerate data, configured minimum sample | detected linear or squared-return dependence | sensitivity-only unless independence is affirmatively supported | never treated as dependence robust |
| HAC | adequate sample and explicit bandwidth below sample size | invalid data, degenerate variance, severe instability under conservative policy | weakened under instability when policy permits continued analysis | no universal robustness claim |
| Circular-block bootstrap | adequate sample, explicit valid block length, replications, and seed | invalid data, degenerate variance, severe instability under conservative policy | weakened under instability when policy permits continued analysis | never preferred from its result or interval width |

The priority configuration must contain every executable method exactly once;
unknown or unavailable methods fail configuration validation.

## Policy summary

- IID Gaussian inference is ineligible under material linear or squared-return
  dependence and becomes sensitivity-only under material non-Gaussianity or
  when independence evidence is merely inconclusive.
- Mertens/PSR is an IID non-Gaussian approximation and is ineligible as a sole
  method under temporal dependence. PSR is frequentist, not a Bayesian
  posterior probability.
- HAC requires an adequate sample and explicit bandwidth. It uses the Sharpe
  influence function, so its Bartlett/Newey–West long-run variance includes
  dependence in returns and squared returns. It is not valid under arbitrary
  nonstationarity.
- Circular-block bootstrap requires explicit block length, replications, and
  seed. It has no hidden random or expensive default and makes no universal
  validity claim.
- Invalid input, degenerate variance, no primary-valid method, or configured
  severe instability causes global abstention.
- Missing provenance or an unaudited adjustment causes claim-specific
  abstention for multiplicity-adjusted Sharpe claims, plus an information
  request; it does not abort otherwise valid ordinary inference.

No-detected-dependence eligibility is conditional on configured tests and
practical bounds, is sensitivity-only, and carries the explicit caveat that IID
was not proved. Contradictory evidence is aggregated conservatively and cannot
be erased by input ordering; all method references resolve to returned evidence
identifiers.

## Abstention classification

| Trigger | Classification |
| --- | --- |
| invalid/nonfinite data or degenerate variance | global workflow abstention |
| method sample/configuration/assumption failure | method-specific ineligibility |
| missing provenance or unimplemented audited multiplicity adjustment | claim-specific abstention |
| incomplete provenance | request for additional information |
| multiple defensible methods | sensitivity escalation |
| substantive conclusion disagreement | explicit unresolved sensitivity result |
| severe instability | policy-driven global abstention, or qualified weakened methods when configured |

## Sensitivity

All remaining methods in configured priority are run as sensitivity methods.
`ConclusionComparison` records whether confidence intervals, tests, or
frequentist PSR summaries support exceeding the supplied benchmark. Missingly
comparable outputs remain `null`; disagreement is surfaced and never hidden.
Test significance is one minus the configured confidence level, and the
frequentist PSR probability cutoff is its complement. Both values are included
in the comparison output.

Annualization is not inferred. The Phase 2 workflow operates on per-period
excess returns and an unannualized benchmark. Phase 1 annualized descriptive
Sharpe remains available only through an explicit `periods_per_year` argument.
Every workflow result records `sample_frequency: per-period`,
`annualization_convention: not-annualized`, `benchmark_scale: per-period`, and
the applicable data fingerprint and provenance reference.
