# Deterministic diagnostics

Diagnostics target inference-relevant properties, not an exact
data-generating-process label. Every threshold, lag set, and minimum sample is
loaded from `configs/diagnostics/default.yaml`.

## Data quality

The quality stage checks one-dimensional finite input, sample size, variance,
value concentration and repetition, and optional timestamp duplicates,
missingness, ordering, and spacing irregularity. It never imputes or silently
drops observations. Repetition is a warning indicator for stale or smoothed
prices, not proof of either mechanism.

## Dependence

Linear-return and squared-centered-return dependence are separate evidence
items. The maximum configured lag defines `h`; the calculation includes every
lag from one through `h`, while retaining the configured reporting lags as
metadata. Sample autocorrelation centers the series once and divides every lag
cross-product by the full lag-zero sum of squares. The Ljung–Box statistic is

`Q = n(n + 2) sum_{k=1}^h(rho_k^2 / (n - k))`.

The reference uses chi-square degrees of freedom `h`; no fitted-model parameter
adjustment is made. A maximum lag at or above the sample size, insufficient
sample, or constant transformed series produces inconclusive evidence. The
chi-square reference is asymptotic. Dependence is material when the
configured significance threshold or practical autocorrelation bound is
crossed. Otherwise the result is inconclusive: failure to reject does not
prove independence. Squared-return evidence suggests volatility clustering but
does not identify GARCH or another specific volatility model.

## Distribution shape

Skewness and raw Pearson kurtosis use plug-in central moments; Gaussian raw
kurtosis is three. Jarque–Bera uses
`n/6 * (skewness^2 + (raw_kurtosis - 3)^2 / 4)` and an asymptotic chi-square
reference with two degrees of freedom. Material skewness, heavy raw kurtosis,
or the configured omnibus threshold weakens Gaussian inference. The heavy-tail
cutoff is a heuristic warning rather than a tail-index test. Finite-fourth-
moment plausibility is always explicitly heuristic: a finite sample cannot
prove that a population fourth moment exists or does not exist.

## Stability

The lightweight structural-change check compares configurable split-sample
means using a standard-error-scaled difference and compares sample variances by
ratio. Both segments must meet the configured minimum size. Mean and variance
triggers are recorded separately, so conflicting component evidence remains
visible. These cutoffs are policy thresholds, not calibrated change-point
tests. The diagnostic is sensitive to the configured split: an abrupt break
near another location or gradual drift may have lower power. No detected split
change proves neither stationarity nor absence of drift; detected instability
risk does not identify an exact change point or structural-change model.

## Provenance

Selection provenance is optional and typed. `complete` requires trial and
search counts, failed-trial retention, trial-dependence status, searched
universes, and searched windows. Missing or partial provenance generates
explicit uncertainty. Multiplicity-sensitive claims abstain separately when
complete provenance or an audited adjustment is unavailable; valid ordinary
single-series inference continues. DSR is not implemented in Phase 2.

## Threshold classification

| Configuration | Classification |
| --- | --- |
| dependence and distribution `significance_level` | statistical significance level |
| `near_zero_variance`, minimum samples, minimum segment size | numerical validity requirement |
| method minimum samples, practical autocorrelation, stability cutoffs, method priority, instability abstention | routing or diagnostic policy threshold |
| repetition, concentration, irregular spacing, skewness, and heavy-kurtosis cutoffs | heuristic warning threshold |

HAC bandwidth, bootstrap block length, replications, and seed are mandatory
method configuration rather than inferred thresholds. Sensitivity significance
is one minus the configured confidence level. No other numerical decision
cutoff is hidden in the implementation. Schema bounds for probabilities,
positive counts, increasing lags, and a split fraction strictly between 0.1 and
0.9 are numerical/configuration validity requirements. Incomplete-gamma
iteration limits and floating-point tolerances are numerical convergence
safeguards and never classify evidence.
