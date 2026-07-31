# Phase 2 plan and implementation boundary

Phase 2 establishes the strongest deterministic, framework-independent routing
baseline before any agent work.

The implemented pipeline is:

`returns + optional timestamps/provenance -> validation -> diagnostics -> typed evidence -> eligibility -> deterministic route -> primary/sensitivity inference -> audit result`

Configuration is split into diagnostic thresholds, conservative routing policy,
and a lightweight smoke experiment. Unknown keys are rejected. HAC bandwidth,
bootstrap block length, replications, seed, confidence level, benchmark, and
provenance requirement are explicit. Stability also requires explicit minimum
segment size. Method priority rejects missing or unavailable executable
methods. Default commands do not run large simulations.

The method catalog describes IID Gaussian, Mertens/PSR, HAC, circular-block
bootstrap, sensitivity comparison, abstention, and future DSR requirements.
Phase 2 reuses the audited Phase 1 ordinary Python functions for all inference.

Evaluation hooks expose valid- and invalid-method selection rates, abstention,
unnecessary abstention, oracle agreement, evidence completeness, and routing
consistency. A full synthetic benchmark is deferred.

## Out of scope

- LLM APIs, LangGraph, LangChain, agents, UI, or network access;
- exact data-generating-process identification;
- DSR without fully specified provenance and a separate mathematical audit;
- HAC or bootstrap power calculations;
- universal guarantees for asymptotic diagnostics or bootstrap validity;
- automatic frequency inference or implicit annualization;
- a broad structural-break library or full synthetic benchmark.

The Phase 2 audit corrected the incomplete-gamma upper-tail calculation,
standardized Ljung–Box accumulation over every lag through `h`, made evidence
aggregation order-invariant, prevented IID methods from becoming primary after
mere non-rejection, separated multiplicity abstention from ordinary inference,
and removed a hidden PSR comparison threshold. Fixed fixtures cover trusted
statsmodels Ljung–Box outputs, IID/skewed/AR/GARCH-like data, abrupt mean and
variance breaks, gradual drift, repetition, insufficient samples, provenance,
and contradictory evidence.

Remaining review choices are policy rather than hidden mathematics: practical
autocorrelation bounds, diagnostic and method minimum samples, stability
cutoffs and split location, repetition warnings, HAC bandwidth, and bootstrap
block length/replications.
