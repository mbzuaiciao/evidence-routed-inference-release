# Research question

## Motivation and testbed

Procedure selection is part of inference, yet software and free-form reasoning
often treat assumptions as unstructured caveats after a method has already
been chosen. In finite samples, observed returns rarely identify an exact
data-generating process (DGP). ERI therefore does **not** attempt exact DGP
recovery. It detects inference-relevant properties—such as linear and
squared-return dependence, non-Gaussian shape, instability, data defects, and
missing selection provenance—and represents their support, conflict, and
uncertainty as typed evidence.

Sharpe-ratio inference is the initial testbed because familiar estimators lead
to materially different uncertainty procedures under IID Gaussian,
non-Gaussian, dependent, unstable, and post-selection conditions. Typed
evidence makes those distinctions machine-checkable. Abstention is a first-
class action when no procedure or claim is adequately supported, rather than a
failure to return an answer.

The broader AI question is whether constraining reasoning through explicit
evidence and decision contracts improves reliable tool selection compared with
free-form reasoning. The deterministic router must be benchmarked first: it is
the reproducible controller, establishes what evidence alone achieves, and
prevents later agent gains from being confused with statistical or policy
changes.

## Primary question

> Under finite-sample model ambiguity, does decomposing statistical assumptions into typed, uncertainty-aware evidence reduce invalid procedure selection and improve selective calibration relative to free-form reasoning under identical tool access?

Selective calibration is evaluated **conditional on non-abstention**. It asks
whether reported confidence intervals and tests have their intended behavior
among cases where a system elects to make an inferential claim. Coverage alone
is insufficient: a system could appear calibrated by abstaining on nearly
everything, so coverage is reported with non-abstention, harmful
non-abstention, and unnecessary-abstention rates.

## Secondary questions

1. Which diagnostic evidence and uncertainty fields most reduce invalid method
   selection across dependence, non-Gaussianity, instability, and provenance
   ambiguity?
2. How does deterministic evidence routing trade interval efficiency against
   harmful non-abstention and unnecessary abstention relative to fixed methods,
   a hand-authored decision tree, and a simulation oracle?
3. Under identical tools and information, does future structured evidence
   generation improve over monolithic free-form reasoning and unconstrained
   multi-agent discussion without changing the audited statistical functions
   or deterministic eligibility policy?

## Scope

Phases 1 and 2 implement the statistical functions, typed diagnostics, and
deterministic router needed to pose the question. They do not answer it.
Phase 2.5 fixes the claims, benchmark, baselines, metrics, and threats before
Phase 3 simulation. Agent comparisons are deferred until the deterministic
benchmark is established.

## Phase 4B refinement

With the deterministic benchmark and guarded agent layer implemented, the
next primary question is narrower: **under what information conditions does
agentic evidence acquisition improve statistical procedure selection?** The
comparison separates full information, budgeted numeric evidence, and open-
world provenance. It includes an adaptive deterministic baseline so any gain
from choosing another diagnostic is not automatically attributed to agency.

This is a preregistered question, not a superiority claim. Mock runs validate
implementation only; scientific comparison requires matched live-model runs.
