# Phase 3 benchmark design

## Phase 3B validation layer

Phase 3B does not expand the Phase 3A design. It locks the eight scenario IDs,
the four evaluated systems, and the validity-first and efficiency-balanced
profiles. Three explicit tiers increase oracle-training and evaluation draws:
a 25/25 scaling pilot, a 250/250 hackathon run, and a 1,000/2,000 AAAI design.
The pilot is operational validation only.

Scientific seeds are allocated from versioned, deterministic, disjoint
namespaces for oracle training, development evaluation, hackathon evaluation,
AAAI evaluation, and frozen confirmation. Systems share each evaluation draw,
while oracle actions are selected from separate training draws before
evaluation. Wall-clock time may name a run directory but never selects a
scientific draw.

Phase 3B reports each loss profile separately. Aggregate summaries are
supplemented by scenario-level eligibility, oracle action and stability,
selection, coverage, abstention, regret, runtime, and paired effects. The
benchmark remains a controlled closed world and cannot establish exact DGP
recovery or generalization to unregistered processes.

## Purpose and unit of evaluation

The benchmark tests procedure selection under finite-sample model ambiguity.
Each replicate hides the generating family and regime from the evaluated
system, supplies the same observed series and declared metadata to every
baseline, and retains simulation truth only for scoring. Regime factors include
sample size, effect size, dependence strength, tail shape, break magnitude, and
provenance completeness. Seeds and configurations are recorded.

## Hidden data-generating families

| Family | Relevant properties | Generally eligible | Generally ineligible or qualified | Sensitivity / expected abstention | Targeted failure mode |
| --- | --- | --- | --- | --- | --- |
| IID Gaussian | independent, stationary, finite moments | IID Gaussian; Mertens/PSR; HAC and block bootstrap as configured sensitivities | none solely from family | low abstention at adequate `n` | unnecessary robustness or abstention |
| IID skewed | independent, non-Gaussian, finite fourth moment | Mertens/PSR; HAC; bootstrap | Gaussian primary is weakened | compare Gaussian sensitivity; abstain if moments/sample support are inadequate | Gaussian undercoverage |
| IID heavy-tailed, finite fourth moment | independent, high raw kurtosis | Mertens/PSR when numerically adequate; HAC/bootstrap with explicit settings | Gaussian primary weakened | escalate sensitivity; abstain for degenerate/unstable samples | tail-driven undercoverage |
| AR | linear serial dependence | HAC; circular-block bootstrap | IID Gaussian and Mertens/PSR as sole primary | dependence-strength and bandwidth sensitivity | invalid IID selection |
| MA | finite-order linear dependence | HAC; circular-block bootstrap | IID-only primary | lag/bandwidth sensitivity | missed short-memory dependence |
| AR plus volatility dependence | dependence in returns and squared returns | HAC influence-function inference; block bootstrap | IID-only primary | bandwidth/block-length sensitivity | checking returns but not squared returns |
| GARCH-like | volatility clustering, potentially weak linear dependence | HAC or block bootstrap under stationary regimes satisfying the registered finite-fourth-moment condition | IID-only primary; registered robust methods outside their moment conditions | abstain/qualify near persistence or moment boundaries | treating no linear autocorrelation as IID |
| Stochastic-volatility-like | latent volatility dependence | HAC or block bootstrap when stationary/regular | IID-only primary | sensitivity across persistence and tail regimes | unobserved variance dependence |
| Mean structural break | nonstationary mean | no current method receives universal validity | stationary methods qualified or ineligible by policy | global abstention under severe instability | confident inference across a mean break |
| Variance structural break | nonstationary variance | no current method receives universal validity | stationary methods qualified or ineligible by policy | global abstention under severe instability | confident inference across a variance break |
| Gradual drift | nonstationarity with weak split signal | no universally valid current method | stationary methods qualified | expected abstention may be imperfect; vary drift location/slope | false reassurance from one split check |
| Repeated or smoothed observations | stale/rounded/smoothed process risk | case-dependent robust sensitivity only | methods relying on unqualified raw-return interpretation | warning and possible abstention when variance/quality is compromised | artifacts mistaken for precision |
| Complete selection provenance | known search universe, windows, trial counts, failures, and dependence | ordinary inference; future audited multiplicity procedure if implemented | none solely from provenance | sensitivity remains method-dependent | provenance ignored despite availability |
| Incomplete selection provenance | unknown selection/search multiplicity | ordinary single-series inference may remain eligible | multiplicity-adjusted claim | claim-specific abstention and information request | overstated post-selection evidence |

Eligibility in the table is a benchmark expectation, not a family label made
available to the router. Finite samples can produce conflicting or
inconclusive evidence. The benchmark must include boundary regimes where the
correct action is sensitivity escalation or abstention rather than effortless
family identification.

## Baselines

- **Fixed procedures:** always IID Gaussian, always Mertens/PSR, always HAC
  with preregistered bandwidth, and always block bootstrap with preregistered
  block length and replications.
- **Decision tree:** a transparent hand-authored rule using the same diagnostic
  outputs but without the full typed conflict/uncertainty representation.
- **Deterministic ERI router:** the Phase 2 system and preregistered ablations.
- **Simulation oracle:** for each hidden family, parameter regime, and sample
  size, select the eligible procedure minimizing expected loss estimated over
  independent repeated simulations. The oracle uses generating truth only for
  benchmark scoring and is never a deployable method or per-replicate
  clairvoyant selector.

## Oracle loss

For policy weights `w`, expected loss is

`w_c * coverage_error + w_w * normalized_width + w_i * invalid_selection + w_h * harmful_nonabstention + w_u * unnecessary_abstention + w_k * optional_cost`.

Coverage error measures deviation from nominal coverage; width is normalized
within a scientifically meaningful scale. Invalid selection penalizes choosing
a method outside the preregistered eligibility set. Harmful non-abstention
penalizes issuing an inferential claim in a regime where no current method is
adequately supported. Unnecessary abstention applies when at least one method
is eligible and adequately calibrated. Optional cost covers computation or
latency only if measured consistently. Weights are policy choices, not learned
truth: report multiple preregistered weight profiles, Pareto summaries, and
weight sensitivity rather than one favorable scalar ranking.

## Metrics

- valid-method selection, invalid-method selection, and oracle agreement;
- coverage error, type-I error, power, and normalized interval width;
- selective coverage and selective type-I error conditional on
  non-abstention;
- non-abstention, harmful non-abstention, unnecessary abstention, and
  claim-specific abstention;
- evidence completeness, contradiction rate, routing consistency, and
  sensitivity disagreement;
- selective risk, routing/oracle regret, and, when applicable, latency or
  computational cost.

Every selective metric is accompanied by its denominator and non-abstention
rate. Results are stratified by family, regime, and sample size, with Monte
Carlo uncertainty and paired comparisons using common random numbers.

## Closed world and future open world

Phase 3 is a **closed-world** benchmark: families, parameter grids, eligibility
rules, loss components, policies, and baselines are frozen before final runs.
It measures controlled selection, not universal robustness. A later
**open-world** benchmark may introduce unseen mechanisms, ambiguous metadata,
natural datasets, or evidence-generation agents. Open-world cases must be held
out from policy construction and must preserve identical tool and information
access across compared reasoning systems.

## Implemented Phase 3A slice

Phase 3A implements only IID Gaussian, AR(1), GARCH(1,1)-style, and structural
break families, evaluated by always-IID-Gaussian, always-HAC, the unchanged
deterministic router, and a regime-level empirical oracle. It uses shared
draws, disjoint oracle-training/evaluation seeds, versioned independent
eligibility labels, explicit loss profiles, and timestamped ledgers.

This infrastructure does not itself establish improved selection or
calibration. Smoke runs validate contracts only. The mini configuration is the
first candidate descriptive experiment and remains much smaller than the full
closed-world design above.

## Phase 4B matched information design

Phase 4B retains the exact eight Phase 3B scenarios and creates information
variants without changing the underlying return draw. Its unit is `scenario ×
return seed × information regime × evidence condition`; all six systems share
fingerprints for returns, initial evidence, prose, tools, budgets, visible
method metadata, benchmark Sharpe, and periodicity. Only the evaluator-only
oracle sees simulation truth.

Regime A supplies full diagnostics, regime B masks diagnostics behind a fixed
budget, and regime C supplies natural-language provenance with evaluator-only
truth. The adaptive deterministic policy is the critical baseline for
distinguishing adaptive testing from LLM reasoning. Deadline configuration is
240 matched cells before system expansion; it is not run automatically.
