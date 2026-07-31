# Scientific claims and evidence status

## Implemented and directly inspectable

These are implementation facts, not benchmark findings:

- Phase 1 exposes audited Sharpe estimation and inference as ordinary Python
  functions: IID Gaussian, Mertens/PSR, influence-function HAC, circular-block
  bootstrap, and Gaussian power/sample-size calculations.
- Phase 2 produces typed diagnostic evidence for data quality, dependence in
  returns and squared returns, distribution shape, split stability, and
  provenance; it applies deterministic eligibility, sensitivity, and
  abstention policy.
- Routing is deterministic, auditable, and configurable; diagnostic evidence
  is typed rather than a free-form method recommendation.
- Failure to reject is not treated as affirmative assumption support. Linear
  and squared-return dependence remain distinct, and incomplete provenance
  produces claim-specific rather than unnecessary global abstention.
- Annualization, frequency, HAC bandwidth, bootstrap block length,
  replications, seed, thresholds, and routing priority are explicit.
- Phase 4A implements exactly four typed, provider-neutral agent roles with
  deterministic mock execution and an optional injected live-provider shell.
  No LangGraph/LangChain dependency, UI, default network execution, or
  comparative agent-performance evidence is implemented. Phase 3A remains a
  bounded synthetic benchmark slice, not the full preregistered benchmark.
- Phase 3A now provides a configuration-driven four-family benchmark, an
  independent eligibility registry, shared-draw baseline evaluation, a
  train/evaluation-separated empirical oracle, and auditable output ledgers.
  Infrastructure and smoke correctness are implementation facts. One audited
  mini run now provides preliminary, descriptive, low-resolution results only;
  it is insufficient for a general superiority claim.
- Phase 3B adds a validation layer over the same scenarios: Wilson intervals,
  paired shared-draw effects, scenario-clustered regret/runtime uncertainty,
  both loss profiles, oracle-stability labels, deterministic seed namespaces,
  and a confirmation freeze mechanism. These are implementation facts; no
  Phase 3B performance claim exists until a qualifying output is reviewed.

## Phase 3B claim gates

Pilot results are `smoke-only`. Unreviewed larger runs may be described only as
`preliminary-descriptive` with exact scenario, loss-profile, denominator, and
interval scope. `benchmark-supported` requires at least the registered
hackathon evaluation size and explicit independent review.
`confirmation-supported` additionally requires a valid frozen critical-state
manifest and previously unseen confirmation seeds. None of these levels imply
exact DGP recovery, real-data calibration, or an agent comparison.

## Audited Phase 3B hackathon evidence

The version-0.2 hackathon ledgers passed independent reconstruction of row
counts, shared draws, seed separation, selection, selective coverage,
abstention, losses, regret, and oracle stability. They support reviewed
preliminary descriptions bounded to the eight scenarios. They do not support a
general superiority or calibration claim. The router-versus-HAC
scenario-clustered invalid-selection interval includes zero; the router's
pooled selective-coverage Wilson interval is below nominal 95% at its upper
endpoint; and both IID empirical-oracle choices are unstable. The saved
version-0.2 router-versus-Gaussian paired-coverage rows used an over-restrictive
eligibility filter and must not be cited. Version 0.3 corrects future summaries
without overwriting the scientific runs.

## Preliminary Phase 3A evidence

Run `20260718T005503717951Z-mini` uses eight evaluation draws per scenario and
eight disjoint oracle-training draws per scenario. Its aggregate rates can be
reconstructed from the 256-row ledger, but each scenario-level proportion has
12.5-percentage-point resolution. The run is useful for checking direction,
accounting, and failure modes; it does not establish universal calibration,
robustness, or superiority. Its numbers must not be promoted into the AAAI
abstract as confirmatory evidence.

## Phase 3 claims to evaluate

The following are hypotheses, not current results.

| Proposed claim | Independent variable | Dependent variable | Baseline | Required experiment | What would falsify it |
| --- | --- | --- | --- | --- | --- |
| Typed evidence routing reduces invalid procedure selection. | Selection system, crossed with hidden family, regime, and sample size. | Invalid-method selection rate. | Fixed methods and hand-authored decision tree. | Repeated simulations from the preregistered closed-world benchmark with identical observations and method access. | ERI is no better than the strongest non-oracle baseline, or improves only by abstaining indiscriminately. |
| Abstention improves conditional reliability. | Abstention enabled versus disabled or threshold-matched policy. | Selective risk and conditional coverage/type-I error, reported with non-abstention. | Same router forced to choose and fixed methods. | Paired benchmark evaluation with matched data and explicit abstention-cost sensitivity. | Conditional reliability does not improve, or gains require practically unacceptable unnecessary abstention. |
| Typed evidence improves selective calibration. | Selection system and benchmark condition. | Conditional coverage error and conditional type-I error among non-abstentions. | Fixed methods, decision tree, and deterministic-router ablations. | Evaluate intervals and tests against known simulation truth while reporting non-abstention jointly. | Conditional calibration is not improved, or any apparent gain disappears at matched non-abstention rates. |
| Uncertainty-aware evidence improves the validity/coverage trade-off. | Full evidence representation versus ablations that remove uncertainty, practical bounds, or conflict handling. | Harmful non-abstention, unnecessary abstention, coverage error, and width. | Deterministic router with the corresponding fields removed. | Paired simulations using identical draws and policy-weight sensitivity analysis. | Ablations equal or dominate the full representation across plausible loss weights. |
| ERI approaches oracle routing performance. | Selection system across preregistered regimes. | Expected-loss regret and oracle agreement. | Fixed methods, decision tree, and regime-level simulation oracle. | Estimate oracle actions on independent simulations and score held-out shared draws. | ERI regret is not lower than non-oracle baselines or oracle agreement reflects only a dominant always-select action. |
| Evidence routing remains useful under reasonable policy choices. | Diagnostic thresholds, routing priorities, and oracle loss weights. | Ranking, regret, calibration, and abstention metrics. | Default policy and alternative preregistered policies. | Multiverse/sensitivity analysis without selecting settings from favorable outcomes. | Conclusions reverse across ordinary policy ranges or depend on one tuned configuration. |
| The architecture is robust to controlled distribution shift. | Development family set versus held-out family or parameter shift. | Change in invalid selection, selective risk, regret, and abstention. | Fixed methods, decision tree, and in-distribution ERI result. | Freeze policy, then evaluate unseen mechanisms or parameter ranges without retuning. | ERI degrades to or below simple baselines, or fails to increase abstention when support weakens. |

## Future agent claims

Only after the deterministic benchmark is fixed should the project test the
claim that structured evidence generation outperforms free-form reasoning
under identical tool and token access.

| Proposed claim | Independent variable | Dependent variable | Baseline | Required experiment | What would falsify it |
| --- | --- | --- | --- | --- | --- |
| A structured evidence agent improves selection over free-form reasoning. | Prompting/orchestration condition with identical tools, information, model, budget, and statistical backend. | Invalid selection, selective calibration, evidence completeness, consistency, cost, and abstention. | Free-form reasoning and the deterministic router without an agent. | Blinded repeated trials on held-out open-world tasks after Phase 3, with outputs reduced to the same typed contracts. | No robust improvement over free-form reasoning, or gains result from unequal tools, information, or compute. |
| Agentic evidence acquisition helps when evidence is incomplete, contradictory, or partly natural language. | Static evidence versus controlled acquisition access. | Correct information requests, evidence completeness, harmful non-abstention, cost, and regret. | Deterministic router given the initial evidence only. | Held-out tasks where useful evidence can be acquired through a matched, logged tool interface. | Acquisition does not improve decisions net of cost or introduces unsupported evidence. |
| Evidence-constrained agents outperform unconstrained multi-agent discussion. | Typed evidence/decision constraints versus unconstrained discussion. | Invalid selection, selective risk, consistency, abstention, and cost. | Unconstrained multi-agent discussion with the same models, tools, information, turns, and tokens. | Randomized repeated open-world evaluation with blinded scoring and fixed statistical backend. | Constrained agents do not improve reliability or merely consume more effective compute/context. |

No future-agent claim attributes novelty to LangGraph. A graph framework may
provide execution, persistence, or observability, but the scientific variables
are evidence structure, uncertainty representation, and decision behavior.

## Phase 4A implementation claims

It is now an implementation fact that four specialized agents can propose
bounded diagnostic requests, extract explicitly supported provenance, cite
typed evidence when challenging a route, request eligible sensitivities, and
generate a traceable report. A deterministic adjudicator validates every
proposal and the existing router remains final authority.

No comparative performance claim follows. Phase 3B measures the deterministic
router, not Phase 4A agents. Agent effects on calibration, selection validity,
abstention, coverage, regret, or scientific productivity remain unevaluated.

## Phase 4B preregistered claims

Phase 4B tests four falsifiable propositions about full information, adaptive
budgeted observation, open-world provenance, and deterministic constraints.
The three registered primary outcomes are invalid primary selection, harmful
non-abstention, and routing regret. The benchmark may find no advantage, an
advantage confined to incomplete information, or degradation from agency.

Until a reviewed live deadline run exists, the strongest permitted statement
is implementation-only. A live pilot supports cost and feasibility language
only. Mock behavior is never agent-performance evidence.
## Phase 4E operational evidence boundary

Successful Alibaba connectivity, the planner probe, the single integration
case, and the completed three-cell operational micro-pilot support only
compatibility and feasibility statements. The completed gates used zero retries
and repairs. They do not
support benchmark, confirmation, general reliability, or comparative
agent-performance claims. Three cells are not used to construct inferential
confidence intervals. The 18-cell pilot remains separately gated and
unauthorized.
