# Phase 3 benchmark metrics

## Phase 3B uncertainty and paired reporting

Every binomial metric carries a two-sided Wilson score interval and its
explicit numerator and denominator. This applies to valid and invalid
selection, abstention, harmful non-abstention, unnecessary abstention,
selective coverage, oracle agreement, and execution failure. Coverage remains
not applicable when the target is undefined and is conditional on a produced
interval.

Mean regret and runtime include Monte Carlo standard errors and normal
intervals. Scenario rows use draw-level observations; aggregate rows use eight
equal-weight scenario means and are labeled scenario-clustered. These summaries
do not treat the four systems sharing one draw as independent evidence.

Router comparisons with always-IID-Gaussian, always-HAC, and the oracle are
paired on `(scenario_id, evaluation_seed)`. Coverage pairs require both systems
to have coverage-applicable output for a defined target. Paired files state
cell counts, exclusions, denominator definitions, standard errors, intervals,
and aggregation level. No p-values are added solely for presentation.

Metrics are computed from one ledger record per scenario, evaluation seed, and
routing system. Coverage uses the configured confidence level and is evaluated
only when a stable per-period target is defined.

| Metric | Definition |
| --- | --- |
| Valid-method selection rate | Successful analytically eligible selections divided by all evaluation rows. Abstentions and execution failures remain in the denominator. |
| Invalid-method selection rate | Ineligible, target-inapplicable, or failed non-abstaining actions divided by all non-abstaining rows. |
| Confidence-interval coverage | Covered intervals divided by non-abstaining rows with a defined target. A failed attempt is uncovered. |
| Selective coverage | The same coverage calculation, explicitly interpreted as conditional on non-abstention and a defined target. |
| Abstention rate | Fraction of runs with global abstention. |
| Harmful non-abstention | Failure to abstain when the stable target is undefined or no method is analytically eligible. |
| Unnecessary abstention | Abstention when the target is defined and at least one method is analytically eligible. |
| Oracle agreement | Fraction whose action, including abstention, equals the regime-level oracle action. |
| Routing regret | Per-draw selected-action loss minus the oracle-action counterfactual loss. |
| Runtime | Wall-clock routing/workflow and inference time, stored separately where the interface permits. |

Every generated metric row emits its numerator and denominator alongside the
rate. Failed executions remain in run, non-abstention, invalid-selection, and
failure denominators. On a defined target they enter coverage as uncovered but
cannot enter the interval-width denominator because they produce no interval.
Structural-break rows have
`target_defined=false`, `coverage_applicable=false`, and `covered=null`; they
never enter ordinary coverage or width denominators.

Routing regret is

$$
\operatorname{Regret}
=
L(\text{selected action})-L(\text{oracle action}).
$$

The configured loss is

$$
L = w_c e_c + w_w \widetilde W + w_i I + w_h H + w_u U
  + w_f F + w_a A + w_t \widetilde T,
$$

where $e_c$ is coverage error, $\widetilde W$ is normalized interval width,
$I$ is invalid selection, $H$ is harmful non-abstention, $U$ is unnecessary
abstention, $F$ is execution failure, $A$ is the explicit base abstention cost,
and $\widetilde T$ is normalized runtime. Width is divided by
$\max(s_W,|S|)$ and runtime by $s_T$, where positive scales $s_W$ and $s_T$
are mandatory loss-profile fields. Coverage-inapplicable rows contribute no
coverage term; their null `covered` state distinguishes exclusion from observed
zero coverage. Oracle training uses aggregate empirical coverage error; ledger
rows retain per-draw components for auditability.

The frozen regime-level oracle is not per-draw clairvoyant. Consequently, a
selected action can have a negative held-out per-draw loss difference when it
happens to outperform the frozen oracle action on that draw. Such values are
retained. Only differences within $10^{-12}$ of zero are normalized to exact
zero, and the ledger stores both selected and oracle losses so the identity can
be checked directly. A materially negative value must therefore be explained
as held-out sampling variation, not hidden by clipping.

The weights in `loss-profiles.yaml` are policy choices. The validity-first
profile emphasizes invalid selection, coverage, and harmful non-abstention.
The efficiency-balanced profile gives more weight to width, unnecessary
abstention, and runtime. Neither is universally correct.
