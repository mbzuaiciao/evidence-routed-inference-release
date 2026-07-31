# Phase 3B oracle stability

The Phase 3B oracle remains the Phase 3A regime-level empirical comparator. It
is restricted to registry-eligible actions, trained on draws disjoint from all
evaluation namespaces, and frozen before evaluation begins. Evaluation losses
never alter the oracle action.

## Stability experiment

For each scenario and loss profile, deterministic seed blocks repeat empirical
oracle selection at increasing training sizes. The full AAAI design evaluates
8, 25, 50, 100, 250, 500, and 1,000 training draws. Pilot and hackathon tiers
use the prefixes supported by their final oracle sizes.

Each row records:

- the modal selected action and every block action;
- empirical expected losses for eligible actions;
- the minimum and mean best-versus-second-best loss margin;
- action-change frequency across blocks; and
- whether the loss margin is within the configured practical-tie tolerance.

## Status labels

- `unique eligible action`: the registry leaves only one permissible action.
- `unstable`: deterministic blocks select different actions.
- `weak-margin`: blocks agree, but the smallest best-versus-second-best margin
  is no greater than the preregistered practical-tie tolerance.
- `stable`: blocks agree and the margin exceeds that tolerance.

The tie tolerance is a policy-reporting threshold, not a statistical test. A
stable label means stable under the tested sizes and deterministic blocks, not
that the action is universally optimal. Loss profiles can legitimately select
different actions because their weights encode different policy trade-offs.

Overall oracle agreement and regret must be stratified by final stability
status. Agreement on `stable` scenarios, on `unique eligible action` scenarios,
and on `unstable` scenarios answer different questions. Unstable oracle choices
do not invalidate router eligibility or coverage evaluation, but they weaken a
definitive best-method interpretation of agreement and regret. Phase 3B outputs
therefore include agreement numerators, denominators, and rates by stability
stratum.
