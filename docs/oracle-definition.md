# Phase 3 oracle definition

The oracle is a regime-level empirical comparator, not a system that merely
knows the process-family label and not a per-draw clairvoyant selector.

For each scenario—defined by family, parameter regime, and sample size—the
oracle:

1. obtains method labels from the versioned benchmark registry, independently
   of diagnostic or router output;
2. excludes every method labeled ineligible or target-not-well-defined;
3. generates each oracle-training draw once and reuses that immutable draw for
   every remaining action;
4. estimates expected loss under the selected explicit loss profile;
5. selects minimum expected loss, breaking exact ties by the fixed registry
   order; and
6. freezes that action before final evaluation draws are generated.

Oracle-training and evaluation seeds must be nonempty, unique, and disjoint;
configuration validation rejects overlap. All evaluated routing systems share
each final simulation draw. For a structural-break scenario with no stable
full-sample target, ordinary methods are target-not-well-defined and abstention
is the registered oracle action.

Expected oracle loss uses empirical coverage error across repeated training
draws rather than interpreting a single covered interval as calibrated. An
analytically ineligible method cannot win through favorable finite Monte Carlo
noise. Changing loss weights can change the oracle; this is a policy
sensitivity, not a contradiction.

For Gaussian AR(1), nonzero autoregression makes IID-only primary inference
ineligible. For the registered Gaussian-innovation GARCH(1,1) process, HAC and
block-bootstrap primary eligibility additionally requires

$$
3\alpha^2 + 2\alpha\beta + \beta^2 < 1,
$$

the finite-fourth-moment condition under standard-normal innovations. If that
condition fails, IID, Mertens, HAC, and block-bootstrap actions are ineligible
and registered abstention is the only eligible action. These are benchmark
truth labels used for scoring; the router never receives the process-family or
eligibility label and is not claimed to identify the exact DGP.

The oracle is frozen using training draws, so a non-oracle action may beat it
on an individual held-out draw. The reported per-draw difference remains
$L(\text{selected})-L(\text{oracle})$ and is not clipped. This sampling effect
does not imply evaluation leakage or make the oracle per-draw clairvoyant.

The smoke configuration uses two training seeds only to validate correctness.
It is not precise enough for a scientific oracle-performance claim. The mini
configuration remains descriptive and must report its limited Monte Carlo
resolution.

## Phase 3B stability and confirmation

Phase 3B repeats oracle estimation across deterministic, disjoint seed blocks
at increasing training sizes. It records action losses, the best-versus-second
best margin, action-change frequency, practical ties, and a `stable`,
`weak-margin`, `unstable`, or `unique eligible action` status. These labels
describe the tested blocks and sizes; they do not prove universal optimality.

The final oracle for each profile and tier uses only that tier's registered
oracle-training draws and is frozen before evaluation. Development, hackathon,
AAAI, and frozen-confirmation evaluation seeds cannot affect oracle choice. A
confirmation-supported run additionally requires an unchanged critical-state
manifest hash.
