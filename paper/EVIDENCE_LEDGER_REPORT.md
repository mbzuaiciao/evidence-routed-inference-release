# Paper Evidence Ledger Report

Status: **proposed — awaiting user review**

This report inventories the canonical quantitative and implementation-grounded
evidence currently permitted for the paper. The machine-readable claim ledger is
[`paper/evidence_ledger.yaml`](evidence_ledger.yaml).

No new experiment was run. Phase 5B values come from the approved deterministic
recomputation in:

`external-artifact://phase5b-v2/`

## 1. Canonical Phase 4J design and accounting

| Quantity | Canonical value | Evidence status |
|---|---:|---|
| Scenario families | 2 | Frozen configuration |
| Return seeds | 3 | Frozen configuration |
| Information regimes | 3 | Frozen configuration |
| Matched cells | 18 | Archive manifest and scientific dossier |
| Systems | 6 | Frozen configuration |
| Intended and retained outputs | 108 | Archive manifest |
| Completed outputs | 89 | Archive manifest |
| Failed outputs | 19 | Archive manifest |
| Abstentions | 50 | Archive manifest |
| Successful completed live outputs | 35 | Archive manifest |
| Provider logical attempts | 118 | Archive manifest |
| HTTP requests dispatched | 118 | Archive manifest |
| Provider responses received | 118 | Archive manifest |

Phase 4J used Alibaba `qwen3.7-max` in the pinned Singapore workspace
`[omitted-release-workspace-id]`, with a 4,096-token output ceiling, 90-second timeout,
zero retries, zero repairs, and thinking disabled.

The benchmark is a single-provider, 18-cell, two-scenario pilot. It is not a
population-representative benchmark.

## 2. Canonical Phase 4J system metrics

Each system has 18 retained cell outcomes.

| System | Mean regret | Invalid | Harmful non-abstention | Claim valid | Abstentions | Complete | Failed | Live calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 0.000 | 0.000 | 0.000 | 1.000 | 9 | 18 | 0 | 0 |
| Adaptive deterministic policy | 0.0013 | 0.000 | 0.000 | 1.000 | 9 | 18 | 0 | 0 |
| Unconstrained specialized agents | 0.682 | 0.056 | 0.000 | 0.944 | 9 | 18 | 0 | 72 |
| Monolithic agent | 4.084 | 0.278 | 0.056 | 0.722 | 9 | 17 | 1 | 34 |
| Static deterministic router | 4.890 | 0.222 | 0.222 | 0.778 | 9 | 18 | 0 | 0 |
| Evidence-constrained Phase 4A agents | 20.375 | 1.000 | 0.000 | 0.000 | 5 | 0 | 18 | 12 |

The constrained-system row is an operational failure result, not a measure of
model reasoning quality. All 18 units failed legacy schema or evidence-snapshot
integration.

The adaptive deterministic policy is the strongest deployable Phase 4J
baseline by observed mean regret. It is also benchmark-specific: it encodes
eligibility-relevant rules and can acquire diagnostics. Its advantage over
static routing was concentrated in structural-break cells:

- adaptive overall mean regret: `0.0013` (canonical value
  `0.001297193836243632`);
- static overall mean regret: `4.890`;
- structural-break static mean regret: `9.778`;
- GARCH static and adaptive mean regret: `0.003`;
- high-loss structural-break cells: 4.

## 3. Canonical Phase 4J paired comparisons

Differences are first system minus static deterministic router. All use 18
matched cells and only two scenario clusters.

| Comparison | Metric | Mean difference | Clustered SE | 95% interval |
|---|---|---:|---:|---:|
| Unconstrained specialized − static | Routing regret | −4.208 | 5.569 | [−15.124, 6.707] |
| Unconstrained specialized − static | Invalid selection | −0.167 | 0.278 | [−0.711, 0.378] |
| Unconstrained specialized − static | Harmful non-abstention | −0.222 | 0.222 | [−0.658, 0.213] |
| Monolithic − static | Routing regret | −0.806 | 4.305 | [−9.244, 7.632] |
| Monolithic − static | Invalid selection | 0.056 | 0.278 | [−0.489, 0.600] |
| Adaptive deterministic − static | Routing regret | −4.889 | 4.889 | [−14.471, 4.693] |

Every clustered interval crosses zero. These are descriptive paired estimates,
not conventional significance findings.

## 4. Canonical Phase 5B condition metrics

Phase 5B uses the preregistered failure treatment:

`routing_regret = max(0, assigned_failed_loss - oracle_loss)`

Each condition retains all 18 cells.

| Condition | Mean regret | Median regret | Invalid | Harmful | Claim valid | Abstention | Failure | Complete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — M-short-2 | 4.723122 | 0.002880 | 0.388889 | 0.166667 | 0.611111 | 0.333333 | 0.055556 | 17/18 |
| B — M-unified-2 | 1.389345 | 0 | 0.111111 | 0.111111 | 0.888889 | 0.611111 | 0.055556 | 17/18 |
| C — M-unified-4 | 0.250287 | 0 | 0 | 0 | 1 | 0.833333 | 0 | 18/18 |
| D — Specialized-4 | 3.598322 | 0.001440 | 0.222222 | 0.111111 | 0.777778 | 0.444444 | 0 | 18/18 |

The two reconstructed provider-schema failures are:

1. Condition A, structural mean-break seed `41100003`, conflicting
   provenance: assigned loss `1.0`, oracle loss `0`, regret `1.0`.
2. Condition B, persistent GARCH seed `41100001`, explicit provenance:
   assigned loss `1.0`, oracle loss `0.246805`, regret `0.753195`.

No selected method is imputed. Invalid-selection and final-claim-validity values
follow fail-closed evaluator semantics. Harmful and abstention flags reproduce
the contemporaneous transcript reconstruction and are not directly observed
model decisions.

## 5. Canonical Phase 5B contrasts

Negative regret differences favor the first condition.

| Contrast | All-cell mean | Completed-pair mean | Completed pairs | 95% cluster-bootstrap interval |
|---|---:|---:|---:|---:|
| C−B | −1.139058 | −1.206062 | 17 | [−2.444444, 0.166328] |
| B−A | −3.333777 | −3.735074 | 16 | [−4.111998, −2.555556] |
| D−C | 3.348035 | 3.348035 | 18 | [1.807181, 4.888889] |
| D−A | −1.124800 | −2.426259 | 17 | [−2.138489, −0.111111] |

Only two scenario clusters are available. These intervals are unstable and
must not be used for decisive population-level significance claims.

### Primary C−B sensitivity boundary

The overall C−B mean favors C, but the mechanism result is fragile:

- largest cell supplies `88.1%` of absolute contrast magnitude;
- trimming that cell changes the mean to `+0.088056`;
- GARCH-specific mean is `+0.166328`;
- structural-break mean is `−2.444444`;
- diagnostic-subset mean is `0`;
- the clustered interval crosses zero.

Accordingly, the ledger does not authorize the claim that sequential execution
is definitively dominant or that C broadly and uniformly outperforms B.

The B−A direction remains negative after trimming and in both scenario
families. D−C remains positive after trimming and in both scenario families.
D−A is heterogeneous; its non-diagnostic-subset mean is exactly zero.

## 6. Retired and sensitivity-only values

The following values are explicitly retired from main-text use:

| Quantity | Retired value | Canonical replacement | Disposition |
|---|---:|---:|---|
| Condition B mean regret | 1.403057 | 1.389345 | Sensitivity/provenance only |
| C−B all-cell mean | −1.152770 | −1.139058 | Sensitivity/provenance only |
| B−A all-cell mean | −3.320066 | −3.333777 | Sensitivity/provenance only |

The retired values directly assigned regret `1.0` to every failure. The
canonical analysis first assigns failed loss `1.0` and then subtracts the cell
oracle, as preregistered.

## 7. Phase 5B operational evidence classes

### Original-artifact-backed

- intended units: 72;
- completed units: 70;
- failed units: 2;
- elapsed time: 2,396.74 seconds;
- planned dispatches: 216.

The 216 value is planned, not observed.

### Transcript-backed only

- attempted calls: 214;
- responses: 212;
- input tokens: 488,640;
- output tokens: 71,820;
- estimated cost: $0.2816.

The recovered original run directory does not contain the raw request/response
ledger needed to independently verify these operational values.

## 8. Implementation provenance

The Phase 5B freeze records:

`3bbc07062fa9cf4fe08ed5713dd2f7ea06edae8b`

This is not a valid Git identifier. All four frozen runtime source hashes match:

`3bbc070df3a43439464d75f3b662285ab6146c88`

This is an implementation-identifier recording error. Historical freeze,
authorization, and run files remain unchanged. The recovery manifest
establishes integrity from recovery time, not execution-time integrity.

The Phase 4J scientific dossier references raw archive
`archives/phase4j-alibaba-qwen37-18cell-compatibility-live-pilot.tar.gz` with
SHA-256
`6bf634651ea48b941488ebcbf3b4f8dd718c54796410af3bfaca7e59e517396`.
That archive is not present in this clean worktree. Phase 4J numbers are
therefore anchored here to the checked-in scientific dossier and archive
manifest; recovering the archive would improve independent artifact-level
traceability.

## 9. Central claims and proposed wording

### Directly supported

**Instruction effect**

> On this 18-cell, single-model benchmark, adding richer unified instructions
> reduced mean routing regret from 4.723 in Condition A to 1.389 in Condition B;
> the paired B−A difference was −3.334.

**Observed Condition C profile**

> Condition C had the lowest observed mean regret (0.250), zero invalid primary
> selections, zero harmful non-abstentions, and full claim validity, while
> abstaining in 83.3% of cells.

**Tested D versus C architecture**

> The tested specialized workflow D had higher mean regret than unified
> sequential C (D−C = 3.348) on this benchmark.

**Deterministic baseline**

> The adaptive deterministic policy was the strongest deployable Phase 4J
> baseline by observed mean regret (approximately 0.0013), with its advantage over static
> routing concentrated in structural-break cells.

### Pilot-descriptive interpretation

> Instruction design appears to explain a substantial portion of the
> improvement over short monolithic prompting.

> Unified sequential execution may prevent severe structural-break errors, but
> the overall C−B contrast is concentrated and changes sign under a
> largest-cell trimming sensitivity.

> The tested specialized architecture was less robust than unified sequential
> C in this benchmark; this comparison does not isolate a pure persona effect.

### Mechanistic hypotheses—not findings

> Externally preserved intermediate state may help prevent tail errors.

> Context fragmentation or interface boundaries may contribute to Condition
> D's weaker observed performance.

> Structured workflows may primarily improve tail-error prevention rather than
> ordinary-cell performance.

## 10. Prohibited claims

The paper must not claim that:

- sequential execution is definitively the dominant mechanism;
- C broadly or uniformly outperforms B;
- preregistered Case 3 is conclusively established;
- Case 4 is universally disconfirmed;
- persona specialization is generally harmful;
- multi-agent systems are generally inferior;
- C−D isolates a pure persona effect;
- Phase 5B establishes population-level significance;
- all 216 planned calls were physically dispatched;
- exact Phase 5B token and cost values are original-artifact verified.

## 11. Unresolved issues

1. Phase 5B uncertainty uses only two scenario clusters.
2. The primary C−B result is dominated by one structural-break cell and is not
   robust to trimming.
3. Phase 5B failure harmful/abstention flags are transcript-defined
   reconstructions rather than observed model actions.
4. Phase 5B lacks a surviving canonical raw request/response ledger.
5. The freeze contains the implementation-identifier recording error described
   above.
6. The Phase 4J raw archive is referenced and hashed but absent from this clean
   worktree.
7. Neither Phase 4J nor Phase 5B provides independent replication across
   providers, models, or a larger scenario-cluster set.

These issues limit claim strength but do not prevent careful
**pilot-descriptive** use of the canonical values recorded in the ledger.
