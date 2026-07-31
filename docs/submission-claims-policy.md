# Submission claims policy

Phase 3B artifacts carry one conservative `permitted_claim_level`. The level is
a ceiling on wording; authors may always use weaker language.

## Claim levels

### `smoke-only`

Permitted: the pipeline runs end to end, produces typed artifacts, preserves
shared draws, and reports uncertainty. Not permitted: a quantitative benchmark
finding or comparative performance claim. The scaling pilot always has this
level.

### `preliminary-descriptive`

Permitted: exact, clearly bounded observations from an unreviewed development
or hackathon run, with intervals, denominators, scenarios, and loss profile.
Use “in this run” or “across the registered scenarios.” Not permitted:
superiority, confirmation, robustness, generalization, or agent claims.

### `benchmark-supported`

Requires at least the hackathon evaluation size and the explicit independent
review gate. A permissible conditional statement, only when the reviewed
effects, denominators, coverage behavior, scenario sensitivity, and oracle
stability support it, is:

> Across the preregistered closed-world benchmark, the evidence router reduced
> invalid primary-method selections relative to fixed Gaussian and HAC
> baselines while preserving selective interval coverage.

Any qualification from profile sensitivity, oracle instability, exclusions,
small-cluster uncertainty, nominal undercoverage, or scenario-level reversal
must accompany the statement. Passing software tests or setting review metadata
does not by itself make a directional scientific sentence permissible.

### `confirmation-supported`

Requires a valid freeze hash and execution on the previously unseen frozen
confirmation namespace. Wording must still remain closed-world and
scenario-bounded. Confirmation does not establish real-data calibration,
universal validity, exact DGP recovery, or causal superiority.

## Submission distinction

Hackathon material may emphasize the functioning audited core, typed evidence,
deterministic routing, explicit abstention, reproducibility, reviewed bounded
results, and the implemented Phase 4A agent systems layer. It must state that
the Phase 3B numbers evaluate the deterministic router and that comparative
agent performance remains unevaluated.

AAAI material must distinguish the implemented deterministic evidence router,
the implemented Phase 4A agent systems component, and the future comparative
agent hypothesis. A result-ready abstract may use a supported deterministic
claim and traceable numbers only after independent review. It must not claim an
agent, LangGraph, or multi-agent advantage before matched-access evaluation.

`result_claims.json` is a machine-readable review aid, not an automatic source
of publication prose. Submission text is never rewritten by benchmark code.

## Phase 4A systems claims

Hackathon materials may describe the four specialized agent roles, typed
provider contract, bounded tool requests, deterministic eligibility guardrails,
safe fallback, audit artifacts, and mock scenarios as implemented systems
behavior.

They must not transfer Phase 3B deterministic-router numbers to the agent
layer, imply that agents were a Phase 3B baseline, or claim improved
calibration, selection, coverage, regret, or abstention. The approved agentic
description is:

> An agentic, evidence-routed statistical inference system in which specialized
> agents acquire and critique evidence, while an audited deterministic core
> enforces method eligibility, computes uncertainty, and abstains when validity
> cannot be established.

AAAI wording may state that the agent layer is implemented as a systems
component. Comparative agent-performance evaluation remains future work.
