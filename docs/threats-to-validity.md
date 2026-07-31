# Threats to validity

| Threat | Consequence | Planned mitigation |
| --- | --- | --- |
| Finite-sample diagnostics have low or uneven power. | Inconclusive evidence or missed contraindications, especially at small `n`. | Stratify by sample size, include boundary regimes, report evidence uncertainty, and prefer sensitivity or abstention over proof claims. |
| Exact DGPs are non-identifiable from observed finite samples. | Family labels or diagnostic outputs are overinterpreted. | Target inference-relevant properties only; never claim exact DGP recovery; score decisions rather than label identification. |
| Benchmark families favor the implemented diagnostics. | Inflated router performance and weak external validity. | Preregister families and boundary regimes; include misspecified and held-out open-world mechanisms later. |
| Oracle leakage or per-replicate clairvoyance. | An unattainable comparator and misleading regret. | Define the oracle only by family/regime/sample-size expected loss on independent simulations; never expose truth to evaluated systems. |
| Diagnostic and routing thresholds are tuned on final outcomes. | Optimistic selection and calibration. | Freeze defaults before final evaluation, separate development and evaluation seeds, and report threshold sensitivity. |
| Thresholds are partly policy-dependent. | One value encodes preferences rather than scientific truth. | Label statistical, numerical, heuristic, and routing thresholds separately; publish multiverse and policy sensitivity. |
| Asymptotic approximations are poor at relevant sample sizes. | Miscalibrated tests and intervals despite correct routing. | Include small- and moderate-sample regimes, compare empirical calibration, and document where no current procedure is adequate. |
| HAC bandwidth and bootstrap block length drive conclusions. | Apparent robustness depends on one tuning choice. | Preregister defensible grids, report sensitivity, and never infer either value silently. |
| Loss weights encode the desired winner. | Policy conclusions masquerade as scientific facts. | Publish components separately, use multiple preregistered weight profiles, and report Pareto/weight sensitivity. |
| Selective calibration hides excessive abstention. | Trivial apparent coverage from making almost no claims. | Always report non-abstention, harmful non-abstention, unnecessary abstention, and denominators beside conditional metrics. |
| Fixed baseline configuration is weak or unfair. | ERI wins by baseline neglect. | Give all systems identical observations and methods; preregister defensible bandwidth/block settings and baseline sensitivity. |
| Family labels oversimplify finite-sample eligibility. | Scoring treats ambiguous cases as obvious. | Define eligibility by regime and estimand, include uncertainty zones, and audit labels independently. |
| Monte Carlo error or seed dependence. | Unstable rankings and false precision. | Use independent seeded replicates, common random numbers for paired comparisons, Monte Carlo intervals, and replication checks. |
| Multiple metrics and benchmark slices invite selective reporting. | Spurious claims. | Preregister primary metrics and aggregations, publish all planned strata, and distinguish exploratory analyses. |
| Current diagnostics miss breaks, drift, or infinite-moment behavior. | Harmful non-abstention despite typed evidence. | Include adversarial boundary cases, document diagnostic limits, and treat unresolved support as a reason for sensitivity or abstention. |
| Simulation truth does not capture empirical finance. | Limited ecological validity. | Frame Phase 3 as controlled evidence; follow with carefully documented real-data case studies without pretending their truth is known. |
| Future results depend on LLM model, prompt, or sampling. | Agent conclusions fail to generalize or reproduce. | Defer agents; later cross models/prompts, log versions and seeds, and separate architecture effects from model effects. |
| Agent comparisons receive unequal context, tools, or compute. | The effect of evidence structure is confounded. | Later match model, prompts, tools, information, token budget, seeds, and statistical backend. |
| A deterministic controller is superior to agents. | Agentic complexity adds variance, cost, or unsupported claims. | Treat the deterministic router as a first-class baseline and accept deterministic superiority as a valid outcome. |
| Sharpe inference is too narrow. | Claims do not generalize to other estimands or domains. | Limit current claims to the testbed; later replicate typed eligibility and abstention on independently audited inference tasks. |
| Framework choice is mistaken for the contribution. | Engineering novelty is overstated. | Keep methods and routing framework-independent; describe LangGraph only as optional execution infrastructure. |
| Post-selection provenance is incomplete. | Ordinary inference is misread as multiplicity-adjusted evidence. | Keep claim-specific abstention and information requests; do not implement DSR without a separately audited specification. |

These mitigations reduce, but do not remove, construct and external-validity
limits. In particular, no finite diagnostic suite proves a data-generating
process, stationarity, IID behavior, or existence of population moments.
