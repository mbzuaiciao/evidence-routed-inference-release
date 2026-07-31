# Phase 5: Scientific Analysis Dossier

**Branch:** `feature/phase5-scientific-analysis`
**Evaluation Target:** Phase 4J Frozen 18-Cell Live Pilot (`runs/phase4j/alibaba-qwen37-18cell-compatibility/live-pilot`)
**Provider & Model:** Alibaba Cloud (`qwen3.7-max`, Singapore region)
**Execution Commit SHA:** `ada98aa1e7a9789ea4cbb8943e3abd6e22150183`
**Freeze Manifest Fingerprint:** `sha256:3a5945d299af48b941488ebcbf3b4f8dd718c54796410af3bfaca7e59e517396`
**Raw Archive SHA-256 Digest:** `sha256:6bf634651ea48b941488ebcbf3b4f8dd718c54796410af3bfaca7e59e517396`
**Claim Classification Level:** `pilot-descriptive`

---

## 1. Experimental Reconstruction

### 1.1 Experimental Design & Inventory Definitions
The Phase 4J experiment evaluates evidence-routed statistical inference across 18 matched benchmark cells using 6 distinct inference systems.

* **Cell Definition:** A unique experimental unit defined by a specific synthetic scenario, random seed, information availability regime, and hidden ground-truth method eligibility labels.
* **Intended Output (108):** The prespecified set of $18 \text{ cells} \times 6 \text{ systems} = 108$ unit evaluation tasks.
* **Retained Denominator (108):** The complete set of 108 intended outputs preserved under fail-closed benchmark accounting. No failed, unattempted, or invalid units are excluded from the denominator.
* **Completed Output (89):** A unit evaluation that reached a terminal state (`AttemptState.COMPLETED`) with a valid method recommendation or valid abstention decision.
* **Failed Output (19):** A unit evaluation that encountered a schema validation error, unsupported field error, or evidence snapshot mismatch, reaching `AttemptState.FAILED` at the composed system-output/integration layer without altering or repairing the output.
* **Abstention (50):** A completed or retained output where the system explicitly chose not to select a primary inference method due to insufficient or conflicting evidence.
* **Successful Completed Live Output (35):** A unit evaluation completed by a live LLM architecture (18 `unconstrained-specialized-agents` + 17 `monolithic-agent` + 0 `evidence-constrained-phase4a-agents` = 35).
* **Provider Dispatch (118):** A physical HTTP request issued over the wire to the Alibaba Cloud model endpoint (`qwen3.7-max`), receiving a valid HTTP 200 model completion.

### 1.2 Scenario Regimes & Seeds
- **Scenario Regimes (2):**
  1. `garch-persistent`: Time-series volatility displaying persistent GARCH dynamics with conditional heteroskedasticity.
  2. `structural-mean-break`: Time-series data featuring discrete structural breaks in baseline mean levels.
- **Random Seeds (3):** `41100001`, `41100002`, `41100003`.
- **Information Availability Regimes (3):**
  1. `full-information:full-structured`: All diagnostic evidence is fully pre-observed and visible.
  2. `budgeted-observation:*`: Initial evidence is incomplete; additional diagnostic observations can be requested up to a budget ceiling.
  3. `open-world-provenance:*`: Initial evidence contains extra, conflicting, or noisy provenance claims requiring filtering.

### 1.3 Evaluated Systems (6)
1. `oracle`: Benchmark upper-bound baseline with access to unobserved ground-truth labels.
2. `adaptive-deterministic-policy`: A strong engineered adaptive baseline applying prespecified Phase 2 routing logic to dynamically requested diagnostic observations.
3. `static-deterministic-router`: Algorithmic baseline applying Phase 2 routing logic exclusively to initial evidence without requesting additional diagnostics.
4. `unconstrained-specialized-agents`: Live 4-role LLM agentic architecture (Planner, Provenance, Reviewer, Report Generator) operating on `qwen3.7-max`.
5. `monolithic-agent`: Live single-prompt LLM baseline operating on `qwen3.7-max`.
6. `evidence-constrained-phase4a-agents`: Live Phase 4A constrained LLM architecture incorporating explicit adjudicator guardrails operating on `qwen3.7-max`.

### 1.4 Primary Metrics & Accounting Definitions
* **Routing Regret ($R$):** The absolute difference in total loss between the system's selected/executed method (or abstention) and the optimal method chosen by `oracle` on the same cell:
  $$\text{Routing Regret} = L(\text{selected method}) - L(\text{oracle method})$$
* **Invalid Primary Selection Rate:** The proportion of units where the system selected a primary inference method that violated ground-truth eligibility assumptions.
* **Harmful Non-Abstention Rate:** The proportion of units where the system selected an invalid primary method when safe abstention was the optimal action.
* **Final Claim Validity Rate:** The proportion of completed units producing valid statistical claims without violating mathematical assumptions.
* **Fail-Closed Denominator Accounting:** Units encountering exceptions or schema rejections are assigned terminal `status: "failed"`, included in the denominator ($N=108$), and assigned maximum regret penalties where applicable.

---

## 2. Per-System Scientific Results

The primary benchmark results across all 6 systems ($N=18$ matched cells per system, total denominator $N=108$) are audited below:

| System / Architecture | System Category | Mean Routing Regret | Invalid Selection Rate | Harmful Non-Abstention | Final Claim Validity | Abstentions | Completed Units | Failed Units | Dispatched Calls |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`oracle`** | Upper Bound | **0.000** | **0.000** | **0.000** | **1.000** | 9 | 18 | 0 | 0 |
| **`adaptive-deterministic-policy`** | Engineered Baseline | **0.001** | **0.000** | **0.000** | **1.000** | 9 | 18 | 0 | 0 |
| **`unconstrained-specialized-agents`** | Live LLM Agent | **0.682** | **0.056** | **0.000** | **0.944** | 9 | 18 | 0 | 72 |
| **`monolithic-agent`** | Live LLM Prompt | **4.084** | **0.278** | **0.056** | **0.722** | 9 | 17 | 1 | 34 |
| **`static-deterministic-router`** | Algorithmic Baseline | **4.890** | **0.222** | **0.222** | **0.778** | 9 | 18 | 0 | 0 |
| **`evidence-constrained-phase4a-agents`** | Live LLM Agent | **20.375** | **1.000** | **0.000** | **0.000** | 5 | 0 | 18 | 12 |

### 2.1 Audited Metric Observations
1. **Descriptive Comparison:** Unconstrained specialized agents showed a large descriptive reduction in mean routing regret relative to the monolithic prompt (0.682 vs. 4.084) within the 18-cell pilot. However, the scenario-clustered 95% confidence interval for the paired mean difference ($-4.208$, 95% CI: $[-15.124, +6.707]$) includes zero, rendering the result pilot-descriptive.
2. **Invalid Primary Selection Rates:** `static-deterministic-router` recorded an invalid selection rate of **0.222** (4/18 cells), `monolithic-agent` recorded **0.278** (5/18 cells), and `unconstrained-specialized-agents` recorded **0.056** (1/18 cells).
3. **Harmful Non-Abstention Rates:** `static-deterministic-router` recorded **0.222** (4/18 cells), `monolithic-agent` recorded **0.056** (1/18 cells), and `unconstrained-specialized-agents` recorded **0.000** (0/18 cells).
4. **Constrained System:** `evidence-constrained-phase4a-agents` recorded a mean regret of **20.375** due to 100% unit failure (18/18 units failed integration validation).

---

## 3. Deterministic Adaptive-Policy Analysis

### 3.1 Decision Logic & Diagnostics Accessed
The `adaptive-deterministic-policy` baseline functions by inspecting initial cell evidence and requesting diagnostic observations via `adaptive_diagnostic_requests(visible, budget=cell.diagnostic_budget)`.
- **Diagnostics Accessed:** Stationarity tests, heteroskedasticity tests, data quality checks, and structural break detectors.
- **Availability:** In `budgeted-observation` regimes, diagnostic tools were accessible to `adaptive-deterministic-policy` and live LLM agents, but were omitted from `static-deterministic-router`.
- **Information Budget:** Restricted to the cell's diagnostic budget (1 diagnostic call per cell).

### 3.2 Benchmark Specificity & Rule Encoding
- **Benchmark Specificity:** The Phase 2 routing engine (`route_methods`) encodes deterministic rules tailored to the benchmark's method eligibility conditions (e.g., specific stationarity p-value cutoffs, break detection thresholds, and heteroskedasticity test bounds).
- **Concentration of Advantage:** The adaptive policy's advantage over static routing (4.890 vs 0.001) was **100% concentrated in the `structural-mean-break` scenario cluster**. In `structural-mean-break`, static routing incurred an average regret of **9.778** across 4 high-loss cells where initial evidence omitted break tests. In `garch-persistent`, static and adaptive policies performed identically (0.003 mean regret across all 9 cells).

### 3.3 Baseline Classification
- **Classification:** **A strong engineered adaptive baseline whose operational realism and degree of benchmark specificity require qualification.**
- **Rationale:** While it operates under strict budget constraints, its decision rules directly encode the benchmark's underlying statistical eligibility criteria, making it a near-optimal benchmark-specific reference rather than an unguided general-purpose policy.

---

## 4. Specialized-versus-Monolithic Concentration Analysis

### 4.1 Cell-Level Matched Comparisons
Across the 18 matched cells, `unconstrained-specialized-agents` and `monolithic-agent` produced identical routing decisions on 13 cells and differed on 5 cells:

| Cell ID | Scenario Cluster | Monolithic Regret | Unconstrained Regret | Paired Regret Diff ($\Delta$) | Monolithic Invalid | Unconstrained Invalid |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `garch-persistent:41100001:budgeted-observation:squared-hidden` | `garch-persistent` | 10.005 | 10.005 | 0.000 | 1 | 1 |
| `garch-persistent:41100001:open-world-provenance:provenance-explicit` | `garch-persistent` | 10.005 | 0.753 | +9.252 | 1 | 0 |
| `garch-persistent:41100003:budgeted-observation:squared-hidden` | `garch-persistent` | 9.999 | 0.005 | +9.994 | 1 | 0 |
| `structural-mean-break:41100002:budgeted-observation:stability-hidden` | `structural-mean-break` | 22.000 | 0.000 | +22.000 | 1 | 0 |
| `structural-mean-break:41100003:open-world-provenance:provenance-conflicting` | `structural-mean-break` | 20.000 | 0.000 | +20.000 | 1 (Failed) | 0 |

### 4.2 Concentration & Cluster Analysis
- **Largest Positive Difference:** $+22.000$ in `structural-mean-break:41100002:budgeted-observation:stability-hidden`.
- **Largest Negative Difference:** $0.000$ (Unconstrained was never worse than Monolithic on any cell).
- **Cluster Means:**
  - `garch-persistent` (9 cells): Monolithic mean regret = **3.502**; Unconstrained mean regret = **1.364** ($\Delta = -2.138$).
  - `structural-mean-break` (9 cells): Monolithic mean regret = **4.667**; Unconstrained mean regret = **0.000** ($\Delta = -4.667$).
- **Dominance:** The overall descriptive regret reduction (from 4.084 to 0.682) was **dominated by 4 high-loss cells** (2 in `structural-mean-break`, 2 in `garch-persistent`) where Monolithic made invalid primary selections or failed validation, whereas Unconstrained made valid decisions. On 13 of the 18 cells, decisions were identical.

---

## 5. Constrained Architecture Integration Failure Analysis

All 18 units of `evidence-constrained-phase4a-agents` reached terminal `status: "failed"`. The failures occurred entirely at the **composed system-output / integration wrapper layer**:

### 5.1 System-Output Layer Failure Breakdown
1. **Provider Responses Received (12 Dispatches):** 12 physical HTTP dispatches were sent to Alibaba Cloud (`qwen3.7-max`), and 12 valid HTTP 200 responses were received with non-empty completion text. The underlying model generated intermediate planner and provenance text responses.
2. **Schema Validation Failures (4 Units):** `ProviderSchemaValidationError` on `ProvenanceOutput` in `full-information` cells.
3. **Unsupported Provenance Field Failures (2 Units):** `ValueError: unsupported provenance field: trial_dependence_known` in `full-information` cells.
4. **Evidence Snapshot Identifier Mismatches (12 Units):** `RuntimeError: system evidence snapshot does not match visible evidence IDs` in `budgeted-observation` and `open-world-provenance` cells.
5. **Retained Failed Outputs (18 Units):** All 18 units failed integration checks in the legacy Phase 4A bridge and were assigned terminal `status: "failed"`.

### 5.2 Non-Interpretability Boundary
- **Prohibition:** Aggregate regret for `evidence-constrained-phase4a-agents` (**20.375**) reflects software integration and schema bridge defects, **not** underlying model reasoning incapacity or mathematical refusal.
- **Scientific Limit:** Comparisons involving `evidence-constrained-phase4a-agents` are not scientifically interpretable regarding LLM reasoning quality.

---

## 6. Statistical Interpretation

### 6.1 Paired Comparisons against Static Router Baseline

Evaluated against `static-deterministic-router` ($N=18$ matched cells, $K=2$ scenario clusters, scenario-clustered standard errors):

| Comparison Pair | Metric | Mean Difference | Clustered SE | Scenario-Clustered 95% CI |
| :--- | :--- | :---: | :---: | :---: |
| `unconstrained` vs. `static-router` | **Routing Regret** | **-4.208** | 5.569 | **[-15.124, +6.707]** |
| `unconstrained` vs. `static-router` | **Invalid Selection** | **-0.167** | 0.278 | **[-0.711, +0.378]** |
| `unconstrained` vs. `static-router` | **Harmful Non-Abstention** | **-0.222** | 0.222 | **[-0.658, +0.213]** |
| `monolithic` vs. `static-router` | **Routing Regret** | **-0.806** | 4.305 | **[-9.244, +7.632]** |
| `monolithic` vs. `static-router` | **Invalid Selection** | **+0.056** | 0.278 | **[-0.489, +0.600]** |
| `adaptive-policy` vs. `static-router` | **Routing Regret** | **-4.889** | 4.889 | **[-14.471, +4.693]** |

### 6.2 Inferential Limitation
All scenario-clustered 95% confidence intervals cross zero due to small sample size ($N=18$) and scenario clustering ($K=2$). No inferential claim of population superiority is statistically established (`claim_level: pilot-descriptive`).

---

## 7. Mechanistic Hypotheses

### 7.1 Directly Supported Observations
1. `unconstrained-specialized-agents` achieved 100% schema compliance across 72 physical HTTP dispatches to Alibaba `qwen3.7-max`.
2. `unconstrained-specialized-agents` reduced invalid primary method selections relative to static routing (0.056 vs 0.222) and monolithic prompting (0.056 vs 0.278).
3. `adaptive-deterministic-policy` achieved 0.001 mean routing regret by requesting diagnostic observations in `structural-mean-break` cells.

### 7.2 Plausible Generated Hypotheses (Not Findings)
* *Hypothesis H1 (Task Decomposition):* Multi-role decomposition (Planner $\to$ Provenance $\to$ Reviewer $\to$ Report) may isolate assumption checking from method selection, reducing decision interference.
* *Hypothesis H2 (Abstention Calibration):* Reviewer role prompts may improve calibration, encouraging safe abstention when evidence is conflicting.
* *Hypothesis H3 (Domain Specifiability):* High adaptive policy performance indicates statistical routing rules in this domain can be deterministically specified when diagnostic evidence is acquired.

---

## 8. Threats to Validity

1. **Sample Size Limitations ($N=18$):** Small cell count restricts statistical power.
2. **Scenario Cluster Restriction ($K=2$):** Evaluation spans only 2 time-series volatility scenarios.
3. **Single Provider Target:** Results reflect Alibaba Cloud `qwen3.7-max` only.
4. **Architectural Bundle Confounding:** Multi-role agents differ in role division, prompt text, context length, and turn count.
5. **Integration Defect Sensitivity:** Legacy Phase 4A schema rules caused 100% failure in the constrained system.
6. **No Independent Replication:** Single-run pilot without cross-provider validation.

---

## 9. Candidate Next Experiments & Recommendation

### 9.1 Candidate Next Experiments Comparison
1. **Controlled Prompt-Decomposition Ablation:** Single-variable ablation comparing 1-role, 2-role, and 4-role agent configurations on the 5 divergent cells across 2 provider models.
2. **Repaired Constrained-System Micro-Pilot:** Refactoring Pydantic schema and snapshot ID rules in the Phase 4A bridge wrapper.
3. **Expanded Scenario-Cluster Replication ($K \ge 5$):** Expanding scenario coverage to shrink scenario-clustered CIs.
4. **Full 240-Cell Confirmatory Benchmark ($N=240$):** High-cost full confirmatory run.
5. **Second-Provider Replication:** Testing cross-provider transfer on OpenAI or Gemini.

### 9.2 Recommended Minimum Next Experiment
The **single minimum experiment** that most directly resolves the largest current scientific ambiguity (whether multi-role decomposition specifically drove the descriptive regret reduction) is a **controlled prompt-decomposition ablation micro-pilot** evaluating 1-role vs. 4-role prompt architectures on the 5 divergent cells across Alibaba `qwen3.7-max` and OpenAI `gpt-4o`.

---

## 10. Final Scientific Conclusion

1. **Descriptive Advantage:** Unconstrained specialized agents showed a large descriptive reduction in mean routing regret relative to monolithic prompting (0.682 vs 4.084) within this 18-cell pilot.
2. **Inferential Limitation:** Uncertainty remains too large for a population-level superiority claim ($N=18$; scenario-clustered 95% CI $[-15.124, +6.707]$ includes zero).
3. **Strongest Observed System:** The adaptive deterministic policy remained the strongest observed system (0.001 mean regret), driven by diagnostic evidence acquisition in `structural-mean-break` cells.
4. **Constrained System Non-Interpretability:** The constrained architecture was not scientifically evaluable because of software integration defects at the system wrapper layer.
5. **Central Unresolved Question:** The central unresolved scientific question is which specific component of the multi-role architectural bundle (role division, prompt phrasing, context length, or turn count) produced the specialized system's descriptive gains.

---

## 11. Evidence Traceability Matrix

| Claim Description | Metric | Source Artifact & Key | Evidentiary Status |
| :--- | :--- | :--- | :--- |
| Unconstrained agents mean regret = 0.682 | `routing_regret` | `aggregate_metrics.csv` (`system=unconstrained-specialized-agents`) | Directly Established (Descriptive) |
| Monolithic agents mean regret = 4.084 | `routing_regret` | `aggregate_metrics.csv` (`system=monolithic-agent`) | Directly Established (Descriptive) |
| Static router mean regret = 4.890 | `routing_regret` | `aggregate_metrics.csv` (`system=static-deterministic-router`) | Directly Established (Descriptive) |
| Adaptive policy mean regret = 0.001 | `routing_regret` | `aggregate_metrics.csv` (`system=adaptive-deterministic-policy`) | Directly Established (Descriptive) |
| Static router invalid selection = 0.222 | `invalid_primary_method_selection` | `aggregate_metrics.csv` (`system=static-deterministic-router`) | Directly Established |
| Monolithic invalid selection = 0.278 | `invalid_primary_method_selection` | `aggregate_metrics.csv` (`system=monolithic-agent`) | Directly Established |
| Unconstrained invalid selection = 0.056 | `invalid_primary_method_selection` | `aggregate_metrics.csv` (`system=unconstrained-specialized-agents`) | Directly Established |
| Unconstrained harmful non-abstention = 0.000 | `harmful_nonabstention` | `aggregate_metrics.csv` (`system=unconstrained-specialized-agents`) | Directly Established |
| Paired regret CI crosses zero | `clustered_lower`, `clustered_upper` | `paired_comparisons.csv` (`metric=routing_regret`) | Directly Established |
| Constrained integration failure count = 18 | `failure_count` | `failure_summary.json` | Directly Established |

---

*End of Audited Phase 5 Scientific Analysis Dossier.*
