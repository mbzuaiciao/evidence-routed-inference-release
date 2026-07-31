# Phase 5B: Controlled Prompt-Decomposition Ablation Plan & Frozen Specification

**Branch:** `feature/phase5-scientific-analysis`
**Primary Evaluation Set:** All 18 Frozen Phase 4J Benchmark Cells ($N=18$)
**Diagnostic Subset:** 5 Prespecified Divergent Cells (Labeled Metadata Flag)
**Evaluation Target Endpoint:** Alibaba Cloud (`qwen3.7-max`, Singapore region)
**Preflight Status:** PASSED (Zero Network Calls Dispatched)

---

## 1. Experimental Objective & Hypotheses

### Objective
Isolate whether the observed descriptive reduction in mean routing regret of `unconstrained-specialized-agents` relative to `monolithic-agent` is specifically attributable to **instruction wording**, **sequential state accumulation**, or **role persona decomposition**, while strictly controlling for nominal condition-level maximum output token capacity across all 18 cells.

### Primary Contrast Definition ($C \text{ vs. } B$)
The primary contrast ($C \text{ vs. } B$) is precisely defined as:
> **“The effect of four-call sequential execution with intermediate-state accumulation under matched unified instructions and matched nominal condition-level maximum output-token capacity.”**

*Note:* This contrast is **not** called a pure "turn-count effect," because input-context accumulation across sequential calls and the distribution of output capacity across dispatches necessarily differ.

### Nominal vs. Actual Output Capacity Qualification
Matching 8,192 maximum output tokens per condition establishes nominal budget equality, but does **not** imply identical actual token consumption or identical computational opportunity. For every cell run, the benchmark records and compares:
* Actual input tokens consumed
* Actual output tokens consumed
* Physical HTTP calls dispatched
* Accumulated context window size (input bytes/tokens per stage)
* Total provider latency (seconds)

---

## 2. Experimental Conditions & Matched Output-Token Capacity

All 4 conditions are evaluated across all 18 benchmark cells ($N=18$):

| Condition Identifier | Code Name | Turn Budget | System Prompt Persona | Max Output Tokens / Call | Total Max Output Capacity / Cell |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Condition A** | `M-short-2` | 2 Calls | Single Monolithic | 4,096 | **8,192 tokens** |
| **Condition B** | `M-unified-2` | 2 Calls | Single Monolithic (4-Role Unified) | 4,096 | **8,192 tokens** |
| **Condition C** | `M-unified-4` | 4 Calls | Single Sequential (4-Role Unified) | 2,048 | **8,192 tokens** |
| **Condition D** | `Specialized-4` | 4 Calls | 4 Personas (Split 4-Role Prompts) | 2,048 | **8,192 tokens** |

### Historical Phase 4J Comparison Note ($D \text{ vs. } A$)
Condition $D \text{ vs. } A$ represents a concurrent architectural-bundle contrast. In historical Phase 4J, `unconstrained-specialized-agents` used 4,096 maximum output tokens per dispatch across 4 calls (total max capacity 16,384 tokens). In Phase 5B, `Specialized-4` uses 2,048 max output tokens per dispatch (total max capacity 8,192 tokens) to match condition-level nominal capacity with B and C. Consequently, Phase 5B `Specialized-4` is **not an exact replication** of the historical Phase 4J specialized condition.

---

## 3. Complete Instruction-Equivalence & Message Wrapper Audit

### 3.1 Identical Canonical System Instruction Text
Condition B (`M-unified-2`) and Condition C (`M-unified-4`) share **100% identical system instruction content**:

```text
Phase 4B reasoner with unified role contracts. Use only supplied information and registered tools.
You must execute the reasoning responsibilities of four specialized sub-roles within this single agent persona:
1. Diagnostic Planner: Request only allowlisted diagnostics for unresolved claims. Do not decide whether a statistical claim is true or calculate statistics.
2. Provenance Analyst: Extract only explicitly supported provenance. Quote exact support, preserve ambiguity, and never invent counts.
3. Skeptical Reviewer: Cite typed evidence or method metadata for every challenge. Never override eligibility or select a method.
4. Report Generator: Use only supplied typed state. Do not calculate new values. Distinguish frequentist PSR from posterior probability and benchmark evidence from this analysis.
```

* **SHA-256 Digest (Condition B & C):** `a2a8658d9216574c6ef2e581f05a0a01eb36e7e43d496d5504b13bbb351923b7`
* **Programmatic Hash Match:** **VERIFIED IDENTICAL** (`verify_instruction_identity()["identical"] == True`).

### 3.2 Execution Wrapper & Message Construction Differences
While system instructions are byte-identical, the complete message constructions differ by experimental design as follows:

1. **User Message Stage Wrappers:** Condition B passes standard monolithic user payload. Condition C passes step-numbered user payload wrappers (`Phase 4B sequential reasoner step 1-planner`, `2-provenance`, etc.).
2. **Intermediate Output Accumulation:** Condition B receives zero intermediate state accumulation. Condition C accumulates prior step JSON outputs (`PlannerOutput`, `ProvenanceOutput`, `ReviewerOutput`) into the `prior_unconstrained_outputs` array in the user message payload for turns 2, 3, and 4.
3. **Output Schemas per Stage:** Condition B enforces `MonolithicOutput` schema for both calls. Condition C enforces `UnconstrainedOutput` schema for all 4 calls.
4. **Tool-Result Placement:** Diagnostic observation results are acquired after turn 1 (planner stage) and included in the visible evidence payload for subsequent turns in both B and C.

---

## 4. Deterministic Interleaved Dispatch Schedule

To eliminate time-of-day bias, provider latency drift, or session ordering bias:
* Dispatches are organized into a **deterministic rotating dispatch schedule** across all 72 cell-units (18 cells $\times$ 4 conditions).
* Conditions rotate across cell index offsets so that every condition appears equally across rotating schedule positions over the 18 cells.
* Total Physical HTTP Dispatches: **216 dispatches** ($18\text{ cells} \times [2 + 2 + 4 + 4] = 216$).

---

## 5. Pre-registered Contrast Interpretations

1. **Condition B vs. Condition A ($B \text{ vs. } A$):**
   *Interpretation:* Measures the effect of the richer unified instruction package as implemented. It is **not** reduced to a pure "wording effect" because prompt length, directive detail, and salience change together.
2. **Condition C vs. Condition B ($C \text{ vs. } B$) — PRIMARY CONTRAST:**
   *Interpretation:* Measures the effect of 4-call sequential execution with intermediate-state accumulation under matched unified instructions and matched nominal condition-level maximum output-token capacity.
3. **Condition D vs. Condition C ($D \text{ vs. } C$):**
   *Interpretation:* Residual architectural contrast associated with role-specific separation, including stage schemas, persona titles, and context allocation. It is **labeled as a residual architectural contrast**, rather than a pure causal persona effect.
4. **Condition D vs. Condition A ($D \text{ vs. } A$):**
   *Interpretation:* Architectural-bundle contrast and concurrent replication of the original Phase 4J comparison.

---

## 6. Pre-Execution Statistical Analysis Plan

* **Primary Estimand:** Mean paired routing regret difference ($C \text{ minus } B$) across all 18 cells:
  $$\Delta_{\text{primary}} = \bar{R}(C) - \bar{R}(B)$$
* **Reporting Metrics per Contrast:**
  * Paired mean difference & paired median difference
  * Improved / tied / worsened cell counts
  * Scenario-family means & seed-level summaries
  * Largest positive and negative paired cell differences
  * Invalid-selection rate difference
  * Harmful non-abstention rate difference
  * Final claim validity rate difference & abstention difference
  * Fail-closed denominator accounting ($N=18$ per condition retained)
* **Uncertainty Note:** Scenario-clustered 95% confidence intervals are reported for continuity, but explicitly noted as unstable due to $K=2$ scenario clusters. All conclusions remain strictly **`pilot-descriptive`**.

---

## 7. Resource & Cost Estimates

* **Primary Cells:** 18 cells ($N=18$)
* **Conditions:** 4 conditions (A, B, C, D)
* **Condition-Cell Units:** 72 units
* **Physical HTTP Dispatches:** 216 dispatches
* **Estimated Input Tokens:** ~400,000 tokens
* **Estimated Output Tokens:** ~75,000 tokens
* **Estimated Financial Cost:** ~$1.30 USD (Alibaba Cloud `qwen3.7-max`)
* **Estimated Total Latency:** ~750 seconds (~12.5 minutes)

---

*Preflight Audit Passed. Design Frozen under digest. Ready for authorization.*
