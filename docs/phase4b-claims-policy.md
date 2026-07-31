# Phase 4B claims policy

Machine-readable claim levels are:

- `implementation-only`: contracts, fixtures, and orchestration work;
- `pilot-descriptive`: live pilot cost and feasibility, not performance;
- `benchmark-supported`: preliminary paired deadline evidence after independent
  review;
- `confirmation-supported`: a frozen confirmation run after independent review.

Mock results can never support agent-performance claims. A pilot cannot support
performance claims. No claim may imply that agents outperform deterministic
routing before reviewed live evidence. Claims must state the regime, systems,
paired denominator, model/version, uncertainty unit, exclusions, and whether
the adaptive deterministic baseline is included.

The result ledger defaults `agent-performance` to unsupported. Promotion is a
reviewed scientific decision, not an automatic consequence of a favorable
sample mean.

Failed or mixed mock/live results cannot be promoted. Deadline claims require
independent review and pinned provider/model metadata. Confirmation additionally
requires a preregistered freeze hash covering prompts, budgets, tools, provider
settings, policies, and unseen cells. Missing usage or sampling-seed metadata
must be disclosed and cannot support a reproducibility claim it does not
establish.

Any explicit provider override marked as changing scientific comparability
forces the run to `implementation-only`. Recording an override does not make an
unreviewed configuration eligible for pilot, benchmark, or confirmation claims.

Phase 4C adapter completion, offline fake-live tests, provider preflight, and a
future connectivity request remain `implementation-only`. Connectivity is not
scientific evidence. A completed 18-cell live pilot can at most support
`pilot-descriptive` feasibility, latency, usage, cost, and failure statements;
it cannot support comparative agent performance.

Having two offline-tested provider adapters does not support a cross-provider
performance claim. Provider-comparison manifests are metadata-only. A Gemini or
OpenAI connectivity check remains implementation-only, and selecting Gemini
first because a project may be free-tier eligible is an operational decision,
not evidence about model quality.

One approved synthetic live case must be manually inspected before the pilot.
Neither that case nor the pilot promotes the claim level. Preliminary
comparative performance evidence requires the separately approved and reviewed
deadline benchmark.
