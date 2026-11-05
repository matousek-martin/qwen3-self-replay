# Contributor instructions

- Work in English in public artifacts; keep summaries concise.
- Scientific results refer to frozen source snapshots, not automatically to refactored code.
- Keep official prompts, sampling, graders and token caps unchanged.
- Never train on SVAMP, MBPP/MBPP+, BFCL or IFEval; screening may only reject overlaps.
- Training defaults to RTX 4090; the original self-replay recipe requires its explicit 2/4 A100-SXM4-80GB scope. Enforce the chosen hardware in allocation and worker startup.
- Historical budgets authorize no new spending. New cloud execution requires an explicit finite authorization and the independent guard.
- Preserve canonical reservations, ambiguous-allocation reconciliation, SHA-verified backups, release and model-type baseline caching.
- Do not backdate commits or invent human review, GPU tests, provenance or successful results.
- Run local tests and artifact verification before release. GPU tests are separately funded.
