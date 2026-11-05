# AI-assisted research and engineering

This project was developed with substantial AI assistance. Martin Matousek directed the research priorities, chose between experiment proposals, set budgets and made continuation/stopping decisions. AI agents contributed suggestions, literature search, implementation, cloud operations, debugging, tests, analysis and documentation.

The author is responsible for this submission. We do not claim that all implementation was handwritten, that every line was manually reviewed, or that agent-generated conclusions are correct merely because an agent reported them.

## How outputs were checked

- Results are tied to preserved evaluator output, responses, configurations, source snapshots and SHA-256 manifests.
- The release recomputes the 256 and 850 official comparisons from saved responses, without generating new answers.
- Numerical tests compare objectives/gradients and checkpoint state; cloud tests simulate failure and reconciliation paths.
- New local checks and historical GPU evidence are identified separately; no new paid GPU validation is implied by the packaging work.
- Failed experiments, corrected analyses and statistical limitations remain visible.

## AI used to build the project versus AI used to train the model

The main self-replay lineage learns from the current student's own verified generations. It does not train on external teacher answers. Separate experiments did use teacher models or supplied coding answers; their lineage and results are documented separately. AI assistance in writing code or reports must not be confused with teacher distillation in the submitted checkpoint.

## History and privacy

The research workspace did not originally have one root Git repository. The public history records the actual evaluator ancestry and subsequent import/refactor commits, with current dates. Artifact timestamps and source hashes supply the earlier experiment history. No backdated development story is manufactured.

Credentials, account balances, private conversations and server-specific notification configuration are excluded from release assets. Disclosures are a concise account of contributions, not a transcript dump.
