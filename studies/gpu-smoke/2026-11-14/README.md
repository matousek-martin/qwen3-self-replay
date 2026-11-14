# Submission GPU smoke check — 2026-11-14

**Passed** on 2× NVIDIA A100-SXM4-80GB using public source `f99f4952ca0eacaa49a5c05ad713548ec858b677`.
The suite took **26.8 minutes**, excluding provisioning and backup. Outputs
were hash-verified and the worker was released; the conservative charge estimate
was **$2.73**, within the separate $6 cap.

| Check | Result |
|---|---|
| Public CLI and frozen-data staging | Passed |
| Four-question synthetic training | Completed |
| Resume from question 2 | Adapter, optimizer, replay and per-rank RNG matched the uninterrupted run bit for bit |
| Maximum-length backward check | 2,048 tokens passed |
| Normal training path | Four original training questions, one per domain, at the full 2,048-token cap; 3 optimizer updates, development validation and a verified complete checkpoint |
| Released adapters 256 and 850 | Hash checks and base → adapter → base routing/effect/restoration passed |
| Official evaluator smoke | 30 questions × 8 responses per adapter, 480 responses total; all four domains generated and graded without failed benchmarks |

These are bounded execution checks, not a retraining of the submitted checkpoints
or a new full benchmark. The real-data probe uses four training and four separate
development examples with reduced calibration; the original data and adapters are
unchanged. No full official baseline was regenerated. Evaluation used one GPU at
a time and sandbox mode `rlimit`; namespace isolation is not established by this run.
Only the main recipe and released adapters were retested.

[Report](report.json), [resume parity](resume-parity.json),
[maximum-length check](maximum-length-check.json),
[full-cap completion](full-cap-completion.json),
[evaluator 256](evaluation-256.json), [evaluator 850](evaluation-850.json),
and [provenance](provenance.json) retain the original receipts and hashes.
[GPU details](gpu.txt) and [installed packages](packages.txt) record the worker.
The final release adds these receipts and documentation to the tested source;
training and inference implementation remain unchanged.
