# Evaluation contract and compatibility changes

The submission uses the official 2,759-question cohort, eight responses per question, seed 9001, temperature 0.6, top-p 0.95 and top-k 20. Per-domain token caps, prompts and graders are defined by the pinned evaluator. Full results retain failed/ungraded flags and original cohort/revision metadata.

The imported evaluator commit is `6f68418`; at the submission preparation audit, upstream `bd52d8d` had identical evaluate.py, compare.py, requirements.txt and mbpp_worker.py in its committed tree. Its README had added rough guidance on desirable performance. I preserve the actual evaluation implementation rather than silently upgrading it.

## Disclosed changes

1. **MBPP oracle memory:** the local worker frees cached reference outputs between tasks, reusing them across the current task's samples. This prevents the 4 GB process limit from accumulating huge previous-task outputs. The patch is in provenance/mbpp-memory.patch, with a regression test. It is not a new correctness criterion.
2. **vLLM 0.11.0 LoRA routing metadata:** the scientific runtime uses the preserved hash-pinned compatibility patch, carrying actual request adapter identity into output metadata. The evaluator's routing assertion remains. Historical GPU preflights tested base/adapter/base routing; the public refactor has not received a new paid GPU preflight.
3. **Baseline cache:** complete SHA-verified baseline records are reused for matching base revision, cohort, decoding, grader and runtime contract on the same GPU model. Physical GPU identity is retained as provenance, not used to invalidate the cache. A required cache miss fails without silently buying new inference.

## Offline reproduction

Extract the released response archive into artifacts/records, then run:

```bash
python scripts/reproduce_results.py
```

The script verifies response hashes and reproduces the official per-domain metrics, aggregate metrics and confidence intervals for checkpoints 256 and 850. This requires no GPU. Paths identifying the newly materialized run naturally differ from the historical comparison; scientific values must match exactly.

Fresh inference remains available through `self-replay evaluate -- --out /absolute/output --adapter /absolute/adapter`; install requirements/gpu.lock first and freeze the official questions with vendor/thinkingcap/utils/eval/benchmarks.py. For a matched cached baseline in managed workflows, use the preserved experiments.baseline_cache / study_eval implementation in a workspace; do not treat a candidate as its baseline.

## Interpretation

The four official domains are equally weighted; BFCL's three subsets are pooled before the four-domain macro. Total output tokens include reasoning, final answers, incorrect and truncated responses. A confidence interval crossing zero is not an equivalence test. Repeated selection and training-seed uncertainty are not included in the official intervals.
