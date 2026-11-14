# GPU smoke checks

Completed checks and their limits are recorded in the dated reports:

- [Latest submission check](../studies/gpu-smoke/2026-11-14/README.md): fresh training, exact resume and both adapters/evaluator on 2×A100.
- [Earlier main recipe and released adapters](../studies/gpu-smoke/2026-11-07/README.md): training, complete-state resume and adapter/evaluator smoke checks on 2×A100.
- [Other methods](../studies/gpu-smoke/2026-11-07-other-methods/README.md): 21 single-4090 training recipes, including failed attempts and repairs.

These are bounded technical checks, not new full training runs or benchmark results. The runners do not allocate GPUs themselves. Use an already authorized worker, pinned requirements, prepared inputs and a finite lease.

```bash
python scripts/gpu_smoke.py --help
python scripts/gpu_method_smoke.py --help
```

The main runner requires frozen data, released adapters, official question/grader files and a lease. Other recipes additionally need their own prepared inputs and parent checkpoints; see [METHODS.md](METHODS.md). New paid execution requires a separate budget decision.
