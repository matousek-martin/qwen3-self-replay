# Isolated on-policy distillation primitives

Feasibility and scope: [study](../../studies/on-policy-distillation/FEASIBILITY.md).
No training CLI, optimizer, API call or cloud allocation is exposed here.
The shared trainer is owned by replay-selection; see the [integration handoff](../../studies/on-policy-distillation/INTEGRATION.md).

`loss.py` computes full-vocabulary forward KL on fixed prefixes, with detached teacher targets and token-mean reduction. The chunked Qwen helper avoids retaining full-sequence vocabulary projections. It supports same-device Qwen3 models and a PEFT-wrapped student; production PEFT/CUDA use still needs integration tests. Token masks are aligned to shifted causal targets and include the first EOS, excluding prompt, padding and suffix after EOS. Prefix sampling is not differentiated.

```bash
OMP_NUM_THREADS=1 /tmp/thinkingcap-opd-test/bin/python -m unittest experiments.on_policy_distillation.test_loss -v
/tmp/thinkingcap-opd-test/bin/python -m experiments.on_policy_distillation.audit_tokenizers \
  --snapshots lab/experiments/on-policy-distillation/feasibility \
  --out /tmp/opd-alignment-fresh.json
```

The temporary CPU test environment contains torch, transformers 4.57.1 and tiktoken; exact versions are recorded in study evidence. Tests only differentiate synthetic tensors/tiny random Qwen configurations and never run an optimizer. Actual student training requires NVIDIA GeForce RTX 4090, enforced by allocation and worker startup. The audit is local-only and refuses to overwrite its output evidence. Downloaded metadata/tokenizer snapshots are pinned; model weights were not downloaded.
