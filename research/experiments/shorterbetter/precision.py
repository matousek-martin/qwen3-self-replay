"""FP32 trainable parameters/Adam state with BF16 autocast for computation.

Do not confuse BF16 compute with storing the optimizer's parameters in BF16:
at LR=1e-6 the latter can repeatedly erase updates before they accumulate.
"""
import json
from pathlib import Path

import torch
from transformers import TrainerCallback


def require_precision_config(settings):
    if (settings['model_init_kwargs'].get('dtype') != 'float32'
            or settings.get('bf16') is not True or settings.get('optim') != 'adamw_torch'):
        raise RuntimeError('Explicit float32 parameter_dtype, BF16 autocast and AdamW required; '
                           'historical BF16-parameter registrations are read-only.')


def inspect_precision(model, optimizer, require_moments=False):
    parameters = {id(p): p for p in model.parameters() if p.requires_grad}
    if not parameters or any(p.dtype != torch.float32 for p in parameters.values()):
        raise RuntimeError('Trainable optimizer parameters must remain float32')
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    if set(parameters) != optimized:
        raise RuntimeError('Optimizer does not own exactly the trainable model parameters')
    moments = []
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state:
                moments.append(state[key])
    if any(value.dtype != torch.float32 for value in moments):
        raise RuntimeError('Adam moments must remain float32')
    if require_moments and not moments:
        raise RuntimeError('No Adam moments after optimizer update')
    return {'status': 'passed' if moments else 'pending_first_update',
            'parameter_dtype': 'float32', 'parameter_tensors': len(parameters),
            'optimizer_moment_dtype': 'float32' if moments else None,
            'optimizer_moment_tensors': len(moments), 'compute_autocast': 'bfloat16'}


class PrecisionGuard(TrainerCallback):
    def __init__(self, output):
        self.output = Path(output)

    def record(self, state, model, optimizer, require_moments):
        report = inspect_precision(model, optimizer, require_moments)
        report['optimizer_step'] = state.global_step
        self.output.mkdir(parents=True, exist_ok=True)
        temporary = self.output / 'precision-check.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(self.output / 'precision-check.json')

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.record(state, model, optimizer, state.global_step > 0)

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.record(state, model, optimizer, True)


def parameter_sample_indices(numel, device=None, limit=512):
    """Use integer arithmetic: FP32 linspace can round numel-1 up to numel."""
    if numel < 1 or limit < 1:
        raise ValueError('Parameter sampling requires positive sizes')
    count = min(limit, numel)
    if count == 1:
        return torch.zeros(1, dtype=torch.int64, device=device)
    return torch.arange(count, dtype=torch.int64, device=device) * (numel - 1) // (count - 1)


class UpdateGuard(TrainerCallback):
    """Measure actual FP32 scalar updates; tensor hash changes alone are insufficient."""
    def __init__(self, output):
        self.output, self.before, self.empty_updates = Path(output), [], 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
        parameters = sorted(parameters, key=lambda item: item[0])[:12]
        self.before = []
        for name, p in parameters:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError('Nonfinite gradient before optimizer update')
            indices = parameter_sample_indices(p.numel(), p.device)
            self.before.append((name, p, indices, p.detach().flatten()[indices].clone()))

    def on_optimizer_step(self, args, state, control, **kwargs):
        changes, samples, delta2 = 0, 0, 0.0
        sampled_after = []
        for name, p, indices, before in self.before:
            after = p.detach().flatten()[indices]
            if not torch.isfinite(after).all():
                raise RuntimeError('Nonfinite parameter after optimizer update')
            changes += int((after != before).sum())
            samples += after.numel()
            delta2 += float((after.float() - before.float()).square().sum())
            sampled_after.append(after)
        if sampled_after and torch.distributed.is_initialized():
            local = torch.cat(sampled_after)
            gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered, local)
            if any(not torch.equal(local, other) for other in gathered):
                raise RuntimeError('Distributed ranks have different sampled parameters after update')
        fraction = changes / samples if samples else 0.0
        self.empty_updates = self.empty_updates + 1 if fraction < .001 else 0
        with (self.output / 'parameter-updates.jsonl').open('a') as stream:
            stream.write(json.dumps({'optimizer_step': state.global_step + 1,
                'sampled_scalars': samples, 'changed_fraction': fraction, 'delta_l2': delta2 ** .5,
                'rank_samples_identical': True}) + '\n')
        if not samples or self.empty_updates >= 3:
            raise RuntimeError('Actual parameter updates absent/negligible for three consecutive steps')
        self.before = []
