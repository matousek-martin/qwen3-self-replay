"""Measured CUDA/DDP/resume gates on the rented worker, before fresh full training."""
import copy
import json
from pathlib import Path
import shutil
import time

from .distributed import atomic_json
from .prepare import sha
from .throughput import read_rows


def engineering_protocol(protocol):
    p = copy.deepcopy(protocol)
    p['id'] += '-engineering'
    p['execution'].update(phase='benchmark', gated_full_training=False)
    p['training'].update(rollout_batches=8, optimizer_updates_total=32, training_completions_per_arm=4096)
    p['monitoring']['every_rollout_batches'] = 7
    return p


def seed_resume(source, destination, world, step=28, copy_function=shutil.copy2):
    source, destination = Path(source), Path(destination)
    checkpoint = source/f'checkpoint-{step}'
    for name in ('optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'model.safetensors',
                 *([f'rng_state_{rank}.pth' for rank in range(world)] if world > 1 else ['rng_state.pth'])):
        if not (checkpoint/name).is_file():
            raise RuntimeError('Incomplete GPU resume checkpoint: ' + name)
    if json.loads((checkpoint/'trainer_state.json').read_text())['global_step'] != step:
        raise RuntimeError('Unexpected GPU checkpoint boundary')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(checkpoint, destination/checkpoint.name, copy_function=copy_function)
    shutil.copy2(source/'run-provenance.json', destination/'run-provenance.json')
    for p in source.rglob('*.jsonl'):
        if p.parent != source and not p.parent.name.startswith('rank-'):
            continue
        target = destination/p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        retained = []
        for row in read_rows(p):
            at = row.get('optimizer_step', -1)
            is_next_rollout = row.get('split') == 'train' or row.get('mode') == 'train' or row.get('phase') == 'train'
            if at <= step and not (at == step and is_next_rollout):
                retained.append(row)
        target.write_text(''.join(json.dumps(row)+'\n' for row in retained))


def verify_gpu_checks(source, resumed, protocol, lease):
    source, resumed = Path(source), Path(resumed)
    world = protocol['execution']['world_size']
    for rank in range(world):
        folder = source/f'rank-{rank:03d}'
        for filename in ('gpu-memory-check.json', 'gpu-sampling-check.json', 'precision-check.json'):
            if json.loads((folder/filename).read_text())['status'] != 'passed':
                raise RuntimeError('GPU engineering check failed: ' + filename)
        updates = read_rows(folder/'parameter-updates.jsonl')
        if len(updates) != 32 or not all(r['rank_samples_identical'] for r in updates):
            raise RuntimeError('Missing distributed optimizer evidence')
        generations = [r for r in read_rows(folder/'generation-events.jsonl')
                       if r.get('mode') == 'train' and r.get('status') == 'finished' and not r.get('engineering')]
        if {r['optimizer_step'] for r in generations} != set(range(0,32,4)):
            raise RuntimeError('Missing full-length rollout timing or weight synchronization')
        if any(r['vllm_loaded_step'] != r['optimizer_step'] for r in generations):
            raise RuntimeError('Stale vLLM weights')
        def completions(path):
            return [(r['prompt_id'], r['completion_ids_sha256'], r['reward'])
                    for r in read_rows(path/f'rank-{rank:03d}'/'rollouts.jsonl')
                    if r['split'] == 'train' and r['optimizer_step'] == 28]
        if not completions(source) or completions(source) != completions(resumed):
            raise RuntimeError('GPU resume did not reproduce prompts/completions/rewards')
        if json.loads((resumed/f'rank-{rank:03d}'/'precision-check.json').read_text())['status'] != 'passed':
            raise RuntimeError('GPU resume lost optimizer precision')
    from safetensors import safe_open
    import torch
    max_difference = 0.
    with safe_open(source/'model/model.safetensors', framework='pt', device='cpu') as a, \
         safe_open(resumed/'model/model.safetensors', framework='pt', device='cpu') as b:
        if set(a.keys()) != set(b.keys()):
            raise RuntimeError('GPU resume export tensor names differ')
        for name in a.keys():
            left, right = a.get_tensor(name), b.get_tensor(name)
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-8)
            max_difference = max(max_difference, float((left-right).abs().max()))
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(resumed/'model', local_files_only=True)
    AutoTokenizer.from_pretrained(resumed/'model', local_files_only=True)
    if config.model_type != 'qwen3':
        raise RuntimeError('Unexpected exported architecture')
    curve = json.loads((source/'learning-curve.json').read_text())['checkpoints']
    if [r['optimizer_step'] for r in curve] != [0,28] or any(r['questions'] != 128 for r in curve):
        raise RuntimeError('Missing matched validation on engineering worker')
    timing = json.loads((source/'throughput.json').read_text())
    if timing['status'] != 'measured_projection':
        raise RuntimeError('Insufficient measured throughput')
    # Account for every engineering stage already spent on this same rental,
    # then fresh full-model initialization, 300 rollouts and 31 validations.
    predicted = (time.time()-lease['created_at'] + 600 +
                 1.2*(300*timing['slowest_measured_rollout_seconds']+31*timing['slowest_validation_seconds']) + 900)
    timing.update(projected_full_job_hours=predicted/3600,
                  projected_full_job_reserved_usd=predicted/3600*lease['budget_rate'],
                  gpu_resume_max_abs_weight_difference=max_difference)
    atomic_json(source.parent/'full-run-projection.json', timing)
    if not protocol['execution'].get('completion_driven_authorized') and (
            predicted > 8*3600 or predicted*lease['budget_rate']/3600 > lease['max_cost_usd']):
        raise RuntimeError('Measured full run exceeds the authorized eight-hour/cost envelope')
    return {'trainer_gpu_smoke': True, 'distributed_gpu_check': True, 'resume_equivalence': True,
            'export_reload': True, 'matched_validation': True, 'cost_projection': True,
            'gpu_resume_max_abs_weight_difference': max_difference,
            'engineering_provenance_sha256': sha(source/'run-provenance.json')}
