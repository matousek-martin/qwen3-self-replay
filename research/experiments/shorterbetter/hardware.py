"""Scoped user-authorized accelerated ShorterBetter hardware, never a global fallback."""
SCOPE = 'shorterbetter-scaled-v1'
ENTRY = 'experiments/shorterbetter/run-scaled.sh'
# Per-GPU reservation ceilings, not provider price claims.
GPU_RATE_CEILINGS = {
    'NVIDIA GeForce RTX 4090': .99,
    'NVIDIA A100-SXM4-80GB': 2.20,
    'NVIDIA A100 80GB PCIe': 2.20,
    'NVIDIA H100 80GB HBM3': 4.50,
    'NVIDIA H100 PCIe': 3.50,
    'NVIDIA H100 NVL': 4.00,
    'NVIDIA H200': 6.50,
}
WORKER_NAMES = {
    'NVIDIA H200': {'NVIDIA H200', 'NVIDIA H200 SXM'},
}


def allocation_spec(args):
    """Return None for historical jobs; validate the whole scoped request otherwise."""
    scope = getattr(args, 'hardware_scope', None)
    if scope == 'coding-curriculum-256-v1':
        from experiments.coding_curriculum.budget import allocation_spec as curriculum_spec
        return curriculum_spec(args)
    if scope == 'coding-abc-768-v1':
        from experiments.coding_abc.budget import allocation_spec as abc_spec
        return abc_spec(args)
    if scope == 'coding-frontier-768-v1':
        from experiments.coding_frontier.budget import allocation_spec as frontier_spec
        return frontier_spec(args)
    if scope == 'opencoder-mode-probe-v1':
        from experiments.opencoder_sft.mode_budget import allocation_spec as mode_spec
        return mode_spec(args)
    if scope == 'opencoder-sft-768-v1':
        from experiments.opencoder_sft.runtime import allocation_spec as oc_spec
        return oc_spec(args)
    if scope == 'coding-replay-768-v1':
        from experiments.replay_scaled.coding_budget import allocation_spec as coding_spec
        return coding_spec(args)
    if scope == 'coding-length-probe-v1':
        from experiments.replay_scaled.length_budget import allocation_spec as probe_spec
        return probe_spec(args)
    if scope == 'replay-evaluation-v1':
        from experiments.replay_scaled.evaluation_budget import allocation_spec as evaluation_spec
        return evaluation_spec(args)
    if scope == 'replay-scaled-a100-v1':
        from experiments.replay_scaled.core import allocation_spec as replay_spec
        return replay_spec(args)
    count = getattr(args, 'gpu_count', 1)
    if scope is None:
        if count != 1:
            raise ValueError('Multiple GPUs require a scoped registration')
        return None
    if (scope != SCOPE or not args.name.startswith('shorterbetter-scaled-')
            or args.entry != ENTRY or args.gpu not in GPU_RATE_CEILINGS
            or type(count) is not int or count not in (1, 2, 4, 8)
            or (getattr(args, 'max_cost_usd', None) is None
                and getattr(args, 'completion_budget_scope', None) != 'finish_registered_1200_updates')):
        raise ValueError('Invalid scoped ShorterBetter hardware request')
    return {'scope': scope, 'gpu': args.gpu, 'count': count,
            'max_gpu_rate': GPU_RATE_CEILINGS[args.gpu]}


def verify_worker(names, lease):
    if lease.get('hardware_scope') == 'coding-curriculum-256-v1':
        from experiments.coding_curriculum.budget import verify_worker as curriculum_worker
        return curriculum_worker(names, lease)
    if lease.get('hardware_scope') == 'coding-abc-768-v1':
        from experiments.coding_abc.budget import verify_worker as abc_worker
        return abc_worker(names, lease)
    if lease.get('hardware_scope') == 'coding-frontier-768-v1':
        from experiments.coding_frontier.budget import verify_worker as frontier_worker
        return frontier_worker(names, lease)
    if lease.get('hardware_scope') == 'opencoder-mode-probe-v1':
        from experiments.opencoder_sft.mode_budget import verify_worker as mode_worker
        return mode_worker(names, lease)
    if lease.get('hardware_scope') == 'opencoder-sft-768-v1':
        from experiments.opencoder_sft.runtime import verify_worker as oc_worker
        return oc_worker(names, lease)
    if lease.get('hardware_scope') == 'coding-replay-768-v1':
        from experiments.replay_scaled.coding_budget import verify_worker as coding_worker
        return coding_worker(names, lease)
    if lease.get('hardware_scope') == 'coding-length-probe-v1':
        from experiments.replay_scaled.length_budget import verify_worker as probe_worker
        return probe_worker(names, lease)
    if lease.get('hardware_scope') == 'replay-evaluation-v1':
        from experiments.replay_scaled.evaluation_budget import verify_worker as evaluation_worker
        return evaluation_worker(names, lease)
    if lease.get('hardware_scope') == 'replay-scaled-a100-v1':
        from experiments.replay_scaled.core import verify_worker as replay_worker
        return replay_worker(names, lease)
    if (lease.get('hardware_scope') != SCOPE
            or not lease.get('job_name', '').startswith('shorterbetter-scaled-')
            or lease.get('gpu') not in GPU_RATE_CEILINGS
            or lease.get('gpu_count') not in (1, 2, 4, 8)):
        raise RuntimeError('Invalid accelerated worker lease')
    allowed = WORKER_NAMES.get(lease['gpu'], {lease['gpu']})
    if len(names) != lease['gpu_count'] or any(n not in allowed for n in names):
        raise RuntimeError(f'Actual hardware differs from reservation: {names!r}')
    return names
