"""Canonical controller hook for one authorized same-pod checkpoint continuation."""
import json


def maybe_continue(cloud, pod, name, result):
    with cloud.job_lock(name) as acquired:
        if not acquired:
            return True
        with cloud.ledger() as state:
            job = dict(state['jobs'][name])
        continuation = dict(job.get('completion_continuation', {}))
        if not continuation or continuation.get('phase') == 'resumed':
            return False
        if not cloud.completion_budget_authorized(name, job):
            raise ValueError('Continuation has no scoped budget authorization')
        base = cloud.STATE / name
        if continuation.get('phase') != 'preparing':
            if not result or result['exit_code'] != 0:
                return False
            finished = json.loads((base / 'output/training/training-finished.json').read_text())
            if finished.get('training_completed'):
                return False
            checkpoint = base / 'output/training/checkpoint-20'
            if not (checkpoint / 'trainer_state.json').exists():
                return False
            continuation['phase'] = 'preparing'
            continuation['first_segment_optimizer_steps'] = finished['optimizer_steps']
            cloud.update(name, completion_continuation=continuation)
        # The original archive remains intact beside the new active output.
        old = base / 'output-segment-0'
        if not old.exists():
            (base / 'output').rename(old)
        (base / 'output').mkdir(exist_ok=True)
        response = json.loads(cloud.remote(pod,
            'cd /workspace/project && python3 -m experiments.shorterbetter.completion_control resume',
            timeout=300))
        if response.get('phase') != 'launched':
            raise ValueError('Continuation launch not confirmed')
        continuation.update(phase='resumed', checkpoint_step=20, worker_transition=response)
        with cloud.ledger() as state:
            row = state['jobs'][name]
            row.update(completion_continuation=continuation, backup_verified=False, state='running')
            row.pop('worker_dead_observed_at', None)
        return True
