"""Operational one-time checkpoint handoff for an explicitly extended pilot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from experiments.worker_monitor import atomic_json, process_observation

ROOT = Path('/workspace/project')


def seed_continuation(old, new, step):
    """Preserve original logs separately and copy only the accepted checkpoint lineage."""
    old, new = Path(old), Path(new)
    checkpoint = old / 'training' / f'checkpoint-{step}'
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if step != 20 or state['global_step'] != step:
        raise ValueError('Unexpected continuation boundary')
    for filename in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'model.safetensors']:
        if not (checkpoint / filename).is_file():
            raise ValueError('Incomplete continuation checkpoint: ' + filename)
    train = new / 'training'
    train.mkdir(parents=True, exist_ok=True)
    target = train / checkpoint.name
    shutil.copytree(checkpoint, target, copy_function=shutil.copy2, dirs_exist_ok=True)
    for filename in ['readiness-at-start.json', 'packages.txt', 'gpu.txt']:
        if (old / filename).exists():
            shutil.copy2(old / filename, new / filename)
    for filename in ['run-provenance.json', 'learning-curve.json', 'learning-curve.png']:
        if (old / 'training' / filename).exists():
            shutil.copy2(old / 'training' / filename, train / filename)
    for path in (old / 'training').glob('*.jsonl'):
        if path.name == 'budget-stop.jsonl':
            continue
        rows = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            at = row.get('optimizer_step', -1)
            if at > step:
                continue
            # At boundary 20, generation/reward records from the next rollout
            # belong to the discarded tail, while eval/metrics at 20 are retained.
            next_rollout = (row.get('split') == 'train' or row.get('mode') == 'train'
                            or row.get('phase') == 'train')
            if at == step and next_rollout:
                continue
            rows.append(line)
        (train / path.name).write_text(''.join(line + '\n' for line in rows))
    atomic_json(new / 'continuation-lineage.json', {
        'checkpoint_step': step, 'original_segment': str(old),
        'discarded_tail_preserved': True, 'scientific_parameters_unchanged': True,
        'reason': 'User removed spending cap; original process had a captured deadline.'})


def watch(config):
    """Request a graceful stop only once a complete step-20 checkpoint exists."""
    evidence = ROOT / 'completion-stop.json'
    if evidence.exists():
        return
    identity = config['trainer_identity']
    checkpoint = ROOT / 'output/training/checkpoint-20'
    while True:
        observed = process_observation(identity['pid'], identity)
        if observed.get('status') != 'alive' or not observed.get('identity_verified'):
            return
        state = checkpoint / 'trainer_state.json'
        if state.exists() and all((checkpoint / f).exists() for f in
                                 ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'model.safetensors']):
            if json.loads(state.read_text())['global_step'] != 20:
                raise ValueError('Checkpoint identity mismatch')
            # Intent before signal: restarting this helper must never signal twice.
            atomic_json(evidence, {'requested_at': time.time(), 'trainer_identity': identity,
                                   'checkpoint_step': 20})
            os.kill(identity['pid'], signal.SIGTERM)
            return
        time.sleep(5)


def resume(config):
    """Idempotent same-pod handoff; called only after the old archive was verified."""
    transition = ROOT / 'completion-transition.json'
    with (ROOT / 'completion-transition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if transition.exists() and json.loads(transition.read_text()).get('phase') == 'launched':
            print(transition.read_text())
            return
        old, new = ROOT / 'output-segment-0', ROOT / 'output'
        if not old.exists():
            state = json.loads((new / 'worker-state.json').read_text())
            if state['state'] != 'finished' or state['exit_code'] != 0:
                raise ValueError('Old supervisor is not successfully finished')
            new.rename(old)
        if (new / 'worker-state.json').exists():
            # Recover a controller timeout after Popen without launching twice.
            state = json.loads((new / 'worker-state.json').read_text())
            atomic_json(transition, {'phase': 'launched', 'supervisor': state['supervisor'],
                                     'reconciled': True, 'at': time.time()})
            print(transition.read_text())
            return
        seed_continuation(old, new, 20)
        env = dict(os.environ, THINKINGCAP_COMPLETION_DRIVEN='1')
        with (new / 'supervisor.log').open('ab') as log:
            child = subprocess.Popen(['python3', 'experiments/worker_monitor.py', 'run',
                '--entry', 'experiments/shorterbetter/run-completion.sh', '--output', 'output',
                '--cwd', str(ROOT)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
        atomic_json(transition, {'phase': 'launched', 'supervisor_pid': child.pid, 'at': time.time()})
        print(transition.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['watch', 'resume'])
    args = parser.parse_args()
    config = json.loads((ROOT / 'completion-authorization.json').read_text())
    if config.get('authorized') is not True or config.get('scope') != 'finish_registered_80_updates':
        raise ValueError('No scoped completion authorization')
    globals()[args.action](config)


if __name__ == '__main__':
    main()
