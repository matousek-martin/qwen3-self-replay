"""Serial development evaluations; never opens the confirmation partition."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from cloud import ROOT, STATE, write
from confirmation import analyze

STUDY = ROOT / 'lab/experiments/study'
DATA = ROOT / 'lab/experiments/data/study'
INITIAL = ['sft-short', 'sft-random', 'sft-compatible', 'dpo', 'sft-dpo-matched',
           'dpo-16', 'dpo-32', 'dpo-48']
RL = ['grpo', 'dr-grpo', 'rl-fixed', 'rl-adaptive', 'rl-difficulty', 'rl-progressive', 'rl-replay']


def stage(job):
    name = job['arm']
    source = Path(job['output']) / 'training' / name / 'adapter'
    status = json.loads((source.parent / 'status.json').read_text())
    if status['status'] != 'completed' or status['completed_groups'] != 128:
        raise RuntimeError('Cannot label partial training as completed')
    target = DATA / 'adapters' / name
    digest = hashlib.sha256((source / 'adapter_model.safetensors').read_bytes()).hexdigest()
    config_digest = hashlib.sha256((source / 'adapter_config.json').read_bytes()).hexdigest()
    if target.exists():
        if hashlib.sha256((target / 'adapter_model.safetensors').read_bytes()).hexdigest() != digest:
            raise RuntimeError('Refusing to overwrite a staged candidate')
        if hashlib.sha256((target / 'adapter_config.json').read_bytes()).hexdigest() != config_digest:
            raise RuntimeError('Staged adapter configuration differs from its training archive')
    else:
        temporary = STUDY / 'staging' / name
        shutil.copytree(source, temporary)
        temporary.rename(target)
    path = DATA / 'adapters.json'
    manifest = json.loads(path.read_text())
    manifest[name] = {'source': str(source), 'checkpoint_sha256': digest,
                      'config_sha256': config_digest,
                      'purpose': 'completed matched128-group RL arm', 'job': job['job']}
    write(path, manifest)


def evaluate(names, stamp, index):
    write(DATA / 'evaluation.json', {'schema_version': 1, 'split_manifest': 'splits.json',
          'subset': 'tuning', 'candidates': ['base'] + names, 'samples': 2,
          'workers': 4, 'candidate_timeout_seconds': 0})
    name = 'study-eval-' + str(index).zfill(2) + '-' + stamp
    with (STATE / (name + '.controller.log')).open('w') as log:
        process = subprocess.run([sys.executable, str(ROOT / 'experiments/cloud.py'), 'run',
            '--name', name, '--entry', 'experiments/run-study-eval.sh', '--hours', '6'],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    job = json.loads((STATE / 'ledger.json').read_text())['jobs'].get(name, {})
    output = STATE / name / 'output/study'
    complete = job.get('state') == 'complete' or (job.get('state') == 'delete_requested' and job.get('release_reason') == 'complete')
    success = bool(process.returncode == 0 and complete and job.get('backup_verified'))
    result = {'job': name, 'candidates': names, 'success': success, 'output': str(output),
              'cleanup_confirmed': bool(job.get('released_at')), 'analyses': {}}
    if success:
        try:
            finished = json.loads((output / 'study-result.json').read_text())
            if not finished.get('complete') or set(finished.get('completed_candidates', [])) != set(['base'] + names):
                raise ValueError('Study did not complete every requested candidate')
            for candidate in names:
                report = analyze(output / 'runs/base', output / 'runs' / candidate,
                                 DATA / 'splits.json', 'tuning')
                path = STUDY / 'development' / (candidate + '.json')
                write(path, report)
                result['analyses'][candidate] = str(path)
        except (OSError, ValueError, KeyError) as exc:
            result.update(success=False, error=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Continue a terminal needs_review campaign without repeating successful batches')
    args = parser.parse_args()
    path = STUDY / 'evaluation-campaign.json'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if path.exists():
        state = json.loads(path.read_text())
        if not args.resume or state['state'] != 'needs_review':
            raise SystemExit('Existing campaign must be terminal needs_review and explicitly resumed')
        jobs = json.loads((STATE / 'ledger.json').read_text())['jobs']
        if any(not jobs.get(batch['job'], {}).get('released_at') for batch in state['batches']):
            raise SystemExit('Previous evaluation rentals must have confirmed cleanup before resuming')
        state.setdefault('resumed_at', []).append(stamp)
        state['state'] = 'running'
    else:
        if args.resume:
            raise SystemExit('No existing study to resume')
        state = {'started_at': stamp, 'state': 'running', 'batches': [], 'evaluated': [],
                 'confirmation': 'not opened; awaiting all approach results and finalist selection'}
    write(path, state)
    pending = [name for name in INITIAL if name not in state['evaluated']]
    if pending:
        initial = evaluate(pending, stamp, len(state['batches']))
        state['batches'].append(initial)
        if not initial['success']:
            state['state'] = 'needs_review'
            write(path, state)
            return
        state['evaluated'].extend(pending)
        write(path, state)
    while not all(name in state['evaluated'] for name in RL):
        training = json.loads((STUDY / 'training-campaign.json').read_text())
        ready = [j for j in training['jobs'] if j['success'] and j['arm'] not in state['evaluated']]
        if len(ready) >= 2 or (ready and training['state'] != 'running'):
            for job in ready:
                stage(job)
            batch = evaluate([j['arm'] for j in ready], stamp, len(state['batches']))
            state['batches'].append(batch)
            if not batch['success']:
                state['state'] = 'needs_review'
                write(path, state)
                return
            state['evaluated'].extend(j['arm'] for j in ready)
            write(path, state)
        elif training['state'] != 'running':
            state.update(state='needs_review', reason='Training ended without all required arms')
            write(path, state)
            return
        else:
            live = subprocess.run(['systemctl', 'is-active', '--quiet',
                                   'thinkingcap-study-training.service']).returncode == 0
            if not live:
                state.update(state='needs_review', reason='Training controller is no longer active')
                write(path, state)
                return
            state['last_verified_training_service_at'] = datetime.now(timezone.utc).isoformat()
            write(path, state)
            time.sleep(45)
    state['state'] = 'development_complete'
    write(path, state)


if __name__ == '__main__':
    main()
