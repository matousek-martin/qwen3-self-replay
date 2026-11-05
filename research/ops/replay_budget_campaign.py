"""Adopt the registered replay campaign using the current shared budget controller.

Scientific source/data remain sealed. Only controller orchestration is replaced;
known jobs are watched, never allocated or trained again.
"""
import json
from pathlib import Path
import subprocess
import sys

from isolated_cloud import ROOT, verify
import experiments
from experiments import cloud

RELEASE = ROOT / 'lab/isolated/launch-20261025/replay-selection'


def main():
    verify(RELEASE)
    experiments.__path__ = [str(RELEASE / 'experiments'), *experiments.__path__]
    from experiments import replay_selection_campaign as original
    original.LOCAL = ROOT / 'lab/experiments/replay-selection-v1'
    with original.campaign_lock():
        launch = original.read_launch()
        original.recipe.load_protocol(original.STAGED / 'protocol.json')
        path = original.LOCAL / 'campaign.json'
        state = json.loads(path.read_text())
        if original.digest(original.STAGED / 'launch.json') != state['launch_sha256']:
            raise RuntimeError('Registered replay launch changed')
        print(json.dumps({'usable_usd': cloud.USABLE, 'ceiling_usd': cloud.CEILING,
                          'release': str(RELEASE), 'state': str(cloud.STATE)}), flush=True)
        try:
            for seed in launch['seeds']:
                verify(RELEASE)
                original.verify_sources(launch)
                trial = state['trials'][str(seed)]
                with cloud.ledger() as ledger:
                    job = dict(ledger['jobs'].get(trial['job'], {}))
                action = original.existing_action(job)
                if action != 'verify':
                    state['state'] = 'running'
                    trial['state'] = 'running' if action == 'watch' else 'allocating'
                    original.write(path, state)
                    command = [sys.executable, str(ROOT / 'ops/isolated_cloud.py'), action,
                               '--release', str(RELEASE), '--name', trial['job'],
                               '--entry', launch['entries'][str(seed)],
                               '--hours', str(launch['initial_reservation_hours_per_worker'])]
                    with (original.LOCAL / f'seed{seed}.controller.log').open('a') as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                with cloud.ledger() as ledger:
                    job = dict(ledger['jobs'][trial['job']])
                result = cloud.verify_backup(trial['job'])
                role = json.loads((cloud.STATE / trial['job'] / 'output/role-result.json').read_text())
                if (result['exit_code'] != 0 or job.get('state') != 'complete'
                        or not job.get('released_at') or not job.get('backup_verified')
                        or role.get('seed') != seed or not role.get('development_complete')
                        or role.get('protocol_sha256') != launch['protocol_sha256']):
                    raise RuntimeError('Incomplete registered replay result')
                trial.update(state='complete', released=True, backup_verified=True,
                             charged_estimate_usd=job['charged_estimate_usd'])
                original.write(path, state)
            state.update(state='development_complete', completed_at=original.now())
            original.write(path, state)
        except BaseException as exc:
            state.update(state='needs_inspection', error=str(exc), updated_at=original.now())
            original.write(path, state)
            raise


if __name__ == '__main__':
    main()
