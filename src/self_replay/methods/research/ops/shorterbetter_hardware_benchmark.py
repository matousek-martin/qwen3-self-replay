"""Prepare/stage/run one of three matched bounded GPU benchmarks.

Preparation is offline; only the explicit run subcommand calls the canonical
allocator. Existing launch pointers and historical registrations are untouched.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.shorterbetter.hardware_benchmark import SETUPS, SCOPE, ROLLOUTS, MAX_HOURS, validate, compare
from experiments.shorterbetter.hardware import SCOPE as HARDWARE_SCOPE, ENTRY
from experiments.shorterbetter.config import training_arguments, assert_launch_ready
from experiments.shorterbetter.train import check_data, check_dataset_protocol, monitor_rows
from experiments.shorterbetter.prepare import sha
from experiments.shorterbetter.jsonl import read_rows
from experiments import cloud
from ops.isolated_cloud import seal, verify

BASE = ROOT/'lab/experiments/shorterbetter-multidomain'
TEMPLATE = BASE/'a100x2-20261028t154708z/protocol.json'


def build_protocol(setup, data, manifest):
    gpu, count, cap, _ = SETUPS[setup]
    p = json.loads(TEMPLATE.read_text())
    p.update(id='shorterbetter-hardware-comparison-'+setup, status='prepared_not_locked', paid_launch_ready=False,
             question='Which measured GPU configuration offers the best runtime/cost for our full-data trainer?',
             scope='Four full rollouts: one warmup plus three measured; not an efficacy experiment.')
    p['training'].update(rollout_batches=ROLLOUTS, optimizer_updates_total=ROLLOUTS*4,
                         training_completions_per_arm=ROLLOUTS*512, world_size_initial=count)
    rows = read_rows(data/'monitor.jsonl')
    ids = [r['id'] for domain in ('math', 'coding', 'tools', 'instructions')
           for r in [r for r in rows if r['task'] == domain][:4]]
    p['data'] = {'directory': str(data.relative_to(ROOT)), 'manifest_sha256': sha(data/'manifest.json'),
                 'audit_sha256': sha(data/'audit.json'), 'train_questions': manifest['counts']['train.jsonl'],
                 'train_sha256': manifest['files']['train.jsonl'], 'monitor_sha256': manifest['files']['monitor.jsonl'],
                 'cover_all_training_rows': True, 'monitor_subset_questions': 16, 'monitor_ids': ids,
                 'development_questions': manifest['counts']['development.jsonl'],
                 'confirmation_questions': manifest['counts']['confirmation.jsonl'],
                 'domain_counts': manifest['domain_counts'], 'screening': manifest['screening']}
    p['execution'].pop('accelerated_multidomain_benchmark', None)
    p['execution'].update(hardware_comparison=setup, world_size=count, gpu=gpu)
    p['monitoring'].update(every_rollout_batches=ROLLOUTS)
    p['budget'] = {'proposed_total_pilot_cap_usd': cap, 'cap_authorized': False, 'scope': SCOPE}
    p['authorization'] = {'preparation': True, 'hardware_comparison_requested': True,
                           'full_training_authorized': False, 'new_paid_jobs_started': False,
                           'source_user_instruction': 'Prepare the trainer for 2xA100, 2xH100 SXM, 4xH100 SXM after auditing concurrent changes.'}
    p['phases'] = ['Verify actual GPU memory/sampling/precision and sandbox.',
                   'Fresh base validation, four balanced rollouts, final validation; GPU utilization/topology capture.',
                   'Validate all-rank updates, exposure and complete checkpoint/export; archive and release.']
    p['lineage'] = {'kind': 'Fresh pinned base; full v5 dataset; matched prompts and seeds across hardware.'}
    validate(p)
    check_dataset_protocol(manifest, p)
    monitor_rows(data/'monitor.jsonl', p)
    return p


def prepare(directory, data):
    directory, data = directory.resolve(), data.resolve()
    manifest = check_data(data)
    if manifest['id'] != 'shorterbetter-four-domains-full-v1':
        raise ValueError('This comparison requires the new full v5 dataset')
    directory.mkdir(parents=True, exist_ok=False)
    from experiments.shorterbetter.multidomain.sampler import BalancedRepeatSampler
    import itertools
    rows = read_rows(data/'train.jsonl')
    sampler = BalancedRepeatSampler({'task': [r['task'] for r in rows]}, 1, 64, 1, 17, cover_all=True)
    indices = list(itertools.islice(iter(sampler), ROLLOUTS*64))
    expected = {str(i*4): sorted(rows[j]['id'] for j in indices[i*64:(i+1)*64]) for i in range(ROLLOUTS)}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%Sz').lower()
    plan = {'status': 'prepared_not_launched', 'created_at': datetime.now(timezone.utc).isoformat(),
            'data_directory': str(data.relative_to(ROOT)), 'setups': {}, 'rollouts_per_setup': ROLLOUTS,
            'warmup_rollouts': 1, 'measured_rollouts': 3, 'max_runtime_hours_per_setup': MAX_HOURS,
            'max_combined_reservation_usd': sum(s[2] for s in SETUPS.values()),
            'full_training_authorized': False, 'gpu_integration_verified': False}
    for setup in SETUPS:
        folder = directory/setup
        folder.mkdir()
        p = build_protocol(setup, data, manifest)
        p['data']['expected_train_prompt_ids_by_step'] = expected
        cloud.write(folder/'protocol.json', p)
        cloud.write(folder/'resolved-training-arguments.json', training_arguments(p, 'shorterbetter', 17, 'output/training'))
        plan['setups'][setup] = {'directory': setup, 'job_name': 'shorterbetter-scaled-hardware-'+setup+'-'+stamp,
                                 'gpu': SETUPS[setup][0], 'gpu_count': SETUPS[setup][1],
                                 'max_cost_usd': SETUPS[setup][2], 'max_gpu_rate': SETUPS[setup][3]}
    cloud.write(directory/'plan.json', plan)
    return plan


def verify_prepared(directory):
    directory = directory.resolve()
    plan = json.loads((directory/'plan.json').read_text())
    protocols = []
    for setup in SETUPS:
        p = json.loads((directory/setup/'protocol.json').read_text())
        validate(p)
        data = ROOT/p['data']['directory']
        manifest = check_data(data)
        if sha(data/'manifest.json') != p['data']['manifest_sha256'] or sha(data/'audit.json') != p['data']['audit_sha256']:
            raise ValueError('Prepared dataset evidence changed')
        check_dataset_protocol(manifest, p)
        monitor_rows(data/'monitor.jsonl', p)
        training_arguments(p, 'shorterbetter', 17, 'output/training')
        protocols.append(p)
    for p in protocols[1:]:
        for key in ('model', 'reward', 'data', 'arms', 'seeds', 'monitoring', 'stopping'):
            if p[key] != protocols[0][key]:
                raise ValueError('Unmatched comparison field: '+key)
        t = {k: v for k, v in p['training'].items() if k != 'world_size_initial'}
        if t != {k: v for k, v in protocols[0]['training'].items() if k != 'world_size_initial'}:
            raise ValueError('Training exposure differs across hardware')
    return plan


def stage(directory, setup, proof_path):
    """Freeze checked code/data without authorizing a rental or modifying old runs."""
    plan = verify_prepared(directory)
    proof = json.loads(proof_path.read_text())
    if proof.get('status') != 'passed' or not proof.get('cpu_ddp_world_sizes') == [2, 4]:
        raise ValueError('CPU regression and two/four-rank checks required')
    for relative, digest in proof['source_sha256'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Source changed after verification: '+relative)
    entry = plan['setups'][setup]
    folder = directory/setup
    if (folder/'launch.json').exists():
        launch = json.loads((folder/'launch.json').read_text())
        if verify(ROOT/launch['release']) != launch['payload_sha256']:
            raise ValueError('Release changed')
        return launch
    release = ROOT/'lab/isolated'/entry['job_name']
    release.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns('.git', '__pycache__', '.venv*', 'runs', '.env*', '*.log', 'launch')
    for tree in ('experiments', 'ThinkingCap-Test'):
        tree_ignore = shutil.ignore_patterns('.git', '__pycache__', '.venv*', 'runs', '.env*', '*.log', 'questions', 'img') if tree == 'ThinkingCap-Test' else ignore
        shutil.copytree(ROOT/tree, release/tree, ignore=tree_ignore)
    launch_dir = release/'experiments/shorterbetter/launch'
    launch_dir.mkdir()
    p = json.loads((folder/'protocol.json').read_text())
    data = ROOT/p['data']['directory']
    p['data']['directory'] = 'data/multidomain'
    p.update(status='locked', paid_launch_ready=True)
    p['budget']['cap_authorized'] = True
    cloud.write(launch_dir/'protocol.json', p)
    evidence = {k: True for k in ('data_audit', 'reward_parity', 'backend_cpu_parity', 'distributed_cpu_check',
                  'stopping_tests', 'backup_release_integration', 'four_domain_verifiers', 'balanced_sampler', 'data_domains')}
    evidence.update(protocol_sha256=sha(launch_dir/'protocol.json'), budget_reservation=False, pilot_cap_enforced=False,
                    trainer_gpu_smoke=False, distributed_gpu_check=False, resume_equivalence=False,
                    matched_validation=False, export_reload=False, cost_projection=False,
                    cpu_verification_sha256=sha(proof_path))
    assert_launch_ready(p, dict(evidence, budget_reservation=True, pilot_cap_enforced=True))
    cloud.write(launch_dir/'readiness.json', evidence)
    shutil.copy2(proof_path, launch_dir/'cpu-verification.json')
    old = ROOT/'lab/isolated/shorterbetter-retry4-20261027T075430Z/lab/experiments/data'
    payload = release/'lab/experiments/data'
    payload.mkdir(parents=True)
    for tree in ('wheels', 'shorterbetter/sources'):
        shutil.copytree(old/tree, payload/tree, copy_function=os.link)
    (payload/'pilot.jsonl').write_text('{}\n')
    (payload/'multidomain').mkdir()
    for filename in ('train.jsonl', 'monitor.jsonl', 'manifest.json', 'audit.json'):
        shutil.copy2(data/filename, payload/'multidomain'/filename)
    shutil.copy2(old.parents[1]/'gpu-requirements.lock', release/'lab/gpu-requirements.lock')
    digest = seal(release)
    launch = dict(entry, release=str(release.relative_to(ROOT)), payload_sha256=digest,
                  scope=SCOPE, setup=setup, max_runtime_hours=MAX_HOURS, pilot_rollouts=ROLLOUTS,
                  full_training_authorized=False, status='staged_not_launched')
    cloud.write(folder/'launch.json', launch)
    return launch


def run(directory, setup):
    launch = json.loads((directory/setup/'launch.json').read_text())
    release = ROOT/launch['release']
    if verify(release) != launch['payload_sha256']:
        raise ValueError('Sealed source payload changed')
    name = launch['job_name']
    jobs = json.loads((cloud.STATE/'ledger.json').read_text())['jobs']
    if name in jobs:
        raise RuntimeError('Job already registered; reconcile it instead of renting a duplicate')
    if cloud.list_pods():
        raise RuntimeError('Sequential comparison requires previous GPU pods to be released')
    if any(n.startswith('shorterbetter-') and not j.get('released_at') for n, j in jobs.items()):
        raise RuntimeError('Unreleased ShorterBetter reservation needs reconciliation')
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('Need at least20GiB free for verified checkpoint backup')
    authorization = dict(launch, authorized=True, source_user_instruction='User requested these three matched hardware tests; no full training campaign.')
    cloud.write(cloud.STATE/'authorizations'/f'{name}.json', authorization)
    command = [sys.executable, str(ROOT/'ops/isolated_cloud.py'), 'run', '--release', str(release),
               '--name', name, '--entry', ENTRY, '--gpu', launch['gpu'], '--gpu-count', str(launch['gpu_count']),
               '--hardware-scope', HARDWARE_SCOPE, '--hours', '1', '--max-runtime-hours', str(MAX_HOURS),
               '--max-cost-usd', str(launch['max_cost_usd']), '--data-upload-timeout', '900', '--direct-pinned-wheels']
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'stage', 'run', 'compare'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=BASE/'data-v5-full')
    parser.add_argument('--setup', choices=SETUPS)
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare': result = prepare(args.directory, args.data)
    elif args.action == 'verify': result = verify_prepared(args.directory)
    elif args.action == 'stage':
        if not args.setup or not args.proof: parser.error('stage requires --setup and --proof')
        result = stage(args.directory, args.setup, args.proof)
    elif args.action == 'run':
        if not args.setup: parser.error('run requires --setup')
        result = run(args.directory, args.setup)
    else:
        plan = json.loads((args.directory/'plan.json').read_text())
        jobs = json.loads((cloud.STATE/'ledger.json').read_text())['jobs']
        rows = []
        for entry in plan['setups'].values():
            output = cloud.STATE/entry['job_name']/'output'
            file = output/'hardware-benchmark.json'
            job = jobs.get(entry['job_name'], {})
            pilot = output/'multidomain-pilot-result.json'
            if (file.exists() and pilot.exists() and job.get('released_at') and job.get('backup_verified')
                    and json.loads(pilot.read_text()).get('status') == 'passed'):
                cloud.verify_backup(entry['job_name'])
                row = json.loads(file.read_text())
                row['whole_rental_conservative_estimate_usd'] = job.get('charged_estimate_usd')
                row['whole_rental_hours'] = (job['released_at']-job['created_at'])/3600
                rows.append(row)
        result = compare(rows)
        result['measurements'] = rows
        cloud.write(args.directory/'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
