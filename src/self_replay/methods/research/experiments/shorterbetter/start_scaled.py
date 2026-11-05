"""Launch pinned TRL using torchrun DDP; global batches stay unchanged."""
import json
from pathlib import Path
import subprocess
import sys
import time
from .config import assert_launch_ready
from .prepare import sha
from .distributed import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / 'experiments/shorterbetter/launch'
    path = directory / 'protocol.json'
    protocol = json.loads(path.read_text())
    lease = json.loads((ROOT / 'worker-lease.json').read_text())
    evidence = json.loads((directory / 'readiness.json').read_text())
    if evidence['protocol_sha256'] != sha(path):
        raise ValueError('Protocol hash mismatch')
    cap = protocol['budget']['proposed_total_pilot_cap_usd']
    evidence['budget_reservation'] = lease['max_cost_usd'] == cap
    evidence['pilot_cap_enforced'] = bool(lease.get('hard_deadline') and lease['hard_deadline'] > time.time() + 600)
    evidence['completion_driven_authorized'] = (lease.get('completion_budget_scope') == 'finish_registered_1200_updates'
        and cap is None and lease.get('hard_deadline') is None)
    if (lease['gpu_count'] != protocol['execution']['world_size']
            or lease['gpu'] != protocol['execution']['gpu']):
        raise RuntimeError('Allocation and registered hardware differ')
    if protocol['execution'].get('verification_only'):
        from .verify_cheap import run
        run(protocol, path, lease, evidence)
        return
    if protocol['execution'].get('multidomain'):
        if protocol['execution'].get('full_run_continuation'):
            from .multidomain.continuation import run
        elif protocol['execution'].get('multidomain_full_training'):
            from .multidomain.full_run import run
        else:
            from .multidomain.pilot import run
        run(protocol, path, lease, evidence)
        return
    if protocol['execution'].get('resume_repair'):
        from .resume_repair import run
        run(protocol, path, lease, evidence)
        return
    gated = protocol['execution'].get('gated_full_training') is True
    if gated:
        from .launch_checks import engineering_protocol
        engineering = engineering_protocol(protocol)
        engineering_path = ROOT/'output/engineering-protocol.json'
        atomic_json(engineering_path, engineering)
        engineering_evidence = dict(evidence, protocol_sha256=sha(engineering_path))
        assert_launch_ready(engineering, engineering_evidence)
    else:
        assert_launch_ready(protocol, evidence)
    evidence_path = ROOT / 'output/readiness-at-start.json'
    atomic_json(evidence_path, evidence)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc_per_node=' + str(lease['gpu_count']), '--max_restarts=0',
        '-m', 'experiments.shorterbetter.train', '--protocol', str(path), '--arm', 'shorterbetter',
        '--seed', str(protocol['seeds'][0]), '--execute', '--readiness', str(evidence_path),
        '--job-name', lease['job_name']]
    main_command = list(command)
    if gated:
        engineering_evidence_path = ROOT/'output/engineering-readiness.json'
        atomic_json(engineering_evidence_path, engineering_evidence)
        command[command.index('--protocol')+1] = str(engineering_path)
        command[command.index('--readiness')+1] = str(engineering_evidence_path)
    atomic_json(ROOT/'output/phase.json', {'phase': 'startup_verification', 'time': time.time()})
    # An isolated actual optimizer update catches runtime/numerical failures
    # before the expensive baseline. The measured run reloads the fresh model.
    subprocess.run(command + ['--out', str(ROOT / 'output/startup-check'), '--startup-check'], check=True)
    for rank in range(lease['gpu_count']):
        result = json.loads((ROOT / f'output/startup-check/rank-{rank:03d}/parameter-update-check.json').read_text())
        if result['status'] != 'passed':
            raise RuntimeError('Distributed startup update verification failed')
    if gated:
        from .launch_checks import seed_resume, verify_gpu_checks
        source, resumed = ROOT/'output/engineering-check', ROOT/'output/resume-check'
        atomic_json(ROOT/'output/phase.json', {'phase': 'throughput_verification', 'time': time.time()})
        subprocess.run(command + ['--out', str(source)], check=True)
        atomic_json(ROOT/'output/phase.json', {'phase': 'gpu_resume_verification', 'time': time.time()})
        seed_resume(source, resumed, lease['gpu_count'])
        subprocess.run(command + ['--out', str(resumed), '--resume', str(resumed/'checkpoint-28')], check=True)
        evidence.update(verify_gpu_checks(source, resumed, protocol, lease))
        assert_launch_ready(protocol, evidence)
        atomic_json(evidence_path, evidence)
    atomic_json(ROOT/'output/phase.json', {'phase': 'main_training', 'time': time.time()})
    subprocess.run(main_command + ['--out', str(ROOT / 'output/training')], check=True)
    atomic_json(ROOT/'output/phase.json', {'phase': 'finished', 'time': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic_json(ROOT/'output/launch-failure.json', {'type': type(exc).__name__,
            'reason': 'runtime_projection_exceeded' if 'exceeds the authorized' in str(exc) else 'engineering_or_training_failure',
            'time': time.time()})
        raise
