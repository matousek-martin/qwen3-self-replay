"""Bounded GPU pilots. Credentials stay on the control server.

Run via a systemd service; an independent guard enforces the spending limit.
Only this campaign's uniquely named pods can be managed by this module.
"""
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import math
import re
import secrets
import shutil
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ops'))
from runpod import api as runpod_api, ssh_args, summary
from ops import vastai


def api(path, method='GET', payload=None):
    if path.startswith('pods/vast:'):
        return vastai.pod_api(path, method, payload)
    return runpod_api(path, method, payload)


def resolve_uv():
    """Resolve the controller binary before renting, including systemd's small PATH."""
    explicit = os.environ.get('SELF_REPLAY_UV')
    if explicit:
        candidates = [Path(explicit)]
    else:
        found = shutil.which('uv')
        candidates = ([Path(found)] if found else []) + [Path.home() / '.local/bin/uv']
    for candidate in candidates:
        if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError('Executable uv is required before allocation; set SELF_REPLAY_UV to its absolute path')

STATE = ROOT / 'lab/experiments/cloud'
CEILING = 0.0
USABLE = CEILING  # Public reproductions require a fresh finite authorization.
MAX_GPU_RATE = 1.00
MAX_LIVE_PODS = 4  # Global concurrency bound; each public job needs its own finite grant.
# User-required hardware. No fallback, including when capacity is unavailable.
GPU_CANDIDATES = ('NVIDIA GeForce RTX 4090',)
ALLOCATION_WAIT_SECONDS = 600
ABSENCE_REQUIRED_READS = 12
ABSENCE_REQUIRED_SECONDS = 300
LATE_ARRIVAL_WATCH_SECONDS = 24 * 3600


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


@contextmanager
def ledger():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = STATE / 'ledger.json'
        data = json.loads(path.read_text()) if path.exists() else {'ceiling_usd': CEILING, 'jobs': {}}
        yield data
        write(path, data)


def update(name, **values):
    with ledger() as state:
        state['jobs'][name].update(values)


def liability(jobs):
    return sum(j['charged_estimate_usd'] if j.get('released_at') else
               max(j['reserved_usd'], max(0, time.time() - j['created_at']) / 3600 * j['budget_rate'])
               for j in jobs.values())


def budget_allows(jobs):
    from experiments.shorterbetter.budget import ticket
    ordinary = {}
    for name, job in jobs.items():
        authorized = ticket(STATE, name, job)
        if authorized:
            if authorized.get('max_cost_usd') is not None and liability({name: job}) > authorized['max_cost_usd']:
                return False
        else:
            ordinary[name] = job
    return liability(ordinary) <= USABLE


def placement_constraints(name, quote):
    """Optional operational placement policy; never fall back outside it."""
    path = STATE / 'placement-policy.json'
    if not path.exists():
        return None
    policy = json.loads(path.read_text())
    if not any(name.startswith(prefix) for prefix in policy['job_prefixes']):
        return None
    allowed = set(policy['data_center_ids'])
    return sorted(dc['id'] for dc in quote.get('dataCenters', [])
                  if dc['id'] in allowed and dc.get('availability') in ('LOW', 'MEDIUM', 'HIGH'))


def remote(pod, command, timeout=120):
    return subprocess.run(ssh_args(pod) + [command], text=True, capture_output=True,
                          check=True, timeout=timeout).stdout


def sync(pod, source, target, upload=False, exclusions=(), timeout=90, preserve_hardlinks=False):
    """Bound total wall time, not only rsync's idle-I/O timeout."""
    conn = ssh_args(pod)
    host, transport = conn[-1], shlex.join(conn[:-1])
    src = str(source) if upload else host + ':' + str(source)
    dst = host + ':' + str(target) if upload else str(target)
    args = ['rsync', '-az', '--timeout=30', '-e', transport]
    if preserve_hardlinks:
        args.append('-H')
    if not upload:
        # Full optimizer checkpoints can exceed one bounded transfer. Preserve
        # partial blocks separately; only complete files replace live backups.
        args += ['--partial-dir=.rsync-partial']
    for pattern in exclusions:
        args += ['--exclude', pattern]
    subprocess.run(args + [src, dst], check=True, text=True, capture_output=True,
                   timeout=max(1, timeout))


def backup(pod, name, timeout=90):
    target = STATE / name / 'output'
    target.mkdir(parents=True, exist_ok=True)
    with ledger() as state:
        job=state['jobs'].get(name,{})
        bounded_mirror = bool(job.get('incremental_funding') or job.get('bounded_checkpoint_mirror'))
        hardlinks=job.get('preserve_hardlinks',False)
    options = {'preserve_hardlinks': True} if hardlinks else {}
    if hardlinks and remote(pod, 'test -f /workspace/project/output/exit.json && echo finished || echo active').strip() == 'active':
        # Engineering models are consumed and hashed by resume verification;
        # scientific snapshots and all logs are always mirrored. On failure,
        # the final archive includes any engineering artifacts still present.
        options['exclusions'] = ('/verification-*/checkpoint-*', '/verification-*/model')
    sync(pod, '/workspace/project/output/', str(target) + '/', timeout=timeout,
         **options)
    try:
        from experiments.baseline_cache import ingest
        ingest(target,STATE.parent/'baseline-cache')
    except (OSError,ValueError,KeyError) as exc:
        # Partial backup or corrupt cache must not interfere with archive/release safeguards.
        update(name,baseline_cache_import_error=str(exc))
    if bounded_mirror:
        from experiments.shorterbetter.backup_retention import maintain
        maintain(sys.modules[__name__], pod, target)


def verify_backup(name):
    """Require the final marker, nonempty checksums, and the hashed result."""
    base = STATE / name / 'output'
    manifest = base / 'SHA256SUMS'
    lines = manifest.read_text().splitlines()
    if not lines:
        raise RuntimeError('Empty archive checksum manifest')
    seen = set()
    for line in lines:
        digest, relative = line.split(maxsplit=1)
        path = (base / relative).resolve()
        if not path.is_relative_to(base.resolve()) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise RuntimeError('Invalid archive path or digest')
        if path in seen or not path.is_file():
            raise RuntimeError('Duplicate or missing archive file: ' + relative)
        seen.add(path)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != digest:
            raise RuntimeError('Incomplete backup: ' + relative)
    if (base / 'result.json').resolve() not in seen:
        raise RuntimeError('Archive manifest does not cover result.json')
    if (base / 'result.json').read_bytes() != (base / 'exit.json').read_bytes():
        raise RuntimeError('Completion marker does not match hashed result')
    result = json.loads((base / 'result.json').read_text())
    if type(result.get('exit_code')) is not int:
        raise RuntimeError('Invalid exit code in result')
    supervisor_path = base / 'worker-state.json'
    if supervisor_path.exists():
        supervisor = json.loads(supervisor_path.read_text())
        if supervisor.get('state') != 'finished':
            raise RuntimeError('Supervisor has not confirmed worker completion')
        if supervisor.get('exit_code') != result['exit_code']:
            raise RuntimeError('Archive exit status disagrees with supervisor exit status')
    return result


@contextmanager
def job_lock(name):
    """Serialize controller/guard cleanup without waiting behind another cleanup."""
    path = STATE / name / 'release.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def is_missing(exc):
    return isinstance(exc, RuntimeError) and any(p in str(exc) for p in ('Runpod HTTP 404:', 'Vast HTTP 404:'))


def release(name, reason, verified=False):
    with job_lock(name) as acquired:
        if not acquired:
            return
        with ledger() as state:
            job = dict(state['jobs'][name])
        if not job.get('pod_id') or job.get('released_at'):
            return
        # The ID was obtained from our successful create or an exact random-name
        # reconciliation. Do not make deletion depend on another GET/SSH request.
        if not job.get('identity_confirmed'):
            pod = api('pods/' + job['pod_id'])
            if pod['name'] != job['pod_name']:
                raise RuntimeError('Pod identity mismatch')
            update(name, identity_confirmed=True)
        if verified:
            verify_backup(name)
        # No backup here: deadline cleanup must never wait for artifact transfer.
        # The controller continuously syncs, and verifies final output beforehand.
        update(name, delete_requested_at=time.time(), state='deleting')
        try:
            api('pods/' + job['pod_id'], 'DELETE')
        except Exception as exc:
            if not is_missing(exc):
                update(name, delete_error=str(exc))
                raise
        # DELETE may be accepted asynchronously. Keep the full reservation until
        # the provider confirms termination or absence; the guard reconciles it.
        update(name, state='delete_requested', release_reason=reason,
               backup_verified=verified or job.get('backup_verified', False))
        confirm_release(name)


def confirm_release(name):
    with ledger() as state:
        job = dict(state['jobs'][name])
    try:
        pod = api('pods/' + job['pod_id'])
        terminal = pod.get('status') in ('TERMINATED', 'DELETED')
    except Exception as exc:
        if not is_missing(exc):
            update(name, confirmation_error=str(exc))
            return False
        terminal = True
    if not terminal:
        return False
    elapsed = max(0, time.time() - job['created_at'])
    update(name, released_at=time.time(), state=job.get('release_reason', 'released'),
           charged_estimate_usd=round(elapsed / 3600 * job['budget_rate'], 4))
    return True


def renew_budget(name):
    """Reserve another hour atomically; a lease is a cost bound, not a run timer."""
    with ledger() as state:
        job = state['jobs'][name]
        now = time.time()
        if job.get('released_at') or job.get('delete_requested_at'):
            return False
        hard_deadline = job.get('hard_deadline')
        if hard_deadline is not None and now >= hard_deadline - 300:
            job['budget_extension_denied_at'] = now
            return False
        if job['deadline'] - now > 1800:
            return True
        deadline = max(job['deadline'], now) + 3600
        if hard_deadline is not None:
            deadline = min(deadline, hard_deadline)
        reserved = max(job['reserved_usd'], (deadline - job['created_at']) / 3600 * job['budget_rate'])
        proposed = dict(job, deadline=deadline, reserved_usd=reserved)
        jobs = dict(state['jobs'], **{name: proposed})
        allowed = budget_allows(jobs)
        if job.get('hardware_scope') == 'coding-curriculum-256-v1':
            from experiments.coding_curriculum.budget import ticket as curriculum_ticket
            allowed = bool(curriculum_ticket(STATE,name,proposed) and reserved <= job['max_cost_usd'])
        if job.get('hardware_scope') == 'coding-abc-768-v1':
            from experiments.coding_abc.budget import ticket as abc_ticket
            allowed = bool(abc_ticket(STATE,name,proposed) and reserved <= job['max_cost_usd'])
        if job.get('hardware_scope') == 'coding-frontier-768-v1':
            # This named campaign has its own finite, prepaid envelope. An
            # unrelated job falling back to the legacy ledger must not revoke
            # an already funded survey lease. The ticket includes predecessor
            # charges and the three-arm60USD bound; the account guard remains.
            from experiments.coding_frontier.budget import ticket as frontier_ticket
            authorization = frontier_ticket(STATE, name, proposed)
            allowed = bool(authorization and reserved <= authorization['max_cost_usd']
                           and liability({name: proposed}) <= authorization['max_cost_usd'])
        if not allowed and not completion_budget_authorized(name, job):
            job['budget_extension_denied_at'] = now
            return False
        job.setdefault('budget_extensions', []).append({
            'at': now, 'previous_deadline': job['deadline'], 'deadline': deadline,
            'reserved_usd': reserved, 'reason': 'worker remains active'})
        job.update(deadline=deadline, reserved_usd=reserved)
        return True


def completion_budget_authorized(name, job):
    """A scoped user override may renew an existing rental, never allocate another."""
    override = job.get('completion_budget_override', {})
    return (override.get('authorized') is True and override.get('job_name') == name
            and override.get('pod_id') == job.get('pod_id')
            and override.get('scope') == 'finish_registered_80_updates'
            and job.get('completion_driven') is True)


def worker_probe(pod):
    command = 'python3 /workspace/project/experiments/worker_monitor.py probe --directory /workspace/project/output'
    return json.loads(remote(pod, command, timeout=30))


def completion_archive_authorized(name, job, probe):
    """Keep named completed jobs until copying finishes, within their existing cap."""
    from experiments.shorterbetter.budget import ticket
    authorization = ticket(STATE, name, job)
    allowed = authorization and authorization.get('scope') in (
        'finish_registered_1200_updates', 'shorterbetter_4090_verification', 'shorterbetter_multidomain_pilot',
        'shorterbetter_multidomain_a100x2_benchmark', 'shorterbetter_hardware_comparison_v1', 'shorterbetter_vast_bounded_v1', 'replay_vast_bounded_v1', 'replay_runpod_resume_v1', 'coding-length-probe-v1', 'coding-replay-768-v1', 'opencoder-sft-768-v1', 'opencoder-mode-probe-v1', 'coding-frontier-768-v1', 'coding-abc-768-v1', 'replay_runpod_evaluation_v1')
    if authorization and authorization.get('scope') in ('shorterbetter_4090_verification', 'shorterbetter_multidomain_pilot',
                                                      'shorterbetter_multidomain_a100x2_benchmark', 'shorterbetter_hardware_comparison_v1', 'shorterbetter_vast_bounded_v1', 'replay_vast_bounded_v1', 'replay_runpod_resume_v1', 'coding-length-probe-v1', 'coding-replay-768-v1', 'opencoder-sft-768-v1', 'opencoder-mode-probe-v1', 'coding-frontier-768-v1', 'coding-abc-768-v1', 'replay_runpod_evaluation_v1'):
        allowed = allowed and time.time() < (job.get('hard_deadline') or 0) - 300
    return bool(allowed and probe.get('worker_status') == 'dead'
                and probe.get('exit_marker', {}).get('status') == 'valid')


def protect_replay_backup_space(pod, name, job):
    """Shorten this funded continuation if its local backup reserve gets low."""
    continuation=((name.startswith('replay-scaled-runpod-resume-') and job.get('provider')=='runpod')
                  or (name.startswith('replay-scaled-vast-resume1024-') and job.get('provider')=='vast'))
    if (not continuation or job.get('hardware_scope')!='replay-scaled-a100-v1' or job.get('storage_stop_requested_at')):
        return
    space=os.statvfs(STATE)
    free=(space.f_bfree if os.geteuid()==0 else space.f_bavail)*space.f_frsize
    if free>=3*1024**3:return
    deadline=min(job['hard_deadline'],time.time()+900)
    script=("import json,os;from pathlib import Path;"
            "p=Path('/workspace/project/worker-lease.json');v=json.loads(p.read_text());"
            f"assert v['job_name']=={name!r};v['hard_deadline']=min(v['hard_deadline'],{deadline!r});"
            "t=p.with_suffix('.storage-tmp');t.write_text(json.dumps(v));os.replace(t,p)")
    try:
        remote(pod,'python3 -c '+shlex.quote(script),timeout=20)
        update(name,hard_deadline=deadline,storage_stop_requested_at=time.time(),backup_free_bytes=free)
    except (RuntimeError,OSError,subprocess.SubprocessError) as exc:
        update(name,storage_guard_error=str(exc))


def observe_completion(pod, name, controller=True):
    """One bounded poll; a missing controller heartbeat never proves worker death."""
    if controller:
        update(name, heartbeat=time.time())
    with ledger() as state:
        job = dict(state['jobs'][name])
        deadline = state['jobs'][name]['deadline']
        preparing = state['jobs'][name].get('completion_continuation', {}).get('phase') == 'preparing'
    protect_replay_backup_space(pod,name,job)
    if vast_spending_exhausted(name, job):
        release(name, 'vast_account_spending_limit')
        return True
    if preparing:
        from ops.shorterbetter_completion import maybe_continue
        maybe_continue(sys.modules[__name__], pod, name, None)
        return False
    if time.time() >= deadline:
        # Cost enforcement must not wait for a slow artifact transfer. An
        # active worker can renew; an unreachable one cannot spend unboundedly.
        try:
            initial_probe = worker_probe(pod)
            with ledger() as state:
                initial_job = dict(state['jobs'][name])
            alive = (initial_probe.get('worker_status') == 'alive'
                     or completion_archive_authorized(name, initial_job, initial_probe))
        except (subprocess.SubprocessError, OSError, ValueError, RuntimeError):
            alive = False
        if not alive or not renew_budget(name):
            release(name, 'budget_limit' if alive else 'budget_lease_expired_unreachable')
            return True
    try:
        backup_timeout = 90
        with ledger() as state:
            current = state['jobs'][name]
            if current.get('backup_timeout_seconds') is not None:
                backup_timeout = min(300, max(1, int(current['backup_timeout_seconds'])),
                                     max(1, current['deadline'] - time.time() - 60))
        backup(pod, name, timeout=backup_timeout)
        update(name, backup_at=time.time())
    except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
        update(name, backup_error=str(exc))
    if (STATE / name / 'output/exit.json').exists():
        try:
            if not (STATE / name / 'output/worker-state.json').is_file():
                raise RuntimeError('Completion-driven archive is missing supervisor state')
            result = verify_backup(name)
        except (OSError, ValueError, RuntimeError) as exc:
            update(name, verification_error=str(exc))
        else:
            with ledger() as state:
                repair_job = dict(state['jobs'][name])
            if (result['exit_code'] != 0 and repair_job.get('technical_repair_authorized') is True
                    and time.time() < repair_job.get('technical_repair_hold_until', 0)):
                from experiments.shorterbetter.budget import ticket
                authorization = ticket(STATE, name, repair_job)
                if authorization and authorization.get('scope') == 'finish_registered_1200_updates':
                    update(name, backup_verified=True, technical_repair_archive_ready=True)
                    renew_budget(name)
                    return False
            with ledger() as state:
                continuation = state['jobs'][name].get('completion_continuation')
            if continuation:
                from ops.shorterbetter_completion import maybe_continue
                if maybe_continue(sys.modules[__name__], pod, name, result):
                    return False
            release(name, 'complete' if result['exit_code'] == 0 else 'failed', verified=True)
            return True
    try:
        probe = worker_probe(pod)
        update(name, worker_observation=probe, worker_observed_at=time.time(),
               needs_inspection=bool(probe.get('health', {}).get('needs_inspection')))
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
        probe = {'worker_status': 'unknown'}
        update(name, probe_error=str(exc), probe_error_at=time.time())
    status = probe.get('worker_status', 'unknown')
    if status == 'dead':
        with ledger() as state:
            archive_job = dict(state['jobs'][name])
        if completion_archive_authorized(name, archive_job, probe) and renew_budget(name):
            update(name, final_archive_pending=True)
            return False
        # The supervisor has reaped the workload. Preserve what exists even
        # when a failed shell could not publish a valid completion archive.
        with ledger() as state:
            job = state['jobs'][name]
            first_seen = job.setdefault('worker_dead_observed_at', time.time())
            wait_for_final_copy = time.time() - first_seen < 120 and time.time() < job['deadline']
        if wait_for_final_copy:
            return False
        release(name, 'worker_exited_without_verified_result')
        return True
    if status == 'alive':
        renew_budget(name)
    with ledger() as state:
        job = dict(state['jobs'][name])
    remaining = job['deadline'] - time.time()
    if remaining <= 300 and not job.get('budget_stop_requested_at'):
        try:
            remote(pod, 'python3 /workspace/project/experiments/worker_monitor.py stop --directory /workspace/project/output', timeout=30)
            update(name, budget_stop_requested_at=time.time())
        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
            update(name, budget_stop_error=str(exc))
    if remaining <= 0:
        release(name, 'budget_limit' if status == 'alive' else 'budget_lease_expired_unreachable')
        return True
    return False


def is_ambiguous_create(job):
    attempts = job.get('allocation_attempts') or []
    if attempts and attempts[-1].get('pod_name') == job.get('pod_name') and attempts[-1].get('outcome') == 'ambiguous':
        return True
    return bool(re.search(r'Runpod HTTP 5[0-9]{2}:', str(job.get('error', ''))))


def reset_absence(name, reason):
    with ledger() as state:
        job = state['jobs'][name]
        job.update(first_absence_at=None, last_absence_at=None, absence_count=0,
                   absence_observations=[], absence_reset_at=time.time(), absence_reset_reason=reason)


def record_complete_absence(name, expected_name, observed_pods):
    """Our conservative inference, not a provider guarantee for failed POSTs.

    The CLI documents bounded absence for known-resource polling. Applying a
    stricter complete-list/time rule to an uncertain POST is our operational
    inference, backed by the retained exact-name late-arrival watch. No retry.
    """
    with ledger() as state:
        job = state['jobs'][name]
        if job.get('released_at') or job.get('pod_id') or job.get('pod_name') != expected_name:
            return False
        now = time.time()
        if job.get('absence_nonce') != expected_name or job.get('first_absence_at') is None:
            job.update(absence_nonce=expected_name, first_absence_at=now, last_absence_at=None, absence_count=0, absence_observations=[])
        if job.get('last_absence_at') is not None and now < job['last_absence_at']:
            job.update(first_absence_at=now, absence_count=0, absence_observations=[])
        job.update(last_absence_at=now, last_absence_check_at=now,
                   absence_count=job['absence_count'] + 1,
                   absence_list_completeness='pagination.hasNextPage=false')
        job.setdefault('absence_observations', []).append({
            'at': now, 'hasNextPage': False, 'observed_pod_ids': sorted(pod['id'] for pod in observed_pods),
            'exact_nonce_matches': 0})
        elapsed = now - job['first_absence_at']
        if not is_ambiguous_create(job) or job['absence_count'] < ABSENCE_REQUIRED_READS or elapsed < ABSENCE_REQUIRED_SECONDS:
            return False
        # Preserve the full failed-create audit and former reservation. The
        # retained nonce remains on a 24-hour watch for delayed propagation.
        job.update(absence_reconciled_at=now, absence_reserved_usd=job['reserved_usd'],
                   absence_reconciliation={'successful_full_reads': job['absence_count'],
                       'first_absence_at': job['first_absence_at'], 'last_absence_at': now,
                       'elapsed_seconds': elapsed, 'required_reads': ABSENCE_REQUIRED_READS,
                       'required_seconds': ABSENCE_REQUIRED_SECONDS,
                       'reasoning': 'Conservative operational inference from known-resource polling behavior; not an official failed-POST guarantee',
                       'reference': 'https://github.com/runpod/runpodctl#waiting-until-a-resource-is-usable'},
                   late_arrival_watch_until=now + LATE_ARRIVAL_WATCH_SECONDS,
                   safely_no_pod=True, state='create_absent_reconciled', released_at=now,
                   charged_estimate_usd=0, reserved_usd=0, allocation_finished_at=now)
        return True


def watch_late_arrival(name):
    """Delete only exact-name late arrivals for an abandoned, refunded create."""
    with job_lock(name) as acquired:
        if not acquired:
            return
        with ledger() as state:
            job = dict(state['jobs'][name])
        now = time.time()
        watched = dict(job.get('late_arrival_pods') or {})
        pending = any(not item.get('confirmed_released_at') for item in watched.values())
        if now > job['late_arrival_watch_until'] and not pending:
            return
        if now <= job['late_arrival_watch_until']:
            try:
                matches = [pod for pod in list_pods(job.get('provider', 'runpod')) if pod['name'] == job['pod_name']]
                update(name, late_arrival_last_check_at=now)
            except Exception as exc:
                matches = []
                update(name, late_arrival_watch_error=str(exc), late_arrival_watch_error_at=now)
            for pod in matches:
                item = watched.setdefault(pod['id'], {'detected_at': now, 'name': pod['name'], 'state': 'detected'})
                item['last_list_observation'] = {key: pod.get(key) for key in ('id', 'name', 'status', 'createdAt')}
                item['last_list_observed_at'] = now
        if not watched:
            return
        # A late pod re-opens its conservative financial liability before any
        # cleanup call. It also occupies admission slots until confirmed gone.
        open_ids = [pod_id for pod_id, item in watched.items() if not item.get('confirmed_released_at')]
        if open_ids:
            with ledger() as state:
                current = state['jobs'][name]
                current.pop('released_at', None)
                current.update(state='late_arrival_cleanup', safely_no_pod=False,
                    pod_id=open_ids[0], duplicate_pod_ids=open_ids, identity_confirmed=False,
                    late_arrival_pods=watched, late_arrival_detected_at=current.get('late_arrival_detected_at', now),
                    reserved_usd=max(job.get('absence_reserved_usd', 0),
                        (now - job['created_at']) / 3600 * job['budget_rate']) * len(open_ids))
        for pod_id in open_ids:
            item = watched[pod_id]
            try:
                try:
                    pod = api('pods/' + pod_id)
                except Exception as exc:
                    if not is_missing(exc):
                        raise
                    pod = None
                if pod is not None and pod.get('name') != job['pod_name']:
                    raise RuntimeError('Late-arrival pod identity mismatch; refusing deletion')
                item['identity_checked_at'] = time.time()
                item['last_get_observation'] = None if pod is None else {key: pod.get(key) for key in ('id', 'name', 'status', 'createdAt')}
                if pod is not None and pod.get('status') not in ('TERMINATED', 'DELETED'):
                    item.update(state='delete_requested', delete_requested_at=time.time())
                    api('pods/' + pod_id, 'DELETE')
                    try:
                        pod = api('pods/' + pod_id)
                    except Exception as exc:
                        if not is_missing(exc):
                            raise
                        pod = None
                    item['confirmation_observation'] = None if pod is None else {key: pod.get(key) for key in ('id', 'name', 'status', 'createdAt')}
                    item['confirmation_observed_at'] = time.time()
                    if pod is not None and pod.get('name') != job['pod_name']:
                        raise RuntimeError('Late-arrival confirmation identity mismatch')
                if pod is None or pod.get('status') in ('TERMINATED', 'DELETED'):
                    item.update(state='released', confirmed_released_at=time.time())
                else:
                    item['state'] = 'delete_pending_confirmation'
            except Exception as exc:
                item.update(last_error=str(exc), last_error_at=time.time())
            update(name, late_arrival_pods=watched)
        if (open_ids or not job.get('released_at')) and all(item.get('confirmed_released_at') for item in watched.values()):
            conservative = sum(max(0, item['confirmed_released_at'] - job['created_at']) / 3600 * job['budget_rate']
                               for item in watched.values())
            update(name, state='late_arrival_cleaned', released_at=time.time(),
                   charged_estimate_usd=round(conservative, 4), late_arrival_pods=watched)


def vast_spending_exhausted(name, job):
    """Additional account-wide tripwire for variable storage/transfer charges.

Deposits never reset accumulated usage. Explicit concurrent evaluation tickets
use their remaining joint campaign cap for this account-wide signal; each
job's own timed budget stays enforced. Other jobs retain the stricter default.
Transport failures do not bypass timed cleanup.
"""
    if job.get('provider') != 'vast' or job.get('released_at'):
        return False
    try:
        account = vastai.api('v0/users/current/')
        credit = float(account['credit'])
        if not math.isfinite(credit):
            raise ValueError('Invalid account credit')
        with ledger() as state:
            current = state['jobs'][name]
            previous = current.get('last_account_credit', current['initial_account_credit'])
            spent = current.get('account_spend_observed', 0) + max(0, previous - credit)
            current.update(last_account_credit=credit, account_spend_observed=spent,
                           account_checked_at=time.time())
        funding = job.get('incremental_funding', {})
        from experiments.shorterbetter.budget import evaluation_account_limit
        account_limit = evaluation_account_limit(STATE, name, job)
        update(name, account_spending_limit_usd=account_limit)
        if credit <= funding.get('release_credit_usd', 1) or spent >= account_limit - 1:
            return True
        if funding and credit <= funding['stop_credit_usd'] and not job.get('credit_stop_requested_at'):
            pod = api('pods/' + job['pod_id'])
            # ScaledControl observes this file on every rank at the next update,
            # then performs terminal validation and saves a complete checkpoint.
            remote(pod, "python3 - <<'PY'\nimport json,os\nfrom pathlib import Path\np=Path('/workspace/project')\nlease=json.loads((p/'worker-lease.json').read_text())\ntarget=p/'output/stop-request.json'\ntemp=target.with_suffix('.credit-tmp')\ntemp.write_text(json.dumps({'job_name':lease['job_name'],'reason':'low_vast_credit'}))\nos.replace(temp,target)\nPY", timeout=30)
            update(name, credit_stop_requested_at=time.time())
        return False
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        update(name, account_check_error=str(exc))
        return False


def guard_job(name, job):
    try:
        if job.get('late_arrival_watch_until'):
            watch_late_arrival(name)
            return
        if job.get('released_at'):
            return
        if not job.get('pod_id'):
            if job.get('state') != 'creating':
                return
            pods = list_pods(job.get('provider', 'runpod'))
            matches = [p for p in pods if p['name'] == job['pod_name']]
            if len(matches) == 1:
                reset_absence(name, 'exact_nonce_became_visible')
                update(name, pod_id=matches[0]['id'], identity_confirmed=True)
                job['pod_id'] = matches[0]['id']
                if is_ambiguous_create(job):
                    # allocate() raises before uploading or launching anything
                    # after an ambiguous create. A late-created pod is unused.
                    release(name, 'abandoned_create_reconciled')
                    return
            elif matches:
                reset_absence(name, 'multiple_exact_nonce_pods_visible')
                # A randomly suffixed name belongs to this one create attempt.
                # If a provider duplicated it, retain liability and clean all IDs.
                failures = []
                for pod in matches:
                    try:
                        api('pods/' + pod['id'], 'DELETE')
                    except Exception as exc:
                        failures.append(str(exc))
                update(name, duplicate_pod_ids=[p['id'] for p in matches],
                       guard_error='Duplicate create; cleanup attempted: ' + '; '.join(failures))
                return
            else:
                record_complete_absence(name, job['pod_name'], pods)
                if known_capacity_rejection(job) and (
                        not job.get('allocation_attempts') or
                        time.time() - job.get('heartbeat', job['created_at']) > 600 or
                        time.time() >= job['deadline']):
                    update(name, safely_no_pod=True)
                    void_reservation(name, 'capacity_rejected_absent')
                return  # No create is retried here; bounded absence evidence may void only its reservation.
        if job.get('delete_requested_at'):
            if confirm_release(name):
                return
            release(name, job.get('release_reason', 'cleanup_retry'))
            return
        if vast_spending_exhausted(name, job):
            release(name, 'vast_account_spending_limit')
            return
        expired = time.time() >= job['deadline']
        stale = time.time() - job.get('heartbeat', job['created_at']) > 600
        if job.get('completion_driven') and (expired or stale):
            try:
                pod = api('pods/' + job['pod_id'])
            except Exception as exc:
                if not is_missing(exc):
                    raise
                update(name, release_reason='provider_ended')
                confirm_release(name)
                return
            if pod.get('status') in ('TERMINATED', 'DELETED'):
                update(name, release_reason='provider_ended')
                confirm_release(name)
            elif (pod.get('ssh') or {}).get('direct'):
                observe_completion(pod, name, controller=False)
            elif expired:
                release(name, 'budget_lease_expired_unreachable')
            return
        if expired or stale:
            release(name, 'deadline' if expired else 'controller_lost')
    except Exception as exc:
        if not job.get('pod_id') and job.get('state') == 'creating':
            reset_absence(name, str(exc))
        update(name, guard_error=str(exc), guard_error_at=time.time())
        print('Guard cleanup error for ' + name + ': ' + str(exc), file=sys.stderr, flush=True)


def guard():
    with ledger() as state:
        jobs = dict(state['jobs'])
    # Independent cleanup calls: one stalled API must not delay other deadlines.
    with ThreadPoolExecutor(max_workers=max(1, min(16, len(jobs)))) as pool:
        futures = [pool.submit(guard_job, name, job) for name, job in jobs.items()]
        for future in futures:
            future.result()


def capacity_rejected(exc):
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and 'runpod http 400:' in text and any(
        phrase in text for phrase in ('no longer available', 'no longer any instances available', 'insufficient capacity',
                                     'not enough available', 'insufficient gpu', 'no available gpu'))


def known_capacity_rejection(job):
    if capacity_rejected(RuntimeError(job.get('error', ''))):
        return True
    attempts = job.get('allocation_attempts') or []
    return bool(attempts and attempts[-1].get('pod_name') == job.get('pod_name') and
                attempts[-1].get('outcome') == 'capacity_rejected' and
                capacity_rejected(RuntimeError(attempts[-1].get('error', ''))))


def list_pods(provider='runpod'):
    if provider == 'vast':
        return vastai.list_pods()
    if provider != 'runpod':
        raise ValueError('Unknown GPU provider')
    response = api('pods')
    # Absence is evidence only when the authoritative listing is complete.
    # Never silently treat a partial/unknown page as the entire account.
    if not isinstance(response, dict) or not isinstance(response.get('pods'), list):
        raise RuntimeError('Pod listing has no authoritative pagination metadata')
    pagination = response.get('pagination')
    if not isinstance(pagination, dict) or pagination.get('hasNextPage') is not False:
        raise RuntimeError('Pod listing is partial or completeness is unknown')
    if any(not isinstance(p, dict) or not p.get('id') or not p.get('name') for p in response['pods']):
        raise RuntimeError('Pod listing contains invalid pod identities')
    return response['pods']


def void_reservation(name, reason):
    with ledger() as state:
        job = state['jobs'][name]
        if job.get('pod_id') or not job.get('safely_no_pod'):
            raise RuntimeError('Cannot void an uncertain or active allocation')
        job.update(state=reason, released_at=time.time(), charged_estimate_usd=0,
                   reserved_usd=0, allocation_finished_at=time.time())


def record_attempt(name, **changes):
    with ledger() as state:
        state['jobs'][name]['allocation_attempts'][-1].update(changes)


@contextmanager
def allocation_lock(deadline):
    """Serialize quote/create locally, with a bounded wait for another allocator."""
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'allocation.lock').open('a') as stream:
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        try:
            yield acquired
        finally:
            if acquired:
                fcntl.flock(stream, fcntl.LOCK_UN)


def potentially_live_pods(jobs, exclude=None):
    """Deletion acceptance is not release; ambiguous creates also occupy slots."""
    count = 0
    for name, job in jobs.items():
        if name == exclude or job.get('released_at'):
            continue
        known_ids = set(job.get('duplicate_pod_ids') or [])
        if job.get('pod_id'):
            known_ids.add(job['pod_id'])
        if known_ids:
            count += len(known_ids)
        elif job.get('state') == 'creating' or (job.get('pod_name') and job.get('safely_no_pod') is False):
            count += 1
    return count


def reconcile_pending_deletions(exclude, deadline):
    """Read-only provider checks can free slots; never terminate existing work."""
    with ledger() as state:
        pending = [name for name, job in state['jobs'].items()
                   if name != exclude and job.get('pod_id') and job.get('delete_requested_at')
                   and not job.get('released_at')]
    for name in pending:
        if deadline - time.monotonic() < 35:
            break  # Respect the allocation window even if an API check stalls.
        confirm_release(name)


def unhealthy_placement(jobs, gpu, provider, quote=None):
    """Quarantine a failed Vast machine, not every machine with the same GPU."""
    for job in jobs.values():
        if (job.get('gpu') != gpu or job.get('provider', 'runpod') != provider
                or not job.get('released_at') or time.time()-job['created_at'] >= 86400
                or ('Pod startup exceeded' not in (job.get('error') or '')
                    and job.get('release_reason') != 'provider_startup_failure')):
            continue
        if provider != 'vast':
            return True
        if quote:
            previous=job.get('provider_quote', {})
            machine=previous.get('offer', {}).get('machine_id')
            if (machine is not None and machine == quote.get('offer', {}).get('machine_id')):
                return True
            if previous.get('offer_id') == quote.get('offer_id'):
                return True
    return False


def allocate(args, wait_seconds=ALLOCATION_WAIT_SECONDS):
    """Retry only explicit capacity rejections reconciled to no created pod."""
    requested_gpu = getattr(args, 'gpu', None)
    provider = getattr(args, 'provider', 'runpod')
    if provider not in ('runpod', 'vast'):
        raise ValueError('Unknown GPU provider')
    from experiments.shorterbetter.hardware import allocation_spec
    scoped = allocation_spec(args)
    gpu_count = scoped['count'] if scoped else 1
    max_gpu_rate = scoped['max_gpu_rate'] if scoped else MAX_GPU_RATE
    from experiments.shorterbetter.budget import ticket
    authorized = ticket(STATE, args.name, {'gpu': requested_gpu, 'gpu_count': gpu_count,
        'hardware_scope': scoped['scope'] if scoped else None, 'entry': args.entry,
        'max_cost_usd': getattr(args, 'max_cost_usd', None),
        'completion_budget_scope': getattr(args, 'completion_budget_scope', None),
        'provider': provider, 'offer_id': getattr(args, 'vast_offer_id', None),
        'disk_gb': getattr(args, 'vast_disk_gb', 80),
        'max_runtime_hours': getattr(args, 'max_runtime_hours', None)})
    if provider == 'vast' and (not authorized or authorized.get('scope') not in ('public-reproduction-v1', 'shorterbetter_vast_bounded_v1', 'replay_vast_bounded_v1', 'coding-length-probe-v1', 'coding-replay-768-v1', 'opencoder-sft-768-v1', 'opencoder-mode-probe-v1','coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1')):
        raise ValueError('Vast requires a separate named bounded user authorization')
    if getattr(args, 'completion_budget_scope', None) and not authorized:
        raise ValueError('Completion-driven launch requires its named user authorization')
    if authorized:
        max_gpu_rate = min(max_gpu_rate, authorized['max_gpu_rate'])
    if not scoped and requested_gpu is not None and requested_gpu not in GPU_CANDIDATES:
        raise ValueError('Requested GPU is outside the approved candidate list')
    candidates = (requested_gpu,) if requested_gpu else GPU_CANDIDATES
    now = time.time()
    deadline = time.monotonic() + wait_seconds
    # Reserve the maximum eligible rate before selecting a GPU. Retries cannot
    # silently grow the reservation; actual estimates remain conservative.
    budget_rate = max_gpu_rate * gpu_count * 1.25 + (.15 if scoped else .05)
    reserved = budget_rate * args.hours
    job_cap = getattr(args, 'max_cost_usd', None)
    hard_deadline = None
    if job_cap is not None:
        if not math.isfinite(job_cap) or job_cap <= 1 or reserved > job_cap:
            raise ValueError('Invalid job cap or initial lease exceeds cap')
        # Reserve the entire approved envelope before allocation; retain $1
        # within it for deletion/storage/API jitter. Other jobs are unchanged.
        reserved = job_cap
        hard_deadline = now + (job_cap - 1.0) / budget_rate * 3600
    runtime_hours = getattr(args, 'max_runtime_hours', None)
    runpod_resume = bool(provider == 'runpod' and authorized and authorized.get('scope') in ('public-reproduction-v1','replay_runpod_resume_v1','replay_runpod_evaluation_v1'))
    runtime_ceiling = authorized['max_runtime_hours'] if provider == 'vast' or runpod_resume else 8
    if provider != 'vast' and not runpod_resume and authorized and authorized.get('max_runtime_hours') is not None:
        runtime_hours = min(runtime_hours or 8, 8)
    if runtime_hours is not None:
        if not scoped or not math.isfinite(runtime_hours) or not 0 < runtime_hours <= runtime_ceiling:
            raise ValueError('Runtime bound exceeds scoped authorization')
        hard_deadline = min(hard_deadline, now + runtime_hours * 3600)
    vast_credit = None
    if provider == 'vast':
        account = vastai.api('v0/users/current/')
        vast_credit = float(account.get('credit', 0))
        funding = authorized.get('incremental_funding', {})
        minimum_credit = funding.get('minimum_start_credit_usd', reserved)
        if not account.get('can_pay') or not math.isfinite(vast_credit) or vast_credit < minimum_credit:
            raise RuntimeError('Vast credit is below the authorized initial funding requirement')
        # Registration is idempotent and must precede the rental.
        vastai.ensure_ssh_key()
    with ledger() as state:
        if args.name in state['jobs']:
            raise RuntimeError('Job name already used; inspect previous state')
        if not budget_allows(state['jobs']) or (not authorized and liability(state['jobs']) + reserved > USABLE):
            raise RuntimeError('Campaign budget exhausted')
        state['jobs'][args.name] = dict(state='allocation_waiting', created_at=now,
            provider=provider, heartbeat=now, deadline=min(now + args.hours * 3600, hard_deadline) if hard_deadline else now + args.hours * 3600, budget_rate=budget_rate,
            reserved_usd=reserved, entry=args.entry, safely_no_pod=True,
            allocation_attempts=[], allocation_wait_seconds=wait_seconds,
            completion_driven=True, requested_gpu=requested_gpu,
            **({'offer_id': args.vast_offer_id, 'disk_gb': args.vast_disk_gb,
                'max_runtime_hours': runtime_hours, 'initial_account_credit': vast_credit,
                'incremental_funding': authorized.get('incremental_funding', {})} if provider == 'vast' else {}),
            **({'max_runtime_hours': runtime_hours} if runpod_resume else {}),
            **({'bounded_checkpoint_mirror': True, 'preserve_hardlinks': True}
               if authorized and authorized.get('paired_short_experiment') else {}),
            **({'completion_budget_scope': authorized['scope']} if authorized and authorized['scope'] == 'finish_registered_1200_updates' else {}),
            **({'hardware_scope': scoped['scope'], 'gpu_count': gpu_count} if scoped else {}),
            **({'max_cost_usd': job_cap, 'hard_deadline': hard_deadline} if job_cap else {}))
    try:
        while time.monotonic() < deadline:
            update(args.name, heartbeat=time.time())
            with allocation_lock(deadline) as acquired:
                if not acquired:
                    break
                reconcile_pending_deletions(args.name, deadline)
                with ledger() as state:
                    live = potentially_live_pods(state['jobs'], exclude=args.name)
                    state['jobs'][args.name].update(potentially_live_pods=live,
                        allocation_wait_reason='concurrency_limit' if live >= MAX_LIVE_PODS else 'capacity')
                for gpu in (() if live >= MAX_LIVE_PODS else candidates):
                    # Each API call has a 30-second transport timeout. Leave room
                    # for quote, create and a rejection reconciliation in this bound.
                    if deadline - time.monotonic() < 95:
                        break
                    with ledger() as state:
                        unhealthy = unhealthy_placement(state['jobs'], gpu, provider)
                    if unhealthy:
                        continue  # Do not repeatedly rent a placement that never became usable.
                    vast_quote = None
                    if provider == 'vast':
                        vast_quote = vastai.quote(gpu, gpu_count, args.vast_disk_gb,
                            args.vast_offer_id, runtime_hours * 3600 + 1800)
                        if vast_quote is None:
                            continue
                        with ledger() as state:
                            if unhealthy_placement(state['jobs'], gpu, provider, vast_quote):
                                continue
                        q = {'availability': 'HIGH', 'price': {'secure': vast_quote['hourly_ceiling'] / gpu_count}}
                    else:
                        q = api('catalog/gpus/' + urllib.parse.quote(gpu) +
                                f'?include=AVAILABILITY&product=POD&count={gpu_count}&minCudaVersion=12.8&cloud=SECURE')
                    per_gpu_rate = float(q['price']['secure'])
                    rate = per_gpu_rate * gpu_count
                    if q['availability'] == 'NONE' or not math.isfinite(rate) or not 0 < per_gpu_rate <= max_gpu_rate:
                        continue
                    placement = placement_constraints(args.name, q) if provider == 'runpod' else None
                    if placement == []:
                        continue
                    pod_name = 'thinkingcap-pilot-' + args.name + '-' + secrets.token_hex(6)
                    with ledger() as state:
                        job = state['jobs'][args.name]
                        if potentially_live_pods(state['jobs'], exclude=args.name) >= MAX_LIVE_PODS:
                            job['allocation_wait_reason'] = 'concurrency_limit'
                            break
                        job.update(pod_name=pod_name, state='creating', safely_no_pod=False,
                                   heartbeat=time.time(), rate=rate, gpu=gpu, error=None)
                        if vast_quote:
                            job['provider_quote'] = vast_quote
                        if placement is not None:
                            job['requested_data_center_ids'] = placement
                        job['allocation_attempts'].append(dict(pod_name=pod_name, gpu=gpu,
                            rate=rate, started_at=time.time(), outcome='creating'))
                    try:
                        pod = vastai.create(vast_quote, pod_name,
                            **({'direct_ssh': True} if authorized.get('direct_ssh') else {})) if provider == 'vast' else api('pods', 'POST', {'name': pod_name,
                            'templateId': 'runpod-torch-v280', 'cloud': 'SECURE',
                            'gpu': {'id': gpu, 'count': gpu_count, 'minCudaVersion': '12.8'},
                            'disk': 30, 'mounts': {'persistent': {'size': 100 if scoped else 50, 'path': '/workspace'}},
                            'ports': ['22/tcp'], 'startSsh': True, 'startJupyter': False,
                            **({'dataCenterIds': placement} if placement is not None else {})})
                    except Exception as exc:
                        record_attempt(args.name, outcome='capacity_rejected' if capacity_rejected(exc) else 'ambiguous', error=str(exc))
                        update(args.name, error=str(exc))
                        if not capacity_rejected(exc):
                            raise  # No retry or reservation refund after ambiguity.
                        matches = [p for p in list_pods(provider) if p.get('name') == pod_name]
                        if not matches:
                            record_attempt(args.name, outcome='capacity_rejected_absent', reconciled_at=time.time())
                            update(args.name, state='allocation_waiting', safely_no_pod=True,
                                   heartbeat=time.time())
                            continue
                        if len(matches) != 1:
                            raise RuntimeError('Multiple pods after rejected allocation; guard will reconcile')
                        pod = matches[0]  # Unexpected provider success: adopt, never duplicate.
                    if pod.get('name') != pod_name:
                        record_attempt(args.name, outcome='ambiguous', error='Create response identity mismatch')
                        raise RuntimeError('Create response pod identity mismatch')
                    record_attempt(args.name, outcome='created', pod_id=pod['id'], confirmed_at=time.time())
                    update(args.name, pod_id=pod['id'], identity_confirmed=True, state='starting',
                           safely_no_pod=False, allocation_finished_at=time.time())
                    return pod
            remaining = deadline - time.monotonic()
            if remaining < 95:
                break
            time.sleep(min(10, remaining - 94))
        void_reservation(args.name, 'allocation_unavailable')
        raise RuntimeError('No eligible GPU capacity within bounded allocation window; no pod rented')
    except BaseException as exc:
        with ledger() as state:
            job = dict(state['jobs'][args.name])
        if not job.get('released_at') and job.get('safely_no_pod'):
            void_reservation(args.name, 'allocation_failed')
        update(args.name, error=str(exc))
        raise


def launch_command(entry, seconds, directory='/workspace/project', completion_driven=False):
    if seconds <= 0:
        raise ValueError('Worker runtime must be positive')
    # Keep the foreground shell's cd separate from the background command.
    # An asynchronous AND-list retains the SSH stdout/stderr pipes until it ends.
    if completion_driven:
        return ('set -e\ncd ' + shlex.quote(directory) + '\n'
                'THINKINGCAP_COMPLETION_DRIVEN=1 nohup python3 experiments/worker_monitor.py run --entry ' +
                shlex.quote(entry) + ' --output output --cwd ' + shlex.quote(directory) +
                ' > output/supervisor.log 2>&1 < /dev/null &\n'
                "worker_pid=$!\nprintf '%s\\n' \"$worker_pid\" > output/worker.pid\n")
    return ('set -e\ncd ' + shlex.quote(directory) + '\n'
            'nohup timeout --signal=TERM --kill-after=120 ' + str(int(seconds)) +
            ' bash ' + shlex.quote(entry) + ' > output/runner.log 2>&1 < /dev/null &\n'
            "worker_pid=$!\nprintf '%s\\n' \"$worker_pid\" > output/worker.pid\n")


def run(args):
    if not (ROOT / args.entry).is_file():
        raise RuntimeError('Entry script missing')
    uv_binary = resolve_uv()
    pilot_input = ('lab/experiments/data/coding-curriculum/summary.json'
                   if getattr(args, 'hardware_scope', None) == 'coding-curriculum-256-v1'
                   else 'lab/experiments/data/coding-abc/plan.json'
                   if getattr(args, 'hardware_scope', None) == 'coding-abc-768-v1'
                   else 'lab/experiments/data/coding-frontier/plan.json'
                   if getattr(args, 'hardware_scope', None) == 'coding-frontier-768-v1'
                   else 'lab/experiments/data/opencoder-mode-probe/plan.json'
                   if getattr(args, 'hardware_scope', None) == 'opencoder-mode-probe-v1'
                   else 'lab/experiments/data/opencoder-sft/plan.json'
                   if getattr(args, 'hardware_scope', None) == 'opencoder-sft-768-v1'
                   else 'lab/experiments/data/coding-replay/plan.json'
                   if getattr(args, 'hardware_scope', None) == 'coding-replay-768-v1'
                   else 'lab/experiments/data/replay-scaled/plan.json'
                   if getattr(args, 'hardware_scope', None) == 'replay-scaled-a100-v1'
                   else 'lab/experiments/data/coding-length-probe/plan.json'
                   if getattr(args, 'hardware_scope', None) == 'coding-length-probe-v1'
                   else 'lab/experiments/data/pilot.jsonl')
    if getattr(args, 'public_authorization', None):
        pilot_input = pilot_input.replace('lab/experiments/data/', 'data/', 1)
    if not (ROOT / pilot_input).exists():
        raise RuntimeError('Prepare the shared pilot before renting compute')
    if subprocess.run(['systemctl', 'is-active', '--quiet', os.environ.get('SELF_REPLAY_GUARD_UNIT', 'self-replay-guard.timer')]).returncode:
        raise RuntimeError('Independent budget guard timer must be active before provisioning')
    if getattr(args,'resume_startup',False):
        with ledger() as state:existing=dict(state['jobs'][args.name])
        if (existing.get('hardware_scope') not in ('replay-evaluation-v1','coding-frontier-768-v1') or existing.get('state')!='starting'
                or existing.get('released_at') or existing.get('worker_launch_attempted_at')
                or not existing.get('identity_confirmed') or not existing.get('pod_id')
                or existing.get('entry')!=args.entry or existing.get('max_cost_usd')!=args.max_cost_usd):
            raise RuntimeError('Startup resume requires an identified evaluation worker that never launched')
        pod=api('pods/'+existing['pod_id'])
        if pod.get('id')!=existing['pod_id'] or pod.get('name')!=existing['pod_name']:
            raise RuntimeError('Startup resume provider identity mismatch')
        remote(pod,'test ! -e /workspace/project/output/worker.pid && test ! -e /workspace/project/output/worker-state.json',timeout=20)
        source=STATE/args.name/'source'
        if source.exists():source.rename(source.with_name('source-before-startup-resume-'+str(int(time.time()))))
        update(args.name,startup_resumed_at=time.time())
    else:
        pod = allocate(args)
    with ledger() as state:
        job = dict(state['jobs'][args.name])
    now, rate, budget_rate = job['created_at'], job['rate'], job['budget_rate']
    launch_attempted=False
    try:
        actual_rate = pod.get('cost')
        if actual_rate is not None and float(actual_rate) > budget_rate:
            raise RuntimeError('Actual pod rate exceeds reserved budget rate')
        write(STATE / args.name / 'pod.json', summary(pod))
        print('Created', pod['id'], 'for', args.name, 'at', rate, 'USD/hour', flush=True)
        startup_started=time.time() if getattr(args,'resume_startup',False) else job.get('allocation_finished_at',now)
        for attempt in range(90):
            if time.time() - startup_started >= 600:
                raise RuntimeError('Pod startup exceeded ten minutes; no training launched')
            update(args.name, heartbeat=time.time())
            pod = api('pods/' + pod['id'])
            if (pod.get('ssh') or {}).get('direct'):
                try:
                    remote(pod, 'true', timeout=20)
                    break
                except (subprocess.SubprocessError, RuntimeError):
                    pass
            if pod['status'] in ('ERROR', 'TERMINATED', 'EXITED'):
                raise RuntimeError('Startup failed: ' + pod['status'])
            time.sleep(10)
        else:
            raise RuntimeError('SSH startup deadline')
        write(STATE / args.name / 'pod.json', summary(pod))
        remote(pod, 'mkdir -p /workspace/project/output /workspace/project/data; command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)')
        # Upload an immutable local snapshot, retaining the exact source used by
        # each rental even when later experiments change the working tree.
        source = STATE / args.name / 'source'
        source.mkdir(exist_ok=True)
        for folder in ('experiments', 'ThinkingCap-Test'):
            shutil.copytree(ROOT / folder, source / folder,
                            ignore=shutil.ignore_patterns('.git', '__pycache__', '.venv*', 'runs', '.env*', '*.log'))
            sync(pod, str(source / folder) + '/', '/workspace/project/' + folder + '/', upload=True,
                 exclusions=('.git', '__pycache__', '.venv*', 'runs', '.env*', '*.log'))
        data_root = ROOT / ('data' if getattr(args, 'public_authorization', None) else 'lab/experiments/data')
        sync(pod, str(data_root) + '/', '/workspace/project/data/', upload=True, timeout=300)
        shutil.copy2(ROOT / 'lab/gpu-requirements.lock', source / 'requirements.lock')
        write(source / 'manifest.json', {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(source.rglob('*')) if p.is_file()})
        sync(pod, source / 'requirements.lock', '/workspace/project/requirements.lock', upload=True)
        sync(pod, uv_binary, '/workspace/project/uv', upload=True, timeout=300)
        if job.get('provider') == 'vast':
            # CUDA base images lack RunPod's Python installation. Populate uv's
            # interpreter cache before the worker enables offline dependencies.
            remote(pod, '/workspace/project/uv python install 3.11', timeout=300)
        if job.get('max_cost_usd') is not None or job.get('hardware_scope'):
            lease = {'job_name': args.name, 'pod_id': job['pod_id'],
                     'provider': job.get('provider', 'runpod'),
                     'created_at': job['created_at'], 'deadline': job['deadline'],
                     'hard_deadline': job.get('hard_deadline'), 'max_cost_usd': job.get('max_cost_usd'),
                     'budget_rate': job['budget_rate'], 'rate': job['rate'], 'gpu': job['gpu']}
            if job.get('hardware_scope'):
                lease.update(hardware_scope=job['hardware_scope'], gpu_count=job['gpu_count'])
                lease['completion_budget_scope'] = job.get('completion_budget_scope')
            write(source / 'worker-lease.json', lease)
            sync(pod, source / 'worker-lease.json', '/workspace/project/worker-lease.json', upload=True)
        remaining = int(now + args.hours * 3600 - time.time() - 300)
        if remaining <= 0:
            raise RuntimeError('No training time remains before reserved cleanup window')
        launch_attempted=True
        update(args.name,worker_launch_attempted_at=time.time())
        remote(pod, launch_command(args.entry, remaining, completion_driven=True))
        update(args.name, state='running', heartbeat=time.time())
        while True:
            if observe_completion(pod, args.name):
                return
            time.sleep(45)
    except BaseException as exc:
        update(args.name, error=str(exc),startup_failed_before_launch=not launch_attempted)
        if pod:
            try:
                if (pod.get('ssh') or {}).get('direct') and time.time() < now + args.hours * 3600 - 60:
                    backup(pod, args.name, timeout=min(30, max(1, now + args.hours * 3600 - time.time() - 60)))
            except Exception as backup_exc:
                update(args.name, backup_error=str(backup_exc))
            try:
                release(args.name, 'failed')
            except Exception as release_exc:
                update(args.name, release_error=str(release_exc))
        raise


def watch(name):
    """Resume observation only: never allocate, upload, or relaunch GPU work.

    Controller errors leave the detached worker under the independent guard.
    Restarting this command is safe for both active and released jobs.
    """
    while True:
        with ledger() as state:
            job = dict(state['jobs'][name])
        if job.get('released_at'):
            return
        if not job.get('completion_driven') or not job.get('pod_id') or not job.get('identity_confirmed'):
            raise RuntimeError('Watch requires an identified completion-driven job')
        if job.get('delete_requested_at'):
            release(name, job.get('release_reason', 'cleanup_retry'))
        else:
            try:
                pod = api('pods/' + job['pod_id'])
            except Exception as exc:
                if not is_missing(exc):
                    raise
                update(name, release_reason='provider_ended')
                confirm_release(name)
                return
            if pod.get('id') != job['pod_id'] or pod.get('name') != job['pod_name']:
                raise RuntimeError('Provider identity does not match recorded job')
            if pod.get('status') in ('TERMINATED', 'DELETED'):
                update(name, release_reason='provider_ended')
                confirm_release(name)
                return
            observe_completion(pod, name)
        with ledger() as state:
            if state['jobs'][name].get('released_at'):
                return
        time.sleep(45)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    start = sub.add_parser('run')
    start.add_argument('--name', required=True)
    start.add_argument('--entry', required=True)
    start.add_argument('--gpu', choices=GPU_CANDIDATES,
                       help='Require this GPU; no fallback to a different model')
    start.add_argument('--hours', type=float, default=6,
                       help='Initial budget reservation in hours; renewed while active within the campaign ceiling')
    sub.add_parser('guard')
    sub.add_parser('status')
    resume = sub.add_parser('watch', help='Resume polling an existing job without restarting its worker')
    resume.add_argument('--name', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        if not 0 < args.hours <= 12 or not args.name.replace('-', '').isalnum():
            parser.error('Use a safe unique name and an initial reservation up to 12 hours')
        run(args)
    elif args.command == 'guard':
        guard()
    elif args.command == 'watch':
        watch(args.name)
    else:
        with ledger() as state:
            print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
