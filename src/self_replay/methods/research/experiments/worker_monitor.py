"""Completion-driven worker supervision and read-only health probes.

Run:   python worker_monitor.py run --entry experiments/run-study-eval.sh --output output
Probe: python worker_monitor.py probe --output output [--previous earlier-probe.json]

There is no runtime deadline and idle observations never authorize termination.
The caller separately controls spending and handles unreachable rented servers.
Exclude worker-state.json, worker.pid, and .worker-state.*.tmp from final checksums:
the supervisor records its final state after the child's archive is finalized.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SCHEMA_VERSION = 1
PROC = Path('/proc')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_observation(pid, expected=None):
    """Alive/dead/unknown, with PID reuse protection when identity is available."""
    if type(pid) is not int or pid <= 0:
        return {'status': 'unknown', 'pid': pid, 'reason': 'invalid_pid'}
    try:
        raw = (PROC / str(pid) / 'stat').read_text()
        # comm can itself contain spaces or parentheses. Field 22 is starttime.
        fields = raw.rsplit(')', 1)[1].strip().split()
        state, start_ticks = fields[0], int(fields[19])
        boot_id = (PROC / 'sys/kernel/random/boot_id').read_text().strip()
    except FileNotFoundError as exc:
        # Only disappearance of this PID proves absence; a missing boot-id source
        # or any permission/read/parse error means identity is unknown.
        if exc.filename == str(PROC / str(pid) / 'stat'):
            return {'status': 'dead', 'pid': pid, 'reason': 'pid_absent', 'identity_verified': False}
        return {'status': 'unknown', 'pid': pid, 'reason': 'identity_source_missing'}
    except (OSError, ValueError, IndexError) as exc:
        return {'status': 'unknown', 'pid': pid, 'reason': 'process_read_failed', 'error_type': type(exc).__name__}
    identity = {'pid': pid, 'start_ticks': start_ticks, 'boot_id': boot_id}
    verified = bool(expected and type(expected.get('start_ticks')) is int and expected.get('boot_id'))
    if verified and any(expected.get(key) != value for key, value in identity.items()):
        return dict(identity, status='dead', process_state=state, identity_verified=True, reason='pid_identity_changed')
    alive = state not in ('Z', 'X', 'x')
    return dict(identity, status='alive' if alive else 'dead', process_state=state,
                identity_verified=verified, reason='matched_identity' if verified else 'legacy_identity_unverified')


def identity_for(pid):
    observed = process_observation(pid)
    if observed['status'] == 'unknown':
        raise RuntimeError('Cannot establish worker process identity: ' + observed['reason'])
    return {key: observed[key] for key in ('pid', 'start_ticks', 'boot_id') if key in observed}


def run_worker(entry, output, cwd):
    output, cwd = Path(output).resolve(), Path(cwd).resolve()
    entry = (cwd / entry).resolve()
    if not entry.is_file():
        raise ValueError('Worker entry script does not exist')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'worker-state.json'
    if state_path.exists():
        raise ValueError('Worker state already exists; preserve it and use a fresh output directory')
    state = {'schema_version': SCHEMA_VERSION, 'state': 'starting', 'started_at': timestamp(),
             'started_unix': time.time(), 'entry': str(entry), 'supervisor': identity_for(os.getpid()),
             'worker': None, 'exit_code': None, 'runtime_deadline': None}
    atomic_json(state_path, state)
    # Preserve compatibility with callers that previously tracked a timeout PID.
    (output / 'worker.pid').write_text(str(os.getpid()) + '\n')
    child = None
    requested_signals = []
    pending_signals = []

    def forward(signum, frame):
        requested_signals.append(signum)
        if child is None:
            pending_signals.append(signum)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    previous_handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (output / 'runner.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(['bash', str(entry)], cwd=cwd, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(state='running', worker={'pid': child.pid}, process_group=child.pid)
            try:
                state['worker'] = identity_for(child.pid)
            except RuntimeError as exc:
                state['worker_identity_error'] = str(exc)
            atomic_json(state_path, state)
            for signum in pending_signals:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass
            returncode = child.wait()  # No timeout. Reap the child after completion.
        exit_code = returncode if returncode >= 0 else 128 - returncode
        state.update(state='finished', finished_at=timestamp(), finished_unix=time.time(),
                     returncode=returncode, exit_code=exit_code, forwarded_signals=requested_signals)
        atomic_json(state_path, state)
        return exit_code
    except BaseException as exc:
        state.update(state='launch_failed' if child is None else 'supervisor_error', error_type=type(exc).__name__, error=str(exc),
                     observed_at=timestamp(), forwarded_signals=requested_signals)
        atomic_json(state_path, state)
        # Do not invent completion or kill a possibly healthy child on a monitor
        # error. Its identity remains available to the read-only probe/cost guard.
        raise
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)


def meaningful_progress(output, maximum_files=50000):
    latest = []
    scanned = 0
    errors = []
    truncated = False
    excluded = {'worker.pid', 'worker-state.json', 'monitor.log', 'supervisor.log', 'sha256sums'}
    try:
        def onerror(exc):
            errors.append(type(exc).__name__)
        for directory, folders, files in os.walk(output, followlinks=False, onerror=onerror):
            folders[:] = [name for name in folders if not name.startswith('.') and not (Path(directory) / name).is_symlink()]
            for filename in files:
                scanned += 1
                if scanned > maximum_files:
                    truncated = True
                    break
                name = filename.lower()
                path = Path(directory) / filename
                relative = path.relative_to(output).as_posix()
                if (name in excluded or name.startswith('.') or name.startswith('sha256sums') or
                        '.tmp' in name or 'heartbeat' in name or relative in ('exit.json', 'result.json')):
                    continue
                if not (name.endswith(('.json', '.jsonl', '.log', '.safetensors')) or 'tfevents' in name):
                    continue
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    stat = path.stat()
                    latest.append({'path': relative, 'modified_unix': stat.st_mtime, 'bytes': stat.st_size})
                    latest = sorted(latest, key=lambda item: (item['modified_unix'], item['path']), reverse=True)[:5]
                except FileNotFoundError:
                    continue  # Atomic checkpoint replacement can race a probe.
                except OSError as exc:
                    errors.append(type(exc).__name__)
            if truncated:
                break
    except OSError as exc:
        errors.append(type(exc).__name__)
    return {'latest': latest[0] if latest else None, 'recent_files': latest, 'scanned_files': scanned,
            'scan_truncated': truncated, 'scan_error_types': errors[:5]}


def gpu_observation():
    command = ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
        devices = []
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) != 4:
                raise ValueError('Unexpected GPU telemetry columns')
            values = []
            for value in row:
                try:
                    values.append(float(value.strip()))
                except ValueError:
                    values.append(None)
            devices.append(dict(zip(('index', 'utilization_percent', 'memory_utilization_percent', 'memory_used_mib'), values)))
        return {'status': 'available', 'devices': devices}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'status': 'unknown', 'devices': [], 'error_type': type(exc).__name__}


def read_exit_marker(output):
    try:
        value = json.loads((output / 'exit.json').read_text())
        if type(value.get('exit_code')) is not int:
            raise ValueError('Exit marker lacks integer exit_code')
        return {'status': 'valid', 'exit_code': value['exit_code'], 'finished_at': value.get('finished_at')}
    except FileNotFoundError:
        return {'status': 'absent'}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {'status': 'unknown', 'error_type': type(exc).__name__}


def assess(probe, previous=None, idle_seconds=1800, minimum_probe_gap=60):
    worker = probe['worker']
    marker = probe['exit_marker']
    latest = probe['progress']['latest']
    last_progress = latest['modified_unix'] if latest else probe.get('started_unix')
    idle = max(0, probe['observed_unix'] - last_progress) if last_progress is not None else None
    gpu_values = [row['utilization_percent'] for row in probe['gpu'].get('devices', []) if row.get('utilization_percent') is not None]
    gpu_busy = any(value > 0 for value in gpu_values) if gpu_values else None
    same_identity = bool(previous and all(previous.get('worker', {}).get(key) == worker.get(key)
                         for key in ('pid', 'start_ticks', 'boot_id')))
    previous_latest = (previous or {}).get('progress', {}).get('latest')
    repeated = bool(same_identity and idle is not None and idle >= idle_seconds and
                    probe['observed_unix'] - previous.get('observed_unix', probe['observed_unix']) >= minimum_probe_gap and
                    previous.get('health', {}).get('idle_suspect') and previous_latest == latest)
    state = probe.get('supervisor_state')
    confirmed_exit = worker['status'] == 'dead' or state in ('finished', 'launch_failed')
    suspicious = worker['status'] == 'alive' and idle is not None and idle >= idle_seconds
    if confirmed_exit:
        status = 'finished' if marker['status'] == 'valid' else 'worker_exited_without_exit_marker'
    elif worker['status'] == 'unknown':
        status = 'unknown'
    elif marker['status'] == 'valid':
        status = 'finishing'
    elif probe.get('supervisor', {}).get('status') == 'dead' and probe.get('mode') == 'supervised':
        status = 'orphaned_worker'
    elif suspicious and gpu_busy:
        status = 'busy_without_recent_progress'
    elif repeated:
        status = 'idle_suspect_confirmed'
    elif suspicious:
        status = 'idle_suspect'
    else:
        status = 'running'
    return {'status': status, 'worker_exit_confirmed': confirmed_exit, 'idle_seconds': idle,
            'idle_suspect': suspicious, 'repeated_idle_suspect': repeated, 'gpu_busy': gpu_busy,
            'needs_inspection': status in ('unknown', 'orphaned_worker', 'idle_suspect_confirmed', 'worker_exited_without_exit_marker'),
            'idle_authorizes_termination': False}


def probe_worker(output, previous=None, include_gpu=True, idle_seconds=1800):
    output = Path(output).resolve()
    observed = time.time()
    probe = {'schema_version': SCHEMA_VERSION, 'observed_at': timestamp(), 'observed_unix': observed,
             'output': str(output), 'mode': 'unknown', 'supervisor_state': None,
             'supervisor': {'status': 'unknown'}, 'worker': {'status': 'unknown', 'reason': 'missing_worker_identity'},
             'started_unix': None}
    try:
        state = json.loads((output / 'worker-state.json').read_text())
        if state.get('schema_version') != SCHEMA_VERSION or not isinstance(state.get('supervisor'), dict):
            raise ValueError('Invalid supervisor state')
        probe.update(mode='supervised', supervisor_state=state.get('state'),
                     started_unix=state.get('started_unix'), exit_code=state.get('exit_code'))
        probe['supervisor'] = process_observation(state['supervisor'].get('pid'), state['supervisor'])
        if isinstance(state.get('worker'), dict):
            probe['worker'] = process_observation(state['worker'].get('pid'), state['worker'])
        elif state.get('state') in ('finished', 'launch_failed'):
            probe['worker'] = {'status': 'dead', 'reason': 'supervisor_recorded_exit'}
    except FileNotFoundError:
        try:
            legacy_path = output / 'worker.pid'
            pid = int(legacy_path.read_text().strip())
            probe.update(mode='legacy', started_unix=legacy_path.stat().st_mtime)
            probe['worker'] = process_observation(pid)
        except (OSError, ValueError) as exc:
            probe['identity_error_type'] = type(exc).__name__
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        probe['identity_error_type'] = type(exc).__name__
    probe['exit_marker'] = read_exit_marker(output)
    probe['progress'] = meaningful_progress(output)
    probe['gpu'] = gpu_observation() if include_gpu else {'status': 'not_requested', 'devices': []}
    probe['worker_status'] = probe['worker']['status']
    probe['health'] = assess(probe, previous, idle_seconds)
    return probe


def stop_worker(output):
    """Forward TERM through a verified supervisor; never signal a recycled PID."""
    output = Path(output).resolve()
    try:
        state = json.loads((output / 'worker-state.json').read_text())
        expected = state['supervisor']
        if state.get('schema_version') != SCHEMA_VERSION or not isinstance(expected, dict):
            raise ValueError('Invalid supervisor identity')
        observed = process_observation(expected.get('pid'), expected)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'ok': False, 'status': 'refused_unverified_identity', 'error_type': type(exc).__name__}
    if observed['status'] == 'dead':
        child = state.get('worker') or {}
        child_observed = process_observation(child.get('pid'), child)
        if state.get('state') in ('finished', 'launch_failed') or child_observed['status'] == 'dead':
            return {'ok': True, 'status': 'already_exited'}
        return {'ok': False, 'status': 'supervisor_absent_worker_not_confirmed_exited'}
    if observed['status'] != 'alive' or not observed.get('identity_verified'):
        return {'ok': False, 'status': 'refused_unverified_identity'}
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        return {'ok': False, 'status': 'safe_pidfd_signaling_unavailable'}
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['pid'], 0)
        # Recheck after obtaining the pidfd. If PID reuse happened before open,
        # refuse; if it happens afterwards, pidfd still targets the original task.
        rechecked = process_observation(expected['pid'], expected)
        if rechecked['status'] != 'alive' or not rechecked.get('identity_verified'):
            return {'ok': False, 'status': 'identity_changed_before_signal'}
        signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
        return {'ok': True, 'status': 'term_forward_requested', 'pid': expected['pid'], 'signal': 'SIGTERM'}
    except ProcessLookupError:
        return {'ok': True, 'status': 'supervisor_exited_before_signal'}
    except OSError as exc:
        return {'ok': False, 'status': 'signal_unconfirmed', 'error_type': type(exc).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    launch = commands.add_parser('run')
    launch.add_argument('--entry', type=Path, required=True)
    launch.add_argument('--output', type=Path, default=Path('output'))
    launch.add_argument('--cwd', type=Path, default=Path.cwd())
    probe = commands.add_parser('probe')
    probe.add_argument('--output', '--directory', dest='output', type=Path, default=Path('output'))
    probe.add_argument('--previous', type=Path)
    probe.add_argument('--idle-seconds', type=float, default=1800)
    probe.add_argument('--no-gpu', action='store_true')
    stop = commands.add_parser('stop')
    stop.add_argument('--output', '--directory', dest='output', type=Path, default=Path('output'))
    args = parser.parse_args()
    if args.command == 'run':
        raise SystemExit(run_worker(args.entry, args.output, args.cwd))
    if args.command == 'stop':
        result = stop_worker(args.output)
        print(json.dumps(result, separators=(',', ':')))
        raise SystemExit(0 if result['ok'] else 2)
    previous = json.loads(args.previous.read_text()) if args.previous else None
    print(json.dumps(probe_worker(args.output, previous, not args.no_gpu, args.idle_seconds), separators=(',', ':')))


if __name__ == '__main__':
    main()
