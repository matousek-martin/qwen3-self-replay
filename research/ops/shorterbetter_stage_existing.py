"""Resume staging on an existing, never-started authorized ShorterBetter pod.

No allocation path. Preserve the sealed payload and fetch identical public wheels
directly to local worker disk, verified against its original SHA256SUMS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cloud
from experiments.shorterbetter.budget import ticket
from experiments.shorterbetter.hardware import verify_worker
from ops.isolated_cloud import verify

FETCH = r'''
import concurrent.futures, hashlib, json, os, pathlib, time, urllib.request
manifest = pathlib.Path('/opt/thinkingcap-wheels/SHA256SUMS')
def fetch(line):
    expected, filename = line.split()
    filename = filename.lstrip('*')
    if pathlib.Path(filename).name != filename or not filename.endswith('.whl'):
        raise ValueError('Unexpected wheel filename')
    path = manifest.parent/filename
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    package, version = filename.split('-')[:2]
    metadata = json.load(urllib.request.urlopen('https://pypi.org/pypi/'+package+'/'+version+'/json', timeout=30))
    match = next(row for row in metadata['urls'] if row['filename'] == filename)
    if match['digests']['sha256'] != expected or not match['url'].startswith('https://files.pythonhosted.org/'):
        raise ValueError('Public artifact differs from sealed wheel: '+filename)
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(match['url'], timeout=120) as response, path.with_suffix('.partial').open('wb') as out:
                while chunk := response.read(8*1024*1024):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError('Downloaded checksum mismatch')
            path.with_suffix('.partial').replace(path)
            print(filename, 'verified', flush=True)
            return
        except Exception:
            if attempt == 2: raise
            time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, manifest.read_text().splitlines()))
print('All original wheel checksums verified', flush=True)
'''


def main(directory):
    launch = json.loads((directory/'launch.json').read_text())
    name = launch['name'] if 'name' in launch else launch['job_name']
    release = Path(launch['release'])
    payload_hash = verify(release)
    job = json.loads((cloud.STATE/'ledger.json').read_text())['jobs'][name]
    if (job.get('state') != 'starting' or job.get('released_at') or job.get('delete_requested_at')
            or not ticket(cloud.STATE, name, job)):
        raise RuntimeError('Only the existing authorized never-started job can resume staging')
    pod = cloud.api('pods/'+job['pod_id'])
    cloud.remote(pod, 'test ! -e /workspace/project/output/worker-state.json && test ! -e /workspace/project/output/worker.pid && test ! -e /workspace/project/output/exit.json')
    names = cloud.remote(pod, 'nvidia-smi --query-gpu=name --format=csv,noheader').strip().splitlines()
    verify_worker(names, dict(job, job_name=name))
    recovery = {'pod_id': job['pod_id'], 'payload_sha256': payload_hash,
                'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'actual_gpus': names, 'started_at': time.time(), 'method': 'public_wheels_exact_sha256'}
    cloud.write(directory/'staging-recovery.json', recovery)
    ended = threading.Event()
    def heartbeat():
        while not ended.wait(30):
            cloud.update(name, heartbeat=time.time())
            if not cloud.renew_budget(name):
                print('Lease could not be renewed; guard retains control', flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        cloud.update(name, heartbeat=time.time())
        cloud.sync(pod, str(release/'lab/experiments/data')+'/', '/workspace/project/data/',
                   upload=True, exclusions=('wheels',), timeout=300)
        cloud.remote(pod, 'mkdir -p /opt/thinkingcap-wheels')
        cloud.sync(pod, release/'lab/experiments/data/wheels/SHA256SUMS', '/opt/thinkingcap-wheels/SHA256SUMS', upload=True)
        helper = directory/'fetch-pinned-wheels.py'
        helper.write_text(FETCH)
        cloud.sync(pod, helper, '/opt/fetch-pinned-wheels.py', upload=True)
        print(cloud.remote(pod, 'python3 /opt/fetch-pinned-wheels.py', timeout=1800), flush=True)
        cloud.remote(pod, 'mv /workspace/project/data/wheels /workspace/project/data/wheels-incomplete-upload && ln -s /opt/thinkingcap-wheels /workspace/project/data/wheels')
        source = cloud.STATE/name/'source'
        shutil.copy2(release/'lab/gpu-requirements.lock', source/'requirements.lock')
        cloud.write(source/'manifest.json', {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(source.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
        cloud.sync(pod, source/'requirements.lock', '/workspace/project/requirements.lock', upload=True)
        cloud.sync(pod, Path('/root/.local/bin/uv').resolve(), '/workspace/project/uv', upload=True, timeout=300)
        lease = {k: job.get(k) for k in ('pod_id','created_at','deadline','hard_deadline','max_cost_usd',
                 'budget_rate','gpu','hardware_scope','gpu_count','completion_budget_scope')}
        lease['job_name'] = name
        cloud.write(source/'worker-lease.json', lease)
        cloud.sync(pod, source/'worker-lease.json', '/workspace/project/worker-lease.json', upload=True)
        # Policy checks physical GPUs before setup; package checksums remain in the
        # original worker entrypoint. Never change its immutable experiment code.
        cloud.remote(pod, cloud.launch_command(job['entry'], 3600, completion_driven=True))
        cloud.update(name, state='running', heartbeat=time.time(), staging_recovery_finished=time.time())
        recovery['worker_launched_at'] = time.time()
        cloud.write(directory/'staging-recovery.json', recovery)
    except BaseException as exc:
        cloud.update(name, error='Staging recovery: '+str(exc))
        try:
            cloud.backup(pod, name, timeout=60)
        finally:
            cloud.release(name, 'failed')
        raise
    finally:
        ended.set()
    cloud.watch(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    main(parser.parse_args().directory.resolve())
